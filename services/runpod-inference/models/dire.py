from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from .base import DetectorBase, DetectorOutput

log = logging.getLogger("dire")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DireDetector(DetectorBase):
    """DIRE (Diffusion Reconstruction Error) single-model detector.

    Pipeline: image -> ADM DDIM reverse -> DDIM forward reconstruction ->
    |orig - recon| -> ResNet-50 binary classifier -> synthetic probability.

    Reference: compute_dire.py in ZhendongWang6/DIRE
      reverse_fn(model, shape, noise=imgs, clip_denoised=True)
      sample_fn(model, shape, noise=latent, clip_denoised=True)
      dire = th.abs(imgs - recons)
      dire = (dire * 255.0 / 2.0).clamp(0, 255).to(th.uint8)
    """

    name = "dire"

    def __init__(self) -> None:
        self.model = None          # ADM UNet
        self.diffusion = None      # spaced diffusion scheduler
        self.classifier = None     # ResNet-50
        self.device = "cpu"
        self.image_size = int(os.environ.get("DIRE_IMAGE_SIZE", "256"))
        self.use_fp16 = False

    def load(self, device: str) -> None:
        self.device = device
        self.use_fp16 = device == "cuda"

        repo_path = os.environ.get("DIRE_REPO_PATH", "/workspace/dire_v1/repo")
        if repo_path and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
            gd = str(Path(repo_path) / "guided-diffusion")
            if Path(gd).exists() and gd not in sys.path:
                sys.path.insert(0, gd)

        from guided_diffusion.script_util import (
            create_model_and_diffusion,
            model_and_diffusion_defaults,
        )

        respacing = os.environ.get("DIRE_TIMESTEP_RESPACING", "ddim20")
        defaults = model_and_diffusion_defaults()
        defaults.update(dict(
            image_size=self.image_size,
            class_cond=False,
            learn_sigma=True,
            num_channels=256,
            num_head_channels=64,
            num_res_blocks=2,
            attention_resolutions="32,16,8",
            resblock_updown=True,
            use_scale_shift_norm=True,
            use_fp16=False,
            diffusion_steps=1000,
            noise_schedule="linear",
            timestep_respacing=respacing,
        ))
        self.model, self.diffusion = create_model_and_diffusion(**defaults)

        adm_path = os.environ.get(
            "DIRE_ADM_WEIGHTS",
            "/workspace/dire_v1/weights/256x256_diffusion_uncond.pt",
        )
        log.info("Loading ADM weights from %s", adm_path)
        state = torch.load(adm_path, map_location="cpu")
        self.model.load_state_dict(state, strict=False)
        self.model.to(device).eval()
        # NOTE: skip convert_to_fp16() — guided_diffusion's partial fp16
        # conversion leaves the final out conv in float32, causing dtype
        # mismatch at runtime. Float32 is fine for 256x256 on A5000 24GB.
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Classifier: ResNet-50 with fc replaced to Linear(2048, 1)
        # Matches DIRE official: resnet50(num_classes=1) then model.fc = nn.Linear(2048, 1)
        cls_path = os.environ.get(
            "DIRE_CLASSIFIER_WEIGHTS",
            "/workspace/dire_v1/weights/classifier/lsun_adm.pth",
        )
        log.info("Loading classifier weights from %s", cls_path)
        self.classifier = resnet50(num_classes=1)
        sd = torch.load(cls_path, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        missing, unexpected = self.classifier.load_state_dict(sd, strict=False)
        log.info("classifier: missing=%d unexpected=%d", len(missing), len(unexpected))
        self.classifier.to(device).eval()
        for p in self.classifier.parameters():
            p.requires_grad_(False)

        log.info("DireDetector loaded (device=%s, respacing=%s)", device, respacing)

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> DetectorOutput:
        if self.model is None or self.diffusion is None or self.classifier is None:
            raise RuntimeError("DireDetector.load() must be called first")

        # Preprocess: BGR -> RGB -> center crop -> 256x256 -> [-1, 1]
        x = self._preprocess_for_adm(image_bgr).to(self.device)

        shape = x.shape  # (1, 3, 256, 256)

        # DDIM reverse: encode image into latent
        # Official API: ddim_reverse_sample_loop(model, shape, noise=imgs, ...)
        latent = self.diffusion.ddim_reverse_sample_loop(
            self.model, shape, noise=x, clip_denoised=True,
        )

        # DDIM forward: reconstruct from latent
        # Official API: ddim_sample_loop(model, shape, noise=latent, ...)
        recon = self.diffusion.ddim_sample_loop(
            self.model, shape, noise=latent, clip_denoised=True,
        )

        # DIRE map: absolute pixel difference.
        # x and recon are in [-1, 1], so abs diff is in [0, 2].
        # Official: dire = th.abs(imgs - recons) then dire * 255/2 -> uint8
        # We scale to [0, 1] by dividing by 2 to match the uint8/255 normalization.
        dire = (x.float() - recon.float()).abs() / 2.0  # [0, 1]

        # Debug stats
        log.info(
            "DEBUG x: min=%.4f max=%.4f mean=%.4f | recon: min=%.4f max=%.4f mean=%.4f | dire: min=%.4f max=%.4f mean=%.4f",
            x.min().item(), x.max().item(), x.mean().item(),
            recon.min().item(), recon.max().item(), recon.mean().item(),
            dire.min().item(), dire.max().item(), dire.mean().item(),
        )

        # Official demo.py feeds the ORIGINAL image to the classifier with its own
        # preprocessing: PIL Resize(256) -> CenterCrop(224) -> ToTensor -> ImageNet norm.
        # We must re-preprocess image_bgr independently (not reuse the ADM-cropped tensor).
        prob = self._classify_from_bgr(image_bgr)

        # Heatmap for visualization: channel-mean, normalize to [0, 1]
        heatmap = dire.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        hm_max = float(heatmap.max()) if heatmap.size else 0.0
        if hm_max > 0:
            heatmap = heatmap / hm_max

        return DetectorOutput(score=float(prob), heatmap=heatmap)

    def _preprocess_for_adm(self, image_bgr: np.ndarray) -> torch.Tensor:
        """Center-crop to square, resize to image_size, scale to [-1, 1]."""
        h, w = image_bgr.shape[:2]
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        sq = image_bgr[top:top + side, left:left + side]
        sq = cv2.resize(sq, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 127.5 - 1.0  # [0,255] -> [-1, 1]
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()

    def _classify_from_bgr(self, image_bgr: np.ndarray) -> float:
        """ResNet-50 binary classifier on the original image.

        Mirrors official demo.py exactly:
          PIL Resize(256, shortest edge) -> CenterCrop(224) -> ToTensor -> ImageNet norm.
        """
        from PIL import Image as PILImage
        # BGR -> RGB PIL image
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)

        # Resize shortest edge to 256 (same as transforms.Resize(256))
        w, h = pil.size
        if w < h:
            new_w, new_h = 256, int(256 * h / w)
        else:
            new_w, new_h = int(256 * w / h), 256
        pil = pil.resize((new_w, new_h), PILImage.BICUBIC)

        # CenterCrop(224)
        w, h = pil.size
        left = (w - 224) // 2
        top = (h - 224) // 2
        pil = pil.crop((left, top, left + 224, top + 224))

        # ToTensor: [0,255] -> [0,1], then ImageNet normalize
        arr = np.array(pil, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        mean = IMAGENET_MEAN.to(x.device, x.dtype)
        std = IMAGENET_STD.to(x.device, x.dtype)
        x = (x - mean) / std
        x = x.to(self.device)

        logit = self.classifier(x)
        prob = torch.sigmoid(logit).flatten()[0].item()
        log.info("DEBUG classifier: logit=%.4f prob=%.4f", logit.flatten()[0].item(), prob)
        return prob
