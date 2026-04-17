from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
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
            use_fp16=self.use_fp16,
            diffusion_steps=1000,
            noise_schedule="linear",
            timestep_respacing=respacing,
            use_ddim=True,
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
        if self.use_fp16:
            self.model.convert_to_fp16()
        for p in self.model.parameters():
            p.requires_grad_(False)

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

        x = self._preprocess_for_adm(image_bgr).to(self.device)
        if self.use_fp16:
            x = x.half()

        latent = self.diffusion.ddim_reverse_sample_loop(
            self.model, x, clip_denoised=True,
        )
        recon = self.diffusion.ddim_sample_loop(
            self.model, x.shape, noise=latent, clip_denoised=True,
        )

        dire = (x.float() - recon.float()).abs().clamp(0.0, 1.0)

        prob = self._classify(dire)

        heatmap = dire.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        hm_max = float(heatmap.max()) if heatmap.size else 0.0
        if hm_max > 0:
            heatmap = heatmap / hm_max

        return DetectorOutput(score=float(prob), heatmap=heatmap)

    def _preprocess_for_adm(self, image_bgr: np.ndarray) -> torch.Tensor:
        h, w = image_bgr.shape[:2]
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        sq = image_bgr[top:top + side, left:left + side]
        sq = cv2.resize(sq, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()

    def _classify(self, dire: torch.Tensor) -> float:
        x = F.interpolate(dire, size=(256, 256), mode="bilinear", align_corners=False)
        off = (256 - 224) // 2
        x = x[:, :, off:off + 224, off:off + 224]
        mean = IMAGENET_MEAN.to(x.device, x.dtype)
        std = IMAGENET_STD.to(x.device, x.dtype)
        x = (x - mean) / std
        logit = self.classifier(x)
        return torch.sigmoid(logit).flatten()[0].item()
