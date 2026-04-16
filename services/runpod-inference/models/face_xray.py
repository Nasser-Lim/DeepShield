from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from .base import DetectorBase, DetectorOutput

# ---------------------------------------------------------------------------
# UnivFD detector — Universal Fake Image Detector (CVPR 2023, Ojha et al.)
# Repo: https://github.com/WisconsinAIVision/UniversalFakeDetect
#
# Architecture: frozen CLIP ViT-L/14 visual encoder + Linear(768, 1) probe
# Trained with JPEG/blur augmentation → robust to compressed press photos.
# Weights file: fc_weights.pth (~3KB — head-only, 768→1 linear layer)
#
# Slot name "xray" kept for API schema compatibility; semantically this is
# now UnivFD, not FatFormer (see docs/runpod-setup.md).
# ---------------------------------------------------------------------------

WEIGHTS_PATH = os.environ.get(
    "UNIVFD_WEIGHTS",
    "/workspace/ds_weights/univfd_fc_weights.pth",
)

log = logging.getLogger("inference")

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


class FaceXrayDetector(DetectorBase):
    """UnivFD detector (slot 'xray')."""

    name = "xray"

    def load(self, device: str) -> None:
        self.device = device
        self._use_placeholder = True

        try:
            import torch
            import torch.nn as nn
            from transformers import CLIPModel

            # UnivFD uses the CLIP visual trunk only (no text encoder, no
            # visual_projection). HuggingFace CLIPModel.vision_model returns
            # pooled hidden state of dim 1024 → the UnivFD fc is 768→1, so
            # they feed visual_projection output (768) into fc.
            # We use get_image_features() which runs vision_model +
            # visual_projection → (B, 768).
            class _UnivFD(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.clip = CLIPModel.from_pretrained(
                        "openai/clip-vit-large-patch14",
                        ignore_mismatched_sizes=True,
                    )
                    self.fc = nn.Linear(768, 1)

                def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                    out = self.clip.get_image_features(pixel_values=pixel_values)
                    feats = out if isinstance(out, torch.Tensor) else out.pooler_output
                    return self.fc(feats)  # (B, 1) logit → sigmoid

            net = _UnivFD()

            # fc_weights.pth is head-only (fc.weight / fc.bias).
            ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
            else:
                sd = ckpt

            # Accept both {fc.weight, fc.bias} and bare {weight, bias} layouts.
            remapped = {}
            for k, v in sd.items():
                if k in ("weight", "bias"):
                    remapped[f"fc.{k}"] = v
                elif k.startswith("fc."):
                    remapped[k] = v
                else:
                    remapped[k] = v
            missing, unexpected = net.load_state_dict(remapped, strict=False)

            # CLIP backbone keys are expected to be missing (head-only ckpt);
            # fc.weight/bias must be present.
            head_loaded = "fc.weight" not in missing and "fc.bias" not in missing
            if not head_loaded:
                raise RuntimeError(
                    f"UnivFD fc head not found in checkpoint — keys: {list(sd.keys())[:5]}"
                )
            log.info(
                "UnivFD: head loaded, backbone missing=%d unexpected=%d",
                len(missing), len(unexpected),
            )
            self.model = net.to(device).eval()
            self._use_placeholder = False
            log.info("UnivFD: loaded ok")
        except Exception as e:
            log.warning("UnivFD load failed (%s) — using placeholder", e)

    def predict(self, face_bgr: np.ndarray) -> DetectorOutput:
        if self._use_placeholder:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
            hp = np.abs(gray - blur)
            return DetectorOutput(score=0.5, heatmap=hp / (hp.max() + 1e-6))

        import torch
        from torchvision import transforms

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])
        x = tf(face_bgr[:, :, ::-1].copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(x)                    # (1, 1)
            score = float(torch.sigmoid(logit)[0, 0].item())
        return DetectorOutput(score=score, heatmap=None)
