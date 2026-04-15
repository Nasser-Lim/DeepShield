from __future__ import annotations

import hashlib
import logging
import os

import cv2
import numpy as np

from .base import DetectorBase, DetectorOutput

# ---------------------------------------------------------------------------
# SBI detector — Self-Blended Images (CVPR 2022, Shiohara & Yamasaki)
# Architecture: EfficientNet-B4 (timm tf_efficientnet_b4) → sigmoid binary head
# Weights: https://github.com/mapooon/SelfBlendedImages (FFraw.pth ~135MB)
#
# Probe result: keys use "net._conv_stem.*" (underscore prefix = timm <=0.6 style)
# timm.create_model("tf_efficientnet_b4") generates these underscore keys.
# Attribute name must be "net" to match checkpoint prefix.
# ---------------------------------------------------------------------------

WEIGHTS_PATH = os.environ.get(
    "SBI_WEIGHTS",
    "/workspace/weights/sbi_best.pth",
)

log = logging.getLogger("inference")


class EffortDetector(DetectorBase):
    """SBI detector (slot 'effort', weight 0.40)."""

    name = "effort"

    def load(self, device: str) -> None:
        self.device = device
        self._use_placeholder = True

        try:
            import torch
            import torch.nn as nn
            import timm

            class _SBINet(nn.Module):
                def __init__(self):
                    super().__init__()
                    # "tf_efficientnet_b4" produces _conv_stem / _bn0 key names
                    # matching the SBI checkpoint (trained with timm <=0.6).
                    # Attribute must be "net" to match "net.*" checkpoint keys.
                    self.net = timm.create_model(
                        "tf_efficientnet_b4", pretrained=False, num_classes=1
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.net(x)  # (B, 1) logit

            net = _SBINet()
            ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
            else:
                sd = ckpt
            missing, unexpected = net.load_state_dict(sd, strict=False)
            log.info("SBI: missing=%d unexpected=%d", len(missing), len(unexpected))
            if missing:
                log.warning("SBI missing keys (first 5): %s", missing[:5])
            self.model = net.to(device).eval()
            self._use_placeholder = False
            log.info("SBI: loaded ok")
        except Exception as e:
            log.warning("SBI load failed (%s) — using placeholder", e)

    def predict(self, face_bgr: np.ndarray) -> DetectorOutput:
        if self._use_placeholder:
            digest = hashlib.sha1(face_bgr.tobytes()).digest()
            score = (digest[0] * 256 + digest[1]) / 65535.0
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
            return DetectorOutput(score=float(score), heatmap=lap / (lap.max() + 1e-6))

        import torch
        from torchvision import transforms

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        x = tf(face_bgr[:, :, ::-1].copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(x)
            score = float(torch.sigmoid(logit)[0, 0].item())
        return DetectorOutput(score=score, heatmap=None)
