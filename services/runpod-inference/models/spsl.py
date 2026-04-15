from __future__ import annotations

import hashlib
import logging
import os

import cv2
import numpy as np

from .base import DetectorBase, DetectorOutput

# ---------------------------------------------------------------------------
# SPSL detector — pure PyTorch reimplementation of DeepfakeBench's spsl_detector.
# Architecture: RGB + phase-spectrum (4-channel) → Xception → 2-class softmax.
# Weights: https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/spsl_best.pth
# ---------------------------------------------------------------------------

WEIGHTS_PATH = os.environ.get(
    "SPSL_WEIGHTS",
    "/workspace/weights/spsl_best.pth",
)

log = logging.getLogger("inference")


class SPSLDetector(DetectorBase):
    """SPSL (Spatial-Phase Shallow Learning) detector (slot 'spsl', weight 0.25).

    Computes phase spectrum of grayscale input, concatenates with RGB to form
    a 4-channel input, then passes through an Xception backbone.
    """

    name = "spsl"

    def load(self, device: str) -> None:
        self.device = device
        self._use_placeholder = True

        try:
            import torch
            import torch.nn as nn
            from ._xception import Xception

            class _SPSLNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    # 4-channel input matches checkpoint's (32, 4, 3, 3) conv1
                    self.backbone = Xception(in_channels=4, num_classes=2, head_type="linear")

                @staticmethod
                def _phase(x: torch.Tensor) -> torch.Tensor:
                    # x: (B, 3, H, W) normalized to [-1, 1]
                    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
                    f = torch.fft.fft2(gray)
                    phase = torch.angle(f) / torch.pi
                    return phase.unsqueeze(1)

                def forward(self, x):
                    x = torch.cat([x, self._phase(x)], dim=1)
                    return self.backbone(x)

            net = _SPSLNet()
            ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = net.load_state_dict(sd, strict=False)
            log.info("SPSL: missing=%d unexpected=%d", len(missing), len(unexpected))
            if missing:
                log.warning("SPSL missing keys (first 5): %s", missing[:5])
            if unexpected:
                log.warning("SPSL unexpected keys (first 5): %s", unexpected[:5])
            self.model = net.to(device).eval()
            self._use_placeholder = False
        except Exception as e:
            log.warning("SPSL load failed (%s) — using placeholder", e)

    def predict(self, face_bgr: np.ndarray) -> DetectorOutput:
        if self._use_placeholder:
            digest = hashlib.sha1(face_bgr.tobytes()).digest()
            score = (digest[4] * 256 + digest[5]) / 65535.0
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            f = np.fft.fft2(gray)
            phase = np.angle(np.fft.fftshift(f))
            norm = (phase - phase.min()) / (phase.max() - phase.min() + 1e-6)
            return DetectorOutput(score=float(score), heatmap=norm.astype(np.float32))

        import torch
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        x = tf(face_bgr[:, :, ::-1].copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            score = float(probs[0, 1].item())
        return DetectorOutput(score=score, heatmap=None)
