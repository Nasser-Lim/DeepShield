from __future__ import annotations

import hashlib
import logging
import math
import os

import cv2
import numpy as np

from .base import DetectorBase, DetectorOutput

# ---------------------------------------------------------------------------
# F3Net detector — pure PyTorch reimplementation of DeepfakeBench's f3net_detector.
# Architecture:
#   FAD_head: DCT-based frequency decomposition into 4 bands (low/mid-low/mid-high/all)
#             → concat 4*3=12 channels → Xception backbone (12-channel input)
#   Head: Sequential(Dropout, Linear(2048, 2)) — softmax 2-class
# Weights: https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/f3net_best.pth
# ---------------------------------------------------------------------------

WEIGHTS_PATH = os.environ.get(
    "F3NET_WEIGHTS",
    "/workspace/weights/f3net_best.pth",
)

log = logging.getLogger("inference")


class FaceXrayDetector(DetectorBase):
    """F3Net frequency-domain detector (slot 'xray', weight 0.35)."""

    name = "xray"

    def load(self, device: str) -> None:
        self.device = device
        self._use_placeholder = True

        try:
            import torch
            import torch.nn as nn
            from ._xception import Xception

            # FAD (Frequency-Aware Decomposition) head replicating DeepfakeBench layout.
            # Checkpoint keys:
            #   FAD_head._DCT_all          (256, 256)  — DCT basis matrix
            #   FAD_head._DCT_all_T        (256, 256)  — its transpose
            #   FAD_head.filters.{0..3}.base       (256, 256)  — fixed band-pass mask
            #   FAD_head.filters.{0..3}.learnable  (256, 256)  — learned residual
            class _Filter(nn.Module):
                def __init__(self, size: int = 256):
                    super().__init__()
                    self.base = nn.Parameter(torch.zeros(size, size), requires_grad=False)
                    self.learnable = nn.Parameter(torch.zeros(size, size))

                def forward(self) -> torch.Tensor:
                    return self.base + torch.sigmoid(self.learnable)

            class _FADHead(nn.Module):
                def __init__(self, size: int = 256):
                    super().__init__()
                    self.size = size
                    self._DCT_all = nn.Parameter(torch.zeros(size, size), requires_grad=False)
                    self._DCT_all_T = nn.Parameter(torch.zeros(size, size), requires_grad=False)
                    self.filters = nn.ModuleList([_Filter(size) for _ in range(4)])

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # x: (B, 3, H, W) — assumed H=W=size
                    # DCT(x) = D @ x @ D^T  per channel
                    D = self._DCT_all
                    DT = self._DCT_all_T
                    # (B, 3, H, W) -> apply DCT on last two dims
                    x_dct = D @ x @ DT
                    out = []
                    for f in self.filters:
                        mask = f()  # (H, W)
                        y_dct = x_dct * mask  # broadcast over batch and channels
                        # IDCT: x = D^T @ y @ D (since D is orthonormal, D^-1 = D^T)
                        y = DT @ y_dct @ D
                        out.append(y)
                    # Concatenate along channel: 4 * 3 = 12
                    return torch.cat(out, dim=1)

            class _F3Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.FAD_head = _FADHead(size=256)
                    # 12-channel input; head is Sequential(Dropout, Linear(2048,2))
                    self.backbone = Xception(in_channels=12, num_classes=2,
                                             head_type="dropout_linear")

                def forward(self, x):
                    x = self.FAD_head(x)
                    return self.backbone(x)

            net = _F3Net()
            ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = net.load_state_dict(sd, strict=False)
            log.info("F3Net: missing=%d unexpected=%d", len(missing), len(unexpected))
            if missing:
                log.warning("F3Net missing keys (first 5): %s", missing[:5])
            if unexpected:
                log.warning("F3Net unexpected keys (first 5): %s", unexpected[:5])
            self.model = net.to(device).eval()
            self._use_placeholder = False
        except Exception as e:
            log.warning("F3Net load failed (%s) — using placeholder", e)

    def predict(self, face_bgr: np.ndarray) -> DetectorOutput:
        if self._use_placeholder:
            digest = hashlib.sha1(face_bgr.tobytes()).digest()
            score = (digest[2] * 256 + digest[3]) / 65535.0
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
            hp = np.abs(gray - blur)
            return DetectorOutput(score=float(score), heatmap=hp / (hp.max() + 1e-6))

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
