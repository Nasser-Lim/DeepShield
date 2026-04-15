from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .base import DetectorBase, DetectorOutput

# ---------------------------------------------------------------------------
# SPSL detector — wrapper around DeepfakeBench's spsl_detector.py
# Weights: https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/spsl_best.pth
# ---------------------------------------------------------------------------

WEIGHTS_PATH = "/volume/weights/deepfakebench/training/weights/spsl_best.pth"
DEEPFAKEBENCH_ROOT = "/volume/weights/deepfakebench"


class SPSLDetector(DetectorBase):
    """SPSL spectral-phase detector (weight 0.25).

    Placeholder is active when the weights file is absent so the pipeline
    runs end-to-end in local dev without GPU.
    """

    name = "spsl"

    def load(self, device: str) -> None:
        self.device = device
        self._use_placeholder = True

        try:
            import sys, torch
            sys.path.insert(0, DEEPFAKEBENCH_ROOT)
            from training.detectors.spsl_detector import SPSLDetector as Net
            self.model = Net()
            ckpt = torch.load(WEIGHTS_PATH, map_location=device)
            self.model.load_state_dict(ckpt, strict=False)
            self.model.to(device).eval()
            self._use_placeholder = False
        except Exception as e:
            import logging
            logging.getLogger("inference").warning("SPSL load failed (%s) — using placeholder", e)

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
            score = float(torch.sigmoid(self.model(x)).item())
        return DetectorOutput(score=score, heatmap=None)
