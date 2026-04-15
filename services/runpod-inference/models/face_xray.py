from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .base import DetectorBase, DetectorOutput

# ---------------------------------------------------------------------------
# F3Net detector — wrapper around DeepfakeBench's f3net_detector.py
# Weights: https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/f3net_best.pth
# ---------------------------------------------------------------------------

WEIGHTS_PATH = "/volume/weights/deepfakebench/training/weights/f3net_best.pth"
DEEPFAKEBENCH_ROOT = "/volume/weights/deepfakebench"


class FaceXrayDetector(DetectorBase):
    """F3Net frequency-domain detector (slot originally named 'xray', weight 0.35).

    Placeholder is active when the weights file is absent so the pipeline
    runs end-to-end in local dev without GPU.
    """

    name = "xray"

    def load(self, device: str) -> None:
        self.device = device
        self._use_placeholder = True

        try:
            import sys, torch
            sys.path.insert(0, DEEPFAKEBENCH_ROOT)
            from training.detectors.f3net_detector import F3NetDetector
            self.model = F3NetDetector()
            ckpt = torch.load(WEIGHTS_PATH, map_location=device)
            self.model.load_state_dict(ckpt, strict=False)
            self.model.to(device).eval()
            self._use_placeholder = False
        except Exception as e:
            import logging
            logging.getLogger("inference").warning("F3Net load failed (%s) — using placeholder", e)

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
            score = float(torch.sigmoid(self.model(x)).item())
        return DetectorOutput(score=score, heatmap=None)
