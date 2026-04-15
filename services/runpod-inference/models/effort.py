from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .base import DetectorBase, DetectorOutput

# ---------------------------------------------------------------------------
# Xception detector — wrapper around DeepfakeBench's xception_detector.py
# Weights: https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth
# ---------------------------------------------------------------------------

WEIGHTS_PATH = "/volume/weights/deepfakebench/training/weights/xception_best.pth"
DEEPFAKEBENCH_ROOT = "/volume/weights/deepfakebench"


class EffortDetector(DetectorBase):
    """Xception-based detector (slot originally named 'effort', weight 0.40).

    Placeholder is active when the weights file is absent so the pipeline
    runs end-to-end in local dev without GPU.
    """

    name = "effort"

    def load(self, device: str) -> None:
        self.device = device
        self._use_placeholder = True

        try:
            import sys, torch
            sys.path.insert(0, DEEPFAKEBENCH_ROOT)
            from training.detectors.xception_detector import XceptionDetector
            self.model = XceptionDetector()
            ckpt = torch.load(WEIGHTS_PATH, map_location=device)
            self.model.load_state_dict(ckpt, strict=False)
            self.model.to(device).eval()
            self._use_placeholder = False
        except Exception as e:
            import logging
            logging.getLogger("inference").warning("Xception load failed (%s) — using placeholder", e)

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
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        x = tf(face_bgr[:, :, ::-1].copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = float(torch.sigmoid(self.model(x)).item())
        return DetectorOutput(score=score, heatmap=None)
