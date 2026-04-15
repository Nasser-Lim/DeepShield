from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .base import DetectorBase, DetectorOutput


class FaceXrayDetector(DetectorBase):
    """Blending-boundary detector (placeholder).

    Real Face X-ray predicts a boundary map; here we approximate with a
    high-pass response so the UI overlay still reflects local gradients.
    """

    name = "xray"

    def load(self, device: str) -> None:
        self.device = device
        # TODO: load Face X-ray weights
        self._ready = True

    def predict(self, face_bgr: np.ndarray) -> DetectorOutput:
        digest = hashlib.sha1(face_bgr.tobytes()).digest()
        score = (digest[2] * 256 + digest[3]) / 65535.0

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
        high_pass = np.abs(gray - blur)
        heatmap = high_pass / (high_pass.max() + 1e-6)

        return DetectorOutput(score=float(score), heatmap=heatmap)
