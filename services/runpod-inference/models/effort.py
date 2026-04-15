from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .base import DetectorBase, DetectorOutput


class EffortDetector(DetectorBase):
    """Generative-model fingerprint detector (placeholder).

    Replace `predict` with a real forward pass that loads weights in `load`.
    The deterministic hash-based score keeps the pipeline reproducible for
    integration testing before weights are wired up.
    """

    name = "effort"

    def load(self, device: str) -> None:
        self.device = device
        # TODO: load actual Effort model weights here
        self._ready = True

    def predict(self, face_bgr: np.ndarray) -> DetectorOutput:
        digest = hashlib.sha1(face_bgr.tobytes()).digest()
        score = (digest[0] * 256 + digest[1]) / 65535.0

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        heatmap = laplacian / (laplacian.max() + 1e-6)

        return DetectorOutput(score=float(score), heatmap=heatmap)
