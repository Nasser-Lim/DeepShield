from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .base import DetectorBase, DetectorOutput


class SPSLDetector(DetectorBase):
    """Spatial-Phase Shallow Learning detector (placeholder).

    SPSL relies on the phase spectrum of the Fourier transform. We compute
    it here for the visualisation; the numeric score itself is a stable
    hash until real weights are plugged in.
    """

    name = "spsl"

    def load(self, device: str) -> None:
        self.device = device
        # TODO: load SPSL weights
        self._ready = True

    def predict(self, face_bgr: np.ndarray) -> DetectorOutput:
        digest = hashlib.sha1(face_bgr.tobytes()).digest()
        score = (digest[4] * 256 + digest[5]) / 65535.0

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        f = np.fft.fft2(gray)
        phase = np.angle(np.fft.fftshift(f))
        normalised = (phase - phase.min()) / (phase.max() - phase.min() + 1e-6)

        return DetectorOutput(score=float(score), heatmap=normalised.astype(np.float32))
