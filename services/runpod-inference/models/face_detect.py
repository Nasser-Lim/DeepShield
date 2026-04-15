from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("inference")


@dataclass
class FaceCrop:
    bbox: tuple[int, int, int, int]  # x, y, w, h in original image pixels
    patch: np.ndarray                # BGR crop


class FaceDetector:
    """MTCNN-based face detector with Haar cascade fallback.

    MTCNN handles rotated, partially occluded, and small faces far better
    than Haar. Falls back to Haar only if mtcnn is not importable.
    Raises ValueError("no_face_detected") when no face is found so the
    caller can return a 422 error instead of running meaningless inference.
    """

    def __init__(self) -> None:
        self._mtcnn = None
        try:
            # Suppress TensorFlow verbosity
            import os
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            from mtcnn import MTCNN
            self._mtcnn = MTCNN()
        except Exception as e:
            log.warning("MTCNN unavailable (%s) — falling back to Haar cascade", e)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image_bgr: np.ndarray) -> FaceCrop:
        if self._mtcnn is not None:
            return self._detect_mtcnn(image_bgr)
        return self._detect_haar(image_bgr)

    def _detect_mtcnn(self, image_bgr: np.ndarray) -> FaceCrop:
        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._mtcnn.detect_faces(image_rgb)

        # Filter by confidence threshold
        results = [r for r in results if r["confidence"] >= 0.90]

        if not results:
            raise ValueError("no_face_detected")

        # Largest face (by bbox area) wins
        best = max(results, key=lambda r: r["box"][2] * r["box"][3])
        x, y, fw, fh = best["box"]
        # MTCNN can return negative coords on edge cases
        x, y = max(0, x), max(0, y)

        # Pad 20% on each side, clamped to image bounds
        pad_x, pad_y = int(fw * 0.2), int(fh * 0.2)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + fw + pad_x)
        y1 = min(h, y + fh + pad_y)
        patch = image_bgr[y0:y1, x0:x1].copy()
        return FaceCrop(bbox=(x0, y0, x1 - x0, y1 - y0), patch=patch)

    def _detect_haar(self, image_bgr: np.ndarray) -> FaceCrop:
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.1, 3, minSize=(40, 40))

        if len(faces) == 0:
            raise ValueError("no_face_detected")

        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        pad_x, pad_y = int(fw * 0.2), int(fh * 0.2)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + fw + pad_x)
        y1 = min(h, y + fh + pad_y)
        patch = image_bgr[y0:y1, x0:x1].copy()
        return FaceCrop(bbox=(x0, y0, x1 - x0, y1 - y0), patch=patch)
