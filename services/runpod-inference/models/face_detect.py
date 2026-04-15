from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FaceCrop:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    patch: np.ndarray                # BGR crop


class FaceDetector:
    """OpenCV Haar cascade face detector.

    Chosen for zero-download bootstrap. Swap for MTCNN or YOLOv8-face once
    the pod has those weights baked in. Falls back to the full frame if no
    face is found, so downstream detectors always receive a crop.
    """

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image_bgr: np.ndarray) -> FaceCrop:
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.2, 5, minSize=(64, 64))

        if len(faces) == 0:
            raise ValueError("no_face_detected")

        # Largest face wins
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        # Pad the crop by 20 % on each side, clamped to image bounds
        pad_x, pad_y = int(fw * 0.2), int(fh * 0.2)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + fw + pad_x)
        y1 = min(h, y + fh + pad_y)
        patch = image_bgr[y0:y1, x0:x1].copy()
        return FaceCrop(bbox=(x0, y0, x1 - x0, y1 - y0), patch=patch)
