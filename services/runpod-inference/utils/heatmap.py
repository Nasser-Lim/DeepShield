from __future__ import annotations

import cv2
import numpy as np


def colorize(heatmap: np.ndarray) -> np.ndarray:
    """Map a HxW float32 0..1 heatmap to a BGR uint8 JET visualisation."""
    if heatmap.dtype != np.uint8:
        heatmap = np.clip(heatmap, 0.0, 1.0)
        heatmap = (heatmap * 255).astype(np.uint8)
    return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


def overlay_on_face(face_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    coloured = colorize(heatmap)
    if coloured.shape[:2] != face_bgr.shape[:2]:
        coloured = cv2.resize(coloured, (face_bgr.shape[1], face_bgr.shape[0]))
    return cv2.addWeighted(face_bgr, 1 - alpha, coloured, alpha, 0)
