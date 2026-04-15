from __future__ import annotations

import base64
from io import BytesIO

import cv2
import httpx
import numpy as np
from PIL import Image


def load_image_from_url(url: str, timeout: float = 15.0) -> np.ndarray:
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url)
        resp.raise_for_status()
    return _bytes_to_bgr(resp.content)


def load_image_from_b64(b64: str) -> np.ndarray:
    # Allow data-URL prefix
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return _bytes_to_bgr(base64.b64decode(b64))


def encode_png_b64(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _bytes_to_bgr(data: bytes) -> np.ndarray:
    img = Image.open(BytesIO(data)).convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
