from __future__ import annotations

import hashlib
import logging
import os

import cv2
import numpy as np

from .base import DetectorBase, DetectorOutput

# ---------------------------------------------------------------------------
# Xception detector — pure PyTorch reimplementation matching DeepfakeBench's
# checkpoint key layout. Loads weights only, no DeepfakeBench code dependency.
# Weights: https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth
# ---------------------------------------------------------------------------

WEIGHTS_PATH = os.environ.get(
    "XCEPTION_WEIGHTS",
    "/workspace/weights/xception_best.pth",
)

log = logging.getLogger("inference")


class EffortDetector(DetectorBase):
    """Xception detector (slot 'effort', weight 0.40).

    DeepfakeBench checkpoint uses 2-class softmax head (real=0, fake=1).
    We return softmax probability of the 'fake' class.
    """

    name = "effort"

    def load(self, device: str) -> None:
        self.device = device
        self._use_placeholder = True

        try:
            import torch
            from ._xception import DetectorNet

            net = DetectorNet(in_channels=3, num_classes=2, head_type="linear")
            ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = net.load_state_dict(sd, strict=False)
            log.info("Xception: missing=%d unexpected=%d", len(missing), len(unexpected))
            if missing:
                log.warning("Xception missing keys (first 5): %s", missing[:5])
            if unexpected:
                log.warning("Xception unexpected keys (first 5): %s", unexpected[:5])
            self.model = net.to(device).eval()
            self._use_placeholder = False
        except Exception as e:
            log.warning("Xception load failed (%s) — using placeholder", e)

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
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        x = tf(face_bgr[:, :, ::-1].copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            score = float(probs[0, 1].item())  # fake class probability
        return DetectorOutput(score=score, heatmap=None)
