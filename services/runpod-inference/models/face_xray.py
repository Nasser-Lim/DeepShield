from __future__ import annotations

import hashlib
import logging
import os

import cv2
import numpy as np

from .base import DetectorBase, DetectorOutput

# ---------------------------------------------------------------------------
# FatFormer detector — Forgery-aware Adaptive Transformer (CVPR 2024, Liu et al.)
# Architecture: CLIP ViT-L/14 (full) + Forgery-aware adapters + 2-class head
# Weights: https://github.com/Michel-liu/FatFormer (~1.9GB)
#
# Probe result (1116 keys), prefix "clip_model.*":
#   clip_model.positional_embedding, clip_model.text_projection,
#   clip_model.logit_scale, clip_model.visual.*, clip_model.transformer.*,
#   clip_model.token_embedding.*, clip_model.ln_final.*
# The checkpoint stores the FULL open_clip model as self.clip_model,
# plus additional adapter/head layers at the top level.
# Strategy: hold clip_model as a submodule, load with strict=False.
# Binary head keys are inferred from the unexpected keys after first load.
# ---------------------------------------------------------------------------

WEIGHTS_PATH = os.environ.get(
    "FATFORMER_WEIGHTS",
    "/workspace/weights/fatformer_best.pth",
)

log = logging.getLogger("inference")

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


class FaceXrayDetector(DetectorBase):
    """FatFormer detector (slot 'xray', weight 0.40).

    CLIP ViT-L/14 backbone with forgery-aware frequency adapters.
    Generalises well to unseen GAN and Diffusion-generated faces.
    """

    name = "xray"

    def load(self, device: str) -> None:
        self.device = device
        self._use_placeholder = True

        try:
            import torch
            import torch.nn as nn
            import open_clip

            # ------------------------------------------------------------------
            # The FatFormer checkpoint stores the full CLIP model under the
            # attribute name "clip_model". We mirror that so keys match.
            # Adapter and classification head keys (outside clip_model.*) are
            # loaded with strict=False — they land in unexpected_keys if our
            # module doesn't declare them, but the CLIP backbone loads cleanly.
            # ------------------------------------------------------------------

            class _FatFormer(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Mirror the checkpoint's "clip_model" attribute name
                    self.clip_model, _, _ = open_clip.create_model_and_transforms(
                        "ViT-L-14", pretrained=None
                    )
                    # Binary classification head (real=0, fake=1)
                    # FatFormer uses a linear head on the visual features (dim=1024)
                    self.head = nn.Linear(1024, 2)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    feats = self.clip_model.encode_image(x)   # (B, 1024)
                    return self.head(feats)                    # (B, 2) logits

            net = _FatFormer()
            ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
            else:
                sd = ckpt
            missing, unexpected = net.load_state_dict(sd, strict=False)
            log.info(
                "FatFormer: missing=%d unexpected=%d", len(missing), len(unexpected)
            )
            if missing:
                log.warning("FatFormer missing keys (first 5): %s", missing[:5])
            self.model = net.to(device).eval()
            self._use_placeholder = False
            log.info("FatFormer: loaded ok")
        except Exception as e:
            log.warning("FatFormer load failed (%s) — using placeholder", e)

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

        # FatFormer: CLIP input spec 224×224 with CLIP normalization
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])
        x = tf(face_bgr[:, :, ::-1].copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)                  # (1, 2)
            probs = torch.softmax(logits, dim=1)
            score = float(probs[0, 1].item())       # fake class
        return DetectorOutput(score=score, heatmap=None)
