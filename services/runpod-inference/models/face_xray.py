from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from .base import DetectorBase, DetectorOutput

# ---------------------------------------------------------------------------
# FatFormer detector — Forgery-aware Adaptive Transformer (CVPR 2024, Liu et al.)
# Architecture: CLIP ViT-L/14 (open_clip) with forgery-aware adapters
# Weights: https://github.com/Michel-liu/FatFormer (~1.9GB)
#
# Probe result — top-level keys (no "clip_model" wrapper):
#   image_encoder.*          — CLIP visual encoder (ViT-L/14)
#   text_encoder.*           — CLIP text encoder
#   language_guided_alignment.* — LGA module (ctx, token_prefix/suffix, MLP)
#   text_guided_interactor.* — cross-attention between text and image features
#   norm1, linear1, linear2, norm2  — 2-layer MLP classification head
#   logit_scale              — CLIP logit scale
#
# Strategy: build an open_clip CLIP model, then rename its submodules to match
# the checkpoint's top-level attribute names before loading.
# ---------------------------------------------------------------------------

WEIGHTS_PATH = os.environ.get(
    "FATFORMER_WEIGHTS",
    "/workspace/ds_weights/fatformer_best.pth",
)

log = logging.getLogger("inference")

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


class FaceXrayDetector(DetectorBase):
    """FatFormer detector (slot 'xray', weight 0.40)."""

    name = "xray"

    def load(self, device: str) -> None:
        self.device = device
        self._use_placeholder = True

        # FatFormer inference requires the learned language-guided alignment
        # prompts (language_guided_alignment.ctx + text_encoder) and the
        # text-guided interactor cross-attention. Re-implementing the full
        # inference path from weights alone is non-trivial — leaving as a
        # follow-up task. For now the slot uses a deterministic placeholder
        # and the ensemble weight is reduced (apps/api/app/config.py).
        log.warning("FatFormer: full inference not yet implemented — placeholder active")

    def predict(self, face_bgr: np.ndarray) -> DetectorOutput:
        if self._use_placeholder:
            # Neutral score (0.5) — slot effectively abstains until inference
            # path is implemented. Combined with weight_xray=0.0 in config.py
            # this slot makes no contribution to the final verdict.
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
            hp = np.abs(gray - blur)
            return DetectorOutput(score=0.5, heatmap=hp / (hp.max() + 1e-6))

        import torch
        from torchvision import transforms

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])
        x = tf(face_bgr[:, :, ::-1].copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)               # (1, 2)
            probs = torch.softmax(logits, dim=1)
            score = float(probs[0, 1].item())    # fake class
        return DetectorOutput(score=score, heatmap=None)
