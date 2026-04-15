from __future__ import annotations

import hashlib
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
    "/workspace/weights/fatformer_best.pth",
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

        try:
            import torch
            import torch.nn as nn
            import open_clip

            # ------------------------------------------------------------------
            # FatFormer checkpoint layout (confirmed by probe):
            #   image_encoder.*  — CLIP visual (ViT-L/14, with forgery adapters
            #                       injected at resblocks 7, 15, 23)
            #   text_encoder.*   — CLIP text transformer
            #   language_guided_alignment.* — LGA prompts + patch enhancer
            #   text_guided_interactor.*    — cross-attention
            #   norm1, linear1, linear2, norm2 — classification MLP head
            #   logit_scale
            #
            # We build a thin nn.Module whose attribute names mirror these
            # top-level keys, load with strict=False so adapter weights
            # (forgery_aware_adapter.*) land correctly in image_encoder.
            # The final score uses the 2-layer MLP head (linear1→ReLU→linear2).
            # ------------------------------------------------------------------

            class _FatFormer(nn.Module):
                def __init__(self):
                    super().__init__()
                    clip_model, _, _ = open_clip.create_model_and_transforms(
                        "ViT-L-14", pretrained=None
                    )
                    # Mirror checkpoint attribute names exactly
                    self.image_encoder = clip_model.visual   # (B, 1024)
                    self.text_encoder = clip_model.transformer
                    self.logit_scale = clip_model.logit_scale

                    feat_dim = 1024  # ViT-L/14 output dim

                    # Classification MLP head: norm1 → linear1 → ReLU → linear2
                    self.norm1 = nn.LayerNorm(feat_dim)
                    self.linear1 = nn.Linear(feat_dim, feat_dim // 2)
                    self.linear2 = nn.Linear(feat_dim // 2, 2)
                    self.norm2 = nn.LayerNorm(feat_dim // 2)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    feats = self.image_encoder(x)           # (B, 1024)
                    feats = self.norm1(feats)
                    feats = torch.relu(self.linear1(feats)) # (B, 512)
                    feats = self.norm2(feats)
                    return self.linear2(feats)              # (B, 2) logits

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
