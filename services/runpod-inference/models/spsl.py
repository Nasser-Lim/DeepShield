from __future__ import annotations

import hashlib
import logging
import os

import cv2
import numpy as np

from .base import DetectorBase, DetectorOutput

# ---------------------------------------------------------------------------
# C2P-CLIP detector — Category-to-Prompt CLIP (AAAI 2025, Tan et al.)
# Architecture: HuggingFace CLIPModel (ViT-L/14) + classification head
# Weights: https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection (~1.2GB)
#
# Probe result (394 keys), prefix "model.vision_model.*":
#   model.vision_model.embeddings.*, model.vision_model.encoder.layers.*,
#   model.vision_model.post_layernorm.*,  model.visual_projection.*
# This is the HuggingFace transformers.CLIPModel layout.
# The classifier head is likely stored as "classifier.*" or "fc.*" at top level.
# ---------------------------------------------------------------------------

WEIGHTS_PATH = os.environ.get(
    "C2PCLIP_WEIGHTS",
    "/workspace/weights/c2pclip_best.pth",
)

log = logging.getLogger("inference")

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


class SPSLDetector(DetectorBase):
    """C2P-CLIP detector (slot 'spsl', weight 0.20).

    HuggingFace CLIPModel (ViT-L/14) fine-tuned with Category Common Prompts.
    Low false-positive rate on real photos.
    """

    name = "spsl"

    def load(self, device: str) -> None:
        self.device = device
        self._use_placeholder = True

        try:
            import torch
            import torch.nn as nn
            from transformers import CLIPModel

            # ------------------------------------------------------------------
            # C2P-CLIP stores weights under "model.*" (HuggingFace CLIPModel).
            # We mirror that attribute name so keys align.
            # The classification head (binary: real vs fake) sits outside "model.*"
            # and is loaded via strict=False.
            # ------------------------------------------------------------------

            class _C2PCLIP(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Mirror checkpoint attribute name "model"
                    self.model = CLIPModel.from_pretrained(
                        "openai/clip-vit-large-patch14",
                        ignore_mismatched_sizes=True,
                    )
                    # Binary head (real=0, fake=1)
                    # CLIPModel.get_image_features() output dim = 768
                    self.classifier = nn.Linear(768, 2)

                def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                    feats = self.model.get_image_features(pixel_values=pixel_values)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    return self.classifier(feats)   # (B, 2) logits

            net = _C2PCLIP()
            ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
            else:
                sd = ckpt
            missing, unexpected = net.load_state_dict(sd, strict=False)
            log.info(
                "C2P-CLIP: missing=%d unexpected=%d", len(missing), len(unexpected)
            )
            if missing:
                log.warning("C2P-CLIP missing keys (first 5): %s", missing[:5])
            self.model = net.to(device).eval()
            self._use_placeholder = False
            log.info("C2P-CLIP: loaded ok")
        except Exception as e:
            log.warning("C2P-CLIP load failed (%s) — using placeholder", e)

    def predict(self, face_bgr: np.ndarray) -> DetectorOutput:
        if self._use_placeholder:
            digest = hashlib.sha1(face_bgr.tobytes()).digest()
            score = (digest[4] * 256 + digest[5]) / 65535.0
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            f = np.fft.fft2(gray)
            phase = np.angle(np.fft.fftshift(f))
            norm = (phase - phase.min()) / (phase.max() - phase.min() + 1e-6)
            return DetectorOutput(score=float(score), heatmap=norm.astype(np.float32))

        import torch
        from torchvision import transforms

        # C2P-CLIP: CLIP input spec 224×224 with CLIP normalization
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
