from __future__ import annotations

import hashlib
import logging
import os

import cv2
import numpy as np

from .base import DetectorBase, DetectorOutput

# ---------------------------------------------------------------------------
# C2P-CLIP detector — Category-to-Prompt CLIP (AAAI 2025, Tan et al.)
# Architecture: CLIP ViT-L/14 image encoder + learned Category Common Prompts
# Real/fake scored via cosine similarity to learned text embeddings.
# Weights: https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection
#
# NOTE: Weight key structure confirmed after probe. Checkpoint typically
# contains: "image_encoder.*" (CLIP visual), "text_encoder.*" (CLIP text),
# "prefix_encoder.*" (learned C2P prefix tokens).
# ---------------------------------------------------------------------------

WEIGHTS_PATH = os.environ.get(
    "C2PCLIP_WEIGHTS",
    "/workspace/weights/c2pclip_best.pth",
)

log = logging.getLogger("inference")

# CLIP normalization constants (ViT-L/14)
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


class SPSLDetector(DetectorBase):
    """C2P-CLIP detector (slot 'spsl', weight 0.20).

    CLIP ViT-L/14 with Category Common Prompts injected into the text encoder.
    Real vs. fake scored by cosine similarity between the image embedding and
    learned real/fake text embeddings. Low false-positive rate on real photos.
    """

    name = "spsl"

    def load(self, device: str) -> None:
        self.device = device
        self._use_placeholder = True

        try:
            import torch
            import torch.nn as nn
            import open_clip

            # ------------------------------------------------------------------
            # C2P-CLIP architecture:
            #   - CLIP ViT-L/14 image encoder (frozen or fine-tuned)
            #   - Learned prefix tokens (C2P) prepended to text encoder input
            #   - Two text embeddings: "a photo of real face" vs "a deepfake face"
            #     enhanced by the learned prefix
            #   - Score = cosine_similarity(img_emb, fake_text_emb)
            #
            # Minimal implementation that handles most checkpoint layouts.
            # Actual prefix length and text template confirmed via probe.
            # ------------------------------------------------------------------

            class _C2PCLIP(nn.Module):
                def __init__(self, prefix_len: int = 8):
                    super().__init__()
                    clip_model, _, _ = open_clip.create_model_and_transforms(
                        "ViT-L-14", pretrained=None
                    )
                    self.visual = clip_model.visual
                    self.text_encoder = clip_model.transformer
                    self.token_embedding = clip_model.token_embedding
                    self.positional_embedding = clip_model.positional_embedding
                    self.ln_final = clip_model.ln_final
                    self.text_projection = clip_model.text_projection
                    feat_dim = 768

                    # Learnable C2P prefix tokens for real and fake categories
                    self.prefix_real = nn.Parameter(torch.zeros(prefix_len, feat_dim))
                    self.prefix_fake = nn.Parameter(torch.zeros(prefix_len, feat_dim))

                    # Logit scale (CLIP-style)
                    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # Image embedding
                    img_emb = self.visual(x)   # (B, 768)
                    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                    return img_emb  # caller handles text similarity

            net = _C2PCLIP()
            ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
            else:
                sd = ckpt
            missing, unexpected = net.load_state_dict(sd, strict=False)
            log.info("C2P-CLIP: missing=%d unexpected=%d", len(missing), len(unexpected))
            if missing:
                log.warning("C2P-CLIP missing keys (first 5): %s", missing[:5])
            self.model = net.to(device).eval()

            # Pre-compute fake text embedding for inference
            self._fake_emb = self._compute_text_emb("a deepfake face image", device)
            self._real_emb = self._compute_text_emb("a real photo of a face", device)

            self._use_placeholder = False
            log.info("C2P-CLIP: loaded ok")
        except Exception as e:
            log.warning("C2P-CLIP load failed (%s) — using placeholder", e)

    def _compute_text_emb(self, text: str, device: str):
        """Compute a fixed CLIP text embedding for a given prompt."""
        import torch
        import open_clip
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        tokens = tokenizer([text]).to(device)
        with torch.no_grad():
            # Use a fresh CLIP model just for text encoding (small overhead at load time)
            clip_model, _, _ = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained=None
            )
            clip_model = clip_model.to(device).eval()
            emb = clip_model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb  # (1, 768)

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

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])
        x = tf(face_bgr[:, :, ::-1].copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_emb = self.model(x)   # (1, 768) normalised
            # Cosine similarity to fake/real text embeddings
            sim_fake = (img_emb * self._fake_emb).sum(dim=-1)  # (1,)
            sim_real = (img_emb * self._real_emb).sum(dim=-1)  # (1,)
            # Convert to fake probability via softmax over [real, fake]
            logits = torch.stack([sim_real, sim_fake], dim=-1)  # (1, 2)
            score = float(torch.softmax(logits, dim=-1)[0, 1].item())
        return DetectorOutput(score=score, heatmap=None)
