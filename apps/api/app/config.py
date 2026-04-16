from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8080

    runpod_inference_url: str = "http://localhost:8000"
    runpod_inference_timeout: float = 60.0

    # Ensemble weights (must sum to 1.0)
    # effort = SBI    (CVPR 2022)  — EfficientNet-B4, face blending artifacts
    # xray   = UnivFD (CVPR 2023)  — CLIP ViT-L/14 linear probe, JPEG-robust
    # spsl   = C2P-CLIP (AAAI 2025) — CLIP ViT-L/14 with category prompts
    #
    # Equal weight across backbones: SBI (EfficientNet) vs UnivFD/C2P-CLIP
    # (both CLIP-based, but trained with different objectives and datasets).
    weight_effort: float = 0.35
    weight_xray: float = 0.35
    weight_spsl: float = 0.30

    # Verdict thresholds
    threshold_safe: float = 0.30
    threshold_risk: float = 0.70


@lru_cache
def get_settings() -> Settings:
    return Settings()
