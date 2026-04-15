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
    # F3Net (xray) has systematic upward bias on real photos → lower its weight.
    # Xception (effort) is most reliable; SPSL complements phase artifacts.
    weight_effort: float = 0.50
    weight_xray: float = 0.20
    weight_spsl: float = 0.30

    # Verdict thresholds
    # Raised to compensate for F3Net's positive bias on real-camera images.
    threshold_safe: float = 0.35
    threshold_risk: float = 0.75


@lru_cache
def get_settings() -> Settings:
    return Settings()
