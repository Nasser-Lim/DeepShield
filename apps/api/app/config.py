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
    weight_effort: float = 0.40
    weight_xray: float = 0.35
    weight_spsl: float = 0.25

    # Verdict thresholds
    threshold_safe: float = 0.30
    threshold_risk: float = 0.70


@lru_cache
def get_settings() -> Settings:
    return Settings()
