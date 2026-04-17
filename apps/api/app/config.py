from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8080

    runpod_inference_url: str = "http://localhost:8000"
    # DDIM reverse+forward loop (ddim20) takes ~2-3s per image on A5000;
    # allow headroom for cold starts and concurrent requests.
    runpod_inference_timeout: float = 120.0

    # Verdict thresholds on the single DIRE probability
    threshold_safe: float = 0.30
    threshold_risk: float = 0.70


@lru_cache
def get_settings() -> Settings:
    return Settings()
