from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["safe", "caution", "risk"]


class ModelScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    heatmap_b64: str | None = None
    score_raw: float | None = Field(default=None, ge=0.0, le=1.0)
    score_tta: float | None = Field(default=None, ge=0.0, le=1.0)


class InferResponse(BaseModel):
    effort: ModelScore
    xray: ModelScore
    spsl: ModelScore
    face_bbox: list[int]
    overlay_b64: str
    jpeg_tta_delta: float | None = None


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    size: int


class AnalyzeResult(BaseModel):
    id: str
    verdict: Verdict
    final_score: float
    scores: InferResponse
