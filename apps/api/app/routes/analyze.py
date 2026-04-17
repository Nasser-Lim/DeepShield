from __future__ import annotations

import logging
import uuid

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile

from ..schemas.analysis import AnalyzeResult
from ..services.runpod_client import RunPodClient
from ..services.verdict import verdict_from_score

log = logging.getLogger("analyze")
router = APIRouter(prefix="/analyze", tags=["analyze"])

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif"}


@router.post("", response_model=AnalyzeResult)
async def create_analysis(image: UploadFile = File(...)) -> AnalyzeResult:
    if image.content_type not in ALLOWED_MIME:
        raise HTTPException(415, f"unsupported media type: {image.content_type}")

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty upload")

    client = RunPodClient()

    try:
        upload = await client.upload(
            raw,
            filename=image.filename or "image.jpg",
            content_type=image.content_type or "image/jpeg",
        )
    except Exception as e:
        log.exception("upload to RunPod failed")
        raise HTTPException(502, f"upload failed: {e}") from e

    try:
        infer = await client.infer(file_id=upload.file_id)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, str(e)) from e
    except Exception as e:
        log.exception("inference failed")
        raise HTTPException(502, f"inference failed: {e}") from e

    agg = verdict_from_score(infer.dire.score)

    return AnalyzeResult(
        id=uuid.uuid4().hex,
        verdict=agg.verdict,
        final_score=agg.final_score,
        scores=infer,
    )
