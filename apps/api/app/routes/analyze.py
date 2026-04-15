from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..schemas.analysis import AnalyzeResult
from ..services.ensemble import aggregate
from ..services.runpod_client import RunPodClient

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

    # Step 1: Upload to RunPod pod storage
    try:
        upload = await client.upload(
            raw,
            filename=image.filename or "image.jpg",
            content_type=image.content_type or "image/jpeg",
        )
    except Exception as e:
        log.exception("upload to RunPod failed")
        raise HTTPException(502, f"upload failed: {e}") from e

    # Step 2: Run inference using the stored file_id
    try:
        infer = await client.infer(file_id=upload.file_id)
    except Exception as e:
        log.exception("inference failed")
        raise HTTPException(502, f"inference failed: {e}") from e

    # Step 3: Weighted ensemble vote
    agg = aggregate(infer.effort.score, infer.xray.score, infer.spsl.score)

    return AnalyzeResult(
        id=uuid.uuid4().hex,
        verdict=agg.verdict,
        final_score=agg.final_score,
        scores=infer,
    )
