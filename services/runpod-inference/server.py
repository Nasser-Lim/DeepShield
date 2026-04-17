from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from models.base import DetectorBase
from models.dire import DireDetector
from utils.heatmap import overlay_on_image
from utils.io import encode_png_b64, load_image_from_b64, load_image_from_url, _bytes_to_bgr

log = logging.getLogger("inference")
logging.basicConfig(level=logging.INFO)

DEVICE = os.getenv("DEVICE", "cpu")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/deepshield/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


# ── schemas ────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    size: int


class InferRequest(BaseModel):
    file_id: str | None = None
    image_url: str | None = None
    image_b64: str | None = None


class ModelScoreOut(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    heatmap_b64: str | None = None


class InferResponse(BaseModel):
    dire: ModelScoreOut
    overlay_b64: str


# ── app lifecycle ──────────────────────────────────────────────────────────

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading DIRE detector on %s", DEVICE)
    detector: DetectorBase = DireDetector()
    detector.load(DEVICE)
    state["detector"] = detector
    yield
    state.clear()


app = FastAPI(title="DeepShield DIRE Inference", lifespan=lifespan)


# ── endpoints ─────────────────────────────────────────────────────────────

@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "detectors": ["dire"] if "detector" in state else []}


@app.post("/upload", response_model=UploadResponse)
async def upload(image: UploadFile = File(...)) -> UploadResponse:
    if image.content_type not in ALLOWED_MIME:
        raise HTTPException(415, f"unsupported media type: {image.content_type}")

    raw = await image.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large (max 20 MB)")

    file_id = uuid.uuid4().hex
    ext = Path(image.filename or "img.jpg").suffix or ".jpg"
    dest = UPLOAD_DIR / f"{file_id}{ext}"
    dest.write_bytes(raw)
    log.info("Stored upload %s (%d bytes)", dest.name, len(raw))
    return UploadResponse(file_id=file_id, filename=dest.name, size=len(raw))


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest) -> InferResponse:
    """Run DIRE inference on a full image (no face detection).

    Accepts one of:
    - file_id   – previously uploaded via /upload
    - image_url – publicly reachable URL
    - image_b64 – base64-encoded image data
    """
    if not any([req.file_id, req.image_url, req.image_b64]):
        raise HTTPException(400, "file_id, image_url or image_b64 required")

    try:
        if req.file_id:
            matches = list(UPLOAD_DIR.glob(f"{req.file_id}*"))
            if not matches:
                raise HTTPException(404, f"file_id not found: {req.file_id}")
            image = _bytes_to_bgr(matches[0].read_bytes())
        elif req.image_url:
            image = load_image_from_url(req.image_url)
        else:
            image = load_image_from_b64(req.image_b64)  # type: ignore[arg-type]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"image load failed: {e}") from e

    detector: DetectorBase = state["detector"]
    out = await asyncio.to_thread(detector.predict, image)

    if out.heatmap is not None:
        overlay = overlay_on_image(image, out.heatmap)
        heatmap_b64 = encode_png_b64(overlay)
    else:
        overlay = image
        heatmap_b64 = None
    overlay_b64 = encode_png_b64(overlay)

    return InferResponse(
        dire=ModelScoreOut(score=out.score, heatmap_b64=heatmap_b64),
        overlay_b64=overlay_b64,
    )
