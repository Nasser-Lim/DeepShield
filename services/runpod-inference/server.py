from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from models.base import DetectorBase
from models.effort import EffortDetector
from models.face_detect import FaceDetector
from models.face_xray import FaceXrayDetector
from models.spsl import SPSLDetector
from utils.heatmap import overlay_on_face
from utils.io import encode_png_b64, load_image_from_b64, load_image_from_url, _bytes_to_bgr

log = logging.getLogger("inference")
logging.basicConfig(level=logging.INFO)

DEVICE = os.getenv("DEVICE", "cpu")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/deepshield/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


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
    # JPEG TTA (Test-Time Augmentation): score on Q=75 re-encoded crop.
    # Large raw−tta gap suggests the model is keying on JPEG artifacts,
    # not AI generation fingerprints.
    score_raw: float | None = Field(default=None, ge=0.0, le=1.0)
    score_tta: float | None = Field(default=None, ge=0.0, le=1.0)


class InferResponse(BaseModel):
    effort: ModelScoreOut
    xray: ModelScoreOut
    spsl: ModelScoreOut
    face_bbox: list[int]
    overlay_b64: str
    # TTA-adjusted final scores used by the ensemble. When jpeg_tta_delta
    # is high (>0.3), the verdict should be treated with low confidence.
    jpeg_tta_delta: float | None = None


# ── app lifecycle ──────────────────────────────────────────────────────────

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading detectors on %s", DEVICE)
    detectors: list[DetectorBase] = [EffortDetector(), FaceXrayDetector(), SPSLDetector()]
    for d in detectors:
        d.load(DEVICE)
    state["detectors"] = {d.name: d for d in detectors}
    state["face_detector"] = FaceDetector()
    yield
    state.clear()


app = FastAPI(title="DeepShield Inference", lifespan=lifespan)


# ── endpoints ─────────────────────────────────────────────────────────────

@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "detectors": list(state.get("detectors", {}))}


@app.post("/upload", response_model=UploadResponse)
async def upload(image: UploadFile = File(...)) -> UploadResponse:
    """Store an image on the pod and return a file_id for /infer."""
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
    """Run triple-detector inference.

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

    face_detector: FaceDetector = state["face_detector"]
    detectors: dict[str, DetectorBase] = state["detectors"]
    try:
        crop = face_detector.detect(image)
    except ValueError as e:
        if str(e) == "no_face_detected":
            raise HTTPException(422, "이미지에서 얼굴을 감지할 수 없습니다. 얼굴이 포함된 이미지를 업로드해 주세요.")
        raise

    # JPEG TTA: re-encode the face crop at Q=75 so models see a
    # "compressed-twice" copy. If scores drop sharply on the re-encoded
    # copy, the model was keying on JPEG artifacts, not AI fingerprints.
    import cv2 as _cv2
    ok, enc = _cv2.imencode(".jpg", crop.patch, [_cv2.IMWRITE_JPEG_QUALITY, 75])
    if ok:
        crop_tta = _cv2.imdecode(enc, _cv2.IMREAD_COLOR)
    else:
        crop_tta = crop.patch  # fallback: no TTA if encode fails

    async def run(det: DetectorBase, patch):
        return await asyncio.to_thread(det.predict, patch)

    (
        effort_raw, xray_raw, spsl_raw,
        effort_tta, xray_tta, spsl_tta,
    ) = await asyncio.gather(
        run(detectors["effort"], crop.patch),
        run(detectors["xray"],   crop.patch),
        run(detectors["spsl"],   crop.patch),
        run(detectors["effort"], crop_tta),
        run(detectors["xray"],   crop_tta),
        run(detectors["spsl"],   crop_tta),
    )

    # Final score per detector = mean(raw, tta). Averaging damps
    # predictions that are unstable under benign re-compression.
    def _mean(a, b) -> float:
        return float((a.score + b.score) / 2.0)

    effort_score = _mean(effort_raw, effort_tta)
    xray_score   = _mean(xray_raw,   xray_tta)
    spsl_score   = _mean(spsl_raw,   spsl_tta)

    # Reuse the raw heatmap for overlay (TTA crop has no meaningful heatmap).
    effort_out = type(effort_raw)(score=effort_score, heatmap=effort_raw.heatmap)
    xray_out   = type(xray_raw)(  score=xray_score,   heatmap=xray_raw.heatmap)
    spsl_out   = type(spsl_raw)(  score=spsl_score,   heatmap=spsl_raw.heatmap)

    # Confidence signal for downstream UI: absolute gap between raw and
    # TTA scores averaged across detectors. >0.3 = low-confidence verdict.
    tta_delta = float(
        (abs(effort_raw.score - effort_tta.score)
         + abs(xray_raw.score - xray_tta.score)
         + abs(spsl_raw.score - spsl_tta.score)) / 3.0
    )

    # Use the first available heatmap for overlay, or just the face crop
    _overlay_heatmap = next(
        (o.heatmap for o in [xray_out, effort_out, spsl_out] if o.heatmap is not None),
        None,
    )
    if _overlay_heatmap is not None:
        overlay = overlay_on_face(crop.patch, _overlay_heatmap)
    else:
        overlay = crop.patch
    overlay_b64 = encode_png_b64(overlay)

    def pack(out, raw, tta) -> ModelScoreOut:
        heatmap_b64 = None
        if out.heatmap is not None:
            vis = overlay_on_face(crop.patch, out.heatmap)
            heatmap_b64 = encode_png_b64(vis)
        return ModelScoreOut(
            score=out.score,
            heatmap_b64=heatmap_b64,
            score_raw=float(raw.score),
            score_tta=float(tta.score),
        )

    return InferResponse(
        effort=pack(effort_out, effort_raw, effort_tta),
        xray=pack(xray_out, xray_raw, xray_tta),
        spsl=pack(spsl_out, spsl_raw, spsl_tta),
        face_bbox=list(crop.bbox),
        overlay_b64=overlay_b64,
        jpeg_tta_delta=round(tta_delta, 4),
    )
