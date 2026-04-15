from __future__ import annotations

from fastapi import APIRouter

from ..services.runpod_client import RunPodClient

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict:
    runpod = RunPodClient()
    try:
        infer = await runpod.healthz()
    except Exception as e:  # noqa: BLE001
        infer = {"ok": False, "error": str(e)}
    return {"ok": True, "inference": infer}
