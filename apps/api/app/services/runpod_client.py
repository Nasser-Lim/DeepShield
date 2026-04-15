from __future__ import annotations

import httpx

from ..config import get_settings
from ..schemas.analysis import InferResponse, UploadResponse


class RunPodClient:
    def __init__(self) -> None:
        s = get_settings()
        self._base = s.runpod_inference_url.rstrip("/")
        self._timeout = s.runpod_inference_timeout

    async def upload(self, raw: bytes, filename: str, content_type: str) -> UploadResponse:
        """Upload image bytes to the pod and return a file_id."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(
                f"{self._base}/upload",
                files={"image": (filename, raw, content_type)},
            )
            r.raise_for_status()
            return UploadResponse.model_validate(r.json())

    async def infer(
        self,
        *,
        file_id: str | None = None,
        image_url: str | None = None,
        image_b64: str | None = None,
    ) -> InferResponse:
        payload = {"file_id": file_id, "image_url": image_url, "image_b64": image_b64}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            r = await client.post(f"{self._base}/infer", json=payload)
            if not r.is_success:
                # Propagate the RunPod error detail directly to the caller
                try:
                    detail = r.json().get("detail", r.text)
                except Exception:
                    detail = r.text
                raise httpx.HTTPStatusError(
                    detail, request=r.request, response=r
                )
            return InferResponse.model_validate(r.json())

    async def healthz(self) -> dict:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{self._base}/healthz")
            r.raise_for_status()
            return r.json()
