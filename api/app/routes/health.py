"""Liveness and readiness (section 4.3). They answer different questions."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

from ..config import get_settings
from ..services.readiness import check_readiness

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Is the process alive? Must never touch Chroma, Redis or the LLM."""
    return {"status": "alive"}


@router.get("/readyz")
def readyz(request: Request, response: Response) -> dict[str, object]:
    settings = getattr(request.app.state, "settings", None) or get_settings()
    chroma_client = getattr(request.app.state, "chroma_client", None)
    if chroma_client is None:
        retriever = getattr(request.app.state, "retriever", None)
        if retriever is not None and hasattr(retriever, "client"):
            chroma_client = retriever.client  # type: ignore[union-attr]
    redis_client = getattr(request.app.state, "redis_client", None)

    result = check_readiness(settings, chroma_client=chroma_client, redis_client=redis_client)
    if result.get("status") != "ready":
        response.status_code = 503
    return result
