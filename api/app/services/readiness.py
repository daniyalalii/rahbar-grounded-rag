"""Readiness check service (sections 4.3 and 4.4).

Verifies that Chroma is reachable via heartbeat, the target collection exists
and is non-empty, Redis is reachable via ping, and the collection's embedding
model metadata matches EMBED_MODEL.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import Settings

log = logging.getLogger("rahbar.readiness")


def check_readiness(
    settings: Settings,
    chroma_client: Any | None = None,
    redis_client: Any | None = None,
    timeout: float = 2.0,
) -> dict[str, object]:
    """Verify dependencies and collection embedding pin.

    Returns a dict with {"status": "ready", "checks": {...}} or
    {"status": "not ready", "failed": "...", "detail": "..."}.

    When RETRIEVER=stub the API never queries Chroma, so the Chroma collection
    check and the embedding-pin check are skipped.  Only Redis is verified.
    This lets the CI overlay (k8s/overlays/ci) roll out the API without an
    Ollama-backed indexer run.
    """
    if settings.retriever == "stub":
        # Stub retriever does not use Chroma; skip the collection check and
        # the embedding-pin check.  Only verify that Redis is reachable.
        if redis_client is None:
            try:
                import redis as _redis

                redis_client = _redis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    socket_timeout=timeout,
                    socket_connect_timeout=timeout,
                )
            except Exception as err:
                return {
                    "status": "not ready",
                    "failed": "redis",
                    "detail": f"Could not create Redis client: {err}",
                }
        try:
            if not redis_client.ping():
                return {
                    "status": "not ready",
                    "failed": "redis",
                    "detail": f"Redis ping returned False at {settings.redis_host}:{settings.redis_port}",
                }
        except Exception as err:
            return {
                "status": "not ready",
                "failed": "redis",
                "detail": f"Redis unreachable at {settings.redis_host}:{settings.redis_port}: {err}",
            }
        return {
            "status": "ready",
            "checks": {
                "retriever": "stub",
                "chroma": "skipped (stub retriever)",
                "redis": "ok",
            },
        }

    # 1. Chroma reachable (heartbeat)
    if chroma_client is None:
        try:
            import httpx

            # Short HTTP probe before client initialization to avoid hangs
            httpx.get(
                f"http://{settings.chroma_host}:{settings.chroma_port}/api/v2/heartbeat",
                timeout=timeout,
            ).raise_for_status()

            import chromadb
            from chromadb.config import Settings as ChromaSettings

            chroma_client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            if hasattr(chroma_client, "_server") and hasattr(chroma_client._server, "_session"):
                chroma_client._server._session.timeout = httpx.Timeout(timeout)
        except Exception as err:
            return {
                "status": "not ready",
                "failed": "chroma",
                "detail": f"Chroma unreachable at {settings.chroma_host}:{settings.chroma_port}: {err}",
            }

    try:
        chroma_client.heartbeat()
    except Exception as err:
        return {
            "status": "not ready",
            "failed": "chroma",
            "detail": f"Chroma heartbeat failed: {err}",
        }

    # 2. Collection exists and is non-empty (count > 0)
    try:
        collection = chroma_client.get_collection(name=settings.collection)
        count = collection.count()
        if count <= 0:
            return {
                "status": "not ready",
                "failed": "collection-empty",
                "detail": f"Collection '{settings.collection}' is empty (count={count})",
            }
    except Exception as err:
        return {
            "status": "not ready",
            "failed": "collection-empty",
            "detail": f"Collection '{settings.collection}' missing or empty: {err}",
        }

    # 3. Redis reachable (ping with short timeout)
    if redis_client is None:
        try:
            import redis

            redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                socket_timeout=timeout,
                socket_connect_timeout=timeout,
            )
        except Exception as err:
            return {
                "status": "not ready",
                "failed": "redis",
                "detail": f"Could not create Redis client for {settings.redis_host}:{settings.redis_port}: {err}",
            }

    try:
        if not redis_client.ping():
            return {
                "status": "not ready",
                "failed": "redis",
                "detail": f"Redis ping returned False at {settings.redis_host}:{settings.redis_port}",
            }
    except Exception as err:
        return {
            "status": "not ready",
            "failed": "redis",
            "detail": f"Redis unreachable at {settings.redis_host}:{settings.redis_port}: {err}",
        }

    # 4. Collection metadata embedding model matches EMBED_MODEL
    metadata = getattr(collection, "metadata", None) or {}
    indexed_model = metadata.get("embed_model")
    if indexed_model != settings.embed_model:
        log.error(
            "embedding model mismatch: indexed model %s vs EMBED_MODEL %s",
            indexed_model,
            settings.embed_model,
            extra={
                "indexed_model": indexed_model,
                "embed_model": settings.embed_model,
            },
        )
        return {
            "status": "not ready",
            "failed": "embedding-pin",
            "detail": f"embedding model mismatch: indexed model {indexed_model} vs EMBED_MODEL {settings.embed_model}",
        }

    return {
        "status": "ready",
        "checks": {
            "chroma": "ok",
            "collection": settings.collection,
            "count": count,
            "redis": "ok",
            "embed_model": settings.embed_model,
        },
    }

