"""Stage 4: Embed. One vector per chunk, from the pinned model only.

Calls the Ollama /api/embed endpoint in a single batched request.
"""

from __future__ import annotations

import httpx

from .config import IndexerSettings

# Timeout for the embedding call (seconds). Large batches may be slow on CPU.
_TIMEOUT = 120.0


def embed(texts: list[str], settings: IndexerSettings) -> list[list[float]]:
    """Embed with settings.embed_model and nothing else.

    With Ollama: POST {ollama_url}/api/embed with {"model": ..., "input": [texts]}
    returns {"embeddings": [[...], ...]}. Batch the calls; one request per chunk is slow.
    """
    if not texts:
        return []

    url = settings.ollama_url.rstrip("/") + "/api/embed"
    payload = {"model": settings.embed_model, "input": texts}

    response = httpx.post(url, json=payload, timeout=_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    embeddings: list[list[float]] = data["embeddings"]

    if len(embeddings) != len(texts):
        raise ValueError(
            f"embed: expected {len(texts)} vectors, got {len(embeddings)}"
        )

    return embeddings
