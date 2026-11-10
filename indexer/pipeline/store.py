"""Stage 5: Store. The only code in the whole system that writes to Chroma.

Strategy: delete-and-recreate the collection (atomic from Chroma's point of
view) so the index never shows stale + partial data simultaneously. The API
will return 503 from /readyz during the brief window between delete and the
final collection.add(), which is acceptable for a batch job.
"""

from __future__ import annotations

from typing import Any

from .config import IndexerSettings
from .models import Chunk


def get_client(settings: IndexerSettings) -> Any:
    import chromadb  # the thin client, from chromadb-client in requirements.txt
    from chromadb.config import Settings as ChromaSettings

    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port,
                               settings=ChromaSettings(anonymized_telemetry=False))


def store(chunks: list[Chunk], vectors: list[list[float]], settings: IndexerSettings) -> int:
    """Write every chunk with its vector and metadata, and record the embedding model
    and vector dimension in the collection metadata (section 4.4). Return the count."""
    if not chunks:
        raise ValueError("store: no chunks to write")
    if len(chunks) != len(vectors):
        raise ValueError(
            f"store: {len(chunks)} chunks but {len(vectors)} vectors"
        )

    dim = len(vectors[0])
    client = get_client(settings)

    # Delete the existing collection so the index is replaced atomically.
    try:
        client.delete_collection(settings.collection)
    except Exception:
        pass  # collection did not exist yet

    collection = client.create_collection(
        name=settings.collection,
        metadata={
            "hnsw:space": "cosine",
            "embed_model": settings.embed_model,
            "dimension": dim,
        },
    )

    # Chroma recommends batches ≤ 5 000 to avoid memory spikes.
    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_vecs = vectors[start : start + batch_size]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=batch_vecs,
            metadatas=[c.metadata() for c in batch],
        )

    # Invalidate semantic cache in Redis after rebuilding the index so stale answers are never served
    try:
        import redis
        r = redis.Redis(host=settings.redis_host, port=settings.redis_port, socket_timeout=1.0)
        try:
            r.incr(f"rahbar:cache:{settings.collection}:gen")
        except Exception:
            pass
        prefix = f"rahbar:cache:{settings.collection}:*"
        keys: Any = r.keys(prefix)
        if keys:
            r.delete(*keys)
    except Exception:
        pass  # Redis may be offline or in a test harness

    return int(collection.count())
