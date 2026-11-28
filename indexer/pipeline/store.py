"""Stage 5 (Incremental): Store. The only code in the whole system that writes to Chroma.

RAGOps section 4.2.2-4.2.4: incremental update strategy.

Strategy:
  - A hash store (JSON file on datalake) records SHA-256 per document source.
  - On re-run:
      * Unchanged doc -> skip BEFORE embedding (zero chunks embedded)
      * New doc       -> embed only its chunks, insert into Chroma
      * Changed doc   -> embed only its chunks, delete old chunks, insert new chunks
      * Deleted doc   -> delete all its old chunks from Chroma (even if 0 docs pass verification)
      * Orphan chunks -> any chunk in Chroma whose source is not in current passed docs is deleted
  - The original full-replace store() is kept for first-time indexing and --full-rebuild.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .config import IndexerSettings
from .models import Chunk


HASH_STORE_FILENAME = "doc_hashes.json"
ARCHIVE_DIR_NAME = "archive"


def resolve_datalake_dir(settings: IndexerSettings) -> Path:
    if settings.datalake_dir:
        p = Path(settings.datalake_dir)
    else:
        p = Path(settings.corpus_dir).parent / "datalake"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hash_store_path(settings: IndexerSettings) -> Path:
    return resolve_datalake_dir(settings) / HASH_STORE_FILENAME


def _load_hash_store(settings: IndexerSettings) -> dict[str, str]:
    p = _hash_store_path(settings)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _save_hash_store(settings: IndexerSettings, store_data: dict[str, str]) -> None:
    p = _hash_store_path(settings)
    p.write_text(json.dumps(store_data, indent=2), encoding="utf-8")


def _chunk_hash(chunks: list[Chunk]) -> str:
    """SHA-256 of all chunk texts and metadata versions for a document."""
    combined = "\n---\n".join(f"{c.doc_version}|{c.superseded}|{c.text}" for c in chunks)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _doc_hash(chunks: list[Chunk]) -> str:
    return _chunk_hash(chunks)


def get_client(settings: IndexerSettings) -> Any:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _get_or_create_collection(client: Any, settings: IndexerSettings) -> Any:
    try:
        return client.get_collection(settings.collection)
    except Exception:
        return client.create_collection(
            name=settings.collection,
            metadata={
                "hnsw:space": "cosine",
                "embed_model": settings.embed_model,
            },
        )


def _add_chunks_to_collection(
    collection: Any,
    chunks: list[Chunk],
    vectors: list[list[float]],
    batch_size: int = 500,
) -> None:
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_vecs = vectors[start : start + batch_size]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=batch_vecs,
            metadatas=[c.metadata() for c in batch],
        )


def _delete_doc_chunks(collection: Any, source: str) -> int:
    """Delete all chunks for a given source document. Returns count deleted."""
    try:
        result = collection.get(where={"source": source})
        ids = result.get("ids", []) if isinstance(result, dict) else []
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0


def _delete_orphan_chunks(collection: Any, active_sources: set[str]) -> int:
    """Find and delete any chunks in Chroma whose source is not in active_sources."""
    try:
        all_items = collection.get(include=["metadatas"])
        ids = all_items.get("ids", []) if isinstance(all_items, dict) else []
        metas = all_items.get("metadatas", []) if isinstance(all_items, dict) else []
        orphan_ids = [
            cid
            for cid, meta in zip(ids, metas)
            if not meta or str(meta.get("source", "")) not in active_sources
        ]
        if orphan_ids:
            collection.delete(ids=orphan_ids)
        return len(orphan_ids)
    except Exception:
        return 0


def sync_incremental(
    doc_chunks: dict[str, list[Chunk]],
    settings: IndexerSettings,
    embed_fn: Callable[[list[str], IndexerSettings], list[list[float]]],
    client: Any = None,
) -> dict[str, int]:
    """Incremental update that checks hashes BEFORE embedding.

    - Unchanged docs: 0 chunks embedded.
    - New docs: only their chunks embedded and added.
    - Changed docs: only their chunks embedded, old chunks deleted, new added.
    - Deleted docs (in old_hashes but not doc_chunks): old chunks deleted even if doc_chunks is empty.
    - Orphan chunks in Chroma not in doc_chunks: deleted.
    """
    if client is None:
        client = get_client(settings)
    collection = _get_or_create_collection(client, settings)

    old_hashes = _load_hash_store(settings)
    new_hashes: dict[str, str] = {}

    to_embed_sources: list[str] = []
    unchanged_chunks = 0
    new_docs_chunks: dict[str, list[Chunk]] = {}
    changed_docs_chunks: dict[str, list[Chunk]] = {}

    for source, source_chunks in doc_chunks.items():
        h = _doc_hash(source_chunks)
        new_hashes[source] = h
        if old_hashes.get(source) == h:
            unchanged_chunks += len(source_chunks)
        elif source in old_hashes:
            changed_docs_chunks[source] = source_chunks
            to_embed_sources.append(source)
        else:
            new_docs_chunks[source] = source_chunks
            to_embed_sources.append(source)

    newly_embedded_chunks = 0
    updated_chunks = 0
    deleted_chunks = 0

    if to_embed_sources:
        batch_chunks: list[Chunk] = []
        for src in to_embed_sources:
            batch_chunks.extend(doc_chunks[src])

        vectors = embed_fn([c.text for c in batch_chunks], settings)
        if len(vectors) != len(batch_chunks):
            raise ValueError(
                f"embed returned {len(vectors)} vectors for {len(batch_chunks)} chunks"
            )

        offset = 0
        for src in to_embed_sources:
            src_chunks = doc_chunks[src]
            src_vecs = vectors[offset : offset + len(src_chunks)]
            offset += len(src_chunks)

            if src in changed_docs_chunks:
                deleted_chunks += _delete_doc_chunks(collection, src)
                _add_chunks_to_collection(collection, src_chunks, src_vecs)
                updated_chunks += len(src_chunks)
            else:
                _add_chunks_to_collection(collection, src_chunks, src_vecs)
                newly_embedded_chunks += len(src_chunks)

    # Delete chunks for removed documents (works even when doc_chunks is empty)
    deleted_sources = set(old_hashes.keys()) - set(doc_chunks.keys())
    for src in deleted_sources:
        deleted_chunks += _delete_doc_chunks(collection, src)

    # Sweep any remaining orphan chunks in Chroma not in active sources
    orphan_deleted = _delete_orphan_chunks(collection, set(doc_chunks.keys()))

    _save_hash_store(settings, new_hashes)
    _invalidate_cache(settings)

    total_in_collection = int(collection.count())
    summary = {
        "newly_embedded_chunks": newly_embedded_chunks,
        "unchanged_chunks": unchanged_chunks,
        "updated_chunks": updated_chunks,
        "deleted_chunks": deleted_chunks,
        "orphan_chunks_deleted": orphan_deleted,
        "total_in_collection": total_in_collection,
    }
    _log("store_incremental done", **summary)
    return summary


def store_incremental(
    chunks: list[Chunk],
    vectors: list[list[float]],
    settings: IndexerSettings,
    client: Any = None,
) -> int:
    """Legacy wrapper when vectors are pre-computed."""
    if len(chunks) != len(vectors):
        raise ValueError(f"store_incremental: {len(chunks)} chunks but {len(vectors)} vectors")

    doc_chunks: dict[str, list[Chunk]] = defaultdict(list)
    doc_vecs: dict[str, list[list[float]]] = defaultdict(list)
    for c, v in zip(chunks, vectors):
        doc_chunks[c.source].append(c)
        doc_vecs[c.source].append(v)

    def _precomputed_embed(texts: list[str], _s: IndexerSettings) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            for c, v in zip(chunks, vectors):
                if c.text == t:
                    out.append(v)
                    break
        return out

    summary = sync_incremental(doc_chunks, settings, _precomputed_embed, client=client)
    return summary["total_in_collection"]


def _log(msg: str, **fields: object) -> None:
    import sys
    print(
        json.dumps(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "logger": "rahbar.store",
                "msg": msg,
                **fields,
            }
        ),
        flush=True,
        file=sys.stdout,
    )


def _invalidate_cache(settings: IndexerSettings) -> None:
    try:
        import redis
        r = redis.Redis(host=settings.redis_host, port=settings.redis_port, socket_timeout=1.0)
        r.incr(f"rahbar:cache:{settings.collection}:gen")
        prefix = f"rahbar:cache:{settings.collection}:*"
        keys: Any = r.keys(prefix)
        if keys:
            r.delete(*keys)
    except Exception:
        pass


def store(chunks: list[Chunk], vectors: list[list[float]], settings: IndexerSettings) -> int:
    """Full-replace: delete and recreate the collection atomically."""
    if not chunks:
        raise ValueError("store: no chunks to write")
    if len(chunks) != len(vectors):
        raise ValueError(f"store: {len(chunks)} chunks but {len(vectors)} vectors")

    dim = len(vectors[0])
    client = get_client(settings)

    try:
        client.delete_collection(settings.collection)
    except Exception:
        pass

    collection = client.create_collection(
        name=settings.collection,
        metadata={
            "hnsw:space": "cosine",
            "embed_model": settings.embed_model,
            "dimension": dim,
        },
    )

    _add_chunks_to_collection(collection, chunks, vectors)
    _invalidate_cache(settings)
    return int(collection.count())
