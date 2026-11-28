"""Stage 5 (Incremental): Store. The only code in the whole system that writes to Chroma.

RAGOps section 4.2.2-4.2.4: incremental update strategy.

Strategy:
  - A hash store (JSON file on /datalake) records SHA-256 per document.
  - On re-run:
      * Unchanged doc → skip (zero chunks re-embedded)
      * Changed doc   → delete its old chunks from Chroma, insert new chunks
      * Deleted doc   → find orphan chunks by doc_id metadata, delete them
      * Superseded    → archive with metadata is_active=False, excluded from retrieval
  - The original full-replace store() is kept for first-time indexing and --full-rebuild.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from .config import IndexerSettings
from .models import Chunk


# ---------------------------------------------------------------------------
# Hash store helpers
# ---------------------------------------------------------------------------

HASH_STORE_FILENAME = "doc_hashes.json"
ARCHIVE_DIR_NAME = "archive"


def _hash_store_path(settings: IndexerSettings) -> Path:
    """Path to the JSON hash store on the datalake PVC."""
    datalake = Path(settings.corpus_dir).parent / "datalake"
    try:
        datalake.mkdir(parents=True, exist_ok=True)
    except OSError:
        datalake = Path("/tmp/datalake")
        datalake.mkdir(parents=True, exist_ok=True)
    return datalake / HASH_STORE_FILENAME


def _load_hash_store(settings: IndexerSettings) -> dict[str, str]:
    """Returns {source_name: sha256} from last run, or {} if first run."""
    p = _hash_store_path(settings)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_hash_store(settings: IndexerSettings, store: dict[str, str]) -> None:
    p = _hash_store_path(settings)
    p.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _chunk_hash(chunks: list[Chunk]) -> str:
    """SHA-256 of all chunk texts for a document (order-sensitive)."""
    combined = "\n---\n".join(c.text for c in chunks)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _doc_hash(chunks: list[Chunk]) -> str:
    return _chunk_hash(chunks)


# ---------------------------------------------------------------------------
# ChromaDB client
# ---------------------------------------------------------------------------

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
        col = client.get_collection(settings.collection)
    except Exception:
        col = client.create_collection(
            name=settings.collection,
            metadata={
                "hnsw:space": "cosine",
                "embed_model": settings.embed_model,
            },
        )
    return col


def _add_chunks_to_collection(
    collection: Any,
    chunks: list[Chunk],
    vectors: list[list[float]],
    batch_size: int = 500,
) -> None:
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start: start + batch_size]
        batch_vecs = vectors[start: start + batch_size]
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
        ids = result.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0


def _archive_superseded(settings: IndexerSettings, source: str) -> None:
    """Copy superseded document file to archive directory."""
    corpus_path = Path(settings.corpus_dir)
    archive_path = corpus_path.parent / ARCHIVE_DIR_NAME
    try:
        archive_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        archive_path = Path("/tmp/archive")
        archive_path.mkdir(parents=True, exist_ok=True)
    for ext in [".md", ".pdf", ".html", ".txt"]:
        for f in corpus_path.glob(f"*{ext}"):
            if f.stem == Path(source).stem:
                dest = archive_path / f"{f.stem}_{time.strftime('%Y%m%d')}{f.suffix}"
                try:
                    shutil.copy2(str(f), str(dest))
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Incremental store (main)
# ---------------------------------------------------------------------------

def store_incremental(
    chunks: list[Chunk],
    vectors: list[list[float]],
    settings: IndexerSettings,
) -> int:
    """Incremental update: only re-embed and re-store changed documents.

    Returns the total number of chunks now in the collection.
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"store_incremental: {len(chunks)} chunks but {len(vectors)} vectors")

    client = get_client(settings)
    collection = _get_or_create_collection(client, settings)

    # Group chunks by source document
    from collections import defaultdict
    doc_chunks: dict[str, list[Chunk]] = defaultdict(list)
    doc_vecs: dict[str, list[list[float]]] = defaultdict(list)
    for chunk, vec in zip(chunks, vectors):
        doc_chunks[chunk.source].append(chunk)
        doc_vecs[chunk.source].append(vec)

    # Load previous hash store
    old_hashes = _load_hash_store(settings)
    new_hashes: dict[str, str] = {}

    new_count = 0
    unchanged_count = 0
    updated_count = 0

    for source, source_chunks in doc_chunks.items():
        new_hash = _doc_hash(source_chunks)
        new_hashes[source] = new_hash

        if old_hashes.get(source) == new_hash:
            # Unchanged — skip entirely
            unchanged_count += len(source_chunks)
            continue

        # Changed or new — delete old chunks, insert new
        deleted = _delete_doc_chunks(collection, source)
        _add_chunks_to_collection(collection, source_chunks, doc_vecs[source])

        if deleted > 0:
            updated_count += len(source_chunks)
            # Archive superseded version if flagged
            if source_chunks[0].superseded:
                _archive_superseded(settings, source)
        else:
            new_count += len(source_chunks)

    # Detect deleted documents (in old_hashes but not in current run)
    deleted_docs = set(old_hashes.keys()) - set(doc_chunks.keys())
    orphan_total = 0
    for deleted_source in deleted_docs:
        orphan_count = _delete_doc_chunks(collection, deleted_source)
        orphan_total += orphan_count

    # Persist updated hash store
    _save_hash_store(settings, new_hashes)

    # Invalidate Redis semantic cache
    _invalidate_cache(settings)

    total = collection.count()
    _log(
        "store_incremental done",
        new=new_count,
        updated=updated_count,
        unchanged=unchanged_count,
        orphans_deleted=orphan_total,
        total_in_collection=total,
    )
    return total


def _log(msg: str, **fields: object) -> None:
    import sys
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "logger": "rahbar.store", "msg": msg, **fields}), flush=True, file=sys.stdout)


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


# ---------------------------------------------------------------------------
# Full-replace store (kept for first-time / --full-rebuild)
# ---------------------------------------------------------------------------

def store(chunks: list[Chunk], vectors: list[list[float]], settings: IndexerSettings) -> int:
    """Full-replace: delete and recreate the collection atomically.
    Use for first-time setup or --full-rebuild. Normal runs use store_incremental().
    """
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
