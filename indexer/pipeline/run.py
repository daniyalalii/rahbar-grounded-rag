"""The indexer Job's entry point. It runs once and exits (section 4.2).

    python -m pipeline.run --dummy    # week 1: three fake chunks, no model needed
    python -m pipeline.run            # the real pipeline: extract, verify, chunk, incremental embed & store

Exit code 0 means the index was written; anything else fails the Job, and
Kubernetes retries it up to the Job's backoffLimit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

from .config import IndexerSettings, get_settings
from .models import Chunk


def log(msg: str, **fields: object) -> None:
    print(
        json.dumps(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "logger": "rahbar.indexer",
                "msg": msg,
                **fields,
            }
        ),
        flush=True,
    )


def _front_matter(path: Path) -> dict[str, str]:
    """Read `key: value` lines between leading '---' markers (used by eval/fixtures)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        if key.strip():
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta


def discover(corpus_dir: str) -> list[tuple[Path, str, bool]]:
    """List (path, doc_version, superseded) for every document to index."""
    root = Path(corpus_dir)
    index = root / "_index.json"
    if index.exists():
        entries = json.loads(index.read_text(encoding="utf-8"))
        out = []
        for e in entries:
            p = root / e["file"]
            if p.exists():
                out.append((p, str(e.get("doc_version", "unknown")), bool(e.get("superseded", False))))
        return out
    found = []
    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in {".pdf", ".md", ".html", ".txt"}:
            continue
        if path.name.lower().startswith("readme"):
            continue
        meta = _front_matter(path) if path.suffix.lower() == ".md" else {}
        ver = meta.get("doc_version", meta.get("version", "unknown"))
        found.append((path, ver, meta.get("superseded", "false").lower() == "true"))
    return found


def run_pipeline(
    settings: IndexerSettings,
    embed_fn: Optional[Callable[[list[str], IndexerSettings], list[list[float]]]] = None,
    client: Any = None,
) -> int:
    from .chunk import chunk
    from .clean import clean
    from .embed import embed
    from .extract import extract
    from .models import Document
    from .store import sync_incremental
    from .verify import verify_corpus

    documents = discover(settings.corpus_dir)

    # --- Stage 1b: Verification (RAGOps 4.2.1-4.2.4) ---
    extracted: list[tuple[Path, Document]] = []
    for path, doc_version, superseded in documents:
        doc = clean(extract(str(path), doc_version, superseded))
        extracted.append((path, doc))

    corpus_parent = Path(settings.corpus_dir).parent
    quarantine_dir = settings.quarantine_dir or str(corpus_parent / "quarantine")
    archive_dir = settings.archive_dir or str(corpus_parent / "archive")
    report_dir = settings.report_dir or str(corpus_parent.parent / "eval" / "results")

    passed_docs, report = verify_corpus(
        extracted, quarantine_dir, report_dir, archive_dir=archive_dir
    )
    log(
        "verification complete",
        total=report.total,
        passed=report.passed,
        quarantined=report.quarantined,
        archived=report.archived,
        embedding_backend=report.embedding_backend,
    )
    if report.embedding_warning:
        log("verification warning", warning=report.embedding_warning)

    for r in report.documents:
        if not r.overall_passed:
            failed_checks = [
                k for k in ("quality", "completeness", "recency", "uniqueness")
                if not getattr(r, k).passed
            ]
            log(
                r.disposition,
                file=r.file,
                failed_checks=failed_checks,
                injection=r.injection_detected,
            )

    # --- Stage 2: Chunk verified docs only (before embedding!) ---
    doc_chunks: dict[str, list[Chunk]] = defaultdict(list)
    for path, doc in passed_docs:
        pieces = chunk(doc, settings.chunk_size, settings.chunk_overlap)
        log("chunked", source=path.name, chunks=len(pieces))
        if pieces:
            doc_chunks[doc.source].extend(pieces)

    # --- Stages 3-5: Hash check -> Embed ONLY new/changed -> Incremental store ---
    # Note: sync_incremental runs even when doc_chunks is empty so deleted docs are purged!
    summary = sync_incremental(
        doc_chunks,
        settings,
        embed_fn=embed_fn or embed,
        client=client,
    )
    log("incremental summary", **summary)
    return summary["newly_embedded_chunks"] + summary["updated_chunks"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the RAHBAR index, then exit.")
    parser.add_argument("--dummy", action="store_true", help="write three fake chunks")
    parser.add_argument("--fetch", action="store_true", help="fetch and verify real corpus before indexing")
    args = parser.parse_args()
    settings = get_settings()
    log(
        "indexer starting",
        dummy=args.dummy,
        fetch=args.fetch or settings.fetch_corpus,
        embed_model=settings.embed_model,
        collection=settings.collection,
        chroma=f"{settings.chroma_host}:{settings.chroma_port}",
        corpus_dir=settings.corpus_dir,
        pod=os.environ.get("HOSTNAME", "-"),
    )
    started = time.perf_counter()
    try:
        if args.fetch or settings.fetch_corpus:
            from .fetch import fetch_and_verify
            log("fetching real corpus", target_dir=settings.corpus_dir)
            fetch_and_verify(settings.corpus_dir)
            log("corpus fetch and verification complete")

        if args.dummy:
            from .dummy import write_dummy_index
            from .store import get_client

            count = write_dummy_index(settings, get_client(settings))
        else:
            count = run_pipeline(settings)
    except NotImplementedError as exc:
        log("indexer stage not implemented yet", error=str(exc))
        return 2
    except Exception as exc:
        log("indexer failed", error=str(exc))
        return 1
    log("indexer finished", embedded_or_updated=count, seconds=round(time.perf_counter() - started, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
