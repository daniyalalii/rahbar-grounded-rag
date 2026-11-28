"""RAGOps Part A3: Superseded Version Rollback / Restore.

Lists archived document versions and restores a specified archived version
back into the active corpus (marking superseded=false) and re-indexes it
so it is retrievable again.

Usage:
    python -m pipeline.rollback --list
    python -m pipeline.rollback --restore rules-v1.md
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from .config import IndexerSettings, get_settings


def _archive_dir(settings: IndexerSettings) -> Path:
    if settings.archive_dir:
        return Path(settings.archive_dir)
    return Path(settings.corpus_dir).parent / "archive"


def list_archived(settings: IndexerSettings) -> list[dict]:
    """Return metadata dicts for all archived document versions."""
    arch = _archive_dir(settings)
    if not arch.exists():
        return []
    items = []
    for meta_file in sorted(arch.glob("*.meta.json")):
        try:
            items.append(json.loads(meta_file.read_text(encoding="utf-8")))
        except Exception:
            continue
    return items


def restore_archived_doc(
    target: str,
    settings: IndexerSettings,
    embed_fn: Optional[Callable[[list[str], IndexerSettings], list[list[float]]]] = None,
    client: Any = None,
    reindex: bool = True,
) -> dict:
    """Restore an archived document version to corpus_dir and re-index it."""
    arch = _archive_dir(settings)
    corpus_dir = Path(settings.corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    archived_items = list_archived(settings)
    match: Optional[dict] = None
    for item in archived_items:
        if target in (
            item.get("original_file"),
            item.get("archived_file"),
            item.get("source"),
        ):
            match = item
            break

    if match is None:
        raise FileNotFoundError(f"Archived document '{target}' not found in {arch}")

    src_file = arch / match["archived_file"]
    if not src_file.exists():
        raise FileNotFoundError(f"Archived payload missing: {src_file}")

    dest_file = corpus_dir / match["original_file"]
    shutil.copy2(str(src_file), str(dest_file))

    # Ensure markdown front matter has superseded: false if present
    if dest_file.suffix.lower() == ".md":
        text = dest_file.read_text(encoding="utf-8", errors="replace")
        if re.search(r"(?m)^superseded\s*:", text):
            text = re.sub(r"(?m)^superseded\s*:.*$", "superseded: false", text)
            dest_file.write_text(text, encoding="utf-8")

    # Update _index.json if present
    idx_file = corpus_dir / "_index.json"
    if idx_file.exists():
        entries = json.loads(idx_file.read_text(encoding="utf-8"))
        updated = False
        for e in entries:
            if e.get("file") == match["original_file"]:
                e["superseded"] = False
                updated = True
        if not updated:
            entries.append({
                "file": match["original_file"],
                "doc_version": match.get("doc_version", "1.0"),
                "superseded": False,
            })
        idx_file.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    result = {
        "restored_file": str(dest_file),
        "source": match.get("source"),
        "doc_version": match.get("doc_version"),
        "reindexed_chunks": 0,
    }

    if reindex:
        from .chunk import chunk
        from .clean import clean
        from .embed import embed
        from .extract import extract
        from .store import _add_chunks_to_collection, _delete_doc_chunks, _get_or_create_collection, get_client

        doc = clean(extract(str(dest_file), str(match.get("doc_version", "1.0")), False))
        chunks = chunk(doc, settings.chunk_size, settings.chunk_overlap)
        if chunks:
            efn = embed_fn or embed
            vecs = efn([c.text for c in chunks], settings)
            cli = client or get_client(settings)
            col = _get_or_create_collection(cli, settings)
            _delete_doc_chunks(col, doc.source)
            _add_chunks_to_collection(col, chunks, vecs)
            result["reindexed_chunks"] = len(chunks)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGOps Part A3: Archive rollback/restore")
    parser.add_argument("--list", action="store_true", help="List archived document versions")
    parser.add_argument("--restore", metavar="FILE", help="Restore an archived file (e.g. rules-v1.md)")
    parser.add_argument("--no-reindex", action="store_true", help="Only restore file, do not re-index")
    args = parser.parse_args()

    settings = get_settings()
    if args.list:
        items = list_archived(settings)
        if not items:
            print("No archived versions found.")
            return 0
        for it in items:
            print(f"- {it.get('original_file')} (source={it.get('source')}, version={it.get('doc_version')}, archived_at={it.get('archived_at')})")
        return 0

    if args.restore:
        res = restore_archived_doc(args.restore, settings, reindex=not args.no_reindex)
        print(json.dumps(res, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

