"""Tests for RAGOps Part A: verification, incremental updates, archive/rollback, and Batch B."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from pipeline.config import IndexerSettings
from pipeline.models import Document, Page
from pipeline.rollback import list_archived, restore_archived_doc
from pipeline.run import run_pipeline
from pipeline.verify import (
    _deterministic_bow_vector,
    check_completeness,
    check_quality,
    check_recency,
    verify_corpus,
)


class InMemoryCollection:
    def __init__(self, name: str, metadata: dict) -> None:
        self.name = name
        self.metadata = metadata
        self.rows: dict[str, tuple[str, list[float], dict]] = {}

    def add(self, ids, documents, embeddings, metadatas) -> None:
        for i, d, e, m in zip(ids, documents, embeddings, metadatas):
            self.rows[i] = (d, e, m)

    def get(self, where=None, include=None) -> dict[str, Any]:
        ids = []
        metas = []
        docs = []
        for k, (d, _e, m) in self.rows.items():
            if where:
                match = all(m.get(wk) == wv for wk, wv in where.items())
                if not match:
                    continue
            ids.append(k)
            docs.append(d)
            metas.append(m)
        return {"ids": ids, "documents": docs, "metadatas": metas}

    def delete(self, ids) -> None:
        for i in ids:
            self.rows.pop(i, None)

    def count(self) -> int:
        return len(self.rows)


class InMemoryClient:
    def __init__(self) -> None:
        self.collections: dict[str, InMemoryCollection] = {}

    def get_collection(self, name: str) -> InMemoryCollection:
        if name not in self.collections:
            raise ValueError(f"missing {name}")
        return self.collections[name]

    def create_collection(self, name: str, metadata: dict) -> InMemoryCollection:
        col = InMemoryCollection(name, metadata)
        self.collections[name] = col
        return col

    def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)


def _settings(tmp_path: Path, corpus_dir: Path) -> IndexerSettings:
    return IndexerSettings(
        embed_model="nomic-embed-text",
        chroma_host="localhost",
        chroma_port=8000,
        collection="rahbar-test",
        corpus_dir=str(corpus_dir),
        chunk_size=400,
        chunk_overlap=50,
        ollama_url="http://127.0.0.1:9",
        datalake_dir=str(tmp_path / "datalake"),
        quarantine_dir=str(tmp_path / "quarantine"),
        archive_dir=str(tmp_path / "archive"),
        report_dir=str(tmp_path / "reports"),
    )


def _write_md(path: Path, source: str, title: str, body: str, version: str = "1.0", date: str = "") -> None:
    lines = ["---", f"title: {title}", f"source: {source}", f"version: {version}", "page_count: 1"]
    if date:
        lines.append(f"date: {date}")
    lines.extend(["---", "", body])
    path.write_text("\n".join(lines), encoding="utf-8")


def test_completeness_accepts_date_or_version() -> None:
    doc = Document(source="a.md", doc_version="1.0", superseded=False, pages=[Page("a.md", 1, "Complete sentence here with enough content to pass.")])
    # Only version, no date -> PASS
    assert check_completeness(doc, {"source": "a.md", "title": "A", "version": "1.0", "page_count": "1"}).passed is True
    # Only date, no version -> PASS
    assert check_completeness(doc, {"source": "a.md", "title": "A", "date": "2026-01-01", "page_count": "1"}).passed is True
    # Neither date nor version -> FAIL
    res = check_completeness(doc, {"source": "a.md", "title": "A", "page_count": "1"})
    assert res.passed is False
    assert "date or version" in res.reason


def test_deterministic_bow_vector_is_stable() -> None:
    v1 = _deterministic_bow_vector("FAST NUCES attendance policy 80 percent")
    v2 = _deterministic_bow_vector("FAST NUCES attendance policy 80 percent")
    assert v1 is not None and v1 == v2


def test_incremental_zero_embed_add_change_delete_and_empty(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_md(corpus / "doc1.md", "doc1.md", "Doc One", "## Sec 1\n\nThis is document one with plenty of text for testing.")
    _write_md(corpus / "doc2.md", "doc2.md", "Doc Two", "## Sec 2\n\nThis is document two with plenty of text for testing.")

    embed_calls: list[list[str]] = []

    def fake_embed(texts: list[str], _s: IndexerSettings) -> list[list[float]]:
        embed_calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]

    client = InMemoryClient()
    s = _settings(tmp_path, corpus)

    # 1. Initial run: embeds both docs
    n1 = run_pipeline(s, embed_fn=fake_embed, client=client)
    assert n1 == 2
    assert len(embed_calls) == 1
    assert client.get_collection("rahbar-test").count() == 2

    # 2. Unchanged run: MUST embed 0 chunks
    embed_calls.clear()
    n2 = run_pipeline(s, embed_fn=fake_embed, client=client)
    assert n2 == 0
    assert embed_calls == []
    assert client.get_collection("rahbar-test").count() == 2

    # 3. Add doc3: embeds ONLY doc3 (1 chunk)
    embed_calls.clear()
    _write_md(corpus / "doc3.md", "doc3.md", "Doc Three", "## Sec 3\n\nThis is document three with plenty of text for testing.")
    n3 = run_pipeline(s, embed_fn=fake_embed, client=client)
    assert n3 == 1
    assert len(embed_calls) == 1 and len(embed_calls[0]) == 1
    assert client.get_collection("rahbar-test").count() == 3

    # 4. Change doc1: embeds ONLY doc1 (1 chunk), doc2 & doc3 untouched
    embed_calls.clear()
    _write_md(corpus / "doc1.md", "doc1.md", "Doc One", "## Sec 1\n\nUpdated document one content with plenty of text for testing.")
    n4 = run_pipeline(s, embed_fn=fake_embed, client=client)
    assert n4 == 1
    assert len(embed_calls) == 1 and len(embed_calls[0]) == 1
    assert client.get_collection("rahbar-test").count() == 3

    # 5. Delete doc2: removes doc2 chunks, embeds 0
    embed_calls.clear()
    (corpus / "doc2.md").unlink()
    n5 = run_pipeline(s, embed_fn=fake_embed, client=client)
    assert n5 == 0
    assert embed_calls == []
    col = client.get_collection("rahbar-test")
    assert col.count() == 2
    assert col.get(where={"source": "doc2.md"})["ids"] == []

    # 6. Delete remaining docs (0 pass verification): still purges old chunks!
    (corpus / "doc1.md").unlink()
    (corpus / "doc3.md").unlink()
    n6 = run_pipeline(s, embed_fn=fake_embed, client=client)
    assert n6 == 0
    assert col.count() == 0


def test_superseded_archive_and_rollback(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_md(corpus / "rules-v1.md", "academic-rules.md", "Rules v1", "## Old\n\nOld warning policy text with enough characters to pass.", version="1.0", date="2023-08-15")
    _write_md(corpus / "rules-v2.md", "academic-rules.md", "Rules v2", "## New\n\nNew probation policy text with enough characters to pass.", version="2.0", date="2026-06-01")

    def fake_embed(texts: list[str], _s: IndexerSettings) -> list[list[float]]:
        return [[0.5, 0.5] for _ in texts]

    client = InMemoryClient()
    s = _settings(tmp_path, corpus)
    run_pipeline(s, embed_fn=fake_embed, client=client)

    # rules-v1 should be archived, NOT quarantined
    archived = list_archived(s)
    assert len(archived) == 1
    assert archived[0]["original_file"] == "rules-v1.md"

    # Restore rules-v1 and verify it is re-indexed with superseded=False
    res = restore_archived_doc("rules-v1.md", s, embed_fn=fake_embed, client=client, reindex=True)
    assert res["reindexed_chunks"] >= 1
    col = client.get_collection("rahbar-test")
    rows = col.get(where={"source": "rules-v1.md"})
    assert any(m.get("doc_version") == "1.0" and m.get("superseded") is False for m in rows["metadatas"])


def test_batch_b_verification_catches_planted_problems(tmp_path: Path) -> None:
    batch_b = Path(__file__).resolve().parents[2] / "corpus" / "batch_b"
    assert batch_b.exists()

    from pipeline.clean import clean
    from pipeline.extract import extract
    from pipeline.run import discover

    docs = discover(str(batch_b))
    assert len(docs) >= 10
    extracted = [(p, clean(extract(str(p), v, sup))) for p, v, sup in docs]

    passed, report = verify_corpus(
        extracted,
        quarantine_dir=str(tmp_path / "q"),
        report_dir=str(tmp_path / "r"),
        archive_dir=str(tmp_path / "a"),
    )
    by_file = {r.file: r for r in report.documents}
    assert by_file["empty-doc.md"].quality.passed is False
    assert by_file["garbled-doc.md"].quality.passed is False
    assert by_file["truncated-doc.md"].completeness.passed is False
    assert by_file["no-metadata.md"].completeness.passed is False
    assert by_file["rules-v1.md"].recency.passed is False
    assert by_file["rules-v1.md"].disposition == "archived"
    assert by_file["fees-dup-b.md"].uniqueness.passed is False
    assert by_file["fees-near-dup.md"].uniqueness.passed is False
    assert by_file["injected-doc.md"].injection_detected is True
    assert len(passed) >= 4
