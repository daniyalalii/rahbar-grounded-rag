"""Tests for the real pipeline stages (extract, clean, chunk, embed, store).

All fakes — no network, no Ollama, no Chroma server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from pipeline.chunk import chunk
from pipeline.clean import clean
from pipeline.config import IndexerSettings
from pipeline.embed import embed
from pipeline.extract import extract, _parse_front_matter, _split_sections
from pipeline.models import Chunk, Document, Page
from pipeline.store import store


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_MD = """\
---
source: test-doc.md
doc_version: 2026
superseded: false
---
# Test Document

Intro paragraph.

## 1. First Section

This is the first section body. It has two sentences. Very useful text.

## 2. Second Section

Another section with content. More text follows here.
"""

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "eval" / "fixtures"


def _make_settings(**overrides: Any) -> IndexerSettings:
    base = dict(
        embed_model="nomic-embed-text",
        chroma_host="localhost",
        chroma_port=8000,
        collection="rahbar",
        corpus_dir="/corpus",
        chunk_size=400,
        chunk_overlap=50,
        ollama_url="http://localhost:11434",
    )
    base.update(overrides)
    return IndexerSettings(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FakeCollection / FakeClient  (mirrors test_starter_dummy.py)
# ---------------------------------------------------------------------------

class FakeCollection:
    def __init__(self, name: str, metadata: dict) -> None:
        self.name = name
        self.metadata = metadata
        self.rows: dict[str, tuple] = {}

    def add(self, ids, documents, embeddings, metadatas) -> None:
        for i, d, e, m in zip(ids, documents, embeddings, metadatas):
            self.rows[i] = (d, e, m)

    def count(self) -> int:
        return len(self.rows)


class FakeClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def delete_collection(self, name: str) -> None:
        if name not in self.collections:
            raise ValueError("no such collection")
        del self.collections[name]

    def create_collection(self, name: str, metadata: dict) -> FakeCollection:
        col = FakeCollection(name, metadata)
        self.collections[name] = col
        return col


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------

def test_parse_front_matter_basic() -> None:
    meta, body = _parse_front_matter("---\ndoc_version: 2026\nsuperseded: false\n---\n# Hello\n")
    assert meta["doc_version"] == "2026"
    assert meta["superseded"] == "false"
    assert "# Hello" in body


def test_parse_front_matter_absent() -> None:
    meta, body = _parse_front_matter("# No front matter\n")
    assert meta == {}
    assert "# No front matter" in body


def test_split_sections_multiple() -> None:
    body = "Intro.\n\n## A\n\nText A.\n\n## B\n\nText B.\n"
    sections = _split_sections(body)
    # Intro under ''
    assert sections[0][0] == ""
    headings = [h for h, _ in sections]
    assert "A" in headings
    assert "B" in headings


def test_extract_md_returns_document(tmp_path: Path) -> None:
    md = tmp_path / "test.md"
    md.write_text(SAMPLE_MD, encoding="utf-8")
    doc = extract(str(md), "2026", False)
    assert doc.source == "test.md"
    assert doc.doc_version == "2026"
    assert not doc.superseded
    # Sections: intro + "1. First Section" + "2. Second Section"
    assert len(doc.pages) >= 2
    # Every page is page 1 (Markdown has no physical pages)
    assert all(p.page == 1 for p in doc.pages)


def test_extract_md_reads_front_matter_version(tmp_path: Path) -> None:
    md = tmp_path / "doc.md"
    md.write_text(SAMPLE_MD, encoding="utf-8")
    # caller passes wrong version; front matter wins
    doc = extract(str(md), "WRONG", True)
    assert doc.doc_version == "2026"
    assert doc.superseded is False


def test_extract_unsupported_format_raises(tmp_path: Path) -> None:
    doc_file = tmp_path / "file.docx"
    doc_file.write_bytes(b"PK\x03\x04")
    with pytest.raises(ValueError):
        extract(str(doc_file), "2026", False)


def test_extract_html_file(tmp_path: Path) -> None:
    html_file = tmp_path / "page.html"
    html_file.write_text("<h1>Fee Structure</h1><table><tr><th>Item</th><th>Fee</th></tr><tr><td>Tuition</td><td>1000</td></tr></table>", encoding="utf-8")
    doc = extract(str(html_file), "2026", False)
    assert doc.source == "page.html"
    assert len(doc.pages) == 1
    assert "Fee Structure" in doc.pages[0].text
    assert "Tuition" in doc.pages[0].text


def test_extract_fixture_superseded(tmp_path: Path) -> None:
    path = FIXTURE_DIR / "academic-rules-2023.md"
    if not path.exists():
        pytest.skip("fixture not available")
    doc = extract(str(path), "unknown", False)
    assert doc.superseded is True
    assert doc.doc_version == "2023"


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------

def test_clean_strips_trailing_whitespace() -> None:
    page = Page(source="x.md", page=1, text="hello   \nworld  \n")
    doc = Document(source="x.md", doc_version="1", superseded=False, pages=[page])
    cleaned = clean(doc)
    for line in cleaned.pages[0].text.splitlines():
        assert line == line.rstrip(), f"trailing whitespace in: {line!r}"


def test_clean_collapses_blank_lines() -> None:
    text = "a\n\n\n\n\nb"
    page = Page(source="x.md", page=1, text=text)
    doc = Document(source="x.md", doc_version="1", superseded=False, pages=[page])
    cleaned = clean(doc)
    # At most two consecutive blank lines (3 newlines = "line\n\n\nline") allowed;
    # four or more consecutive blank lines must be collapsed.
    result = cleaned.pages[0].text
    assert "\n\n\n\n" not in result


def test_clean_preserves_headings() -> None:
    text = "## My Section\n\nBody text.\n"
    page = Page(source="x.md", page=1, text=text)
    doc = Document(source="x.md", doc_version="1", superseded=False, pages=[page])
    cleaned = clean(doc)
    assert "## My Section" in cleaned.pages[0].text


def test_clean_preserves_metadata() -> None:
    page = Page(source="src.md", page=2, text="text", ocr=True)
    doc = Document(source="src.md", doc_version="v1", superseded=True, pages=[page])
    cleaned = clean(doc)
    assert cleaned.doc_version == "v1"
    assert cleaned.superseded is True
    assert cleaned.pages[0].ocr is True


# ---------------------------------------------------------------------------
# chunk
# ---------------------------------------------------------------------------

def _make_doc_with_text(text: str) -> Document:
    page = Page(source="x.md", page=1, text=text)
    return Document(source="x.md", doc_version="2026", superseded=False, pages=[page])


def test_chunk_produces_chunks() -> None:
    doc = _make_doc_with_text(
        "## 1. Attendance\n\nStudents must attend 80 percent of lectures. "
        "Late arrivals count as half absences. Grade FA is awarded otherwise."
    )
    result = chunk(doc, size=200, overlap=20)
    assert len(result) >= 1
    for c in result:
        assert isinstance(c, Chunk)
        assert c.source == "x.md"
        assert c.section  # non-empty
        assert c.page == 1
        assert c.doc_version == "2026"
        assert c.superseded is False


def test_chunk_respects_size() -> None:
    long_text = "Word " * 200  # ~1000 chars
    doc = _make_doc_with_text("## Big\n\n" + long_text)
    result = chunk(doc, size=100, overlap=0)
    # Every chunk should fit within a reasonable margin of the size limit
    for c in result:
        # overlap prepend may push slightly over size, but raw pieces fit
        assert len(c.text) < 300  # generous bound


def test_chunk_overlap_prefix() -> None:
    # With overlap, the second chunk should contain the tail of the first.
    text = "## S\n\n" + "Alpha beta gamma. " * 30
    doc = _make_doc_with_text(text)
    result = chunk(doc, size=120, overlap=30)
    if len(result) >= 2:
        tail = result[0].text[-30:]
        assert tail in result[1].text


def test_chunk_ids_are_stable() -> None:
    doc = _make_doc_with_text("## Sec\n\nSome content here.")
    r1 = chunk(doc, 200, 20)
    r2 = chunk(doc, 200, 20)
    assert [c.id for c in r1] == [c.id for c in r2]


def test_chunk_ids_are_unique() -> None:
    doc = _make_doc_with_text("## Big\n\n" + "sentence. " * 100)
    result = chunk(doc, size=80, overlap=0)
    ids = [c.id for c in result]
    assert len(ids) == len(set(ids))


def test_chunk_fixture_document() -> None:
    path = FIXTURE_DIR / "academic-rules-2026.md"
    if not path.exists():
        pytest.skip("fixture not available")
    doc = extract(str(path), "2026", False)
    doc = clean(doc)
    result = chunk(doc, size=400, overlap=50)
    assert len(result) >= 3
    sections = {c.section for c in result}
    # Should have at least the three named sections
    assert len(sections) >= 2


# ---------------------------------------------------------------------------
# embed  (fake httpx transport)
# ---------------------------------------------------------------------------

class FakeEmbedTransport(httpx.BaseTransport):  # type: ignore[misc]
    """Returns fake embeddings without touching the network."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def handle_request(self, request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        import json
        body = json.loads(request.content)
        n = len(body["input"])
        embeddings = [[0.1] * self.dim for _ in range(n)]
        return httpx.Response(200, json={"embeddings": embeddings})



def _patched_embed(texts: list[str], settings: IndexerSettings, dim: int = 768) -> list[list[float]]:
    """Call embed() with a fake transport patched in."""
    import pipeline.embed as embed_mod
    original_post = httpx.post

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        client = httpx.Client(transport=FakeEmbedTransport(dim))
        return client.post(url, **kwargs)

    embed_mod.httpx.post = fake_post  # type: ignore[attr-defined]
    try:
        return embed(texts, settings)
    finally:
        embed_mod.httpx.post = original_post  # type: ignore[attr-defined]


def test_embed_returns_one_vector_per_text() -> None:
    settings = _make_settings()
    result = _patched_embed(["hello", "world"], settings, dim=768)
    assert len(result) == 2
    assert len(result[0]) == 768


def test_embed_empty_list() -> None:
    settings = _make_settings()
    assert embed([], settings) == []


def test_embed_raises_on_mismatch() -> None:
    """If Ollama returns wrong number of vectors, embed must raise."""
    settings = _make_settings()
    import pipeline.embed as embed_mod

    class OneBadTransport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:  # type: ignore[override]
            return httpx.Response(200, json={"embeddings": [[0.1] * 768]},
                                  request=request)  # only 1 vector for 2 texts

    original_post = embed_mod.httpx.post

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        client = httpx.Client(transport=OneBadTransport())
        return client.post(url, **kwargs)

    embed_mod.httpx.post = fake_post  # type: ignore[attr-defined]
    try:
        with pytest.raises(ValueError, match="expected 2"):
            embed(["a", "b"], settings)
    finally:
        embed_mod.httpx.post = original_post  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

def _fake_chunks(n: int = 3) -> tuple[list[Chunk], list[list[float]]]:
    dim = 4
    chunks_list = [
        Chunk(id=f"id-{i}", text=f"text {i}", source="x.md",
              section="S", page=1, doc_version="2026", superseded=False)
        for i in range(n)
    ]
    vectors = [[float(i)] * dim for i in range(n)]
    return chunks_list, vectors


def test_store_writes_all_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings()
    fake_client = FakeClient()
    monkeypatch.setattr("pipeline.store.get_client", lambda s: fake_client)

    chunks_list, vectors = _fake_chunks(3)
    count = store(chunks_list, vectors, settings)

    assert count == 3
    col = fake_client.collections["rahbar"]
    assert col.metadata["embed_model"] == settings.embed_model
    assert col.metadata["dimension"] == len(vectors[0])


def test_store_sets_collection_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(embed_model="my-model")
    fake_client = FakeClient()
    monkeypatch.setattr("pipeline.store.get_client", lambda s: fake_client)

    chunks_list, vectors = _fake_chunks(2)
    store(chunks_list, vectors, settings)

    col = fake_client.collections["rahbar"]
    assert col.metadata["embed_model"] == "my-model"
    assert col.metadata["hnsw:space"] == "cosine"


def test_store_replaces_existing_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings()
    fake_client = FakeClient()
    monkeypatch.setattr("pipeline.store.get_client", lambda s: fake_client)

    chunks_list, vectors = _fake_chunks(3)
    store(chunks_list, vectors, settings)
    # Run again — collection must be replaced, not appended.
    new_chunks, new_vectors = _fake_chunks(2)
    count = store(new_chunks, new_vectors, settings)
    assert count == 2


def test_store_raises_on_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings()
    fake_client = FakeClient()
    monkeypatch.setattr("pipeline.store.get_client", lambda s: fake_client)
    with pytest.raises(ValueError, match="no chunks"):
        store([], [], settings)


def test_store_raises_on_length_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings()
    fake_client = FakeClient()
    monkeypatch.setattr("pipeline.store.get_client", lambda s: fake_client)
    chunks_list, _ = _fake_chunks(3)
    with pytest.raises(ValueError, match="vectors"):
        store(chunks_list, [[0.0] * 4], settings)


def test_store_chunk_metadata_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings()
    fake_client = FakeClient()
    monkeypatch.setattr("pipeline.store.get_client", lambda s: fake_client)

    chunks_list, vectors = _fake_chunks(2)
    store(chunks_list, vectors, settings)

    col = fake_client.collections["rahbar"]
    for _doc, _vec, meta in col.rows.values():
        assert set(meta) == {"source", "section", "page", "doc_version", "superseded"}
