"""Stage 3: Chunk. Structure-aware, with overlap.

Strategy (Markdown pipeline):
  1. Each Page already corresponds to one section (see extract.py).
  2. Within a section, split on paragraph boundaries (double newline).
  3. If a paragraph still exceeds CHUNK_SIZE, split it on sentence boundaries.
  4. Merge short fragments greedily until adding the next would exceed CHUNK_SIZE.
  5. Carry the last CHUNK_OVERLAP characters of the previous chunk as a prefix
     for the next one so retrieval has cross-boundary context.

Every chunk carries: source, section, page, doc_version, superseded.
IDs are deterministic: sha1(source + section + chunk_index).
"""

from __future__ import annotations

import hashlib
import re

from .models import Chunk, Document

# Simple sentence splitter: split after . ! ? followed by whitespace or end.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Extract the section heading from page text written by extract.py
# ("## Heading\n\n..." or just the body text).
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


def _make_id(source: str, section: str, index: int) -> str:
    key = f"{source}|{section}|{index}"
    return "c-" + hashlib.sha1(key.encode()).hexdigest()[:16]


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.strip())
    result = [p.strip() for p in parts if p.strip()]
    return result if result else [text.strip()]


def _split_by_words(text: str, size: int) -> list[str]:
    """Last-resort splitter: split on word boundaries to fit within `size` chars."""
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and current_len + addition > size:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += addition
    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


def _merge_into_chunks(pieces: list[str], size: int) -> list[str]:
    """Greedily merge pieces into chunks not exceeding `size` characters."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        # +1 for the space separator between pieces
        addition = len(piece) + (1 if current else 0)
        if current and current_len + addition > size:
            chunks.append(" ".join(current))
            current = [piece]
            current_len = len(piece)
        else:
            current.append(piece)
            current_len += addition

    if current:
        chunks.append(" ".join(current))

    return chunks


def _page_to_chunks(
    page_text: str,
    source: str,
    section: str,
    page: int,
    doc_version: str,
    superseded: bool,
    size: int,
    overlap: int,
    index_offset: int,
) -> list[Chunk]:
    """Convert one page's text into Chunk objects."""
    # Remove the embedded heading line so it's not duplicated in every chunk.
    body = _HEADING_RE.sub("", page_text).strip()

    paragraphs = _split_paragraphs(body)
    # Break overlong paragraphs into sentences, then words if still too long.
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= size:
            pieces.append(para)
        else:
            sentences = _split_sentences(para)
            for sent in sentences:
                if len(sent) <= size:
                    pieces.append(sent)
                else:
                    pieces.extend(_split_by_words(sent, size))

    raw_chunks = _merge_into_chunks(pieces, size)

    result: list[Chunk] = []
    prev_tail = ""
    for i, text in enumerate(raw_chunks):
        # Prepend overlap from the previous chunk.
        if prev_tail:
            text = prev_tail + " " + text
        chunk_id = _make_id(source, section, index_offset + i)
        result.append(
            Chunk(
                id=chunk_id,
                text=text,
                source=source,
                section=section,
                page=page,
                doc_version=doc_version,
                superseded=superseded,
            )
        )
        # Keep last `overlap` characters as the tail for the next chunk.
        prev_tail = text[-overlap:] if overlap and len(text) > overlap else ""

    return result


def chunk(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Split on sections, then paragraphs, then sentences, keeping each chunk under
    `size` tokens, with `overlap` tokens shared between neighbours. Every chunk keeps
    the section heading it sits under, its page, doc_version and superseded."""
    all_chunks: list[Chunk] = []
    index_offset = 0

    for page in doc.pages:
        # Determine the section from the embedded heading (extract.py adds "## Heading").
        m = _HEADING_RE.search(page.text)
        section = m.group(1).strip() if m else doc.source

        pieces = _page_to_chunks(
            page_text=page.text,
            source=doc.source,
            section=section,
            page=page.page,
            doc_version=doc.doc_version,
            superseded=doc.superseded,
            size=size,
            overlap=overlap,
            index_offset=index_offset,
        )
        all_chunks.extend(pieces)
        index_offset += len(pieces)

    return all_chunks
