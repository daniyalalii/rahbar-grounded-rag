"""Stage 1: Extract. PDF (or HTML, or Markdown) to text, page by page.

For the fixture corpus (Markdown only). PDF/OCR paths raise NotImplementedError
so the CI fixtures work without system packages in the test environment.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import Document, Page


def _parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Strip YAML front matter delimited by '---' and return (meta, body)."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict[str, str] = {}
    end = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = i
            break
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    if end is None:
        return {}, text
    body = "".join(lines[end + 1 :])
    return meta, body


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split Markdown body into (heading, text) pairs.

    Text before the first heading is kept under a synthetic '' heading.
    """
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[str, str]] = []
    last_heading = ""
    last_pos = 0
    for m in heading_re.finditer(body):
        if m.start() > last_pos:
            sections.append((last_heading, body[last_pos : m.start()]))
        last_heading = m.group(2).strip()
        last_pos = m.end() + 1  # skip the newline after the heading
    # remainder after last heading
    if last_pos <= len(body):
        sections.append((last_heading, body[last_pos:]))
    # drop empty-text sections (e.g., heading immediately followed by another heading)
    return [(h, t) for h, t in sections if t.strip()]


def extract(path: str, doc_version: str, superseded: bool) -> Document:
    """Return the document's text page by page.

    Only Markdown is implemented. PDF/HTML/TXT raise NotImplementedError.
    Detect pages with no text layer and OCR them; mark those pages ocr=True.
    The OCR libraries need system packages in your Dockerfile (tesseract-ocr,
    poppler-utils), not only pip packages.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix != ".md":
        raise NotImplementedError(
            f"extract: only .md supported in this implementation; got {suffix}. "
            "See STARTER-TODO(3.1) for PDF/OCR path."
        )

    raw = p.read_text(encoding="utf-8")
    front_matter, body = _parse_front_matter(raw)

    # Front matter may override caller-supplied values (run.py already reads
    # front matter via discover(), but we honour it here too for correctness).
    effective_version = front_matter.get("doc_version", doc_version)
    effective_superseded_raw = front_matter.get("superseded", "")
    if effective_superseded_raw:
        effective_superseded = effective_superseded_raw.lower() == "true"
    else:
        effective_superseded = superseded

    sections = _split_sections(body)

    # For Markdown there is a single logical "page"; we emit one Page per
    # section so downstream code gets fine-grained units, all labelled page=1.
    source = p.name
    doc = Document(source=source, doc_version=effective_version,
                   superseded=effective_superseded)
    for heading, text in sections:
        # Embed the heading into the page text so the cleaner/chunker sees it.
        prefix = f"## {heading}\n\n" if heading else ""
        doc.pages.append(Page(source=source, page=1, text=prefix + text, ocr=False))

    return doc
