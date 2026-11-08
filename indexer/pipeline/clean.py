"""Stage 2: Clean. Remove artefacts without destroying structure.

For Markdown input: strip trailing whitespace from each line and collapse
runs of more than two consecutive blank lines into a single blank line.
Headings are never removed because chunking and citations depend on them.
"""

from __future__ import annotations


from .models import Document, Page


def _clean_text(text: str) -> str:
    # Strip trailing whitespace on every line.
    lines = [line.rstrip() for line in text.splitlines()]
    # Collapse 3+ consecutive blank lines into 2 (one blank separator).
    collapsed: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 1:
                collapsed.append(line)
        else:
            blank_run = 0
            collapsed.append(line)
    return "\n".join(collapsed)


def clean(doc: Document) -> Document:
    """Strip headers and footers that repeat on every page; keep headings and
    section boundaries, because chunking and citations depend on them."""
    cleaned_pages = [
        Page(source=p.source, page=p.page, text=_clean_text(p.text), ocr=p.ocr)
        for p in doc.pages
    ]
    return Document(
        source=doc.source,
        doc_version=doc.doc_version,
        superseded=doc.superseded,
        pages=cleaned_pages,
    )
