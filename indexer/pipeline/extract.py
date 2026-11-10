"""Stage 1: Extract. PDF, HTML, Markdown, and TXT to text, page by page.

Supports:
  - Markdown (.md): front-matter stripping, heading-based section splitting.
  - Plain text (.txt): paragraph/page splitting.
  - HTML (.html, .htm): table preservation and clean text extraction.
  - Native PDF (.pdf): page-by-page text and table extraction using pdfplumber / pypdf.
  - Scanned PDF: page-by-page OCR fallback via pytesseract / pdf2image if available;
    honest graceful documentation if OCR binaries are unavailable on the host.
"""

from __future__ import annotations

import html.parser
import re
from pathlib import Path

from .models import Document, Page

# Optional OCR / PDF libraries
try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore

try:
    import pypdf
except ImportError:
    pypdf = None  # type: ignore

try:
    import pytesseract  # type: ignore[import-untyped]
    from pdf2image import convert_from_path
except ImportError:
    pytesseract = None  # type: ignore
    convert_from_path = None  # type: ignore


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
    # drop empty-text sections
    return [(h, t) for h, t in sections if t.strip()]


class _HTMLContentExtractor(html.parser.HTMLParser):
    """Extract readable text and preserve table structures from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_cell = False
        self.current_cell: list[str] = []
        self.current_row: list[str] = []
        self.output_parts: list[str] = []
        self.current_heading = ""
        self.in_heading = False
        self.heading_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.in_heading = True
            self.heading_buffer = []
        elif tag == "table":
            self.in_table = True
            self.output_parts.append("\n\n")
        elif tag == "tr":
            self.current_row = []
        elif tag in ("td", "th"):
            self.in_cell = True
            self.current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.in_heading = False
            heading_text = " ".join("".join(self.heading_buffer).split())
            if heading_text:
                self.output_parts.append(f"\n\n## {heading_text}\n\n")
        elif tag == "table":
            self.in_table = False
            self.output_parts.append("\n\n")
        elif tag == "tr":
            if self.current_row:
                # Format row as a markdown table row or pipe-separated values
                self.output_parts.append("| " + " | ".join(self.current_row) + " |\n")
                self.current_row = []
        elif tag in ("td", "th"):
            self.in_cell = False
            cell_text = " ".join("".join(self.current_cell).split())
            self.current_row.append(cell_text)

    def handle_data(self, data: str) -> None:
        if self.in_heading:
            self.heading_buffer.append(data)
        elif self.in_cell:
            self.current_cell.append(data)
        elif not self.in_table:
            text = data.strip()
            if text:
                self.output_parts.append(" " + text + " ")

    def get_text(self) -> str:
        raw = "".join(self.output_parts)
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def extract_html(path: Path, doc_version: str, superseded: bool) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace")
    parser = _HTMLContentExtractor()
    parser.feed(raw)
    content = parser.get_text()

    source = path.name
    doc = Document(source=source, doc_version=doc_version, superseded=superseded)
    doc.pages.append(Page(source=source, page=1, text=content, ocr=False))
    return doc


def extract_txt(path: Path, doc_version: str, superseded: bool) -> Document:
    raw = path.read_text(encoding="utf-8", errors="replace")
    source = path.name
    doc = Document(source=source, doc_version=doc_version, superseded=superseded)
    doc.pages.append(Page(source=source, page=1, text=raw.strip(), ocr=False))
    return doc


def extract_md(path: Path, doc_version: str, superseded: bool) -> Document:
    raw = path.read_text(encoding="utf-8")
    front_matter, body = _parse_front_matter(raw)

    effective_version = front_matter.get("doc_version", doc_version)
    effective_superseded_raw = front_matter.get("superseded", "")
    if effective_superseded_raw:
        effective_superseded = effective_superseded_raw.lower() == "true"
    else:
        effective_superseded = superseded

    sections = _split_sections(body)
    source = path.name
    doc = Document(
        source=source,
        doc_version=effective_version,
        superseded=effective_superseded,
    )
    for heading, text in sections:
        prefix = f"## {heading}\n\n" if heading else ""
        doc.pages.append(Page(source=source, page=1, text=prefix + text, ocr=False))

    return doc


def _ocr_pdf_page(pdf_path: Path, page_num: int) -> str:
    """Attempt OCR on a specific page of a PDF using pypdf image extraction, pdf2image, and pytesseract."""
    if pytesseract is None:
        return ""

    import os
    import sys
    if sys.platform == "win32":
        default_win_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_win_tesseract):
            pytesseract.pytesseract.tesseract_cmd = default_win_tesseract

    # 1. Try direct image extraction via pypdf (pure Python, does not require poppler binary)
    if pypdf is not None:
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            if 0 <= page_num - 1 < len(reader.pages):
                page = reader.pages[page_num - 1]
                if getattr(page, "images", None) and len(page.images) > 0:
                    import io
                    from PIL import Image
                    for img in page.images:
                        pil_img = Image.open(io.BytesIO(img.data))
                        txt = str(pytesseract.image_to_string(pil_img)).strip()
                        if len(txt) > 20:
                            return txt
        except Exception:
            pass

    # 2. Try pdf2image rendering (used in Linux container with poppler-utils)
    if convert_from_path is not None:
        try:
            images = convert_from_path(
                str(pdf_path),
                first_page=page_num,
                last_page=page_num,
                dpi=200,
            )
            if images:
                return str(pytesseract.image_to_string(images[0])).strip()
        except Exception:
            pass

    return ""


def extract_pdf(path: Path, doc_version: str, superseded: bool) -> Document:
    source = path.name
    doc = Document(source=source, doc_version=doc_version, superseded=superseded)

    # 1. Try pdfplumber first for rich text & table extraction
    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(path)) as pdf:
                for i, p in enumerate(pdf.pages, start=1):
                    # Check for tables on the page
                    tables = p.extract_tables() or []
                    table_text_parts = []
                    for table in tables:
                        for row in table:
                            clean_cells = [
                                " ".join((c or "").split()) for c in row
                            ]
                            if any(clean_cells):
                                table_text_parts.append(
                                    "| " + " | ".join(clean_cells) + " |"
                                )
                    if table_text_parts:
                        table_text = (
                            "\n\n### Tables on Page "
                            + str(i)
                            + "\n"
                            + "\n".join(table_text_parts)
                            + "\n\n"
                        )
                    else:
                        table_text = ""

                    raw_text = p.extract_text() or ""
                    combined = (raw_text + "\n" + table_text).strip()

                    # Detect if page is scanned (empty text)
                    if len(combined) < 20:
                        ocr_text = _ocr_pdf_page(path, i)
                        if ocr_text:
                            doc.pages.append(
                                Page(
                                    source=source,
                                    page=i,
                                    text=ocr_text,
                                    ocr=True,
                                )
                            )
                        else:
                            # Note: image-only page without OCR runtime available
                            doc.pages.append(
                                Page(
                                    source=source,
                                    page=i,
                                    text=f"[Scanned page {i} - text not extractable without OCR service]",
                                    ocr=False,
                                )
                            )
                    else:
                        doc.pages.append(
                            Page(
                                source=source,
                                page=i,
                                text=combined,
                                ocr=False,
                            )
                        )
            return doc
        except Exception:
            pass

    # 2. Fallback to pypdf if pdfplumber fails or is missing
    if pypdf is not None:
        reader = pypdf.PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if len(text) < 20:
                ocr_text = _ocr_pdf_page(path, i)
                if ocr_text:
                    doc.pages.append(
                        Page(source=source, page=i, text=ocr_text, ocr=True)
                    )
                else:
                    doc.pages.append(
                        Page(
                            source=source,
                            page=i,
                            text=f"[Scanned page {i} - text not extractable without OCR service]",
                            ocr=False,
                        )
                    )
            else:
                doc.pages.append(
                    Page(source=source, page=i, text=text, ocr=False)
                )
        return doc

    raise RuntimeError(
        f"extract_pdf: neither pdfplumber nor pypdf available to extract {path}"
    )


def extract(path: str, doc_version: str, superseded: bool) -> Document:
    """Return the document's text page by page.

    Supports Markdown, HTML, TXT, and PDF (with table extraction and OCR detection).
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".md":
        return extract_md(p, doc_version, superseded)
    elif suffix in (".html", ".htm"):
        return extract_html(p, doc_version, superseded)
    elif suffix == ".txt":
        return extract_txt(p, doc_version, superseded)
    elif suffix == ".pdf":
        return extract_pdf(p, doc_version, superseded)
    else:
        raise ValueError(
            f"extract: unsupported file format {suffix} for {path}"
        )
