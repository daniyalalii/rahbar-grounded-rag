"""Stage 1b: Verify. Runs after extract(), before chunk().

RAGOps paper section 4.2.1-4.2.4.

For every document, four checks are performed:
  - quality:      text is not empty, garbled, or extraction-failed
  - completeness: required metadata present (source, title, date OR version, page_count)
                  and text not truncated mid-sentence
  - recency:      a newer version of the same document is NOT already known (and not superseded)
  - uniqueness:   no exact duplicate (hash) or near-duplicate (embedding similarity)

Documents that fail quality/completeness/uniqueness are quarantined and excluded from indexing.
Superseded documents (recency fail) are archived with metadata so they are excluded from
normal retrieval and can be restored via pipeline.rollback.
A verification report JSON is written per run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .models import Document


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NEAR_DUP_THRESHOLD = 0.85        # cosine similarity above this -> near-duplicate
MIN_CONTENT_LENGTH = 50          # chars; below this -> quality fail (too short/empty)
GARBLED_RATIO_THRESHOLD = 0.08   # fraction of symbol noise chars -> garbled
TRUNCATION_SIGNALS = [           # text ending with these patterns -> truncated
    r"\bse$", r"\bthe$", r"\bof$", r"\band$", r"\bto$", r"\ba$", r"\bin$",
]

INJECTION_PATTERNS = [
    r"(?i)system\s*:\s*ignore",
    r"(?i)ignore\s+all\s+previous\s+instructions",
    r"(?i)developer\s+mode",
    r"(?i)you\s+are\s+now\s+in\s+(?:developer|admin|root)\s+mode",
    r"(?i)disregard\s+all\s+(?:previous\s+)?(?:instructions|policies|rules)",
    r"(?i)forget\s+(?:everything|all\s+previous)",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    passed: bool
    reason: str


@dataclass
class VerificationResult:
    file: str
    overall_passed: bool
    quality: CheckResult
    completeness: CheckResult
    recency: CheckResult
    uniqueness: CheckResult
    injection_detected: bool = False
    disposition: str = "passed"  # "passed" | "quarantined" | "archived"

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "overall_passed": self.overall_passed,
            "disposition": self.disposition,
            "quality": asdict(self.quality),
            "completeness": asdict(self.completeness),
            "recency": asdict(self.recency),
            "uniqueness": asdict(self.uniqueness),
            "injection_detected": self.injection_detected,
        }


@dataclass
class VerificationReport:
    run_id: str
    corpus_dir: str
    embedding_backend: str = "ollama"
    embedding_warning: str = ""
    total: int = 0
    passed: int = 0
    quarantined: int = 0
    archived: int = 0
    documents: list[VerificationResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "corpus_dir": self.corpus_dir,
            "embedding_backend": self.embedding_backend,
            "embedding_warning": self.embedding_warning,
            "total": self.total,
            "passed": self.passed,
            "quarantined": self.quarantined,
            "archived": self.archived,
            "documents": [d.to_dict() for d in self.documents],
        }


# ---------------------------------------------------------------------------
# Metadata extraction helpers (MD, PDF, HTML, TXT, _index.json)
# ---------------------------------------------------------------------------

def _read_front_matter(path: Path) -> dict[str, str]:
    """Read YAML front-matter from a Markdown file."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {}
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


def _load_index_manifest(corpus_dir: Path) -> dict[str, dict]:
    """Load _index.json keyed by filename if present."""
    idx_path = corpus_dir / "_index.json"
    if not idx_path.exists():
        return {}
    try:
        entries = json.loads(idx_path.read_text(encoding="utf-8"))
        return {str(e["file"]): e for e in entries if isinstance(e, dict) and "file" in e}
    except Exception:
        return {}


def _extract_html_title(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")[:8192]
        m = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return " ".join(m.group(1).split())
    except Exception:
        pass
    return ""


def _extract_pdf_title(path: Path) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        if reader.metadata and getattr(reader.metadata, "title", None):
            title = str(reader.metadata.title).strip()
            if title and title.lower() not in ("untitled", "none"):
                return title
    except Exception:
        pass
    return ""


def extract_doc_metadata(path: Path, doc: Document, index_entry: Optional[dict] = None) -> dict:
    """Gather metadata from _index.json, front matter, HTML/PDF headers, and Document."""
    meta: dict = {}
    if index_entry:
        if index_entry.get("name"):
            meta["title"] = str(index_entry["name"])
        if index_entry.get("title"):
            meta["title"] = str(index_entry["title"])
        if index_entry.get("file"):
            meta["source"] = str(index_entry["file"])
        if index_entry.get("fetched_at"):
            meta["date"] = str(index_entry["fetched_at"])
        if index_entry.get("date"):
            meta["date"] = str(index_entry["date"])
        if index_entry.get("doc_version"):
            meta["version"] = str(index_entry["doc_version"])
        if "superseded" in index_entry:
            meta["superseded"] = bool(index_entry["superseded"])

    suffix = path.suffix.lower()
    if suffix == ".md":
        fm = _read_front_matter(path)
        meta.update({k: v for k, v in fm.items() if v != ""})
        if "doc_version" in fm and "version" not in meta:
            meta["version"] = fm["doc_version"]
    elif suffix in (".html", ".htm") and not meta.get("title"):
        html_title = _extract_html_title(path)
        if html_title:
            meta["title"] = html_title
    elif suffix == ".pdf" and not meta.get("title"):
        pdf_title = _extract_pdf_title(path)
        if pdf_title:
            meta["title"] = pdf_title

    # Fallback source/version/page_count from extracted Document
    if not meta.get("source") and doc.source:
        # Only infer source if we have at least some metadata or non-md file
        if suffix != ".md" or meta:
            meta["source"] = doc.source

    if not meta.get("version") and doc.doc_version not in ("", "unknown", "None"):
        meta["version"] = doc.doc_version

    if not meta.get("page_count") and doc.pages:
        # For markdown without front matter at all, keep page_count unset if no other metadata exists
        if suffix != ".md" or meta:
            meta["page_count"] = str(len(doc.pages))

    return meta


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _full_text(doc: Document) -> str:
    return "\n".join(p.text for p in doc.pages)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _is_valid_val(val: object) -> bool:
    return bool(val) and str(val).strip() not in ("", "unknown", "None", "null")


def check_quality(doc: Document) -> CheckResult:
    """Quality: not empty, not garbled, extraction not failed."""
    text = _full_text(doc)

    if not text or len(text.strip()) < MIN_CONTENT_LENGTH:
        return CheckResult(passed=False, reason=f"empty or too short ({len(text.strip())} chars)")

    if "\ufffd" in text or "\\x00" in text or "\\xff" in text:
        return CheckResult(passed=False, reason="garbled text: contains replacement characters or raw byte escapes")

    total = len(text)
    symbol_chars = sum(
        1 for ch in text
        if not ch.isalnum() and ch not in " \n\r\t.,;:?!-()\"'[]{}<>/|@#%&*+=~`_"
    )
    if total > 0 and (symbol_chars / total) > GARBLED_RATIO_THRESHOLD:
        return CheckResult(
            passed=False,
            reason=f"garbled text: high symbol noise ratio ({symbol_chars / total:.1%})",
        )

    if "[Scanned page" in text and "text not extractable" in text:
        # Only fail if ALL pages failed extraction
        extractable = [p for p in doc.pages if "text not extractable" not in p.text and len(p.text.strip()) >= MIN_CONTENT_LENGTH]
        if not extractable:
            return CheckResult(passed=False, reason="extraction failed: OCR unavailable for scanned pages")

    return CheckResult(passed=True, reason="ok")


def check_completeness(doc: Document, raw_meta: Optional[dict] = None) -> CheckResult:
    """Completeness: required metadata (source, title, date OR version, page_count) and non-truncated text."""
    if raw_meta is None:
        raw_meta = {}

    missing: list[str] = []
    if not _is_valid_val(raw_meta.get("source")):
        missing.append("source")
    if not _is_valid_val(raw_meta.get("title")):
        missing.append("title")
    if not (_is_valid_val(raw_meta.get("date")) or _is_valid_val(raw_meta.get("version"))):
        missing.append("date or version")

    pc_val = raw_meta.get("page_count")
    if not _is_valid_val(pc_val):
        missing.append("page_count")
    else:
        try:
            if int(str(pc_val).strip()) < 1:
                missing.append("page_count (>=1)")
        except ValueError:
            missing.append("page_count (invalid integer)")

    if missing:
        return CheckResult(
            passed=False,
            reason=f"missing required metadata: {', '.join(missing)}",
        )

    text = _full_text(doc).rstrip()
    last_line = text.split("\n")[-1].strip() if text else ""
    if last_line:
        words = last_line.split()
        if words:
            last_word = words[-1].rstrip(".,;:!?")
            for pattern in TRUNCATION_SIGNALS:
                if re.search(pattern, last_word):
                    return CheckResult(
                        passed=False,
                        reason=f"text appears truncated (ends with '{last_line[-60:].strip()}')",
                    )
            if len(last_line) < 20 and not re.search(r"[.!?|]\s*$", last_line):
                return CheckResult(
                    passed=False,
                    reason=f"text appears truncated (short dangling line: '{last_line}')",
                )

    return CheckResult(passed=True, reason="ok")


def _parse_version(ver: str) -> tuple[int, ...]:
    nums = [int(x) for x in re.findall(r"\d+", str(ver))]
    return tuple(nums) if nums else (0,)


def check_recency(doc: Document, all_docs_meta: list[dict], raw_meta: Optional[dict] = None) -> CheckResult:
    """Recency: no newer version of the same source document exists, and doc is not flagged superseded."""
    meta = raw_meta or {}
    if doc.superseded or meta.get("superseded") is True or str(meta.get("superseded", "")).lower() == "true":
        return CheckResult(
            passed=False,
            reason=f"superseded: document version '{meta.get('version', doc.doc_version)}' is marked superseded",
        )

    source = meta.get("source", doc.source)
    version_str = str(meta.get("version", doc.doc_version))
    this_version = _parse_version(version_str)
    doc_path = meta.get("_path")

    for other in all_docs_meta:
        if other.get("source") == source and (not doc_path or other.get("_path") != doc_path):
            other_version = _parse_version(str(other.get("version", "0")))
            if other_version > this_version:
                return CheckResult(
                    passed=False,
                    reason=(
                        f"superseded: a newer version {other.get('version')} "
                        f"exists for source '{source}'"
                    ),
                )

    return CheckResult(passed=True, reason="no newer version detected")


def check_uniqueness(
    doc: Document,
    hash_registry: dict[str, str],
    embedding_registry: list[tuple[str, list[float]]],
    backend_tracker: Optional[dict[str, str]] = None,
) -> CheckResult:
    """Uniqueness: no exact duplicate (SHA-256) or near-duplicate (cosine sim)."""
    text = _full_text(doc)
    h = _sha256(text)

    if h in hash_registry:
        return CheckResult(
            passed=False,
            reason=f"exact duplicate of '{hash_registry[h]}' (SHA-256 match)",
        )

    vec, method = _embed_text(text)
    if backend_tracker is not None:
        backend_tracker["method"] = method

    if vec is not None:
        for other_name, other_vec in embedding_registry:
            if other_name == doc.source:
                continue
            sim = _cosine(vec, other_vec)
            if sim >= NEAR_DUP_THRESHOLD:
                return CheckResult(
                    passed=False,
                    reason=(
                        f"near-duplicate of '{other_name}' "
                        f"(cosine similarity {sim:.3f} >= {NEAR_DUP_THRESHOLD}, method={method})"
                    ),
                )
        embedding_registry.append((doc.source, vec))

    hash_registry[h] = doc.source
    return CheckResult(passed=True, reason="unique (no exact or near-duplicate found)")


def check_injection(doc: Document) -> bool:
    """Detect prompt injection attempts in document body. Returns True if injection found."""
    text = _full_text(doc)
    return any(re.search(pattern, text) for pattern in INJECTION_PATTERNS)


# ---------------------------------------------------------------------------
# Deterministic embedding helper
# ---------------------------------------------------------------------------

def _deterministic_bow_vector(text: str, dim: int = 512) -> Optional[list[float]]:
    """Deterministic SHA-256 hashed bag-of-words vector (no Python randomized hash())."""
    words = re.findall(r"\w+", text.lower()[:4000])
    if not words:
        return None
    vec = [0.0] * dim
    for w in words:
        digest = hashlib.sha256(w.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        vec[idx] += 1.0
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm > 0 else vec


def _embed_text(text: str) -> tuple[Optional[list[float]], str]:
    """Embed text via Ollama for near-dup detection. Falls back to deterministic SHA-256 BoW."""
    ollama_url = os.environ.get("OLLAMA_URL", "http://host.minikube.internal:11434")
    embed_model = os.environ.get("EMBED_MODEL", "nomic-embed-text")
    try:
        import httpx
        snippet = text[:2000]
        resp = httpx.post(
            f"{ollama_url.rstrip('/')}/api/embed",
            json={"model": embed_model, "input": [snippet]},
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0], "ollama"
    except Exception:
        return _deterministic_bow_vector(text), "deterministic_bow_fallback"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Quarantine & Archive helpers
# ---------------------------------------------------------------------------

def _archive_document(
    path: Path,
    doc: Document,
    meta: dict,
    archive_path: Path,
    reason: str,
) -> None:
    """Archive a superseded document and write a metadata sidecar for rollback."""
    archive_path.mkdir(parents=True, exist_ok=True)
    ver_tag = re.sub(r"[^A-Za-z0-9._-]", "_", str(meta.get("version", doc.doc_version or "unknown")))
    dest_name = f"{path.stem}_v{ver_tag}{path.suffix}"
    dest = archive_path / dest_name
    shutil.copy2(str(path), str(dest))

    sidecar = archive_path / f"{path.stem}_v{ver_tag}.meta.json"
    sidecar_data = {
        "archived_file": dest_name,
        "original_file": path.name,
        "original_path": str(path),
        "source": meta.get("source", doc.source),
        "title": meta.get("title", ""),
        "doc_version": str(meta.get("version", doc.doc_version)),
        "date": str(meta.get("date", "")),
        "reason": reason,
        "archived_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sidecar.write_text(json.dumps(sidecar_data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def verify_corpus(
    documents: list[tuple[Path, Document]],
    quarantine_dir: str,
    report_dir: str,
    archive_dir: Optional[str] = None,
) -> tuple[list[tuple[Path, Document]], VerificationReport]:
    """Run all verification checks on a list of (path, Document) pairs.

    Returns:
        (passed_docs, report)
        where passed_docs is the subset that cleared all checks (ready for chunking),
        and report is the full VerificationReport (also written to disk).
    """
    quarantine_path = Path(quarantine_dir)
    quarantine_path.mkdir(parents=True, exist_ok=True)

    archive_path = Path(archive_dir) if archive_dir else quarantine_path.parent / "archive"
    archive_path.mkdir(parents=True, exist_ok=True)

    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    corpus_dir_path = documents[0][0].parent if documents else Path(".")
    report = VerificationReport(
        run_id=run_id,
        corpus_dir=str(corpus_dir_path),
    )

    index_manifest = _load_index_manifest(corpus_dir_path)

    # Pre-build metadata list for recency checks
    all_docs_meta: list[dict] = []
    extracted_metas: list[dict] = []
    for path, doc in documents:
        meta = extract_doc_metadata(path, doc, index_manifest.get(path.name))
        meta["_path"] = str(path)
        extracted_metas.append(meta)
        all_docs_meta.append({
            "source": meta.get("source", doc.source),
            "version": meta.get("version", doc.doc_version),
            "_path": str(path),
        })

    hash_registry: dict[str, str] = {}
    embedding_registry: list[tuple[str, list[float]]] = []
    backend_tracker: dict[str, str] = {}
    quarantine_manifest: list[dict] = []

    passed_docs: list[tuple[Path, Document]] = []

    for (path, doc), meta in zip(documents, extracted_metas):
        q_check = check_quality(doc)
        c_check = check_completeness(doc, raw_meta=meta)
        r_check = check_recency(doc, all_docs_meta, raw_meta=meta)
        u_check = check_uniqueness(doc, hash_registry, embedding_registry, backend_tracker)
        injection = check_injection(doc)

        overall = q_check.passed and c_check.passed and r_check.passed and u_check.passed

        if overall:
            disposition = "passed"
            report.passed += 1
            passed_docs.append((path, doc))
        elif not r_check.passed and q_check.passed and c_check.passed and u_check.passed:
            disposition = "archived"
            report.archived += 1
            _archive_document(path, doc, meta, archive_path, r_check.reason)
        else:
            disposition = "quarantined"
            report.quarantined += 1
            dest = quarantine_path / path.name
            shutil.copy2(str(path), str(dest))
            quarantine_manifest.append({
                "file": path.name,
                "original_path": str(path),
                "quarantined_path": str(dest),
                "reasons": {
                    k: chk.reason
                    for k, chk in [
                        ("quality", q_check),
                        ("completeness", c_check),
                        ("recency", r_check),
                        ("uniqueness", u_check),
                    ]
                    if not chk.passed
                },
                "timestamp": run_id,
            })

        result = VerificationResult(
            file=path.name,
            overall_passed=overall,
            disposition=disposition,
            quality=q_check,
            completeness=c_check,
            recency=r_check,
            uniqueness=u_check,
            injection_detected=injection,
        )
        report.documents.append(result)
        report.total += 1

    if backend_tracker.get("method") == "deterministic_bow_fallback":
        report.embedding_backend = "deterministic_bow_fallback"
        report.embedding_warning = (
            "Ollama embedding service was unavailable during verification; "
            "near-duplicate detection used deterministic SHA-256 BoW fallback."
        )
    elif backend_tracker.get("method"):
        report.embedding_backend = backend_tracker["method"]

    if quarantine_manifest:
        (quarantine_path / "quarantine_manifest.json").write_text(
            json.dumps(quarantine_manifest, indent=2), encoding="utf-8"
        )

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_file = report_path / f"verification_{ts}.json"
    report_file.write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )

    return passed_docs, report
