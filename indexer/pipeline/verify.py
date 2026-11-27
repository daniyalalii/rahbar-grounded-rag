"""Stage 1b: Verify. Runs after extract(), before chunk().

RAGOps paper section 4.2.1-4.2.4.

For every document, four checks are performed:
  - quality:      text is not empty, garbled, or extraction-failed
  - completeness: required metadata present (source, title, date/version, page_count)
                  and text not truncated mid-sentence
  - recency:      a newer version of the same document is NOT already known
  - uniqueness:   no exact duplicate (hash) or near-duplicate (embedding similarity)

Documents that fail any check are moved to the quarantine folder and excluded from indexing.
A verification report JSON is written per run to eval/results/.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .models import Document


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REQUIRED_METADATA_KEYS = {"title", "source", "date", "version", "page_count"}
NEAR_DUP_THRESHOLD = 0.85        # cosine similarity above this → near-duplicate (paper standard)
MIN_CONTENT_LENGTH = 50          # chars; below this → quality fail (too short/empty)
GARBLED_RATIO_THRESHOLD = 0.30   # fraction of non-printable/control chars → garbled
TRUNCATION_SIGNALS = [           # text ending with these patterns → truncated
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

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "overall_passed": self.overall_passed,
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
    total: int = 0
    passed: int = 0
    quarantined: int = 0
    documents: list[VerificationResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "corpus_dir": self.corpus_dir,
            "total": self.total,
            "passed": self.passed,
            "quarantined": self.quarantined,
            "documents": [d.to_dict() for d in self.documents],
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _full_text(doc: Document) -> str:
    return "\n".join(p.text for p in doc.pages)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def check_quality(doc: Document) -> CheckResult:
    """Quality: not empty, not garbled, extraction not failed."""
    text = _full_text(doc)

    if not text or len(text.strip()) < MIN_CONTENT_LENGTH:
        return CheckResult(passed=False, reason=f"empty or too short ({len(text.strip())} chars)")

    # Garbled check 1: unicode replacement character or byte corruption markers
    if "\ufffd" in text or "\\x00" in text or "\\xff" in text:
        return CheckResult(passed=False, reason="garbled text: contains replacement characters or raw byte escapes")

    # Garbled check 2: high ratio of symbols / non-alphanumeric noise
    total = len(text)
    symbol_chars = sum(
        1 for ch in text
        if not ch.isalnum() and ch not in " \n\r\t.,;:?!-()\"'[]{}<>"
    )
    if total > 0 and (symbol_chars / total) > 0.08:
        return CheckResult(
            passed=False,
            reason=f"garbled text: high symbol noise ratio ({symbol_chars / total:.1%})",
        )

    # Failed extraction marker
    if "[Scanned page" in text and "text not extractable" in text:
        return CheckResult(passed=False, reason="extraction failed: OCR unavailable for scanned pages")

    return CheckResult(passed=True, reason="ok")


def check_completeness(doc: Document, raw_meta: Optional[dict] = None) -> CheckResult:
    """Completeness: required metadata present and text not abruptly truncated."""
    if raw_meta is None:
        raw_meta = {}

    # Check required metadata fields
    missing = []
    for key in REQUIRED_METADATA_KEYS:
        val = raw_meta.get(key, "")
        if not val or str(val).strip() in ("", "unknown", "None"):
            missing.append(key)

    if missing:
        return CheckResult(
            passed=False,
            reason=f"missing required metadata: {', '.join(sorted(missing))}",
        )

    # Check for truncation: last non-empty line ends abruptly
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
            # Also flag very short last line that doesn't end with punctuation
            if len(last_line) < 20 and not re.search(r"[.!?]\s*$", last_line):
                return CheckResult(
                    passed=False,
                    reason=f"text appears truncated (short dangling line: '{last_line}')",
                )

    return CheckResult(passed=True, reason="ok")


def check_recency(doc: Document, all_docs_meta: list[dict], raw_meta: Optional[dict] = None) -> CheckResult:
    """Recency: no newer version of the same source document exists in this batch/corpus."""
    source = (raw_meta or {}).get("source", doc.source)
    version_str = str((raw_meta or {}).get("version", doc.doc_version))
    try:
        this_version = tuple(int(x) for x in version_str.split(".") if x.isdigit())
    except Exception:
        this_version = (0,)

    doc_path = getattr(doc, "_path", None) or (raw_meta or {}).get("_path", None)
    for other in all_docs_meta:
        if other.get("source") == source and (not doc_path or other.get("_path") != doc_path):
            try:
                other_version = tuple(
                    int(x) for x in str(other.get("version", "0")).split(".") if x.isdigit()
                )
            except Exception:
                other_version = (0,)
            if other_version > this_version:
                return CheckResult(
                    passed=False,
                    reason=(
                        f"superseded: a newer version {'.'.join(str(x) for x in other_version)} "
                        f"exists for source '{source}'"
                    ),
                )

    return CheckResult(passed=True, reason="no newer version detected")


def check_uniqueness(
    doc: Document,
    hash_registry: dict[str, str],
    embedding_registry: list[tuple[str, list[float]]],
) -> CheckResult:
    """Uniqueness: no exact duplicate (SHA-256) or near-duplicate (cosine sim)."""
    text = _full_text(doc)
    h = _sha256(text)

    # Exact duplicate
    if h in hash_registry:
        return CheckResult(
            passed=False,
            reason=f"exact duplicate of '{hash_registry[h]}' (SHA-256 match)",
        )

    # Near-duplicate via cosine similarity
    try:
        vec = _embed_text(text)
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
                            f"(cosine similarity {sim:.3f} >= {NEAR_DUP_THRESHOLD})"
                        ),
                    )
            # Register this doc vector
            embedding_registry.append((doc.source, vec))
    except Exception:
        pass  # embedding unavailable; skip near-dup check

    # Register hash
    hash_registry[h] = doc.source
    return CheckResult(passed=True, reason="unique (no exact or near-duplicate found)")


def check_injection(doc: Document) -> bool:
    """Detect prompt injection attempts in document body. Returns True if injection found."""
    text = _full_text(doc)
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


# ---------------------------------------------------------------------------
# Embedding helper (optional — falls back gracefully)
# ---------------------------------------------------------------------------

def _embed_text(text: str) -> Optional[list[float]]:
    """Embed text via Ollama for near-dup detection. Falls back to token frequency hashing."""
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
        return resp.json()["embeddings"][0]
    except Exception:
        # High-dimension character n-gram / token frequency fallback for offline test
        words = re.findall(r"\w+", text.lower()[:3000])
        if not words:
            return None
        vec = [0.0] * 512
        for w in words:
            h = abs(hash(w)) % 512
            vec[h] += 1.0
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm > 0 else vec


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Front-matter reader (for standalone use without extract.py)
# ---------------------------------------------------------------------------

def _read_front_matter(path: Path) -> dict:
    """Read YAML front-matter from a Markdown file."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {}
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def verify_corpus(
    documents: list[tuple[Path, Document]],
    quarantine_dir: str,
    report_dir: str,
) -> tuple[list[tuple[Path, Document]], VerificationReport]:
    """Run all verification checks on a list of (path, Document) pairs.

    Returns:
        (passed_docs, report)
        where passed_docs is the subset that cleared all checks (ready for chunking),
        and report is the full VerificationReport (also written to disk).
    """
    quarantine_path = Path(quarantine_dir)
    try:
        quarantine_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        quarantine_path = Path("/tmp/quarantine")
        quarantine_path.mkdir(parents=True, exist_ok=True)

    report_path = Path(report_dir)
    try:
        report_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        report_path = Path("/tmp/eval_results")
        report_path.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    report = VerificationReport(
        run_id=run_id,
        corpus_dir=str(documents[0][0].parent) if documents else "",
    )

    # Pre-build metadata list for recency checks
    all_docs_meta: list[dict] = []
    for path, doc in documents:
        fm = _read_front_matter(path) if path.suffix.lower() == ".md" else {}
        all_docs_meta.append({
            "source": fm.get("source", doc.source),
            "version": fm.get("version", doc.doc_version),
            "_path": str(path),
        })

    # Shared registries for uniqueness checks
    hash_registry: dict[str, str] = {}       # sha256 → source name
    embedding_registry: list[tuple[str, list[float]]] = []

    passed_docs: list[tuple[Path, Document]] = []

    for (path, doc), meta in zip(documents, all_docs_meta):
        fm = _read_front_matter(path) if path.suffix.lower() == ".md" else {}

        q_check = check_quality(doc)
        c_check = check_completeness(doc, raw_meta=fm)
        fm_with_path = dict(fm)
        fm_with_path["_path"] = str(path)
        r_check = check_recency(doc, all_docs_meta, raw_meta=fm_with_path)
        u_check = check_uniqueness(doc, hash_registry, embedding_registry)
        injection = check_injection(doc)

        overall = q_check.passed and c_check.passed and r_check.passed and u_check.passed

        result = VerificationResult(
            file=path.name,
            overall_passed=overall,
            quality=q_check,
            completeness=c_check,
            recency=r_check,
            uniqueness=u_check,
            injection_detected=injection,
        )
        report.documents.append(result)
        report.total += 1

        if overall:
            report.passed += 1
            passed_docs.append((path, doc))
        else:
            report.quarantined += 1
            # Move to quarantine
            dest = quarantine_path / path.name
            try:
                shutil.copy2(str(path), str(dest))
            except Exception:
                pass  # quarantine copy failed; file stays in original location

    # Write report to disk
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_file = report_path / f"verification_{ts}.json"
    report_file.write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )

    return passed_docs, report
