"""Production corpus fetcher and verification module for the RAHBAR indexer.

Downloads documents defined in sources.yaml, verifies SHA-256 hashes against
MANIFEST.md, and writes _index.json so the indexing pipeline can discover and index
all real documents with their version and superseded metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
import yaml  # type: ignore[import-untyped]

log = logging.getLogger("rahbar.indexer.fetch")

USER_AGENT = "RAHBAR-coursework-fetcher/1.0 (CS4032 student project; Kubernetes indexer)"
PAUSE_SECONDS = 1.0
DRIVE_VIEW = re.compile(r"https?://drive\.google\.com/file/d/([^/?#]+)")


def normalise_url(url: str) -> str:
    """Encode spaces and convert Drive viewer links to direct downloads."""
    match = DRIVE_VIEW.match(url)
    if match:
        return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    return quote(url, safe=":/?&=%#+,;@!$'()*~")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:80] or "document"


def guess_extension(url: str, content_type: str) -> str:
    content_type = content_type.lower()
    if "pdf" in content_type:
        return ".pdf"
    if "html" in content_type:
        return ".html"
    suffix = Path(urlsplit(url).path).suffix.lower()
    return suffix if suffix in {".pdf", ".html", ".htm", ".docx", ".txt", ".md"} else ".bin"


def http_get(client: httpx.Client, url: str, attempts: int = 3) -> httpx.Response:
    reason = "unknown"
    for attempt in range(1, attempts + 1):
        try:
            resp = client.get(url)
            if resp.status_code < 500:
                return resp
            reason = f"HTTP {resp.status_code}"
        except httpx.TransportError as exc:
            reason = type(exc).__name__
        if attempt < attempts:
            wait = PAUSE_SECONDS * (2 ** attempt)
            time.sleep(wait)
    raise RuntimeError(f"Download failed after {attempts} attempts for {url}: {reason}")


def load_manifest_checksums(manifest_path: Path) -> dict[str, str]:
    """Parse expected SHA-256 checksums from MANIFEST.md."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    text = manifest_path.read_text(encoding="utf-8")
    rows = re.findall(r"\|\s*`([^`]+)`\s*\|\s*[\d,]+\s*\|\s*`([0-9a-f]{64})`\s*\|", text)
    return {file: expected for file, expected in rows}


def fetch_and_verify(
    target_dir: str | Path,
    sources_file: str | Path | None = None,
    manifest_file: str | Path | None = None,
    force: bool = False,
) -> int:
    """Fetch all sources into target_dir, verify checksums, and write _index.json.

    Returns 0 on success. Raises RuntimeError on download failure or checksum mismatch.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    # Locate sources.yaml
    sources_path: Path | None = None
    if sources_file:
        sources_path = Path(sources_file)
    elif os.environ.get("SOURCES_FILE"):
        sources_path = Path(os.environ["SOURCES_FILE"])
    else:
        candidates = [
            Path(__file__).resolve().parent / "sources.yaml",
            Path("corpus/sources.yaml"),
            Path("/app/pipeline/sources.yaml"),
        ]
        for c in candidates:
            if c.exists():
                sources_path = c
                break

    if not sources_path or not sources_path.exists():
        raise FileNotFoundError(f"Cannot find sources.yaml. Checked candidates: {sources_path}")

    # Locate MANIFEST.md
    manifest_path: Path | None = None
    if manifest_file:
        manifest_path = Path(manifest_file)
    elif os.environ.get("MANIFEST_FILE"):
        manifest_path = Path(os.environ["MANIFEST_FILE"])
    else:
        candidates = [
            Path(__file__).resolve().parent / "MANIFEST.md",
            Path("corpus/MANIFEST.md"),
            Path("/app/pipeline/MANIFEST.md"),
        ]
        for c in candidates:
            if c.exists():
                manifest_path = c
                break

    if not manifest_path or not manifest_path.exists():
        raise FileNotFoundError(f"Cannot find MANIFEST.md. Checked candidates: {manifest_path}")

    expected_hashes = load_manifest_checksums(manifest_path)
    sources_data = yaml.safe_load(sources_path.read_text(encoding="utf-8")) or {}
    docs: list[dict[str, Any]] = sources_data.get("documents", [])

    if not docs:
        raise ValueError("No documents defined in sources.yaml")

    entries: list[dict[str, Any]] = []
    failures: list[str] = []

    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=httpx.Timeout(60.0, connect=15.0),
    ) as client:
        for doc in docs:
            name = doc["name"]
            url = normalise_url(doc["url"])
            doc_version = str(doc.get("doc_version", "unknown"))
            superseded = bool(doc.get("superseded", False))

            content_type = ""
            file_name = slug(name) + guess_extension(url, "")
            # Look up expected filename from manifest if available
            matched_file = None
            for mf in expected_hashes:
                if mf.startswith(slug(name)):
                    matched_file = mf
                    break
            if matched_file:
                file_name = matched_file

            file_path = target / file_name

            # Check if file already exists with valid checksum
            if not force and file_path.exists():
                actual_sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
                expected_sha = expected_hashes.get(file_name)
                if expected_sha and actual_sha == expected_sha:
                    entries.append({
                        "name": name,
                        "url": url,
                        "file": file_name,
                        "bytes": file_path.stat().st_size,
                        "sha256": actual_sha,
                        "doc_version": doc_version,
                        "superseded": superseded,
                    })
                    continue

            # Download document
            try:
                resp = http_get(client, url)
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code} received from {url}")

                content_type = resp.headers.get("content-type", "")
                actual_ext = guess_extension(url, content_type)
                if not matched_file:
                    file_name = slug(name) + actual_ext
                    file_path = target / file_name

                content = resp.content
                actual_sha = hashlib.sha256(content).hexdigest()

                # Integrity check against MANIFEST
                expected_sha = expected_hashes.get(file_name)
                if expected_sha and actual_sha != expected_sha:
                    raise ValueError(
                        f"Checksum mismatch for {file_name}: expected {expected_sha}, got {actual_sha}"
                    )

                file_path.write_bytes(content)
                entries.append({
                    "name": name,
                    "url": url,
                    "file": file_name,
                    "bytes": len(content),
                    "sha256": actual_sha,
                    "content_type": content_type.split(";")[0],
                    "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "doc_version": doc_version,
                    "superseded": superseded,
                })
            except Exception as exc:
                failures.append(f"{name} ({url}): {exc}")

            time.sleep(PAUSE_SECONDS)

    if failures:
        error_msg = f"Failed to download/verify {len(failures)} corpus documents:\n" + "\n".join(f"  - {f}" for f in failures)
        raise RuntimeError(error_msg)

    # Write _index.json
    index_file = target / "_index.json"
    index_file.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return 0
