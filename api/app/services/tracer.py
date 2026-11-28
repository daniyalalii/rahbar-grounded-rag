"""RAGOps Part D: Tracer — writes one JSON line per query.

This module is imported by api/app/services/answer.py to emit trace records
to both stdout (kubectl logs) and a /traces/ volume.

Trace fields per RAGOps paper section 3.2.3, 5.1, Table 2.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

TRACE_DIR = os.environ.get("TRACE_DIR", "/traces")


def _traces_path() -> Path:
    p = Path(TRACE_DIR)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        p = Path("/tmp/traces")
        p.mkdir(parents=True, exist_ok=True)
    return p


def new_trace_id() -> str:
    return str(uuid.uuid4())


def emit_trace(
    trace_id: str,
    user_query: str,
    rewritten_query: Optional[str],
    embed_model: str,
    embed_version: str,
    collection_name: str,
    retrieved_chunks: list[dict],
    filters_applied: dict,
    reranked_order: list[str],
    prompt_text: str,
    llm_model: str,
    llm_params: dict,
    response_text: str,
    citations: list[dict],
    guardrail_actions: list[dict],
    latency_stages: dict[str, float],
    cache_hit: bool,
) -> dict:
    """Build, log, and persist one trace record. Returns the trace dict."""

    trace: dict[str, Any] = {
        "trace_id": trace_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_query": user_query,
        "rewritten_query": rewritten_query,
        "embedding_model": embed_model,
        "embedding_version": embed_version,
        "collection_name": collection_name,
        "retrieved_chunks": retrieved_chunks,
        "filters_applied": filters_applied,
        "reranked_order": reranked_order,
        "prompt_hash": hashlib.sha256(prompt_text.encode("utf-8", errors="replace")).hexdigest(),
        "prompt_text": prompt_text,
        "model": llm_model,
        "model_params": llm_params,
        "response": response_text,
        "citations": citations,
        "guardrail_actions": guardrail_actions,
        "latency_per_stage_ms": latency_stages,
        "cache_hit": cache_hit,
    }

    line = json.dumps(trace)

    # 1. Emit to stdout → visible via kubectl logs
    print(line, flush=True)

    # 2. Write to file on PVC for scripts/trace_tool.py
    try:
        trace_file = _traces_path() / "traces.jsonl"
        with trace_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass  # don't crash the query if trace write fails

    return trace
