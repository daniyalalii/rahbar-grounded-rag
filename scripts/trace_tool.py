#!/usr/bin/env python3
"""RAGOps Part D2: Trace Tool — rebuilds the exact context and prompt for a trace id.

Usage:
    python scripts/trace_tool.py <trace_id>
    python scripts/trace_tool.py --list              # list recent trace ids
    python scripts/trace_tool.py --tail 5            # last 5 traces
    python scripts/trace_tool.py <trace_id> --json   # raw JSON output

Reads from $TRACE_DIR/traces.jsonl (default: traces/traces.jsonl in repo root).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE_FILE = ROOT / "traces" / "traces.jsonl"
TRACE_DIR = os.environ.get("TRACE_DIR", str(ROOT / "traces"))


def find_trace_file() -> Path:
    candidates = [
        Path(TRACE_DIR) / "traces.jsonl",
        DEFAULT_TRACE_FILE,
        Path("/traces/traces.jsonl"),
    ]
    for c in candidates:
        if c.exists():
            return c
    sys.exit(f"No trace file found. Looked in: {[str(c) for c in candidates]}")


def load_all_traces(trace_file: Path) -> list[dict]:
    traces = []
    with trace_file.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    traces.append(json.loads(line))
                except Exception:
                    pass
    return traces


def find_by_id(traces: list[dict], trace_id: str) -> dict | None:
    for t in traces:
        if t.get("trace_id") == trace_id:
            return t
    return None


def print_trace(trace: dict, raw: bool = False) -> None:
    if raw:
        print(json.dumps(trace, indent=2))
        return

    print("=" * 70)
    print(f"TRACE ID:    {trace.get('trace_id')}")
    print(f"TIMESTAMP:   {trace.get('timestamp')}")
    print(f"CACHE HIT:   {trace.get('cache_hit')}")
    print("-" * 70)
    print(f"USER QUERY:  {trace.get('user_query')}")
    if trace.get("rewritten_query"):
        print(f"REWRITTEN:   {trace.get('rewritten_query')}")
    print("-" * 70)
    print(f"EMBED MODEL: {trace.get('embedding_model')} (v{trace.get('embedding_version')})")
    print(f"COLLECTION:  {trace.get('collection_name')}")
    print(f"LLM MODEL:   {trace.get('model')}  params={trace.get('model_params')}")
    print(f"PROMPT HASH: {trace.get('prompt_hash')}")
    print("-" * 70)

    # Reconstruct context from retrieved chunks
    chunks = trace.get("retrieved_chunks") or []
    print(f"RETRIEVED CHUNKS ({len(chunks)}):")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] chunk_id={c.get('chunk_id')}  doc_id={c.get('doc_id')}  "
              f"version={c.get('doc_version')}  page={c.get('page')}  score={c.get('score')}")
        if c.get("text"):
            print(f"       text: {c['text']}")

    print("-" * 70)
    print("EXACT PROMPT:")
    if trace.get("prompt_text"):
        print(trace["prompt_text"])
    else:
        print("  (prompt_text not recorded in this trace; only prompt_hash available)")

    print("-" * 70)
    guardrails = trace.get("guardrail_actions") or []
    if guardrails:
        print(f"GUARDRAIL ACTIONS ({len(guardrails)}):")
        for g in guardrails:
            print(f"  {g}")
    else:
        print("GUARDRAIL ACTIONS: none")

    print("-" * 70)
    latency = trace.get("latency_per_stage_ms") or {}
    print("LATENCY (ms):")
    for stage, ms in latency.items():
        print(f"  {stage:<15} {ms:>8.1f} ms")

    print("-" * 70)
    print("RESPONSE:")
    print(f"  {trace.get('response', '')[:500]}")

    cites = trace.get("citations") or []
    if cites:
        print(f"CITATIONS ({len(cites)}):")
        for c in cites:
            print(f"  - {c.get('source')} § {c.get('section')}")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGOps trace inspector")
    parser.add_argument("trace_id", nargs="?", help="Trace ID to inspect")
    parser.add_argument("--list", action="store_true", help="List all trace IDs")
    parser.add_argument("--tail", type=int, metavar="N", help="Print last N traces")
    parser.add_argument("--json", action="store_true", help="Raw JSON output")
    parser.add_argument("--file", type=Path, help="Path to traces.jsonl")
    args = parser.parse_args()

    trace_file = args.file or find_trace_file()
    traces = load_all_traces(trace_file)

    if not traces:
        print(f"No traces found in {trace_file}")
        return 0

    if args.list:
        print(f"{'TRACE ID':<40} {'TIMESTAMP':<25} {'QUERY'}")
        print("-" * 100)
        for t in traces:
            q = str(t.get("user_query", ""))[:50]
            print(f"{t.get('trace_id', ''):<40} {t.get('timestamp', ''):<25} {q}")
        return 0

    if args.tail:
        for t in traces[-args.tail:]:
            print_trace(t, raw=args.json)
        return 0

    if not args.trace_id:
        parser.print_help()
        return 1

    trace = find_by_id(traces, args.trace_id)
    if trace is None:
        print(f"Trace ID not found: {args.trace_id}")
        print(f"Use --list to see all available trace IDs.")
        return 1

    print_trace(trace, raw=args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
