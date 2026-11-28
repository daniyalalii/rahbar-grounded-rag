#!/usr/bin/env python3
"""RAGOps Part D3: Metrics Summary — summarise trace log over a time window.

Usage:
    python scripts/metrics_summary.py
    python scripts/metrics_summary.py --since 2026-11-27
    python scripts/metrics_summary.py --hours 24
    python scripts/metrics_summary.py --file /traces/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = os.environ.get("TRACE_DIR", str(ROOT / "traces"))


def load_traces(path: Path, since: datetime | None = None) -> list[dict]:
    traces = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
                if since:
                    ts_str = t.get("timestamp", "")
                    try:
                        ts = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                        if ts < since:
                            continue
                    except Exception:
                        pass
                traces.append(t)
            except Exception:
                pass
    return traces


def summarise(traces: list[dict]) -> dict:
    if not traces:
        return {"total": 0}

    total = len(traces)
    cache_hits = sum(1 for t in traces if t.get("cache_hit"))
    abstentions = sum(
        1 for t in traces
        if "corpus does not cover" in str(t.get("response", "")).lower()
        or "unable to answer" in str(t.get("response", "")).lower()
    )
    injection_flags = sum(
        1 for t in traces
        if any(g.get("type") == "injection" for g in (t.get("guardrail_actions") or []))
    )

    # Latency per stage
    stage_latencies: dict[str, list[float]] = {}
    all_latencies = []
    for t in traces:
        latency = t.get("latency_per_stage_ms") or {}
        total_lat = sum(latency.values())
        if total_lat > 0:
            all_latencies.append(total_lat)
        for stage, ms in latency.items():
            stage_latencies.setdefault(stage, []).append(ms)

    p95_total = sorted(all_latencies)[int(len(all_latencies) * 0.95)] if all_latencies else 0
    p50_total = sorted(all_latencies)[len(all_latencies) // 2] if all_latencies else 0

    stage_p95 = {stage: sorted(vals)[int(len(vals) * 0.95)]
                 for stage, vals in stage_latencies.items() if vals}

    # Top-1 retrieval similarity distribution
    top1_scores = []
    for t in traces:
        chunks = t.get("retrieved_chunks") or []
        if chunks:
            score = chunks[0].get("score", 0)
            if isinstance(score, (int, float)):
                top1_scores.append(float(score))

    sim_buckets = Counter()
    for s in top1_scores:
        if s >= 0.8:
            sim_buckets[">=0.8"] += 1
        elif s >= 0.6:
            sim_buckets["0.6-0.8"] += 1
        elif s >= 0.4:
            sim_buckets["0.4-0.6"] += 1
        else:
            sim_buckets["<0.4"] += 1

    return {
        "total_queries": total,
        "cache_hit_rate": round(cache_hits / total, 4),
        "abstention_rate": round(abstentions / total, 4),
        "injection_attempts_flagged": injection_flags,
        "latency_p50_ms": round(p50_total, 1),
        "latency_p95_ms": round(p95_total, 1),
        "latency_p95_per_stage_ms": {k: round(v, 1) for k, v in stage_p95.items()},
        "top1_similarity_distribution": dict(sim_buckets),
        "mean_top1_similarity": round(mean(top1_scores), 4) if top1_scores else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise RAGOps trace log")
    parser.add_argument("--file", type=Path, help="Path to traces.jsonl")
    parser.add_argument("--since", help="ISO date filter, e.g. 2026-11-27")
    parser.add_argument("--hours", type=int, help="Last N hours")
    args = parser.parse_args()

    trace_file = args.file
    if trace_file is None:
        candidates = [Path(TRACE_DIR) / "traces.jsonl", Path("/traces/traces.jsonl")]
        for c in candidates:
            if c.exists():
                trace_file = c
                break
    if trace_file is None or not trace_file.exists():
        print("No trace file found.")
        return 1

    since: datetime | None = None
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    elif args.hours:
        from datetime import timedelta
        since = datetime.now(tz=timezone.utc) - timedelta(hours=args.hours)

    traces = load_traces(trace_file, since=since)
    summary = summarise(traces)

    print(f"\nTrace file:  {trace_file}")
    if since:
        print(f"Since:       {since.isoformat()}")
    print("=" * 55)
    print(f"  Total queries:          {summary.get('total_queries', 0)}")
    print(f"  Cache hit rate:         {summary.get('cache_hit_rate', 0):.1%}")
    print(f"  Abstention rate:        {summary.get('abstention_rate', 0):.1%}")
    print(f"  Injection flags:        {summary.get('injection_attempts_flagged', 0)}")
    print(f"  Latency p50:            {summary.get('latency_p50_ms', 0):.0f} ms")
    print(f"  Latency p95:            {summary.get('latency_p95_ms', 0):.0f} ms")
    print(f"  Mean top-1 sim score:   {summary.get('mean_top1_similarity', 0):.4f}")
    print("\n  Top-1 similarity distribution:")
    for bucket, count in sorted(summary.get("top1_similarity_distribution", {}).items()):
        print(f"    {bucket}: {count}")
    print("\n  p95 latency per stage:")
    for stage, ms in summary.get("latency_p95_per_stage_ms", {}).items():
        print(f"    {stage}: {ms:.0f} ms")
    print("=" * 55)

    return 0


if __name__ == "__main__":
    sys.exit(main())
