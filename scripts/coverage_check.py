#!/usr/bin/env python3
"""RAGOps Part E: Query Coverage Check.

Checks whether live user queries are covered by the golden evaluation set
(cosine similarity >= --sim-threshold). Flags uncovered queries and groups them by topic.

Separates:
  1. Per-query similarity cutoff (--sim-threshold, default 0.60)
  2. Minimum required overall coverage percentage (--min-coverage, default 0.85)

Usage:
    python scripts/coverage_check.py
    python scripts/coverage_check.py --live scripts/live_queries.jsonl \\
                                     --golden eval/golden.jsonl \\
                                     --sim-threshold 0.60 \\
                                     --min-coverage 0.85

Exit code 1 if coverage_pct < --min-coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.minikube.internal:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")


def embed_texts(texts: list[str], ollama_url: str, model: str) -> list[list[float]]:
    import httpx
    r = httpx.post(
        f"{ollama_url.rstrip('/')}/api/embed",
        json={"model": model, "input": texts},
        timeout=60.0,
    )
    r.raise_for_status()
    return r.json()["embeddings"]


def _deterministic_bow(text: str, dim: int = 512) -> list[float]:
    words = re.findall(r"\w+", text.lower())
    v = [0.0] * dim
    for w in words:
        digest = hashlib.sha256(w.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        v[idx] += 1.0
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v] if norm > 0 else v


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGOps query coverage check")
    parser.add_argument("--live", type=Path, default=ROOT / "scripts" / "live_queries.jsonl")
    parser.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden.jsonl")
    parser.add_argument("--sim-threshold", "--threshold", dest="sim_threshold", type=float, default=0.60,
                        help="Per-query cosine similarity cutoff")
    parser.add_argument("--min-coverage", type=float, default=0.85,
                        help="Minimum required overall coverage percentage (0.0-1.0)")
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--embed-model", default=EMBED_MODEL)
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "coverage.json")
    args = parser.parse_args()

    if not args.live.exists():
        sys.exit(f"Live queries file not found: {args.live}")
    if not args.golden.exists():
        sys.exit(f"Golden set not found: {args.golden}")

    live_items = load_jsonl(args.live)
    golden_items = load_jsonl(args.golden)

    live_queries = [item["question"] for item in live_items]
    golden_queries = [item["question"] for item in golden_items]

    print(f"Live queries:   {len(live_queries)}")
    print(f"Golden set:     {len(golden_queries)}")
    print(f"Sim cutoff:     {args.sim_threshold}")
    print(f"Min coverage:   {args.min_coverage:.1%}")
    print(f"Embedding via:  {args.ollama_url} model={args.embed_model}")

    embedding_backend = "ollama"
    try:
        all_texts = live_queries + golden_queries
        all_vecs = embed_texts(all_texts, args.ollama_url, args.embed_model)
        live_vecs = all_vecs[:len(live_queries)]
        golden_vecs = all_vecs[len(live_queries):]
    except Exception as exc:
        embedding_backend = "deterministic_bow_fallback"
        print(f"[WARNING] Remote embedding unavailable ({exc}). Using deterministic SHA-256 BoW fallback.")
        live_vecs = [_deterministic_bow(q) for q in live_queries]
        golden_vecs = [_deterministic_bow(q) for q in golden_queries]

    covered = []
    uncovered = []
    rows = []

    for i, (lq, lv) in enumerate(zip(live_queries, live_vecs)):
        sims = [cosine(lv, gv) for gv in golden_vecs]
        best_idx = max(range(len(sims)), key=lambda j: sims[j])
        best_sim = sims[best_idx]
        best_golden = golden_queries[best_idx]

        is_covered = best_sim >= args.sim_threshold
        topic = live_items[i].get("topic", "unknown")

        row = {
            "live_query": lq,
            "topic": topic,
            "best_golden_match": best_golden,
            "best_similarity": round(best_sim, 4),
            "covered": is_covered,
        }
        rows.append(row)
        if is_covered:
            covered.append(lq)
        else:
            uncovered.append({"query": lq, "topic": topic, "best_sim": round(best_sim, 4)})

    coverage_pct = len(covered) / len(live_queries) if live_queries else 0.0
    print(f"\nCoverage: {len(covered)}/{len(live_queries)} = {coverage_pct:.1%}")

    if uncovered:
        print(f"\nUncovered queries ({len(uncovered)}):")
        topics: dict[str, list[str]] = defaultdict(list)
        for u in uncovered:
            topics[u["topic"]].append(u["query"])
        for topic, qs in sorted(topics.items()):
            print(f"\n  Topic: {topic}")
            for q in qs:
                print(f"    - {q}")

    passed = coverage_pct >= args.min_coverage
    result = {
        "sim_threshold": args.sim_threshold,
        "min_coverage": args.min_coverage,
        "embedding_backend": embedding_backend,
        "total_live_queries": len(live_queries),
        "covered": len(covered),
        "uncovered_count": len(uncovered),
        "coverage_pct": round(coverage_pct, 4),
        "passed": passed,
        "uncovered_queries": uncovered,
        "per_query": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nResults saved to {args.out}")

    if not passed:
        print(f"\n[FAIL] Coverage {coverage_pct:.1%} < minimum {args.min_coverage:.1%}")
        print("Add more golden questions from uncovered topics to close the gap.")
        return 1

    print(f"\n[PASS] Coverage {coverage_pct:.1%} >= minimum {args.min_coverage:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
