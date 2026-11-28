#!/usr/bin/env python3
"""RAGOps Part E: Query Coverage Check.

Checks whether live user queries are covered by the golden evaluation set
(cosine similarity >= threshold). Flags uncovered queries and groups them by topic.

Usage:
    python scripts/coverage_check.py
    python scripts/coverage_check.py --live scripts/live_queries.jsonl \\
                                     --golden eval/golden.jsonl \\
                                     --threshold 0.85

Exit code 1 if coverage < threshold (for CI/CD gate).
"""

from __future__ import annotations

import argparse
import json
import os
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
        timeout=120.0,
    )
    r.raise_for_status()
    return r.json()["embeddings"]


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
    parser.add_argument("--threshold", type=float, default=0.85, help="Coverage similarity threshold")
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
    print(f"Threshold:      {args.threshold}")
    print(f"Embedding via:  {args.ollama_url} model={args.embed_model}")
    print("Embedding all queries...")

    try:
        all_texts = live_queries + golden_queries
        all_vecs = embed_texts(all_texts, args.ollama_url, args.embed_model)
        live_vecs = all_vecs[:len(live_queries)]
        golden_vecs = all_vecs[len(live_queries):]
    except Exception as exc:
        print(f"[WARNING] Remote embedding unavailable ({exc}). Using internal vector fallback.")
        import re
        def _bow(text: str) -> list[float]:
            words = re.findall(r"\w+", text.lower())
            v = [0.0] * 512
            for w in words:
                v[abs(hash(w)) % 512] += 1.0
            norm = sum(x * x for x in v) ** 0.5
            return [x / norm for x in v] if norm > 0 else v
        live_vecs = [_bow(q) for q in live_queries]
        golden_vecs = [_bow(q) for q in golden_queries]

    covered = []
    uncovered = []
    rows = []

    for i, (lq, lv) in enumerate(zip(live_queries, live_vecs or [None] * len(live_queries))):
        if lv is not None:
            sims = [cosine(lv, gv) for gv in golden_vecs]
            best_idx = max(range(len(sims)), key=lambda j: sims[j])
            best_sim = sims[best_idx]
            best_golden = golden_queries[best_idx]
        else:
            # Fallback: word overlap
            lwords = set(lq.lower().split())
            best_sim = max(
                (len(lwords & set(gq.lower().split())) / max(len(lwords), 1) for gq in golden_queries),
                default=0.0
            )
            best_golden = golden_queries[0]
            best_idx = 0

        is_covered = best_sim >= args.threshold
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

    coverage_pct = len(covered) / len(live_queries) if live_queries else 0
    print(f"\nCoverage: {len(covered)}/{len(live_queries)} = {coverage_pct:.1%}")

    if uncovered:
        print(f"\nUncovered queries ({len(uncovered)}):")
        topics: dict[str, list[str]] = defaultdict(list)
        for u in uncovered:
            topics[u["topic"]].append(u["query"])
        for topic, qs in topics.items():
            print(f"\n  Topic: {topic}")
            for q in qs:
                print(f"    - {q}")

    # Save results
    result = {
        "threshold": args.threshold,
        "total_live_queries": len(live_queries),
        "covered": len(covered),
        "uncovered_count": len(uncovered),
        "coverage_pct": round(coverage_pct, 4),
        "uncovered_queries": uncovered,
        "per_query": rows,
    }
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nResults saved to {args.out}")

    if coverage_pct < args.threshold:
        print(f"\n[FAIL] Coverage {coverage_pct:.1%} < threshold {args.threshold:.1%}")
        print("Add more golden questions from uncovered topics to close the gap.")
        return 1

    print(f"\n[PASS] Coverage {coverage_pct:.1%} >= threshold {args.threshold:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
