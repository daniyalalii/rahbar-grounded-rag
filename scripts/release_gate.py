#!/usr/bin/env python3
"""RAGOps Part C2: Release Gate Script.

Reads thresholds from a config file (or ConfigMap env vars) and compares
a candidate evaluation result against a baseline.

Usage:
    python scripts/release_gate.py --baseline eval/results/baseline.json \\
                                   --candidate eval/results/candidate.json

    # With custom thresholds file (mirrors gate-configmap.yaml):
    python scripts/release_gate.py --baseline eval/results/baseline.json \\
                                   --candidate eval/results/candidate.json \\
                                   --thresholds k8s/base/gate-configmap.yaml

Exit code 0 = PASS, 1 = FAIL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Default thresholds (can be overridden by ConfigMap env vars or --thresholds file)
DEFAULTS = {
    "RECALL_AT_5_MAX_DROP": 0.05,       # recall@5 may not fall more than 5 points vs baseline
    "FAITHFULNESS_MIN": 0.70,           # faithfulness must be >= 0.70
    "P95_LATENCY_MAX_MS": 5000.0,       # p95 latency must be <= 5000 ms
    "ABSTENTION_CORRECTNESS_MIN": 1.0,  # must abstain on ALL unanswerable questions
    "CITATION_VALIDITY_MIN": 0.90,      # citation validity >= 90%
}


def load_thresholds(path: Path | None) -> dict[str, float]:
    thresholds = dict(DEFAULTS)

    # Override from environment variables (injected from ConfigMap in K8s)
    for key in DEFAULTS:
        env_val = os.environ.get(key)
        if env_val:
            try:
                thresholds[key] = float(env_val)
            except ValueError:
                pass

    # Override from YAML/text config file
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("apiVersion") or \
               line.startswith("kind") or line.startswith("metadata") or line.startswith("data"):
                continue
            if ":" in line:
                k, _, v = line.partition(":")
                k = k.strip().upper().replace("-", "_")
                v = v.strip().strip('"').strip("'")
                if k in thresholds:
                    try:
                        thresholds[k] = float(v)
                    except ValueError:
                        pass

    return thresholds


def get_metric(result: dict, *keys: str) -> float | None:
    """Navigate nested dict with fallback keys."""
    for key_path in keys:
        parts = key_path.split(".")
        val = result
        try:
            for p in parts:
                val = val[p]
            if isinstance(val, (int, float)):
                return float(val)
        except (KeyError, TypeError):
            continue
    return None


def run_gate(baseline: dict, candidate: dict, thresholds: dict[str, float]) -> list[str]:
    failures = []

    base_r5 = get_metric(baseline, "level2_retrieval.recall_at_5", "level2.recall_at_5", "recall_at_5")
    cand_r5 = get_metric(candidate, "level2_retrieval.recall_at_5", "level2.recall_at_5", "recall_at_5")

    if base_r5 is not None and cand_r5 is not None:
        drop = round(base_r5 - cand_r5, 4)
        max_drop = thresholds["RECALL_AT_5_MAX_DROP"]
        if drop > max_drop:
            failures.append(
                f"recall@5 regression: {base_r5:.4f} → {cand_r5:.4f} "
                f"(drop={drop:.4f}, max_allowed={max_drop})"
            )
    else:
        failures.append("Could not read recall@5 from baseline or candidate results")

    faith = get_metric(candidate, "level3_e2e.faithfulness_mean", "level3.faithfulness_mean", "faithfulness_mean")
    min_faith = thresholds["FAITHFULNESS_MIN"]
    if faith is not None:
        if faith < min_faith:
            failures.append(f"faithfulness {faith:.4f} < threshold {min_faith}")
    else:
        failures.append("Could not read faithfulness from candidate results")

    p95 = get_metric(candidate, "level3_e2e.latency_p95_ms", "level3.latency_p95_ms", "latency_p95_ms")
    max_p95 = thresholds["P95_LATENCY_MAX_MS"]
    if p95 is not None:
        if p95 > max_p95:
            failures.append(f"p95 latency {p95:.0f}ms > threshold {max_p95:.0f}ms")
    else:
        failures.append("Could not read p95 latency from candidate results")

    abstain = get_metric(candidate, "level3_e2e.abstention_correctness", "level3.abstention_correctness", "abstention_correctness")
    min_abstain = thresholds["ABSTENTION_CORRECTNESS_MIN"]
    if abstain is not None:
        if abstain < min_abstain:
            failures.append(f"abstention correctness {abstain:.4f} < threshold {min_abstain}")
    else:
        failures.append("Could not read abstention correctness from candidate results")

    cv = get_metric(candidate, "level3_e2e.citation_validity_mean", "level3.citation_validity_mean", "citation_validity_mean")
    min_cv = thresholds["CITATION_VALIDITY_MIN"]
    if cv is not None:
        if cv < min_cv:
            failures.append(f"citation validity {cv:.4f} < threshold {min_cv}")
    else:
        failures.append("Could not read citation validity from candidate results")

    return failures


def print_comparison(baseline: dict, candidate: dict) -> None:
    def g(d: dict, *keys: str) -> str:
        v = get_metric(d, *keys)
        return f"{v:.4f}" if v is not None else "N/A"

    rows = [
        ("L1 similarity gap", "level1_embedding.similarity_gap",       "level1.similarity_gap"),
        ("L1 embed ms/chunk", "level1_embedding.embed_time_per_chunk_ms", "level1.embed_time_per_chunk_ms"),
        ("L2 recall@3",       "level2_retrieval.recall_at_3",         "level2.recall_at_3"),
        ("L2 recall@5",       "level2_retrieval.recall_at_5",         "level2.recall_at_5"),
        ("L2 MRR",            "level2_retrieval.mrr",                  "level2.mrr"),
        ("L2 nDCG@5",         "level2_retrieval.ndcg_at_5",           "level2.ndcg_at_5"),
        ("L3 faithfulness",   "level3_e2e.faithfulness_mean",         "level3.faithfulness_mean"),
        ("L3 abstention",     "level3_e2e.abstention_correctness",    "level3.abstention_correctness"),
        ("L3 citation valid", "level3_e2e.citation_validity_mean",    "level3.citation_validity_mean"),
        ("L3 p50 lat (ms)",   "level3_e2e.latency_p50_ms",           "level3.latency_p50_ms"),
        ("L3 p95 lat (ms)",   "level3_e2e.latency_p95_ms",           "level3.latency_p95_ms"),
    ]
    print(f"\n{'Metric':<25} {'Baseline':>12} {'Candidate':>12} {'Delta':>10}")
    print("-" * 65)
    for label, *keys in rows:
        bv_str = g(baseline, *keys)
        cv_str = g(candidate, *keys)
        try:
            delta = float(cv_str) - float(bv_str)
            delta_str = f"{delta:+.4f}"
        except ValueError:
            delta_str = "N/A"
        print(f"  {label:<23} {bv_str:>12} {cv_str:>12} {delta_str:>10}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGOps release gate")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline eval result JSON")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate eval result JSON")
    parser.add_argument("--thresholds", type=Path, default=None, help="gate-configmap.yaml or thresholds file")
    args = parser.parse_args()

    if not args.baseline.exists():
        sys.exit(f"Baseline not found: {args.baseline}")
    if not args.candidate.exists():
        sys.exit(f"Candidate not found: {args.candidate}")

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    thresholds = load_thresholds(args.thresholds)

    print(f"Baseline:  {args.baseline.name}  (label={baseline.get('label', 'unknown')})")
    print(f"Candidate: {args.candidate.name}  (label={candidate.get('label', 'unknown')})")
    print("\nThresholds in effect:")
    for k, v in thresholds.items():
        print(f"  {k}: {v}")

    print_comparison(baseline, candidate)

    failures = run_gate(baseline, candidate, thresholds)

    print("\n" + "=" * 65)
    if failures:
        print("RELEASE GATE: ✗ FAIL")
        for f in failures:
            print(f"  ✗ {f}")
        print("Candidate will NOT be rolled out.")
        return 1
    print("RELEASE GATE: ✓ PASS")
    print("Candidate is cleared for rolling update.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
