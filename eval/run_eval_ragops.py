#!/usr/bin/env python3
"""RAGOps Assignment 03 - Full Three-Level Evaluation Suite.

Part B: Offline testing at three levels (paper 4.2.5, Table 1)

Levels:
  Level 1 - Module (embedding): paraphrase similarity gap, embed time, determinism
  Level 2 - Component (retrieval): recall@3, recall@5, MRR, nDCG@5, noise test
  Level 3 - End-to-end (answers): faithfulness, abstention, citations, latency p50/p95

Usage:
    python eval/run_eval_ragops.py --api http://localhost:8000
    python eval/run_eval_ragops.py --api http://localhost:8000 --golden eval/golden.jsonl
    python eval/run_eval_ragops.py --api http://localhost:8000 --label candidate --save-baseline
    python eval/run_eval_ragops.py --api http://localhost:8000 --compare eval/results/baseline_*.json

Exit code 0 = gate passed, 1 = gate failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Optional

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

random.seed(42)

# ---------------------------------------------------------------------------
# Paraphrase pairs for Level 1 (embedding module test)
# ---------------------------------------------------------------------------

PARAPHRASE_PAIRS = [
    ("What CGPA is needed to avoid academic warning?",
     "What is the minimum GPA required to stay off academic probation?"),
    ("What percentage of attendance is required for the final exam?",
     "How much class attendance must a student have to sit the final examination?"),
    ("What is the maximum time to complete a BS degree?",
     "How many years does a student have to finish their undergraduate program?"),
    ("What is the transcript fee?",
     "How much does an additional copy of the academic transcript cost?"),
    ("What is the campus transfer fee?",
     "What does it cost to transfer between university campuses?"),
    ("Can a student repeat a failed course?",
     "Is it possible to redo a course in which the student received a failing grade?"),
    ("What grade point corresponds to B+?",
     "What is the GPA value for a B-plus letter grade?"),
    ("What happens if a student misses more than 20% of classes?",
     "What are the consequences of attendance falling below 80 percent?"),
    ("How long is the summer semester?",
     "What is the duration of the summer term at FAST-NUCES?"),
    ("What is the late registration fee?",
     "How much is charged if a student registers after the deadline?"),
    ("What is the admission fee for the BBA program?",
     "How much must a BBA applicant pay upon admission?"),
    ("What is the PhD admission fee?",
     "How much does it cost to apply for a PhD program?"),
    ("What restrictions apply during academic probation?",
     "What limitations are placed on a student who is on probation?"),
    ("What is the re-admission fee?",
     "How much does it cost to rejoin the university after leaving?"),
    ("What is the duplicate degree fee?",
     "What is the charge for getting a replacement copy of the degree certificate?"),
]

UNRELATED_PAIRS = [
    ("What CGPA is needed to avoid academic warning?",
     "What is the duplicate degree fee?"),
    ("What percentage of attendance is required?",
     "How long is the summer semester?"),
    ("What is the late registration fee?",
     "Can a student repeat a failed course?"),
    ("What is the campus transfer fee?",
     "What grade point corresponds to B+?"),
    ("What is the transcript fee?",
     "What restrictions apply during academic probation?"),
    ("What is the PhD admission fee?",
     "What happens if a student misses more than 20% of classes?"),
    ("What is the re-admission fee?",
     "What is the maximum time to complete a BS degree?"),
    ("How long is the summer semester?",
     "What is the duplicate degree fee?"),
    ("What is the admission fee for the BBA program?",
     "What are the consequences of attendance falling below 80 percent?"),
    ("What limitations apply during probation?",
     "What is the charge for getting a replacement copy of the degree?"),
    ("How much must a BBA applicant pay upon admission?",
     "What happens if a student misses more than 20% of classes?"),
    ("What is the GPA value for a B-plus letter grade?",
     "What is the duration of the summer term?"),
    ("What is the late registration charge?",
     "What is the PhD admission fee?"),
    ("Can a student redo a failed course?",
     "What is the campus transfer fee?"),
    ("What is the minimum GPA to stay off probation?",
     "What is the duplicate degree replacement cost?"),
]

# ---------------------------------------------------------------------------
# Noise documents for Level 2 noise test
# ---------------------------------------------------------------------------

NOISE_DOCS = [
    "The capital of France is Paris. The Eiffel Tower was built in 1889.",
    "Quantum entanglement is a phenomenon where two particles become correlated.",
    "The Amazon rainforest covers approximately 5.5 million square kilometres.",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _embed_via_api(texts: list[str], ollama_url: str, embed_model: str) -> list[list[float]]:
    url = f"{ollama_url.rstrip('/')}/api/embed"
    r = httpx.post(url, json={"model": embed_model, "input": texts}, timeout=60.0)
    r.raise_for_status()
    return r.json()["embeddings"]


def _ask(client: httpx.Client, api: str, question: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = client.post(f"{api.rstrip('/')}/api/ask", json={"question": question}, timeout=120.0)
    r.raise_for_status()
    return r.json(), time.perf_counter() - t0


def _norm(s: str) -> str:
    import re
    s = re.sub(r"\.(pdf|md|html|txt)$", "", (s or "").strip(), flags=re.IGNORECASE)
    return " ".join(s.lower().split())


def _matches(expected: dict, candidate: dict) -> bool:
    exp_src = _norm(str(expected.get("source", "")))
    cand_src = _norm(str(candidate.get("source", "")))
    if exp_src and cand_src and exp_src not in cand_src and cand_src not in exp_src:
        return False
    exp_sec = _norm(str(expected.get("section", "")))
    cand_sec = _norm(str(candidate.get("section", "")))
    if exp_sec and cand_sec:
        return exp_sec in cand_sec or cand_sec in exp_sec
    return True


def recall_at_k(relevant: list[dict], retrieved: list[dict], k: int) -> Optional[float]:
    if not relevant:
        return None
    top_k = retrieved[:k]
    matched = sum(1 for rel in relevant if any(_matches(rel, ret) for ret in top_k))
    return matched / len(relevant)


def mrr(relevant: list[dict], retrieved: list[dict]) -> Optional[float]:
    if not relevant:
        return None
    for i, ret in enumerate(retrieved, 1):
        if any(_matches(rel, ret) for rel in relevant):
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: list[dict], retrieved: list[dict], k: int) -> Optional[float]:
    if not relevant:
        return None
    rel_set = set()
    for rel in relevant:
        rel_set.add((_norm(str(rel.get("source", ""))), _norm(str(rel.get("section", "")))))

    def is_relevant(ret: dict) -> bool:
        rs = _norm(str(ret.get("source", "")))
        rc = _norm(str(ret.get("section", "")))
        return any((rs in s or s in rs) and (rc in c or c in rc) for s, c in rel_set)

    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, ret in enumerate(retrieved[:k], 1)
        if is_relevant(ret)
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def citation_validity(response: dict) -> Optional[float]:
    cites = response.get("citations") or []
    if not cites:
        return None
    retrieved = response.get("retrieved") or []
    if not retrieved:
        return 0.0
    valid = sum(
        1 for c in cites
        if any(
            (_norm(str(c.get("source", ""))) in _norm(str(r.get("source", ""))) or
             _norm(str(r.get("source", ""))) in _norm(str(c.get("source", ""))))
            for r in retrieved
        )
    )
    return valid / len(cites)


def faithfulness_heuristic(response: dict, question_row: dict) -> float:
    """Simple keyword-based faithfulness: does the answer contain the expected keyword?
    Returns 1.0 (faithful) or 0.0 (not faithful). For must-abstain, returns 1.0 if abstained."""
    if question_row.get("must_abstain"):
        return 1.0 if not response.get("answered_from_context", True) else 0.0
    expected = str(question_row.get("expected_answer", "")).lower()
    if not expected:
        return 1.0 if response.get("answered_from_context") else 0.5
    answer_text = str(response.get("answer", "")).lower()
    return 1.0 if expected in answer_text else 0.0


# ---------------------------------------------------------------------------
# Level 1: Module (embedding)
# ---------------------------------------------------------------------------

def run_level1(ollama_url: str, embed_model: str) -> dict:
    print("\n=== Level 1: Module (Embedding) ===")
    results: dict = {}
    errors: list[str] = []

    try:
        # 1a. Paraphrase pairs similarity
        t0 = time.perf_counter()
        all_texts = [t for pair in PARAPHRASE_PAIRS for t in pair]
        vecs = _embed_via_api(all_texts, ollama_url, embed_model)
        embed_total_time = time.perf_counter() - t0

        para_sims = []
        for i, (a, b) in enumerate(PARAPHRASE_PAIRS):
            va, vb = vecs[i * 2], vecs[i * 2 + 1]
            para_sims.append(_cosine(va, vb))

        # 1b. Unrelated pairs
        unrel_texts = [t for pair in UNRELATED_PAIRS for t in pair]
        unrel_vecs = _embed_via_api(unrel_texts, ollama_url, embed_model)
        unrel_sims = []
        for i in range(len(UNRELATED_PAIRS)):
            va, vb = unrel_vecs[i * 2], unrel_vecs[i * 2 + 1]
            unrel_sims.append(_cosine(va, vb))

        mean_para = mean(para_sims)
        mean_unrel = mean(unrel_sims)
        gap = mean_para - mean_unrel

        # 1c. Embedding time per chunk
        embed_time_per_chunk_ms = (embed_total_time / len(all_texts)) * 1000

        # 1d. Determinism check
        test_text = "What is the minimum CGPA to avoid academic warning?"
        v1 = _embed_via_api([test_text], ollama_url, embed_model)[0]
        v2 = _embed_via_api([test_text], ollama_url, embed_model)[0]
        max_diff = max(abs(a - b) for a, b in zip(v1, v2))

        results = {
            "mean_paraphrase_similarity": round(mean_para, 4),
            "mean_unrelated_similarity": round(mean_unrel, 4),
            "similarity_gap": round(gap, 4),
            "embed_time_per_chunk_ms": round(embed_time_per_chunk_ms, 2),
            "determinism_max_diff": max_diff,
            "paraphrase_pairs_count": len(PARAPHRASE_PAIRS),
            "unrelated_pairs_count": len(UNRELATED_PAIRS),
            "embed_model": embed_model,
        }
        print(f"  Paraphrase sim:   {mean_para:.4f}")
        print(f"  Unrelated sim:    {mean_unrel:.4f}")
        print(f"  Gap:              {gap:.4f}")
        print(f"  Embed time/chunk: {embed_time_per_chunk_ms:.1f} ms")
        print(f"  Determinism diff: {max_diff:.2e}")

    except Exception as exc:
        errors.append(f"Level 1 error: {exc}")
        print(f"  [WARNING] Level 1 skipped: {exc}")
        results = {"error": str(exc), "skipped": True}

    results["errors"] = errors
    return results


# ---------------------------------------------------------------------------
# Level 2: Component (retrieval)
# ---------------------------------------------------------------------------

def run_level2(client: httpx.Client, api: str, questions: list[dict]) -> dict:
    print("\n=== Level 2: Component (Retrieval) ===")
    r3_list, r5_list, mrr_list, ndcg_list = [], [], [], []

    for q in questions:
        if q.get("must_abstain"):
            continue
        relevant = [{"source": q.get("ground_truth_doc", ""), "section": q.get("ground_truth_section", "")}]
        try:
            resp, _ = _ask(client, api, q["question"])
            retrieved = resp.get("retrieved") or []
            r3 = recall_at_k(relevant, retrieved, 3)
            r5 = recall_at_k(relevant, retrieved, 5)
            m = mrr(relevant, retrieved)
            n = ndcg_at_k(relevant, retrieved, 5)
            if r3 is not None: r3_list.append(r3)
            if r5 is not None: r5_list.append(r5)
            if m is not None: mrr_list.append(m)
            if n is not None: ndcg_list.append(n)
        except Exception as exc:
            print(f"  [WARNING] {q['id']}: {exc}")

    baseline = {
        "recall_at_3": round(mean(r3_list), 4) if r3_list else 0.0,
        "recall_at_5": round(mean(r5_list), 4) if r5_list else 0.0,
        "mrr": round(mean(mrr_list), 4) if mrr_list else 0.0,
        "ndcg_at_5": round(mean(ndcg_list), 4) if ndcg_list else 0.0,
        "questions_scored": len(r5_list),
    }

    # Noise test: add typos to queries and re-score
    def _add_typo(text: str) -> str:
        words = text.split()
        if len(words) > 2:
            idx = random.randint(0, len(words) - 1)
            w = words[idx]
            if len(w) > 3:
                i = random.randint(1, len(w) - 2)
                words[idx] = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        return " ".join(words)

    r5_noise = []
    for q in questions[:10]:  # sample 10 for noise test
        if q.get("must_abstain"):
            continue
        relevant = [{"source": q.get("ground_truth_doc", ""), "section": q.get("ground_truth_section", "")}]
        try:
            noisy_q = _add_typo(q["question"])
            resp, _ = _ask(client, api, noisy_q)
            retrieved = resp.get("retrieved") or []
            r5 = recall_at_k(relevant, retrieved, 5)
            if r5 is not None:
                r5_noise.append(r5)
        except Exception:
            pass

    noise_r5 = round(mean(r5_noise), 4) if r5_noise else 0.0
    noise_drop = round(baseline["recall_at_5"] - noise_r5, 4)

    results = {
        **baseline,
        "noise_test": {
            "recall_at_5_with_typos": noise_r5,
            "recall_at_5_drop": noise_drop,
            "questions_sampled": len(r5_noise),
        },
    }
    print(f"  recall@3:  {baseline['recall_at_3']:.4f}")
    print(f"  recall@5:  {baseline['recall_at_5']:.4f}")
    print(f"  MRR:       {baseline['mrr']:.4f}")
    print(f"  nDCG@5:    {baseline['ndcg_at_5']:.4f}")
    print(f"  Noise drop (typos): {noise_drop:.4f}")
    return results


# ---------------------------------------------------------------------------
# Level 3: End-to-End (answers)
# ---------------------------------------------------------------------------

def run_level3(client: httpx.Client, api: str, questions: list[dict]) -> dict:
    print("\n=== Level 3: End-to-End (Answers) ===")

    faithfulness_scores, abstain_correct, abstain_total = [], 0, 0
    citation_scores, latencies = [], []
    rows = []

    for q in questions:
        try:
            resp, elapsed = _ask(client, api, q["question"])
            latencies.append(elapsed * 1000)

            faith = faithfulness_heuristic(resp, q)
            faithfulness_scores.append(faith)

            cv = citation_validity(resp)
            if cv is not None:
                citation_scores.append(cv)

            if q.get("must_abstain"):
                abstain_total += 1
                if not resp.get("answered_from_context", True):
                    abstain_correct += 1

            rows.append({
                "id": q["id"],
                "type": q.get("type", "answerable"),
                "must_abstain": bool(q.get("must_abstain")),
                "faithfulness": faith,
                "citation_validity": cv,
                "latency_ms": round(elapsed * 1000, 1),
                "answered_from_context": resp.get("answered_from_context"),
                "answer_snippet": str(resp.get("answer", ""))[:120],
            })
            print(f"  {q['id']:<5} {elapsed*1000:>6.0f}ms  faith={faith:.2f}  {'ABSTAIN' if not resp.get('answered_from_context') else 'ANSWERED'}")
        except Exception as exc:
            print(f"  [WARNING] {q['id']}: {exc}")

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies_sorted else 0

    results = {
        "faithfulness_mean": round(mean(faithfulness_scores), 4) if faithfulness_scores else 0.0,
        "faithfulness_method": "keyword_heuristic",
        "abstention_correctness": round(abstain_correct / abstain_total, 4) if abstain_total else 1.0,
        "abstain_correct": abstain_correct,
        "abstain_total": abstain_total,
        "citation_validity_mean": round(mean(citation_scores), 4) if citation_scores else 1.0,
        "latency_p50_ms": round(p50, 1),
        "latency_p95_ms": round(p95, 1),
        "questions_scored": len(rows),
        "per_question": rows,
    }
    print(f"  Faithfulness: {results['faithfulness_mean']:.4f}")
    print(f"  Abstention:   {results['abstention_correctness']:.4f} ({abstain_correct}/{abstain_total})")
    print(f"  Citation:     {results['citation_validity_mean']:.4f}")
    print(f"  Latency p50:  {p50:.0f}ms  p95: {p95:.0f}ms")
    return results


# ---------------------------------------------------------------------------
# Stochasticity check
# ---------------------------------------------------------------------------

def run_stochasticity(client: httpx.Client, api: str, questions: list[dict]) -> dict:
    """Run level 2 twice with same seed and report metric differences."""
    print("\n=== Stochasticity Check (2x run) ===")
    runs = []
    for run_num in range(2):
        random.seed(42)
        r5_list = []
        for q in questions[:10]:
            if q.get("must_abstain"):
                continue
            relevant = [{"source": q.get("ground_truth_doc", ""), "section": q.get("ground_truth_section", "")}]
            try:
                resp, _ = _ask(client, api, q["question"])
                r5 = recall_at_k(relevant, resp.get("retrieved") or [], 5)
                if r5 is not None:
                    r5_list.append(r5)
            except Exception:
                pass
        runs.append(round(mean(r5_list), 4) if r5_list else 0.0)
        print(f"  Run {run_num+1}: recall@5 = {runs[-1]:.4f}")

    diff = abs(runs[0] - runs[1]) if len(runs) == 2 else 0
    result = {
        "run1_recall_at_5": runs[0] if len(runs) > 0 else 0,
        "run2_recall_at_5": runs[1] if len(runs) > 1 else 0,
        "diff": round(diff, 4),
        "explanation": (
            "Zero diff expected with fixed seeds. Any variation is from non-deterministic LLM generation."
            if diff == 0 else
            f"Small diff of {diff:.4f} due to non-deterministic LLM temperature."
        ),
    }
    print(f"  Diff: {diff:.4f}")
    return result


# ---------------------------------------------------------------------------
# Release gate
# ---------------------------------------------------------------------------

def run_gate(results: dict, baseline: dict | None, thresholds: dict) -> list[str]:
    failures = []
    r5 = results.get("level2", {}).get("recall_at_5", 0.0)
    faith = results.get("level3", {}).get("faithfulness_mean", 0.0)
    p95 = results.get("level3", {}).get("latency_p95_ms", 999999)
    abstain = results.get("level3", {}).get("abstention_correctness", 0.0)
    cv = results.get("level3", {}).get("citation_validity_mean", 0.0)

    if baseline:
        base_r5 = baseline.get("level2", {}).get("recall_at_5", baseline.get("recall_at_5", 0.0))
        max_drop = float(thresholds.get("recall_at_5_max_drop", 0.05))
        drop = round(base_r5 - r5, 4)
        if drop > max_drop:
            failures.append(f"recall@5 dropped {drop:.4f} (baseline {base_r5:.4f} → {r5:.4f}), max allowed drop {max_drop}")

    min_faith = float(thresholds.get("faithfulness_min", 0.70))
    if faith < min_faith:
        failures.append(f"faithfulness {faith:.4f} < threshold {min_faith}")

    max_p95 = float(thresholds.get("p95_latency_max_ms", 5000))
    if p95 > max_p95:
        failures.append(f"p95 latency {p95:.0f}ms > threshold {max_p95:.0f}ms")

    min_abstain = float(thresholds.get("abstention_correctness_min", 1.0))
    if abstain < min_abstain:
        failures.append(f"abstention correctness {abstain:.4f} < threshold {min_abstain}")

    min_cv = float(thresholds.get("citation_validity_min", 0.90))
    if cv < min_cv:
        failures.append(f"citation validity {cv:.4f} < threshold {min_cv}")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_golden(path: Path) -> list[dict]:
    questions = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def load_thresholds(path: Path | None) -> dict:
    defaults = {
        "recall_at_5_max_drop": 0.05,
        "faithfulness_min": 0.70,
        "p95_latency_max_ms": 5000,
        "abstention_correctness_min": 1.0,
        "citation_validity_min": 0.90,
    }
    if path and path.exists():
        try:
            data = {}
            for line in path.read_text().splitlines():
                if ":" in line and not line.strip().startswith("#"):
                    k, _, v = line.partition(":")
                    data[k.strip()] = v.strip().strip('"')
            defaults.update(data)
        except Exception:
            pass
    return defaults


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGOps 3-Level Evaluation Suite")
    parser.add_argument("--api", required=True, help="API base URL e.g. http://localhost:8000")
    parser.add_argument("--golden", type=Path, default=HERE / "golden.jsonl")
    parser.add_argument("--baseline", type=Path, default=HERE / "results" / "baseline.json")
    parser.add_argument("--thresholds", type=Path, default=None, help="gate-configmap.yaml or thresholds file")
    parser.add_argument("--label", default="run", help="label for this run (e.g. 'baseline', 'candidate')")
    parser.add_argument("--save-baseline", action="store_true", help="save results as new baseline")
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://host.minikube.internal:11434"))
    parser.add_argument("--embed-model", default=os.environ.get("EMBED_MODEL", "nomic-embed-text"))
    parser.add_argument("--skip-level1", action="store_true", help="skip embedding module tests (no Ollama needed)")
    args = parser.parse_args()

    if not args.golden.exists():
        sys.exit(f"Golden set not found: {args.golden}")

    questions = load_golden(args.golden)
    thresholds = load_thresholds(args.thresholds)
    baseline = json.loads(args.baseline.read_text()) if args.baseline.exists() else None

    print(f"RAHBAR RAGOps Evaluation — {len(questions)} questions — label: {args.label}")
    print(f"API: {args.api}")

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    # Level 1
    if args.skip_level1:
        level1 = {"skipped": True}
    else:
        level1 = run_level1(args.ollama_url, args.embed_model)

    with httpx.Client(timeout=120.0) as client:
        level2 = run_level2(client, args.api, questions)
        level3 = run_level3(client, args.api, questions)
        stochastic = run_stochasticity(client, args.api, questions)

    results = {
        "timestamp": timestamp,
        "label": args.label,
        "golden": str(args.golden),
        "questions_total": len(questions),
        "api": args.api,
        "embed_model": args.embed_model,
        "level1_embedding": level1,
        "level2_retrieval": level2,
        "level3_e2e": level3,
        "stochasticity": stochastic,
    }

    # Gate
    failures = run_gate(results, baseline, thresholds)

    # Write timestamped result file
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"eval_{args.label}_{timestamp}.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_file}")

    if args.save_baseline:
        args.baseline.parent.mkdir(exist_ok=True)
        args.baseline.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Baseline saved to {args.baseline}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    if not level1.get("skipped"):
        print(f"  L1 similarity gap:   {level1.get('similarity_gap', 'N/A')}")
        print(f"  L1 embed time/chunk: {level1.get('embed_time_per_chunk_ms', 'N/A')} ms")
    print(f"  L2 recall@3:         {level2.get('recall_at_3', 0):.4f}")
    print(f"  L2 recall@5:         {level2.get('recall_at_5', 0):.4f}")
    print(f"  L2 MRR:              {level2.get('mrr', 0):.4f}")
    print(f"  L2 nDCG@5:           {level2.get('ndcg_at_5', 0):.4f}")
    print(f"  L3 faithfulness:     {level3.get('faithfulness_mean', 0):.4f}")
    print(f"  L3 abstention:       {level3.get('abstention_correctness', 0):.4f}")
    print(f"  L3 citation valid:   {level3.get('citation_validity_mean', 0):.4f}")
    print(f"  L3 latency p50:      {level3.get('latency_p50_ms', 0):.0f} ms")
    print(f"  L3 latency p95:      {level3.get('latency_p95_ms', 0):.0f} ms")
    print("=" * 60)

    if failures:
        print("\nGATE FAILED")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("\nGATE PASSED ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
