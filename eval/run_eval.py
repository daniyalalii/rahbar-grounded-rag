#!/usr/bin/env python3
"""Score a running RAHBAR against a golden set, and gate CI on the result (section 5).

    python eval/run_eval.py --api http://<ingress>                 # your corpus
    python eval/run_eval.py --api ... --golden eval/fixtures/golden.fixture.yaml
    python eval/run_eval.py --api ... --update-baseline            # after a deliberate improvement

Exit code 0 when every gate passes, 1 when any fails. CI relies on that exit code.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx
import yaml  # type: ignore[import-untyped]

HERE = Path(__file__).resolve().parent

# The gate from section 5.4. Change these only with a reason written in docs/PIPELINE.md.
MAX_RECALL5_DROP = 0.05        # recall@5 may fall at most 5 points below the baseline
MIN_CITATION_VALIDITY = 0.95   # at least 95% of citations must match retrieved passages
REQUIRED_ABSTENTION = 1.0      # every must-abstain question must be declined


def load_golden(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions = data.get("questions") or []
    if not questions:
        sys.exit(f"no questions in {path}")
    return questions


def ask(client: httpx.Client, api: str, question: str) -> dict:
    """POST one question and return the AskResponse as a dict (see api/app/schemas.py)."""
    response = client.post(f"{api.rstrip('/')}/api/ask", json={"question": question})
    response.raise_for_status()
    return response.json()


def _norm(s: str | None) -> str:
    """Normalize source or section strings for robust matching."""
    if not s:
        return ""
    # strip file extensions, punctuation and lowercase
    cleaned = re.sub(r"\.(pdf|md|html|txt)$", "", s.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"[^\w\s\.-]", "", cleaned)
    return " ".join(cleaned.lower().split())


def _matches(expected: dict, candidate: dict) -> bool:
    """Check if a retrieved passage matches an expected relevant passage."""
    exp_src = _norm(str(expected.get("source", "")))
    cand_src = _norm(str(candidate.get("source", "")))
    if exp_src and cand_src and exp_src not in cand_src and cand_src not in exp_src:
        return False

    exp_sec = _norm(str(expected.get("section", "")))
    cand_sec = _norm(str(candidate.get("section", "")))
    if exp_sec and cand_sec:
        # Match if either is substring of the other or exact token match
        if exp_sec in cand_sec or cand_sec in exp_sec:
            return True
        return False
    return True


def recall_at_k(relevant: list[dict], retrieved: list[dict], k: int) -> float | None:
    """Fraction of the relevant (source, section) pairs found in the top k retrieved.
    Return None when the question has no relevant passages (must-abstain questions)."""
    if not relevant:
        return None

    top_k = retrieved[:k]
    matched = 0
    for rel in relevant:
        for ret in top_k:
            if _matches(rel, ret):
                matched += 1
                break

    return matched / len(relevant)


def citation_validity(response: dict) -> float | None:
    """Fraction of the response's citations that match a passage in response['retrieved'].
    Return None when there are no citations."""
    citations = response.get("citations") or []
    if not citations:
        return None

    retrieved = response.get("retrieved") or []
    if not retrieved:
        return 0.0

    valid = 0
    for cite in citations:
        cite_src = _norm(str(cite.get("source", "")))
        cite_sec = _norm(str(cite.get("section", "")))
        for ret in retrieved:
            ret_src = _norm(str(ret.get("source", "")))
            ret_sec = _norm(str(ret.get("section", "")))
            src_match = (not cite_src) or (cite_src in ret_src) or (ret_src in cite_src)
            sec_match = (not cite_sec) or (cite_sec in ret_sec) or (ret_sec in cite_sec)
            if src_match and sec_match:
                valid += 1
                break

    return valid / len(citations)


def summarise(rows: list[dict]) -> dict:
    """Overall recall@3, recall@5, citation validity and abstention correctness, plus
    recall@5 broken down by question type. This dict becomes results.json."""
    r3_list: list[float] = []
    r5_list: list[float] = []
    r5_by_type: dict[str, list[float]] = {}
    citation_validities: list[float] = []
    abstain_correct = 0
    abstain_total = 0

    for row in rows:
        q_type = row.get("type", "ordinary")
        is_abstain = bool(row.get("must_abstain", False))
        resp = row.get("response") or {}
        retrieved = resp.get("retrieved") or []
        relevant = row.get("relevant") or []

        r3 = recall_at_k(relevant, retrieved, 3)
        r5 = recall_at_k(relevant, retrieved, 5)

        if r3 is not None:
            r3_list.append(r3)
        if r5 is not None:
            r5_list.append(r5)
            r5_by_type.setdefault(q_type, []).append(r5)

        cv = citation_validity(resp)
        if cv is not None:
            citation_validities.append(cv)

        if is_abstain:
            abstain_total += 1
            # Honest abstention: answered_from_context must be False
            if not resp.get("answered_from_context", True):
                abstain_correct += 1

    overall_r3 = round(sum(r3_list) / len(r3_list), 4) if r3_list else 0.0
    overall_r5 = round(sum(r5_list) / len(r5_list), 4) if r5_list else 0.0
    overall_cv = round(sum(citation_validities) / len(citation_validities), 4) if citation_validities else 1.0
    abstain_rate = round(abstain_correct / abstain_total, 4) if abstain_total else 1.0

    breakdown_r5 = {
        t: round(sum(vals) / len(vals), 4) if vals else 0.0
        for t, vals in r5_by_type.items()
    }

    return {
        "recall_at_3": overall_r3,
        "recall_at_5": overall_r5,
        "citation_validity": overall_cv,
        "abstention_correctness": abstain_rate,
        "recall_at_5_by_type": breakdown_r5,
        "must_abstain_total": abstain_total,
        "must_abstain_correct": abstain_correct,
    }


def gate(results: dict, baseline: dict | None) -> list[str]:
    """Return one message per failed gate (section 5.4); an empty list means pass."""
    failures: list[str] = []

    # 1. Baseline regression check: recall@5 may not drop more than MAX_RECALL5_DROP (0.05)
    if baseline is None or not isinstance(baseline, dict) or "recall_at_5" not in baseline or not isinstance(baseline.get("recall_at_5"), (int, float)):
        failures.append("missing or malformed baseline: cannot verify regression gate against baseline")
    else:
        r5 = results.get("recall_at_5", 0.0)
        base_r5 = float(baseline["recall_at_5"])
        # Round to avoid floating point precision issues (e.g., 0.90 - 0.85 = 0.050000000000000044)
        drop = round(base_r5 - r5, 4)
        if drop > MAX_RECALL5_DROP:
            failures.append(
                f"recall@5 dropped by {drop:.4f} (from baseline {base_r5:.4f} to {r5:.4f}), "
                f"exceeding max allowed drop {MAX_RECALL5_DROP}"
            )

    # 2. Citation validity gate: >= MIN_CITATION_VALIDITY (0.95)
    cv = results.get("citation_validity", 1.0)
    if cv < MIN_CITATION_VALIDITY:
        failures.append(
            f"citation validity {cv:.4f} is below required threshold {MIN_CITATION_VALIDITY}"
        )

    # 3. Must-abstain correctness gate: == REQUIRED_ABSTENTION (1.0 = 100%)
    abstain = results.get("abstention_correctness", 1.0)
    if abstain < REQUIRED_ABSTENTION:
        failures.append(
            f"must-abstain correctness {abstain:.4f} is below required 100% ({REQUIRED_ABSTENTION})"
        )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAHBAR against a golden set.")
    parser.add_argument("--api", required=True, help="base URL, e.g. http://192.168.49.2")
    parser.add_argument("--host", help="Host header, when your Ingress rule names a host")
    parser.add_argument("--golden", type=Path, default=HERE / "golden.yaml")
    parser.add_argument("--baseline", type=Path, default=HERE / "baseline.json")
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    parser.add_argument("--update-baseline", action="store_true",
                        help="write these results as the new baseline")
    args = parser.parse_args()

    questions = load_golden(args.golden)
    rows = []
    headers = {"Host": args.host} if args.host else {}
    with httpx.Client(timeout=120.0, headers=headers) as client:
        for q in questions:
            started = time.perf_counter()
            response = ask(client, args.api, q["question"])
            rows.append({
                "id": q["id"], "type": q.get("type", "ordinary"),
                "must_abstain": bool(q.get("must_abstain", False)),
                "relevant": q.get("relevant") or [],
                "expect": q.get("expect"),
                "response": response,
                "seconds": round(time.perf_counter() - started, 2),
            })
            print(f"  {q['id']:<5} {rows[-1]['seconds']:>6.2f}s  {q['question'][:60]}")

    results = summarise(rows)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline.exists() else None
    failures = gate(results, baseline)

    results["golden"] = str(args.golden)
    results["questions"] = len(rows)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults written to {args.out}")
    if args.update_baseline:
        args.baseline.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"baseline updated: {args.baseline}")

    if failures:
        print("\nGATE FAILED")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("\ngate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
