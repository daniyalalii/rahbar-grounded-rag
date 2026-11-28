#!/usr/bin/env python3
"""Run verification check on Batch B and print a clear summary table."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indexer.pipeline.extract import extract
from indexer.pipeline.clean import clean
from indexer.pipeline.verify import verify_corpus

batch_b_dir = Path("corpus/batch_b")
docs = []
for p in sorted(batch_b_dir.glob("*.md")):
    doc = clean(extract(str(p), "1.0", False))
    docs.append((p, doc))

passed, report = verify_corpus(docs, "corpus/quarantine", "eval/results")

print("=" * 75)
print(f"PART A: BATCH B VERIFICATION SUMMARY ({report.run_id})")
print("=" * 75)
print(f"Total documents:   {report.total}")
print(f"Passed (Indexed):  {report.passed}")
print(f"Quarantined:       {report.quarantined}")
print("-" * 75)
print(f"{'STATUS':<6} {'FILENAME':<24} {'REASON / DETAILS'}")
print("-" * 75)

for d in report.documents:
    status = "PASS" if d.overall_passed else "FAIL"
    reasons = []
    if not d.quality.passed:
        reasons.append(f"Quality: {d.quality.reason}")
    if not d.completeness.passed:
        reasons.append(f"Completeness: {d.completeness.reason}")
    if not d.recency.passed:
        reasons.append(f"Recency: {d.recency.reason}")
    if not d.uniqueness.passed:
        reasons.append(f"Uniqueness: {d.uniqueness.reason}")
    if d.injection_detected:
        reasons.append("[!] Guardrail: Prompt injection detected")

    detail = "; ".join(reasons) if reasons else "OK (All 4 checks passed)"
    print(f"{status:<6} {d.file:<24} {detail}")

print("=" * 75)
