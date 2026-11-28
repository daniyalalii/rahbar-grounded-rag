"""Tests for RAGOps Part B/C gate and threshold loading."""

from __future__ import annotations

from pathlib import Path

from eval.run_eval_ragops import load_thresholds, run_gate


def test_run_gate_reads_level2_retrieval_and_level3_e2e() -> None:
    baseline = {
        "level2_retrieval": {"recall_at_5": 0.90},
        "level3_e2e": {
            "faithfulness_mean": 0.85,
            "latency_p95_ms": 1500.0,
            "abstention_correctness": 1.0,
            "citation_validity_mean": 0.95,
        },
    }
    passing = {
        "level2_retrieval": {"recall_at_5": 0.88},
        "level3_e2e": {
            "faithfulness_mean": 0.85,
            "latency_p95_ms": 1800.0,
            "abstention_correctness": 1.0,
            "citation_validity_mean": 0.95,
        },
    }
    failing = {
        "level2_retrieval": {"recall_at_5": 0.70},
        "level3_e2e": {
            "faithfulness_mean": 0.60,
            "latency_p95_ms": 6000.0,
            "abstention_correctness": 0.80,
            "citation_validity_mean": 0.70,
        },
    }
    thresholds = load_thresholds(None)
    assert run_gate(passing, baseline, thresholds) == []
    failures = run_gate(failing, baseline, thresholds)
    assert len(failures) == 5


def test_load_thresholds_reads_k8s_configmap(tmp_path: Path, monkeypatch) -> None:
    cm = tmp_path / "gate-configmap.yaml"
    cm.write_text(
        "apiVersion: v1\nkind: ConfigMap\ndata:\n  RECALL_AT_5_MAX_DROP: '0.02'\n  FAITHFULNESS_MIN: '0.80'\n",
        encoding="utf-8",
    )
    t = load_thresholds(cm)
    assert t["RECALL_AT_5_MAX_DROP"] == 0.02
    assert t["FAITHFULNESS_MIN"] == 0.80

    monkeypatch.setenv("FAITHFULNESS_MIN", "0.92")
    t2 = load_thresholds(None)
    assert t2["FAITHFULNESS_MIN"] == 0.92

