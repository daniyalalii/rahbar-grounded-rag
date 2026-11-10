"""Unit tests for the evaluation engine functions in eval/run_eval.py."""

from __future__ import annotations

from eval.run_eval import citation_validity, gate, recall_at_k, summarise


def test_recall_at_k_perfect() -> None:
    relevant = [{"source": "rules-2026.pdf", "section": "1.21 Attendance"}]
    retrieved = [
        {"source": "rules-2026.pdf", "section": "1.21 Attendance"},
        {"source": "rules-2026.pdf", "section": "2.1 Grading"},
    ]
    assert recall_at_k(relevant, retrieved, 1) == 1.0
    assert recall_at_k(relevant, retrieved, 3) == 1.0


def test_recall_at_k_zero() -> None:
    relevant = [{"source": "rules-2026.pdf", "section": "1.21 Attendance"}]
    retrieved = [
        {"source": "rules-2026.pdf", "section": "5.4 Warning"},
        {"source": "fees-2026.pdf", "section": "Tuition"},
    ]
    assert recall_at_k(relevant, retrieved, 2) == 0.0


def test_recall_at_k_partial() -> None:
    relevant = [
        {"source": "rules-2026.pdf", "section": "1.21 Attendance"},
        {"source": "rules-2026.pdf", "section": "5.4 Warning"},
    ]
    retrieved = [
        {"source": "rules-2026.pdf", "section": "1.21 Attendance"},
        {"source": "other.pdf", "section": "Intro"},
    ]
    assert recall_at_k(relevant, retrieved, 2) == 0.5


def test_recall_at_k_must_abstain_returns_none() -> None:
    assert recall_at_k([], [{"source": "rules.pdf", "section": "1"}], 5) is None


def test_citation_validity_perfect() -> None:
    response = {
        "citations": [{"source": "rules-2026.pdf", "section": "1.21"}],
        "retrieved": [{"source": "rules-2026.pdf", "section": "1.21 Attendance"}],
    }
    assert citation_validity(response) == 1.0


def test_citation_validity_hallucinated() -> None:
    response = {
        "citations": [{"source": "invented.pdf", "section": "99.9"}],
        "retrieved": [{"source": "rules-2026.pdf", "section": "1.21 Attendance"}],
    }
    assert citation_validity(response) == 0.0


def test_citation_validity_no_citations() -> None:
    assert citation_validity({"citations": [], "retrieved": []}) is None


def test_summarise_basic() -> None:
    rows = [
        {
            "id": "q1",
            "type": "ordinary",
            "must_abstain": False,
            "relevant": [{"source": "rules.pdf", "section": "sec1"}],
            "response": {
                "citations": [{"source": "rules.pdf", "section": "sec1"}],
                "retrieved": [{"source": "rules.pdf", "section": "sec1"}],
                "answered_from_context": True,
            },
        },
        {
            "id": "q2",
            "type": "abstain",
            "must_abstain": True,
            "relevant": [],
            "response": {
                "citations": [],
                "retrieved": [],
                "answered_from_context": False,
            },
        },
    ]
    res = summarise(rows)
    assert res["recall_at_3"] == 1.0
    assert res["recall_at_5"] == 1.0
    assert res["citation_validity"] == 1.0
    assert res["abstention_correctness"] == 1.0
    assert res["recall_at_5_by_type"]["ordinary"] == 1.0


def test_gate_pass_no_regression() -> None:
    results = {
        "recall_at_5": 0.90,
        "citation_validity": 0.96,
        "abstention_correctness": 1.0,
    }
    baseline = {"recall_at_5": 0.90}
    failures = gate(results, baseline)
    assert failures == []


def test_gate_pass_4_percentage_point_drop() -> None:
    results = {
        "recall_at_5": 0.86,
        "citation_validity": 0.96,
        "abstention_correctness": 1.0,
    }
    baseline = {"recall_at_5": 0.90}
    failures = gate(results, baseline)
    assert failures == []


def test_gate_pass_exact_5_percentage_point_drop() -> None:
    results = {
        "recall_at_5": 0.85,
        "citation_validity": 0.96,
        "abstention_correctness": 1.0,
    }
    baseline = {"recall_at_5": 0.90}
    failures = gate(results, baseline)
    assert failures == []


def test_gate_fail_greater_than_5_percentage_point_drop() -> None:
    results = {
        "recall_at_5": 0.849,
        "citation_validity": 0.96,
        "abstention_correctness": 1.0,
    }
    baseline = {"recall_at_5": 0.90}
    failures = gate(results, baseline)
    assert len(failures) == 1
    assert "dropped by" in failures[0]


def test_gate_fail_missing_or_malformed_baseline() -> None:
    results = {
        "recall_at_5": 0.90,
        "citation_validity": 0.96,
        "abstention_correctness": 1.0,
    }
    # Missing baseline (None)
    f_none = gate(results, None)
    assert any("missing or malformed baseline" in f for f in f_none)

    # Malformed baseline (wrong keys)
    f_bad = gate(results, {"wrong_key": 0.90})
    assert any("missing or malformed baseline" in f for f in f_bad)


def test_gate_fail_abstain() -> None:
    results = {
        "recall_at_5": 0.90,
        "citation_validity": 0.98,
        "abstention_correctness": 0.5,
    }
    baseline = {"recall_at_5": 0.90}
    failures = gate(results, baseline)
    assert any("must-abstain" in f for f in failures)


def test_gate_fail_citation() -> None:
    results = {
        "recall_at_5": 0.90,
        "citation_validity": 0.90,
        "abstention_correctness": 1.0,
    }
    baseline = {"recall_at_5": 0.90}
    failures = gate(results, baseline)
    assert any("citation validity" in f for f in failures)

