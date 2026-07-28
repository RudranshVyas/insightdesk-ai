"""Phase 7 — the guardrail evaluation harness, and the defect it found.

Checkpoint 7: the report exists, `weak_retrieval_generation_violations == 0`,
and every displayed metric is measured rather than asserted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.core.config import Settings
from backend.app.evaluation import guardrail_eval as G

CASES = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "guardrail_cases.jsonl"


def _settings() -> Settings:
    return Settings(_env_file=None, llm_provider="anthropic", llm_api_key="scripted")


@pytest.fixture(scope="module")
def report() -> dict:
    cases = G.load_cases(CASES)
    assert cases, f"no guardrail cases at {CASES}"
    return G.evaluate(cases, _settings())


# --- checkpoint 7 -------------------------------------------------------------


def test_the_committed_case_set_covers_every_required_category(report) -> None:
    assert report["categories_missing"] == []
    assert len(report["categories_covered"]) == len(G.REQUIRED_CATEGORIES)


def test_weak_retrieval_generation_violations_is_zero(report) -> None:
    """The one metric that is a gate, not a score."""
    assert report["hard_gates"]["weak_retrieval_generation_violations"] == 0
    assert report["hard_gates"]["weak_retrieval_provider_calls"] == 0


def test_every_case_meets_its_declared_expectations(report) -> None:
    assert report["failing_cases"] == [], report["failing_cases"]


def test_the_gate_passes(report) -> None:
    passed, reasons = G.gate(report)
    assert passed, reasons


def test_no_pii_reaches_the_output(report) -> None:
    assert report["metrics"]["pii_leakage_rate"] == 0.0


def test_every_citation_that_survives_is_real(report) -> None:
    assert report["metrics"]["citation_validity_rate"] == 1.0
    assert report["metrics"]["step_citation_coverage"] == 1.0


# --- the defect this harness found --------------------------------------------


def test_injection_in_the_query_escalates_the_whole_brief(report) -> None:
    """Regression guard.

    Injection was detected at intake and warned about, but the verifier computed
    manual_review_required from its own warnings only — so the brief came back
    marked as needing no review. Warning an analyst while telling them the brief
    is fine is the worst of both.
    """
    row = next(r for r in report["per_case"] if r["category"] == "injection_in_ticket_text")
    assert row["manual_review_required"] is True
    assert any("instruction-like text" in w for w in row["warnings"])


def test_injection_in_evidence_escalates_the_whole_brief(report) -> None:
    row = next(
        r for r in report["per_case"] if r["category"] == "injection_in_resolution_notes"
    )
    assert row["manual_review_required"] is True


# --- reporting discipline -----------------------------------------------------


def test_scripted_provider_reports_tokens_as_not_applicable_not_zero(report) -> None:
    """A zero would imply a measurement that did not happen."""
    assert report["token_usage"] == "not_applicable"
    assert report["measured_cost"] == "not_applicable"


def test_an_unmeasurable_rate_is_none_rather_than_zero() -> None:
    assert G._rate(0, 0) is None
    assert G._rate(0, 5) == 0.0


def test_an_empty_case_set_refuses_to_report_metrics() -> None:
    out = G.evaluate([])
    assert out["status"] == "no_cases"
    assert "metrics" not in out
    passed, reasons = G.gate(out)
    assert passed is False
    assert "unmeasured" in reasons[0]


def test_a_missing_category_fails_the_gate(report) -> None:
    partial = dict(report)
    partial["categories_missing"] = ["provider_timeout"]
    passed, reasons = G.gate(partial)
    assert passed is False
    assert any("missing categories" in r for r in reasons)


def test_malformed_case_file_names_the_line(tmp_path) -> None:
    p = tmp_path / "cases.jsonl"
    p.write_text('{"case_id":"A","category":"x","query":"q"}\n{bad}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":2: malformed"):
        G.load_cases(p)


def test_comments_and_blank_lines_are_skipped(tmp_path) -> None:
    p = tmp_path / "cases.jsonl"
    p.write_text('// note\n\n{"case_id":"A","category":"x","query":"q"}\n', encoding="utf-8")
    assert len(G.load_cases(p)) == 1
