"""The retrieval metric regression gate — a CI quality gate, so it needs its own tests."""

from __future__ import annotations

import json

from backend.app.evaluation import regression_gate as G

BASE = {
    "tolerances": {},
    "metrics": {"hybrid": {"hit_at_5": 0.80, "mrr_at_5": 0.60, "ndcg_at_5": 0.55}},
}


def _measured(**overrides) -> dict:
    metrics = {"hit_at_5": 0.80, "mrr_at_5": 0.60, "ndcg_at_5": 0.55}
    metrics.update(overrides)
    return {"hybrid": metrics}


def test_identical_metrics_pass() -> None:
    assert G.gate_result(G.compare(BASE, _measured()))["passed"] is True


def test_a_drop_inside_tolerance_passes() -> None:
    findings = G.compare(BASE, _measured(hit_at_5=0.77))  # -0.03, tolerance 0.05
    assert G.gate_result(findings)["passed"] is True


def test_a_drop_beyond_tolerance_fails() -> None:
    findings = G.compare(BASE, _measured(hit_at_5=0.70))  # -0.10
    result = G.gate_result(findings)
    assert result["passed"] is False
    assert any("hit_at_5" in f for f in result["failures"])
    assert "0.8000 -> 0.7000" in result["failures"][0]


def test_an_improvement_never_fails_the_build() -> None:
    result = G.gate_result(G.compare(BASE, _measured(hit_at_5=0.95)))
    assert result["passed"] is True
    assert result["improvements"], "an improvement should still be surfaced for review"


def test_a_metric_that_stopped_being_measured_is_a_hard_failure() -> None:
    """None is worse than a bad score: the measurement did not happen."""
    result = G.gate_result(G.compare(BASE, _measured(hit_at_5=None)))
    assert result["passed"] is False
    assert "nothing was measured" in result["failures"][0]


def test_a_vanished_config_fails_every_one_of_its_metrics() -> None:
    result = G.gate_result(G.compare(BASE, {"bm25": {"hit_at_5": 0.9}}))
    assert result["passed"] is False
    assert len(result["failures"]) == 3


def test_a_newly_measurable_metric_is_reported_not_failed() -> None:
    base = {"metrics": {"hybrid": {"hit_at_5": None}}}
    result = G.gate_result(G.compare(base, _measured(hit_at_5=0.4)))
    assert result["passed"] is True
    assert result["new_metrics"]


def test_baseline_tolerances_override_the_defaults() -> None:
    strict = {"tolerances": {"hit_at_5": 0.001}, "metrics": BASE["metrics"]}
    assert G.gate_result(G.compare(strict, _measured(hit_at_5=0.79)))["passed"] is False

    loose = {"tolerances": {"hit_at_5": 0.5}, "metrics": BASE["metrics"]}
    assert G.gate_result(G.compare(loose, _measured(hit_at_5=0.40)))["passed"] is True


def test_metrics_absent_from_the_baseline_are_ignored() -> None:
    """A newly added metric cannot retroactively fail an old baseline."""
    measured = _measured()
    measured["hybrid"]["brand_new_metric"] = 0.0
    assert G.gate_result(G.compare(BASE, measured))["passed"] is True


# --- extraction ---------------------------------------------------------------


def test_extract_skips_unavailable_tier1_configs() -> None:
    report = {
        "tier1_leave_one_out": {
            "hybrid": {"available": True, "metrics": {"hit_at_5": 0.8}},
            "dense": {"available": False, "reason": "no issue_type"},
        }
    }
    assert set(G.extract_metrics(report)) == {"hybrid"}


def test_extract_returns_nothing_when_tier2_is_unlabeled() -> None:
    report = {"tier2_manual_labeled": {"status": "not_yet_labeled", "results": {}}}
    assert G.extract_metrics(report, tier="tier2_manual_labeled") == {}


def test_extract_reads_tier2_when_it_is_evaluated() -> None:
    report = {
        "tier2_manual_labeled": {
            "status": "evaluated",
            "results": {"bm25": {"metrics": {"hit_at_5": 0.7}}},
        }
    }
    got = G.extract_metrics(report, tier="tier2_manual_labeled")
    assert got == {"bm25": {"hit_at_5": 0.7}}


# --- round trip ---------------------------------------------------------------


def test_a_recorded_baseline_passes_against_itself(tmp_path) -> None:
    measured = _measured()
    path = tmp_path / "baseline.json"
    G.write_baseline(measured, path, tier="tier1_leave_one_out")
    loaded = G.load_baseline(path)
    assert G.gate_result(G.compare(loaded, measured))["passed"] is True


def test_baseline_records_the_embedding_model_that_produced_it(tmp_path) -> None:
    """Changing the model invalidates the numbers, so the model must be recorded."""
    path = tmp_path / "baseline.json"
    G.write_baseline(
        _measured(), path, tier="tier1_leave_one_out",
        index_manifest={"embedding_model": "all-MiniLM-L6-v2", "corpus_size": 4966},
    )
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["index"]["embedding_model"] == "all-MiniLM-L6-v2"
    assert written["versions"]["index"]
    assert "invalidates" in written["note"]


def test_missing_baseline_loads_as_none(tmp_path) -> None:
    assert G.load_baseline(tmp_path / "absent.json") is None
