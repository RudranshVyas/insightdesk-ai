from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backend.app.core import capability_loader as loader
from backend.app.services import capabilities as cap


def make_df(
    n: int = 800,
    with_notes: bool = True,
    with_escalated: bool = True,
    escalation_rate: float = 0.2,
    with_status: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    areas = ["Payments", "Auth", "Reporting", "Integrations"]
    df = pd.DataFrame(
        {
            "ticket_id": [f"T{i:05d}" for i in range(n)],
            "created_at": pd.date_range("2023-01-01", periods=n, freq="3h"),
            "product_area": rng.choice(areas, n),
            "issue_type": rng.choice(["Technical issue", "Billing inquiry"], n),
            "issue_text": [f"issue number {i} about {rng.choice(areas)} failing" for i in range(n)],
            "resolution_notes": (
                [f"resolved by applying fix {i % 40}" for i in range(n)]
                if with_notes
                else [""] * n
            ),
            "resolution_time_hours": rng.gamma(2.0, 8.0, n),
            "response_time_hours": rng.gamma(1.0, 2.0, n),
            "csat_score": rng.integers(1, 6, n).astype(float),
            "sla_breached": [None] * n,
            "is_resolved": [True] * n if with_status else [None] * n,
        }
    )
    df["escalated"] = (
        rng.random(n) < escalation_rate if with_escalated else pd.Series([None] * n)
    )
    return df


def make_audit(df: pd.DataFrame, **overrides) -> dict:
    esc = df["escalated"]
    # Elementwise on a nullable-boolean Series — see the note in audit.py.
    pos = int((esc == True).sum()) if esc.notna().any() else 0  # noqa: E712
    neg = int((esc == False).sum()) if esc.notna().any() else 0  # noqa: E712
    audit = {
        "source_file": {"sha256": "deadbeef"},
        "text": {"issue_text": {"normalized_unique_ratio": 0.9}},
        "timestamps": {"temporal_split_feasible": True},
        "outcomes": {
            "escalated": (
                {
                    "available": True,
                    "kind": "binary",
                    "definition": "canonical boolean field 'escalated' as mapped",
                    "positive_count": pos,
                    "negative_count": neg,
                    "both_classes_present": pos > 0 and neg > 0,
                    "prevalence": round(pos / (pos + neg), 4) if pos + neg else None,
                    "deterministic_from_status": False,
                }
                if pos or neg
                else {"available": False, "reason": "column absent"}
            ),
            "sla_breached": {"available": False, "reason": "column absent"},
            "resolution_time_hours": {
                "available": True,
                "kind": "numeric",
                "valid_count": int(df["resolution_time_hours"].notna().sum()),
            },
            "csat_score": {"available": True, "kind": "numeric"},
        },
    }
    audit.update(overrides)
    return audit


# --- gates -------------------------------------------------------------------


def test_full_dataset_enables_everything() -> None:
    df = make_df()
    caps = cap.build_capabilities(df, make_audit(df))
    for s in ("analytics", "retrieval", "resolution_generation", "clustering", "risk"):
        assert caps[s]["enabled"], (s, caps[s]["reason"])
    assert caps["risk"]["target"] == "escalated"
    assert caps["risk"]["target_kind"] == "real"


def test_small_corpus_disables_retrieval_with_a_reason() -> None:
    df = make_df(n=20)
    caps = cap.build_capabilities(df, make_audit(df))
    assert caps["retrieval"]["enabled"] is False
    assert "minimum" in caps["retrieval"]["reason"]
    # generation is downstream of retrieval and must go with it
    assert caps["resolution_generation"]["enabled"] is False
    assert "retrieval is disabled" in caps["resolution_generation"]["reason"]


def test_boilerplate_notes_do_not_count_as_usable() -> None:
    df = make_df()
    df["resolution_notes"] = "N/A"
    caps = cap.build_capabilities(df, make_audit(df))
    assert caps["resolution_generation"]["enabled"] is False
    assert caps["retrieval"]["enabled"] is False  # no source cases at all


def test_missing_status_relaxation_is_recorded() -> None:
    df = make_df(with_status=False)
    caps = cap.build_capabilities(df, make_audit(df))
    assert caps["retrieval"]["enabled"] is True
    assert caps["retrieval"]["resolved_status_used"] is False
    assert "relaxation" in caps["retrieval"]
    assert caps["retrieval"]["relaxation"]


def test_analytics_metric_is_dropped_when_its_column_is_absent() -> None:
    df = make_df()
    caps = cap.build_capabilities(df, make_audit(df))
    assert "sla_breach_rate" not in caps["analytics"]["available_metrics"]
    assert "sla_breach_rate" in caps["analytics"]["unavailable_metrics"]
    assert "csat" in caps["analytics"]["available_metrics"]


def test_no_created_at_disables_timeseries() -> None:
    df = make_df()
    df["created_at"] = pd.NaT
    caps = cap.build_capabilities(df, make_audit(df))
    assert "timeseries" not in caps["analytics"]["available_metrics"]
    assert "created_at" in caps["analytics"]["unavailable_metrics"]["timeseries"]


def test_low_text_variation_disables_clustering() -> None:
    df = make_df()
    audit = make_audit(df)
    audit["text"]["issue_text"]["normalized_unique_ratio"] = 0.01
    caps = cap.build_capabilities(df, audit)
    assert caps["clustering"]["enabled"] is False
    assert "variation" in caps["clustering"]["reason"]


# --- risk ladder --------------------------------------------------------------


def test_risk_falls_through_to_derived_target_when_no_real_label() -> None:
    df = make_df(with_escalated=False)
    caps = cap.build_capabilities(df, make_audit(df))
    assert caps["risk"]["enabled"] is True
    assert caps["risk"]["target"] == "long_resolution_risk"
    assert caps["risk"]["target_kind"] == "derived"
    assert "NOT an escalation model" in caps["risk"]["caveat"]


def test_risk_disabled_when_no_target_survives() -> None:
    df = make_df(with_escalated=False)
    df["resolution_time_hours"] = np.nan
    df["csat_score"] = np.nan
    audit = make_audit(df)
    audit["outcomes"]["resolution_time_hours"] = {"available": False, "reason": "all null"}
    audit["outcomes"]["csat_score"] = {"available": False, "reason": "all null"}
    caps = cap.build_capabilities(df, audit)
    assert caps["risk"]["enabled"] is False
    assert caps["risk"]["target"] is None
    assert caps["risk"]["target_kind"] is None
    assert caps["risk"]["ladder_attempts"]


def test_single_class_target_is_rejected() -> None:
    df = make_df(escalation_rate=0.0)
    caps = cap.build_capabilities(df, make_audit(df))
    assert caps["risk"]["target"] != "escalated"


def test_target_determined_by_status_is_rejected() -> None:
    df = make_df()
    audit = make_audit(df)
    audit["outcomes"]["escalated"]["deterministic_from_status"] = True
    caps = cap.build_capabilities(df, audit)
    assert caps["risk"]["target"] != "escalated"
    assert any(
        "determined by status" in a.get("rejected", "")
        for a in caps["risk"].get("ladder_attempts", [])
    )


def test_priority_is_not_a_default_t0_feature() -> None:
    assert "priority" not in cap.T0_CANDIDATE_FIELDS


# --- loader ------------------------------------------------------------------


def test_missing_manifest_reports_everything_disabled(tmp_path) -> None:
    caps = loader.load_capabilities(tmp_path / "nope.json")
    assert caps["available"] is False
    for s in loader.SUBSYSTEMS:
        assert caps[s]["enabled"] is False
        assert "capabilities.json" in caps[s]["reason"]


def test_require_raises_structured_error(tmp_path) -> None:
    path = tmp_path / "capabilities.json"
    caps = {"retrieval": {"enabled": False, "reason": "corpus too small"}}
    path.write_text(json.dumps(caps), encoding="utf-8")

    with pytest.raises(loader.CapabilityDisabled) as exc:
        loader.require("retrieval", loader.load_capabilities(path))

    payload = exc.value.payload()
    assert payload["capability"] == "retrieval"
    assert payload["enabled"] is False
    assert payload["reason"] == "corpus too small"


def test_flipping_a_capability_off_disables_it(tmp_path) -> None:
    """Checkpoint 2: editing the manifest by hand must switch the feature off."""
    path = tmp_path / "capabilities.json"
    path.write_text(json.dumps({"clustering": {"enabled": True, "reason": None}}), "utf-8")
    assert loader.is_enabled("clustering", loader.load_capabilities(path))

    path.write_text(
        json.dumps({"clustering": {"enabled": False, "reason": "manually disabled"}}), "utf-8"
    )
    caps = loader.load_capabilities(path)
    assert not loader.is_enabled("clustering", caps)
    with pytest.raises(loader.CapabilityDisabled):
        loader.require("clustering", caps)


def test_require_metric_gates_individual_metrics(tmp_path) -> None:
    path = tmp_path / "capabilities.json"
    path.write_text(
        json.dumps(
            {
                "analytics": {
                    "enabled": True,
                    "reason": None,
                    "available_metrics": ["ticket_volume"],
                    "unavailable_metrics": {"csat": "no rated tickets"},
                }
            }
        ),
        "utf-8",
    )
    caps = loader.load_capabilities(path)
    loader.require_metric("ticket_volume", caps)  # does not raise
    with pytest.raises(loader.CapabilityDisabled) as exc:
        loader.require_metric("csat", caps)
    assert exc.value.reason == "no rated tickets"
