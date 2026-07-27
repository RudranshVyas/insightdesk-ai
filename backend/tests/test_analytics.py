from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.services import analytics as A
from backend.app.services import ticket_store

ALL = {"resolution_time", "response_time", "escalation_rate", "sla_breach_rate", "csat"}


@pytest.fixture
def df() -> pd.DataFrame:
    """Six tickets, hand-checkable.

    ids 1-3 resolved with durations 2, 4, 6 -> mean 4.0
    id 4 resolved but duration null (bad timestamp)
    id 5 open, no duration            -> must NOT count as a zero
    id 6 resolved, duration 0         -> not a positive duration, excluded
    csat: 5, 3, null, 4, null, null   -> mean 4.0 over 3 responses
    escalated: T, F, null, F, F, T    -> 2/5 known
    sla_breached: T, F, null, null, null, null -> 1/2 eligible
    """
    return pd.DataFrame(
        {
            "ticket_id": [f"T{i}" for i in range(1, 7)],
            "created_at": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-03",
                 "2023-01-04", "2023-01-05", "2023-01-06"]
            ),
            "product_area": ["Payments", "Payments", "Auth", "Auth", "Auth", "Payments"],
            "issue_type": ["Billing"] * 3 + ["Technical"] * 3,
            "status": ["resolved"] * 4 + ["open", "resolved"],
            "is_resolved": [True, True, True, True, False, True],
            "resolution_time_hours": [2.0, 4.0, 6.0, np.nan, np.nan, 0.0],
            "response_time_hours": [1.0, 2.0, np.nan, np.nan, np.nan, 3.0],
            "csat_score": [5.0, 3.0, np.nan, 4.0, np.nan, np.nan],
            "escalated": [True, False, None, False, False, True],
            "sla_breached": [True, False, None, None, None, None],
        }
    )


# --- denominators ------------------------------------------------------------


def test_open_tickets_are_excluded_not_counted_as_zero(df) -> None:
    m = A.compute_metric_block(df, ALL)["resolution_time_hours"]
    assert m["value"] == 4.0
    assert m["denominator"] == 3  # not 6, and the open ticket is not a zero


def test_zero_duration_is_not_a_valid_resolution_time(df) -> None:
    timed = A.get_timed_tickets(df)
    assert "T6" not in set(timed["ticket_id"])


def test_response_time_denominator_counts_only_responded(df) -> None:
    m = A.compute_metric_block(df, ALL)["response_time_hours"]
    assert m["denominator"] == 3
    assert m["value"] == 2.0


def test_escalation_denominator_excludes_unknown_outcomes(df) -> None:
    m = A.compute_metric_block(df, ALL)["escalation_rate"]
    assert m["denominator"] == 5
    assert m["value"] == 0.4


def test_sla_denominator_is_sla_eligible_only(df) -> None:
    m = A.compute_metric_block(df, ALL)["sla_breach_rate"]
    assert m["denominator"] == 2
    assert m["value"] == 0.5


def test_csat_uses_rated_tickets_and_reports_response_count(df) -> None:
    m = A.compute_metric_block(df, ALL)["csat"]
    assert m["value"] == 4.0
    assert m["denominator"] == 3
    assert m["response_count"] == 3
    assert m["response_rate"] == 0.5


def test_every_metric_carries_a_denominator_and_definition(df) -> None:
    for name, m in A.compute_metric_block(df, ALL).items():
        assert "denominator" in m, name
        assert m["definition"], name


def test_empty_denominator_yields_null_not_zero(df) -> None:
    empty = df.iloc[0:0]
    m = A.compute_metric_block(empty, ALL)
    assert m["csat"]["value"] is None
    assert m["csat"]["denominator"] == 0
    assert "note" in m["csat"]


def test_disallowed_metric_is_absent_entirely(df) -> None:
    m = A.compute_metric_block(df, {"resolution_time"})
    assert "csat" not in m
    assert "escalation_rate" not in m
    assert "resolution_time_hours" in m


# --- CSAT semantics are applied identically everywhere -----------------------


def test_csat_helper_is_the_single_source_of_truth(df) -> None:
    """Overview, per-area, and trend CSAT must all come from get_rated_tickets."""
    rated_ids = set(A.get_rated_tickets(df)["ticket_id"])

    overall = A.compute_metric_block(df, ALL)["csat"]
    assert overall["denominator"] == len(rated_ids)

    per_area = A.by_dimension(df, "product_area", ALL)
    area_total = sum(r["metrics"]["csat"]["denominator"] for r in per_area["rows"])
    assert area_total == len(rated_ids)

    ts = A.timeseries(df, ALL, freq="D")
    ts_total = sum(p["csat_denominator"] for p in ts["points"])
    assert ts_total == len(rated_ids)


def test_csat_zero_is_not_reinvented_by_the_helper() -> None:
    """Ingestion decides whether 0 means "no response". Analytics must not
    second-guess it: a 0 that survived ingestion is a real score here."""
    df = pd.DataFrame({"ticket_id": ["A"], "csat_score": [0.0]})
    assert len(A.get_rated_tickets(df)) == 1


# --- rankings ----------------------------------------------------------------


def test_small_categories_are_flagged(df) -> None:
    out = A.by_dimension(df, "product_area", ALL)
    assert out["warning"]
    assert set(out["small_sample_categories"]) == {"Payments", "Auth"}


def test_rows_with_a_null_dimension_are_reported(df) -> None:
    d = df.copy()
    d.loc[0, "product_area"] = None
    out = A.by_dimension(d, "product_area", ALL)
    assert out["excluded_rows_with_null_dimension"] == 1


def test_unavailable_dimension_raises(df) -> None:
    with pytest.raises(ValueError, match="unavailable"):
        A.by_dimension(df, "region", ALL)


def test_timeseries_requires_created_at(df) -> None:
    d = df.copy()
    d["created_at"] = pd.NaT
    with pytest.raises(ValueError, match="created_at"):
        A.timeseries(d, ALL)


# --- filters and pagination ---------------------------------------------------


def test_filters_report_what_was_applied(df) -> None:
    sub, info = ticket_store.apply_filters(df, product_area="Payments")
    assert len(sub) == 3
    assert info["applied"]["product_area"] == ["Payments"]


def test_date_filter_on_a_dataset_without_created_at_is_reported_not_silent(df) -> None:
    d = df.copy()
    d["created_at"] = pd.NaT
    sub, info = ticket_store.apply_filters(d, date_from="2023-01-01")
    assert len(sub) == len(d)
    assert "date_range" in info["ignored"]


def test_filter_on_an_absent_field_is_reported(df) -> None:
    _, info = ticket_store.apply_filters(df, region="EMEA")
    assert "region" in info["ignored"]


def test_pagination(df) -> None:
    page = A.paginate(df, page=2, page_size=4, columns=["ticket_id"])
    assert page["total"] == 6
    assert page["total_pages"] == 2
    assert len(page["items"]) == 2
    assert list(page["items"][0]) == ["ticket_id"]
