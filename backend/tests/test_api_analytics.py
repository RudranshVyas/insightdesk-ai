from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.api import deps
from backend.app.main import app

PREFIX = "/api/v1"


def make_df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "ticket_id": [f"T{i:04d}" for i in range(n)],
            "created_at": pd.date_range("2023-01-01", periods=n, freq="6h"),
            "product_area": rng.choice(["Payments", "Auth"], n),
            "issue_type": rng.choice(["Billing inquiry", "Technical issue"], n),
            "priority": rng.choice(["Low", "High"], n),
            "status": ["resolved"] * n,
            "channel": rng.choice(["Email", "Chat"], n),
            "issue_subject": ["subject"] * n,
            "is_resolved": [True] * n,
            "resolution_time_hours": rng.gamma(2, 5, n),
            "response_time_hours": rng.gamma(1, 2, n),
            "csat_score": rng.integers(1, 6, n).astype(float),
            "escalated": rng.random(n) < 0.2,
        }
    )


def caps_with(metrics: list[str], **overrides) -> dict:
    caps = {
        "analytics": {
            "enabled": True,
            "reason": None,
            "available_metrics": metrics,
            "unavailable_metrics": {"sla_breach_rate": "no SLA outcome column exists"},
        },
        "retrieval": {"enabled": False, "reason": "not built in this test"},
        "resolution_generation": {"enabled": False, "reason": "not built in this test"},
        "clustering": {"enabled": False, "reason": "not built in this test"},
        "risk": {"enabled": False, "reason": "not built in this test"},
    }
    caps.update(overrides)
    return caps


@pytest.fixture
def client():
    df = make_df()
    app.dependency_overrides[deps.get_tickets] = lambda: df
    app.dependency_overrides[deps.get_caps] = lambda: caps_with(
        ["ticket_volume", "resolution_time", "response_time", "escalation_rate",
         "csat", "timeseries"]
    )
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client) -> None:
    assert client.get(f"{PREFIX}/health").json() == {"status": "ok"}


def test_capabilities_endpoint_is_always_reachable(client) -> None:
    r = client.get(f"{PREFIX}/capabilities")
    assert r.status_code == 200
    assert "retrieval" in r.json()


def test_overview_reports_denominators(client) -> None:
    r = client.get(f"{PREFIX}/analytics/overview")
    assert r.status_code == 200
    body = r.json()
    for m in body["metrics"].values():
        assert "denominator" in m and "definition" in m
    assert "sla_breach_rate" not in body["metrics"]
    assert "sla_breach_rate" in body["unavailable_metrics"]


def test_overview_applies_filters(client) -> None:
    r = client.get(f"{PREFIX}/analytics/overview", params={"product_area": "Payments"})
    body = r.json()
    assert body["filters"]["applied"]["product_area"] == ["Payments"]
    assert body["metrics"]["ticket_volume"]["value"] < 300


def test_timeseries(client) -> None:
    r = client.get(f"{PREFIX}/analytics/timeseries", params={"freq": "W"})
    assert r.status_code == 200
    points = r.json()["points"]
    assert points and "ticket_volume" in points[0]
    assert "csat_denominator" in points[0]


def test_product_pain_points_and_issue_types(client) -> None:
    for path in ("product-pain-points", "issue-types"):
        r = client.get(f"{PREFIX}/analytics/{path}")
        assert r.status_code == 200, path
        body = r.json()
        assert body["rows"]
        assert "min_sample_threshold" in body


def test_ticket_list_is_paginated(client) -> None:
    r = client.get(f"{PREFIX}/analytics/tickets", params={"page": 2, "page_size": 10})
    body = r.json()
    assert body["page"] == 2 and len(body["items"]) == 10
    assert body["total"] == 300


def test_ticket_list_page_size_is_capped(client) -> None:
    assert client.get(f"{PREFIX}/analytics/tickets", params={"page_size": 5000}).status_code == 422


# --- capability gating -------------------------------------------------------


def test_disabled_analytics_returns_structured_payload_not_empty_metrics() -> None:
    df = make_df()
    app.dependency_overrides[deps.get_tickets] = lambda: df
    app.dependency_overrides[deps.get_caps] = lambda: {
        "analytics": {"enabled": False, "reason": "no usable ticket table"}
    }
    with TestClient(app) as c:
        r = c.get(f"{PREFIX}/analytics/overview")
    app.dependency_overrides.clear()

    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["enabled"] is False
    assert detail["reason"] == "no usable ticket table"
    assert "metrics" not in detail  # no fabricated zeros


def test_disabled_timeseries_metric_is_refused_with_its_reason() -> None:
    df = make_df()
    app.dependency_overrides[deps.get_tickets] = lambda: df
    caps = caps_with(["ticket_volume"])
    caps["analytics"]["unavailable_metrics"]["timeseries"] = "created_at is unavailable"
    app.dependency_overrides[deps.get_caps] = lambda: caps
    with TestClient(app) as c:
        r = c.get(f"{PREFIX}/analytics/timeseries")
    app.dependency_overrides.clear()

    assert r.status_code == 409
    assert r.json()["detail"]["reason"] == "created_at is unavailable"


def test_missing_dimension_is_refused_with_a_reason() -> None:
    df = make_df().drop(columns=["product_area"])
    app.dependency_overrides[deps.get_tickets] = lambda: df
    app.dependency_overrides[deps.get_caps] = lambda: caps_with(["ticket_volume"])
    with TestClient(app) as c:
        r = c.get(f"{PREFIX}/analytics/product-pain-points")
    app.dependency_overrides.clear()

    assert r.status_code == 409
    assert "unavailable" in r.json()["detail"]["reason"]
