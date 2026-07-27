"""Phase 6 API surface — routes, capability gating, and the trace endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.api import support_brief as route
from backend.app.main import app
from backend.tests.test_support_brief import FakeRetriever, _hit

PREFIX = "/api/v1"


@pytest.fixture
def client(monkeypatch):
    route.reset_retriever(FakeRetriever([_hit("T1"), _hit("T2", group=2, rank=2)]))
    monkeypatch.setattr(
        route, "load_capabilities", lambda: {"retrieval": {"enabled": True, "reason": None}}
    )
    with TestClient(app) as c:
        yield c
    route.reset_retriever(None)


def test_post_returns_a_brief(client) -> None:
    r = client.post(f"{PREFIX}/support-brief", json={"issue_description": "Charged twice"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "deterministic"
    assert body["similar_cases"]
    assert body["request_id"].startswith("req_")
    assert "Human review required" in body["disclaimer"]


def test_blank_description_is_rejected(client) -> None:
    r = client.post(f"{PREFIX}/support-brief", json={"issue_description": "   "})
    assert r.status_code == 422


def test_top_k_is_bounded(client) -> None:
    r = client.post(
        f"{PREFIX}/support-brief", json={"issue_description": "x", "top_k": 500}
    )
    assert r.status_code == 422


def test_disabled_retrieval_returns_a_structured_reason(client, monkeypatch) -> None:
    monkeypatch.setattr(
        route,
        "load_capabilities",
        lambda: {"retrieval": {"enabled": False, "reason": "corpus too small"}},
    )
    r = client.post(f"{PREFIX}/support-brief", json={"issue_description": "x"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "disabled"
    assert body["reason"] == "corpus too small"
    # No fabricated payload standing in for a capability that is off.
    assert "suggested_steps" not in body
    assert "similar_cases" not in body


def test_trace_is_retrievable_and_carries_no_sensitive_content(client) -> None:
    r = client.post(
        f"{PREFIX}/support-brief",
        json={"issue_description": "Charged twice in Reykjavik"},
    )
    request_id = r.json()["request_id"]

    t = client.get(f"{PREFIX}/support-brief/{request_id}/trace")
    assert t.status_code == 200
    trace = t.json()
    assert trace["request_id"] == request_id
    assert trace["provider_calls"] == 0
    assert [s["name"] for s in trace["stage_trace"]][0] == "intake_and_redact"
    assert "Reykjavik" not in str(trace)
    assert "never recorded" in trace["note"]


def test_unknown_trace_id_is_404(client) -> None:
    assert client.get(f"{PREFIX}/support-brief/req_nope/trace").status_code == 404


def test_capabilities_endpoint_lists_the_v2_subsystems(client) -> None:
    r = client.get(f"{PREFIX}/capabilities")
    assert r.status_code == 200
    assert "retrieval" in r.json()
