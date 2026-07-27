"""Phase 2 (spec v2) — provider, agent, ops, and served-vs-audited disclosure.

Checkpoint 2 is the point of this file: flipping a capability off must visibly
disable it, and a disabled capability must never be represented by a zero.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from backend.app.core.capability_loader import (
    CapabilityDisabled,
    load_capabilities,
    require,
)
from backend.app.core.config import Settings
from backend.app.services import capabilities as cap


def _settings(**kwargs) -> Settings:
    # `_env_file=None` keeps a developer's local .env from deciding test outcomes.
    return Settings(_env_file=None, **kwargs)


# The committed fixture is 30 rows — deliberately far below the production
# minimums. These thresholds keep the fixture above the bar so the tests exercise
# the ENABLED branches; the production defaults are asserted elsewhere.
FIXTURE_THRESHOLDS = cap.Thresholds(
    retrieval_min_source_cases=5,
    generation_min_source_cases=5,
    clustering_min_texts=5,
    risk_min_rows=10,
    risk_min_positives=2,
    risk_min_negatives=2,
)


@pytest.fixture
def audit(canonical_df: pd.DataFrame, adapter_report) -> dict:
    from backend.app.services import audit as audit_mod

    return audit_mod.build_audit(
        canonical_df, adapter_report.to_dict(), None, {"sha256": "deadbeef"}
    )


def _caps(canonical_df, audit, **settings_kwargs) -> dict:
    return cap.build_capabilities(
        canonical_df, audit, FIXTURE_THRESHOLDS, settings=_settings(**settings_kwargs)
    )


# --- llm_provider -------------------------------------------------------------


def test_provider_none_is_disabled_with_a_reason() -> None:
    block = cap.describe_llm_provider(_settings(llm_provider="none"))
    assert block["enabled"] is False
    assert "deterministic" in block["reason"]
    assert block["provider"] == "none"


def test_configured_provider_without_a_key_is_disabled() -> None:
    block = cap.describe_llm_provider(_settings(llm_provider="anthropic", llm_api_key=None))
    assert block["enabled"] is False
    assert "no API key" in block["reason"]
    assert block["provider"] == "anthropic"


def test_unknown_provider_is_disabled_not_assumed_to_work() -> None:
    block = cap.describe_llm_provider(
        _settings(llm_provider="some-other-vendor", llm_api_key="k")
    )
    assert block["enabled"] is False
    assert "not implemented" in block["reason"]


def test_configured_provider_with_a_key_is_enabled() -> None:
    block = cap.describe_llm_provider(
        _settings(llm_provider="anthropic", llm_api_key="test-key")
    )
    assert block["enabled"] is True
    assert block["reason"] is None
    assert block["model"]
    # The key itself must never appear in the manifest, which is a public artifact.
    assert "test-key" not in json.dumps(block)


# --- ai_ops -------------------------------------------------------------------


def test_ai_ops_disabled_when_otel_is_off() -> None:
    block = cap.describe_ai_ops(_settings(otel_enabled=False))
    assert block["enabled"] is False
    assert block["exporter"] is None


def test_ai_ops_enabled_reports_the_exporter() -> None:
    block = cap.describe_ai_ops(_settings(otel_enabled=True))
    assert block["enabled"] is True
    assert block["exporter"] == "console"


# --- generation modes ---------------------------------------------------------


def test_deterministic_mode_is_available_without_a_provider(canonical_df, audit) -> None:
    modes = _caps(canonical_df, audit, llm_provider="none")["resolution_generation"][
        "available_modes"
    ]
    assert "deterministic" in modes
    assert "evidence_only" in modes
    assert "llm" not in modes


def test_llm_mode_appears_only_with_a_reachable_provider(canonical_df, audit) -> None:
    caps = _caps(canonical_df, audit, llm_provider="anthropic", llm_api_key="k")
    assert "llm" in caps["resolution_generation"]["available_modes"]


def test_disabled_generation_still_reports_a_modes_list(canonical_df, audit) -> None:
    """A disabled capability returns a structured payload, not a missing key."""
    caps = cap.build_capabilities(
        canonical_df, audit, cap.Thresholds(), settings=_settings()
    )
    block = caps["resolution_generation"]
    assert block["enabled"] is False, "30-row fixture is below the production minimum"
    assert block["available_modes"] == []


# --- analyst agent ------------------------------------------------------------


def test_analyst_agent_disabled_without_a_provider(canonical_df, audit) -> None:
    caps = _caps(canonical_df, audit, llm_provider="none")
    assert caps["analyst_agent"]["enabled"] is False
    assert "provider" in caps["analyst_agent"]["reason"]


def test_analyst_agent_registers_only_enabled_subsystems(canonical_df, audit) -> None:
    caps = _caps(canonical_df, audit, llm_provider="anthropic", llm_api_key="k")
    agent = caps["analyst_agent"]
    assert agent["enabled"] is True

    tools = set(agent["registered_tools"])
    assert "find_similar_cases" in tools
    if not caps["risk"]["enabled"]:
        assert "score_escalation_risk" not in tools
    if not caps["clustering"]["enabled"]:
        assert "list_clusters" not in tools


# --- served vs audited corpus -------------------------------------------------


def test_no_sampling_means_served_equals_audited(canonical_df, audit) -> None:
    r = _caps(canonical_df, audit)["retrieval"]
    assert r["enabled"] is True
    assert r["corpus_size_served"] == r["corpus_size_audited"]
    assert r["note"] is None


def test_sampling_discloses_both_numbers(canonical_df, audit) -> None:
    r = _caps(canonical_df, audit, corpus_serve_limit=5)["retrieval"]
    assert r["corpus_size_served"] == 5
    assert r["corpus_size_audited"] > 5
    assert "audited" in r["note"]


def test_a_limit_above_the_corpus_is_not_treated_as_sampling(canonical_df, audit) -> None:
    r = _caps(canonical_df, audit, corpus_serve_limit=10_000_000)["retrieval"]
    assert r["corpus_size_served"] == r["corpus_size_audited"]
    assert r["note"] is None


# --- evaluation status --------------------------------------------------------


def test_unlabeled_evaluation_set_is_reported_not_faked(tmp_path) -> None:
    s = _settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    assert cap.retrieval_evaluation_status(s) == "manual_set_not_yet_labeled"


def test_labeled_but_unevaluated_is_distinguished(tmp_path) -> None:
    s = _settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    s.ensure_dirs()
    (s.evaluation_dir / "retrieval_queries.jsonl").write_text('{"q": "x"}\n', encoding="utf-8")
    assert cap.retrieval_evaluation_status(s) == "manual_set_labeled_evaluation_not_run"

    (s.retrieval_dir / "evaluation.json").write_text("{}", encoding="utf-8")
    assert cap.retrieval_evaluation_status(s) == "evaluated"


def test_empty_labeled_file_counts_as_unlabeled(tmp_path) -> None:
    s = _settings(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
    s.ensure_dirs()
    (s.evaluation_dir / "retrieval_queries.jsonl").write_text("", encoding="utf-8")
    assert cap.retrieval_evaluation_status(s) == "manual_set_not_yet_labeled"


# --- checkpoint 2: gating actually gates --------------------------------------


def test_loader_knows_every_v2_subsystem(canonical_df, audit, tmp_path) -> None:
    caps = _caps(canonical_df, audit)
    path = tmp_path / "capabilities.json"
    cap.write_capabilities(caps, path)
    loaded = load_capabilities(path)
    for subsystem in (
        "analytics",
        "retrieval",
        "resolution_generation",
        "analyst_agent",
        "clustering",
        "risk",
        "llm_provider",
        "ai_ops",
    ):
        assert subsystem in loaded


def test_disabled_subsystem_raises_with_its_reason(canonical_df, audit, tmp_path) -> None:
    caps = _caps(canonical_df, audit, llm_provider="none")
    path = tmp_path / "capabilities.json"
    cap.write_capabilities(caps, path)
    loaded = load_capabilities(path)

    with pytest.raises(CapabilityDisabled) as exc:
        require("analyst_agent", loaded)
    payload = exc.value.payload()
    assert payload["enabled"] is False
    assert payload["reason"]
    # Never a zero standing in for a measurement that did not happen.
    assert payload.get("value") is None


def test_missing_manifest_disables_everything_rather_than_guessing(tmp_path) -> None:
    loaded = load_capabilities(tmp_path / "does_not_exist.json")
    assert loaded["available"] is False
    for subsystem in ("retrieval", "analyst_agent", "llm_provider", "ai_ops"):
        assert loaded[subsystem]["enabled"] is False
        assert "capabilities.json" in loaded[subsystem]["reason"]
