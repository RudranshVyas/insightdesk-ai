"""Phase 2 — capability manifest.

Derived from the audit, never from optimism. This file is the single source of
truth for what the rest of the app is allowed to do. Every subsystem, API route,
and frontend page is gated on it.

A disabled capability always carries a human-readable `reason`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.core import canonical as C
from backend.app.core.config import Settings, get_settings
from backend.app.core.versions import MANIFEST_VERSION
from backend.app.services import resolution_quality as RQ

# Providers this build knows how to talk to. Anything else is treated as
# "none" rather than assumed to work.
SUPPORTED_LLM_PROVIDERS: frozenset[str] = frozenset({"none", "anthropic"})


@dataclass(frozen=True)
class Thresholds:
    """Minimum data volumes below which a subsystem is not worth claiming.

    These are judgement calls, not measurements. They are recorded in the
    manifest so a reader can see exactly what bar was applied.
    """

    retrieval_min_source_cases: int = 50
    generation_min_source_cases: int = 50
    # A column named `resolution_notes` is a promise the data may not keep. Many
    # exports put the first agent reply there, and a first reply usually asks for
    # information rather than recording a fix. Below this measured rate the
    # column is correspondence, and presenting it as "what resolved the ticket"
    # would be a claim the data does not support.
    generation_min_action_rate: float = 0.05
    clustering_min_texts: int = 200
    clustering_min_normalized_unique_ratio: float = 0.05
    risk_min_rows: int = 500
    risk_min_positives: int = 50
    risk_min_negatives: int = 50


# Features that could plausibly be known at ticket creation. `priority` is
# deliberately absent: in most support workflows an agent sets it during triage,
# i.e. after T0. Phase 8 may re-admit it only as a labelled triage-time variant.
T0_CANDIDATE_FIELDS: tuple[str, ...] = (
    "issue_text",
    "product_area",
    "issue_type",
    "customer_segment",
    "channel",
    "platform",
    "region",
    "sla_plan",
)


def _disabled(reason: str, **extra: Any) -> dict[str, Any]:
    return {"enabled": False, "reason": reason, **extra}


def _enabled(**extra: Any) -> dict[str, Any]:
    return {"enabled": True, "reason": None, **extra}


# --- corpus helpers (shared with Phase 4 so the gate and the builder agree) ---


def usable_resolution_mask(df: pd.DataFrame) -> pd.Series:
    if "resolution_notes" not in df.columns:
        return pd.Series(False, index=df.index)
    notes = df["resolution_notes"].fillna("").astype(str).str.strip()
    return (notes != "") & ~notes.str.lower().isin(C.BOILERPLATE_RESOLUTIONS)


def source_case_mask(df: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    """Rows eligible as retrieval source cases, plus how the decision was made."""
    text_ok = df["issue_text"].fillna("").astype(str).str.strip() != ""
    notes_ok = usable_resolution_mask(df)

    info: dict[str, Any] = {
        "non_empty_issue_text": int(text_ok.sum()),
        "usable_resolution_notes": int(notes_ok.sum()),
    }

    has_resolved_flag = "is_resolved" in df.columns and df["is_resolved"].notna().any()
    if has_resolved_flag:
        resolved_ok = df["is_resolved"].fillna(False).astype(bool)
        info["resolved_status_used"] = True
        info["relaxation"] = None
    else:
        resolved_ok = pd.Series(True, index=df.index)
        info["resolved_status_used"] = False
        info["relaxation"] = (
            "No usable resolved/closed status was available. Tickets with usable "
            "resolution notes are accepted as source cases; this relaxation is "
            "recorded here and surfaced in diagnostics."
        )

    mask = text_ok & notes_ok & resolved_ok
    info["eligible_source_cases"] = int(mask.sum())
    return mask, info


# --- risk target ladder ------------------------------------------------------


def select_risk_target(
    df: pd.DataFrame, audit: dict[str, Any], th: Thresholds
) -> dict[str, Any]:
    """Phase 8 target ladder. Returns the chosen target or an explicit refusal.

    ``target_kind`` must be displayed everywhere the resulting numbers appear.
    """
    outcomes = audit.get("outcomes", {})
    attempts: list[dict[str, Any]] = []

    def _binary_ok(name: str) -> dict[str, Any] | None:
        e = outcomes.get(name, {})
        if not e.get("available"):
            attempts.append({"target": name, "rejected": e.get("reason", "unavailable")})
            return None
        pos, neg = e.get("positive_count", 0), e.get("negative_count", 0)
        if not e.get("both_classes_present"):
            attempts.append({"target": name, "rejected": "only one class present"})
            return None
        if pos < th.risk_min_positives or neg < th.risk_min_negatives:
            attempts.append(
                {
                    "target": name,
                    "rejected": (
                        f"too few examples: {pos} positive / {neg} negative "
                        f"(need {th.risk_min_positives}/{th.risk_min_negatives})"
                    ),
                }
            )
            return None
        if e.get("deterministic_from_status"):
            attempts.append(
                {
                    "target": name,
                    "rejected": (
                        "perfectly determined by status; the column is generated, "
                        "not observed"
                    ),
                }
            )
            return None
        return {
            "target": name,
            "target_kind": "real",
            "definition": e.get("definition", name),
            "positive_count": pos,
            "negative_count": neg,
            "prevalence": e.get("prevalence"),
        }

    for name in ("escalated", "sla_breached"):
        got = _binary_ok(name)
        if got:
            got["attempts"] = attempts
            return got

    # 3. derived long-resolution risk
    e = outcomes.get("resolution_time_hours", {})
    if e.get("available") and e.get("valid_count", 0) >= th.risk_min_rows:
        vals = pd.to_numeric(df.get("resolution_time_hours"), errors="coerce").dropna()
        if len(vals) and vals.nunique() > 2:
            p75 = float(vals.quantile(0.75))
            pos = int((vals > p75).sum())
            neg = int(len(vals) - pos)
            if pos >= th.risk_min_positives and neg >= th.risk_min_negatives:
                return {
                    "target": "long_resolution_risk",
                    "target_kind": "derived",
                    "definition": (
                        f"resolution_time_hours > p75 ({p75:.2f}h), computed over "
                        f"{len(vals)} tickets with a valid positive duration"
                    ),
                    "threshold_hours": p75,
                    "positive_count": pos,
                    "negative_count": neg,
                    "prevalence": round(pos / (pos + neg), 4),
                    "caveat": (
                        "This is NOT an escalation model and must never be described "
                        "as one. It predicts a threshold on resolution duration."
                    ),
                    "attempts": attempts,
                }
            attempts.append(
                {"target": "long_resolution_risk", "rejected": f"{pos}/{neg} class sizes too small"}
            )
        else:
            attempts.append(
                {"target": "long_resolution_risk", "rejected": "resolution_time_hours is degenerate"}
            )
    else:
        attempts.append(
            {
                "target": "long_resolution_risk",
                "rejected": e.get("reason", "insufficient valid durations"),
            }
        )

    # 4. derived low CSAT (a weaker, post-resolution construct)
    e = outcomes.get("csat_score", {})
    if e.get("available"):
        vals = pd.to_numeric(df.get("csat_score"), errors="coerce").dropna()
        pos = int((vals <= 2).sum())
        neg = int(len(vals) - pos)
        if pos >= th.risk_min_positives and neg >= th.risk_min_negatives:
            return {
                "target": "low_csat_risk",
                "target_kind": "derived_post_resolution",
                "definition": f"csat_score <= 2 among {len(vals)} rated tickets",
                "positive_count": pos,
                "negative_count": neg,
                "prevalence": round(pos / (pos + neg), 4),
                "caveat": (
                    "CSAT is recorded after resolution and only for customers who "
                    "responded. This target is a different and weaker construct than "
                    "escalation, and its population is self-selected."
                ),
                "attempts": attempts,
            }
        attempts.append({"target": "low_csat_risk", "rejected": f"{pos}/{neg} class sizes too small"})
    else:
        attempts.append({"target": "low_csat_risk", "rejected": e.get("reason", "unavailable")})

    return {
        "target": None,
        "target_kind": None,
        "attempts": attempts,
        "rejected_reason": (
            "No candidate target survived the ladder. No reliable escalation or SLA "
            "label exists and no derived target has usable class balance."
        ),
    }


# --- manifest ----------------------------------------------------------------


def retrieval_evaluation_status(settings: Settings) -> str:
    """What the UI is allowed to say about retrieval quality.

    Never fabricate a Hit@K. Until a human has graded the Tier 2 query set, the
    honest answer is that the set is not labeled, and the frontend prints
    exactly that instead of a number.
    """
    labeled_set = settings.evaluation_dir / "retrieval_queries.jsonl"
    report = settings.retrieval_dir / "evaluation.json"

    if not labeled_set.exists() or labeled_set.stat().st_size == 0:
        return "manual_set_not_yet_labeled"
    if not report.exists():
        return "manual_set_labeled_evaluation_not_run"
    return "evaluated"


def describe_llm_provider(settings: Settings) -> dict[str, Any]:
    """The provider block. `enabled` means "a request can actually reach a model".

    A provider name with no API key is disabled, not enabled-with-a-warning. The
    whole app is specified to work at `LLM_PROVIDER=none`, so a missing key is a
    normal operating state, not an error.
    """
    provider = (settings.llm_provider or "none").strip().lower()

    if provider == "none":
        return _disabled(
            "LLM_PROVIDER=none; deterministic mode active",
            provider="none",
            model=None,
        )
    if provider not in SUPPORTED_LLM_PROVIDERS:
        return _disabled(
            f"provider '{provider}' is not implemented in this build "
            f"(supported: {sorted(SUPPORTED_LLM_PROVIDERS)}); deterministic mode active",
            provider=provider,
            model=None,
        )
    if not settings.llm_api_key:
        return _disabled(
            f"provider '{provider}' is configured but no API key is set; "
            f"deterministic mode active",
            provider=provider,
            model=settings.llm_model,
        )
    return _enabled(
        provider=provider,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_evidence_cases=settings.llm_max_evidence_cases,
    )


def describe_ai_ops(settings: Settings) -> dict[str, Any]:
    if not settings.otel_enabled:
        return _disabled(
            "OTEL_ENABLED is false; no spans are emitted",
            exporter=None,
        )
    return _enabled(
        exporter=settings.otel_exporter,
        note=(
            "Console exporter only. Span attributes carry operational metadata; "
            "raw ticket text, resolution notes, identifiers, prompts, and provider "
            "responses are never recorded."
        ),
    )


def _served_corpus_size(audited: int, settings: Settings) -> tuple[int, str | None]:
    """Disclose served-vs-audited rather than silently reporting the smaller one."""
    limit = settings.corpus_serve_limit
    if not limit or limit >= audited:
        return audited, None
    return limit, (
        f"This deployment serves a stratified sample of {limit} source cases out "
        f"of {audited} audited locally, to fit the host's memory budget. Every "
        f"audit statistic on this page was computed over all {audited}."
    )


def build_capabilities(
    df: pd.DataFrame,
    audit: dict[str, Any],
    th: Thresholds | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    th = th or Thresholds()
    settings = settings or get_settings()
    n = len(df)

    # --- analytics ----------------------------------------------------------
    metrics: list[str] = ["ticket_volume"]
    metric_reasons: dict[str, str] = {}

    def _consider(metric: str, ok: bool, reason: str) -> None:
        if ok:
            metrics.append(metric)
        else:
            metric_reasons[metric] = reason

    res_valid = int(pd.to_numeric(df.get("resolution_time_hours"), errors="coerce").notna().sum())
    _consider(
        "resolution_time",
        res_valid > 0,
        "no ticket has a valid positive resolution duration",
    )
    resp_valid = int(pd.to_numeric(df.get("response_time_hours"), errors="coerce").notna().sum())
    _consider("response_time", resp_valid > 0, "no response-time information exists")

    esc_known = int(df["escalated"].notna().sum()) if "escalated" in df else 0
    _consider("escalation_rate", esc_known > 0, "no ticket has a known escalation outcome")

    sla_known = int(df["sla_breached"].notna().sum()) if "sla_breached" in df else 0
    _consider("sla_breach_rate", sla_known > 0, "no SLA outcome column exists")

    csat_rated = int(pd.to_numeric(df.get("csat_score"), errors="coerce").notna().sum())
    _consider("csat", csat_rated > 0, "no ticket carries a usable CSAT rating")

    time_ok = "created_at" in df.columns and df["created_at"].notna().any()
    _consider(
        "timeseries",
        bool(time_ok),
        "created_at is unavailable, so no time-based aggregation is possible",
    )

    analytics = _enabled(
        available_metrics=metrics,
        unavailable_metrics=metric_reasons,
        denominators={
            "ticket_volume": n,
            "resolution_time": res_valid,
            "response_time": resp_valid,
            "escalation_rate": esc_known,
            "sla_breach_rate": sla_known,
            "csat": csat_rated,
        },
        groupable_dimensions=[
            f
            for f in ("product_area", "issue_type", "priority", "status", "channel",
                      "platform", "region", "customer_segment", "sla_plan")
            if f in df.columns and df[f].notna().any()
        ],
    )

    # --- retrieval -----------------------------------------------------------
    mask, corpus_info = source_case_mask(df)
    n_source = corpus_info["eligible_source_cases"]
    ids_unique = df["ticket_id"].is_unique and df["ticket_id"].notna().all()

    if not ids_unique:
        retrieval = _disabled("ticket_id is not stable and unique", **corpus_info)
    elif corpus_info["non_empty_issue_text"] < th.retrieval_min_source_cases:
        retrieval = _disabled(
            f"only {corpus_info['non_empty_issue_text']} tickets have issue text "
            f"(minimum {th.retrieval_min_source_cases})",
            **corpus_info,
        )
    elif n_source < th.retrieval_min_source_cases:
        retrieval = _disabled(
            f"only {n_source} tickets qualify as source cases "
            f"(minimum {th.retrieval_min_source_cases})",
            **corpus_info,
        )
    else:
        served, serve_note = _served_corpus_size(n_source, settings)
        retrieval = _enabled(
            **corpus_info,
            corpus_size_audited=n_source,
            corpus_size_served=served,
            note=serve_note,
            evaluation_status=retrieval_evaluation_status(settings),
        )

    # --- resolution generation -----------------------------------------------
    provider = describe_llm_provider(settings)

    # Measure what the "resolution notes" column actually contains before
    # claiming anything about it.
    quality = RQ.assess(df.get("resolution_notes", pd.Series(dtype=str)))

    if not retrieval["enabled"]:
        generation = _disabled(
            f"retrieval is disabled: {retrieval['reason']}",
            available_modes=[],
            resolution_note_quality=quality,
        )
    elif corpus_info["usable_resolution_notes"] < th.generation_min_source_cases:
        generation = _disabled(
            f"only {corpus_info['usable_resolution_notes']} tickets carry non-boilerplate "
            f"resolution notes (minimum {th.generation_min_source_cases})",
            available_modes=["evidence_only"],
            resolution_note_quality=quality,
        )
    elif (
        quality["action_rate"] is not None
        and quality["action_rate"] < th.generation_min_action_rate
    ):
        # The notes exist and are non-boilerplate, but they are correspondence
        # rather than resolutions. Retrieval still works; claiming to show "what
        # resolved it" does not.
        generation = _disabled(
            quality["note"]
            + f" Minimum action rate for this capability is "
            f"{th.generation_min_action_rate:.0%}.",
            available_modes=["evidence_only"],
            resolution_note_quality=quality,
        )
    else:
        # `deterministic` is always available — it needs no key. `llm` appears
        # only when a request could actually reach a provider. `evidence_only`
        # is a per-request fallback, so it is always reachable.
        modes = ["deterministic", "evidence_only"]
        if provider["enabled"]:
            modes.insert(1, "llm")
        generation = _enabled(
            usable_resolution_notes=corpus_info["usable_resolution_notes"],
            resolved_status_used=corpus_info["resolved_status_used"],
            relaxation=corpus_info["relaxation"],
            available_modes=modes,
            resolution_note_quality=quality,
            note=(
                "Generation is additionally gated per request on retrieval strength. "
                "Weak retrieval never reaches a generator."
            ),
        )

    # --- clustering -----------------------------------------------------------
    text_n = corpus_info["non_empty_issue_text"]
    norm_ratio = (
        audit.get("text", {}).get("issue_text", {}).get("normalized_unique_ratio", 0.0) or 0.0
    )
    if text_n < th.clustering_min_texts:
        clustering = _disabled(
            f"only {text_n} tickets have issue text (minimum {th.clustering_min_texts})"
        )
    elif norm_ratio < th.clustering_min_normalized_unique_ratio:
        clustering = _disabled(
            f"issue text has a normalized unique ratio of {norm_ratio}, below "
            f"{th.clustering_min_normalized_unique_ratio}: there is not enough "
            f"variation to cluster meaningfully"
        )
    else:
        clustering = _enabled(
            eligible_texts=text_n,
            normalized_unique_ratio=norm_ratio,
            available_cluster_metrics=[
                m
                for m in ("resolution_time", "escalation_rate", "sla_breach_rate", "csat")
                if m in metrics
            ],
        )

    # --- risk ------------------------------------------------------------------
    t0_available = [f for f in T0_CANDIDATE_FIELDS if f in df.columns and df[f].notna().any()]
    if n < th.risk_min_rows:
        risk = _disabled(
            f"only {n} tickets available (minimum {th.risk_min_rows} to split into "
            f"train/validation/test)",
            target=None,
            target_kind=None,
        )
    elif not t0_available:
        risk = _disabled(
            "no feature is demonstrably available at ticket creation time",
            target=None,
            target_kind=None,
        )
    else:
        sel = select_risk_target(df, audit, th)
        if sel.get("target") is None:
            risk = _disabled(
                sel["rejected_reason"],
                target=None,
                target_kind=None,
                ladder_attempts=sel["attempts"],
            )
        else:
            risk = _enabled(
                target=sel["target"],
                target_kind=sel["target_kind"],
                target_definition=sel["definition"],
                prevalence=sel["prevalence"],
                positive_count=sel["positive_count"],
                negative_count=sel["negative_count"],
                caveat=sel.get("caveat"),
                t0_candidate_features=t0_available,
                temporal_split_feasible=bool(
                    audit.get("timestamps", {}).get("temporal_split_feasible", False)
                ),
                ladder_attempts=sel["attempts"],
            )

    # --- analyst agent (Phase 11) ---------------------------------------------
    # A tool-calling loop with nothing to call, or no model to drive it, is not
    # a capability. Both halves must be present.
    if not retrieval["enabled"]:
        analyst_agent = _disabled(f"retrieval is disabled: {retrieval['reason']}")
    elif not provider["enabled"]:
        analyst_agent = _disabled(f"requires an LLM provider: {provider['reason']}")
    else:
        analyst_agent = _enabled(
            registered_tools=[
                name
                for name, available in (
                    ("get_analytics", analytics["enabled"]),
                    ("list_clusters", clustering["enabled"]),
                    ("get_cluster_tickets", clustering["enabled"]),
                    ("find_similar_cases", retrieval["enabled"]),
                    ("score_escalation_risk", risk["enabled"]),
                )
                if available
            ],
            note=(
                "Tools whose subsystem is disabled are not registered, and the "
                "system prompt lists only registered tools."
            ),
        )

    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_hash": audit.get("source_file", {}).get("sha256"),
        "row_count": n,
        "thresholds": asdict(th),
        "analytics": analytics,
        "retrieval": retrieval,
        "resolution_generation": generation,
        "analyst_agent": analyst_agent,
        "clustering": clustering,
        "risk": risk,
        "llm_provider": provider,
        "ai_ops": describe_ai_ops(settings),
    }


def write_capabilities(caps: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(caps, fh, indent=2, default=str)
