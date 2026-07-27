"""Phase 3 — deterministic analytics. No LLM anywhere in this module.

Two rules govern everything here:

1. Every metric reports its own denominator. A rate whose denominator is not
   stated is not a metric, it is a rumour.
2. A metric that cannot be computed is absent, not zero. Open tickets are never
   counted as zero-hour resolutions and unrated tickets never enter a CSAT mean.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Below this many observations a per-category figure is reported with a warning.
MIN_CATEGORY_SAMPLE = 30


def _f(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else round(v, 4)


def metric(
    value: float | None,
    denominator: int,
    definition: str,
    *,
    unit: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The only shape a metric is allowed to take."""
    out: dict[str, Any] = {
        "value": _f(value),
        "denominator": int(denominator),
        "definition": definition,
    }
    if unit:
        out["unit"] = unit
    if denominator == 0:
        out["value"] = None
        out["note"] = "no ticket qualifies for this metric's denominator"
    elif denominator < MIN_CATEGORY_SAMPLE:
        out["warning"] = (
            f"only {denominator} qualifying tickets; treat this figure as indicative"
        )
    if extra:
        out.update(extra)
    return out


# --- shared population helpers ----------------------------------------------
# Defined once and used by every aggregation, so overview, per-area, per-type,
# and trend figures can never disagree about who counts.


def get_rated_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Tickets with a usable CSAT rating.

    Nulls are excluded. Zeros were already converted to null during ingestion
    *if and only if* the mapping declared `zero_means_no_response: true`; this
    helper never invents that rule for itself.
    """
    if "csat_score" not in df.columns:
        return df.iloc[0:0]
    scores = pd.to_numeric(df["csat_score"], errors="coerce")
    return df[scores.notna()]


def get_resolved_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Tickets that reached a resolved state, per the ingestion-time definition."""
    if "is_resolved" not in df.columns or df["is_resolved"].isna().all():
        return df.iloc[0:0]
    return df[df["is_resolved"].fillna(False).astype(bool)]


def get_timed_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Resolved tickets with a valid, strictly positive resolution duration.

    Open tickets are excluded outright. They are not slow tickets; they are
    tickets with no resolution time at all.
    """
    if "resolution_time_hours" not in df.columns:
        return df.iloc[0:0]
    hours = pd.to_numeric(df["resolution_time_hours"], errors="coerce")
    resolved = get_resolved_tickets(df)
    if len(resolved) == 0:
        return df[hours.notna() & (hours > 0)]
    return resolved[
        pd.to_numeric(resolved["resolution_time_hours"], errors="coerce").notna()
        & (pd.to_numeric(resolved["resolution_time_hours"], errors="coerce") > 0)
    ]


def get_responded_tickets(df: pd.DataFrame) -> pd.DataFrame:
    if "response_time_hours" not in df.columns:
        return df.iloc[0:0]
    hours = pd.to_numeric(df["response_time_hours"], errors="coerce")
    return df[hours.notna() & (hours >= 0)]


def get_escalation_known(df: pd.DataFrame) -> pd.DataFrame:
    """Only tickets whose escalation outcome is actually known."""
    if "escalated" not in df.columns:
        return df.iloc[0:0]
    return df[df["escalated"].notna()]


def get_sla_eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Tickets that carry an SLA outcome. Tickets with no SLA are not "not
    breached" — they are outside the denominator entirely."""
    if "sla_breached" not in df.columns:
        return df.iloc[0:0]
    return df[df["sla_breached"].notna()]


# --- metric computations -----------------------------------------------------


def _mean_hours(sub: pd.DataFrame, col: str) -> float | None:
    if len(sub) == 0:
        return None
    return float(pd.to_numeric(sub[col], errors="coerce").mean())


def _median_hours(sub: pd.DataFrame, col: str) -> float | None:
    if len(sub) == 0:
        return None
    return float(pd.to_numeric(sub[col], errors="coerce").median())


def _rate(sub: pd.DataFrame, col: str) -> float | None:
    if len(sub) == 0:
        return None
    return float(sub[col].astype(bool).mean())


def compute_metric_block(df: pd.DataFrame, allowed: set[str]) -> dict[str, Any]:
    """All allowed metrics for one population. Used for the overview and for
    every per-category row, so the definitions can never drift apart."""
    out: dict[str, Any] = {}

    out["ticket_volume"] = metric(
        len(df), len(df), "count of tickets matching the current filters", unit="tickets"
    )

    if "resolution_time" in allowed:
        timed = get_timed_tickets(df)
        out["resolution_time_hours"] = metric(
            _mean_hours(timed, "resolution_time_hours"),
            len(timed),
            "mean hours from created_at to resolved_at, over resolved tickets with a "
            "valid positive duration; open tickets are excluded, not counted as zero",
            unit="hours",
            extra={"median": _f(_median_hours(timed, "resolution_time_hours"))},
        )

    if "response_time" in allowed:
        responded = get_responded_tickets(df)
        out["response_time_hours"] = metric(
            _mean_hours(responded, "response_time_hours"),
            len(responded),
            "mean hours to first response, over tickets that have response "
            "information; tickets with no recorded response are excluded",
            unit="hours",
            extra={"median": _f(_median_hours(responded, "response_time_hours"))},
        )

    if "escalation_rate" in allowed:
        known = get_escalation_known(df)
        out["escalation_rate"] = metric(
            _rate(known, "escalated"),
            len(known),
            "share of tickets flagged escalated, over tickets whose escalation "
            "outcome is known; unknown outcomes are excluded from the denominator",
            unit="rate",
        )

    if "sla_breach_rate" in allowed:
        eligible = get_sla_eligible(df)
        out["sla_breach_rate"] = metric(
            _rate(eligible, "sla_breached"),
            len(eligible),
            "share of SLA-eligible tickets that breached; tickets with no SLA "
            "outcome are outside the denominator entirely",
            unit="rate",
        )

    if "csat" in allowed:
        rated = get_rated_tickets(df)
        scores = pd.to_numeric(rated["csat_score"], errors="coerce") if len(rated) else None
        out["csat"] = metric(
            float(scores.mean()) if scores is not None and len(scores) else None,
            len(rated),
            "mean CSAT over rated tickets only; the denominator is the response "
            "count, not the ticket count",
            unit="score",
            extra={
                "response_count": len(rated),
                "response_rate": _f(len(rated) / len(df)) if len(df) else None,
            },
        )

    return out


def overview(df: pd.DataFrame, allowed: set[str]) -> dict[str, Any]:
    block = compute_metric_block(df, allowed)
    status_counts = (
        {str(k): int(v) for k, v in df["status"].value_counts(dropna=True).items()}
        if "status" in df.columns
        else {}
    )
    return {
        "metrics": block,
        "status_breakdown": status_counts,
        "open_ticket_count": int(len(df) - len(get_resolved_tickets(df))),
        "resolved_ticket_count": len(get_resolved_tickets(df)),
    }


def timeseries(
    df: pd.DataFrame, allowed: set[str], freq: str = "W"
) -> dict[str, Any]:
    """Volume and available rate metrics over time. Requires created_at."""
    if "created_at" not in df.columns or df["created_at"].notna().sum() == 0:
        raise ValueError("created_at is unavailable; a time series cannot be computed")

    work = df[df["created_at"].notna()].copy()
    work["_bucket"] = pd.to_datetime(work["created_at"]).dt.to_period(freq).dt.start_time

    points: list[dict[str, Any]] = []
    for bucket, sub in work.groupby("_bucket", sort=True):
        block = compute_metric_block(sub, allowed)
        point: dict[str, Any] = {
            "bucket": bucket.isoformat(),
            "ticket_volume": block["ticket_volume"]["value"],
        }
        for key in ("resolution_time_hours", "escalation_rate", "sla_breach_rate", "csat"):
            if key in block:
                point[key] = block[key]["value"]
                point[f"{key}_denominator"] = block[key]["denominator"]
        points.append(point)

    return {
        "freq": freq,
        "points": points,
        "excluded_rows_without_created_at": int(len(df) - len(work)),
        "definitions": {
            k: v["definition"] for k, v in compute_metric_block(work, allowed).items()
        },
    }


def by_dimension(
    df: pd.DataFrame,
    dimension: str,
    allowed: set[str],
    sort_by: str = "ticket_volume",
    limit: int = 20,
) -> dict[str, Any]:
    """Per-category metrics with explicit denominators and sample warnings."""
    if dimension not in df.columns or df[dimension].notna().sum() == 0:
        raise ValueError(f"dimension '{dimension}' is unavailable in this dataset")

    rows: list[dict[str, Any]] = []
    for value, sub in df[df[dimension].notna()].groupby(dimension, sort=False):
        block = compute_metric_block(sub, allowed)
        row: dict[str, Any] = {dimension: str(value), "metrics": block}
        row["_sort"] = {k: (v["value"] if v["value"] is not None else -1) for k, v in block.items()}
        rows.append(row)

    key = sort_by if any(sort_by in r["_sort"] for r in rows) else "ticket_volume"
    rows.sort(key=lambda r: r["_sort"].get(key, -1), reverse=True)
    for r in rows:
        r.pop("_sort")

    small = [
        r[dimension]
        for r in rows
        if r["metrics"]["ticket_volume"]["denominator"] < MIN_CATEGORY_SAMPLE
    ]
    return {
        "dimension": dimension,
        "sorted_by": key,
        "rows": rows[:limit],
        "category_count": len(rows),
        "excluded_rows_with_null_dimension": int(df[dimension].isna().sum()),
        "min_sample_threshold": MIN_CATEGORY_SAMPLE,
        "small_sample_categories": small,
        "warning": (
            f"{len(small)} categories have fewer than {MIN_CATEGORY_SAMPLE} tickets; "
            f"their rankings are unstable"
        )
        if small
        else None,
    }


def paginate(
    df: pd.DataFrame, page: int, page_size: int, columns: list[str] | None = None
) -> dict[str, Any]:
    page = max(1, page)
    page_size = max(1, min(page_size, 200))
    total = len(df)
    start = (page - 1) * page_size
    sub = df.iloc[start : start + page_size]
    if columns:
        sub = sub[[c for c in columns if c in sub.columns]]
    records = [
        {
            k: (None if pd.isna(v) else (v.isoformat() if isinstance(v, pd.Timestamp) else v))
            for k, v in rec.items()
        }
        for rec in sub.to_dict("records")
    ]
    return {
        "items": records,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size,
    }
