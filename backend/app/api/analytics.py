"""Phase 3 analytics routes. Deterministic; capability-gated metric by metric."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.app.api.deps import CommonFilters, capability_error, get_caps, get_tickets
from backend.app.core.capability_loader import CapabilityDisabled, require
from backend.app.services import analytics as A

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _allowed_metrics(caps: dict[str, Any]) -> set[str]:
    try:
        block = require("analytics", caps)
    except CapabilityDisabled as exc:
        raise capability_error(exc) from exc
    return set(block.get("available_metrics") or [])


@router.get("/overview")
def get_overview(
    filters: CommonFilters = Depends(),
    df: pd.DataFrame = Depends(get_tickets),
    caps: dict[str, Any] = Depends(get_caps),
) -> dict[str, Any]:
    allowed = _allowed_metrics(caps)
    sub, filter_info = filters.apply(df)
    return {
        "filters": filter_info,
        "available_metrics": sorted(allowed),
        "unavailable_metrics": caps["analytics"].get("unavailable_metrics", {}),
        **A.overview(sub, allowed),
    }


@router.get("/timeseries")
def get_timeseries(
    freq: str = Query("W", pattern="^(D|W|ME|QE)$", description="D, W, ME, or QE"),
    filters: CommonFilters = Depends(),
    df: pd.DataFrame = Depends(get_tickets),
    caps: dict[str, Any] = Depends(get_caps),
) -> dict[str, Any]:
    allowed = _allowed_metrics(caps)
    if "timeseries" not in allowed:
        raise HTTPException(
            status_code=409,
            detail={
                "capability": "analytics.timeseries",
                "enabled": False,
                "reason": caps["analytics"]
                .get("unavailable_metrics", {})
                .get("timeseries", "created_at is unavailable"),
            },
        )
    sub, filter_info = filters.apply(df)
    return {"filters": filter_info, **A.timeseries(sub, allowed, freq=freq)}


@router.get("/product-pain-points")
def get_product_pain_points(
    sort_by: str = Query("ticket_volume"),
    limit: int = Query(20, ge=1, le=100),
    filters: CommonFilters = Depends(),
    df: pd.DataFrame = Depends(get_tickets),
    caps: dict[str, Any] = Depends(get_caps),
) -> dict[str, Any]:
    allowed = _allowed_metrics(caps)
    sub, filter_info = filters.apply(df)
    try:
        result = A.by_dimension(sub, "product_area", allowed, sort_by=sort_by, limit=limit)
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={"capability": "analytics.product_area", "enabled": False, "reason": str(exc)},
        ) from exc
    return {"filters": filter_info, **result}


@router.get("/issue-types")
def get_issue_types(
    sort_by: str = Query("ticket_volume"),
    limit: int = Query(20, ge=1, le=100),
    filters: CommonFilters = Depends(),
    df: pd.DataFrame = Depends(get_tickets),
    caps: dict[str, Any] = Depends(get_caps),
) -> dict[str, Any]:
    allowed = _allowed_metrics(caps)
    sub, filter_info = filters.apply(df)
    try:
        result = A.by_dimension(sub, "issue_type", allowed, sort_by=sort_by, limit=limit)
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={"capability": "analytics.issue_type", "enabled": False, "reason": str(exc)},
        ) from exc
    return {"filters": filter_info, **result}


@router.get("/tickets")
def list_tickets(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    filters: CommonFilters = Depends(),
    df: pd.DataFrame = Depends(get_tickets),
) -> dict[str, Any]:
    sub, filter_info = filters.apply(df)
    columns = [
        "ticket_id", "created_at", "product_area", "issue_type", "priority",
        "status", "channel", "issue_subject", "resolution_time_hours",
        "escalated", "csat_score",
    ]
    return {"filters": filter_info, **A.paginate(sub, page, page_size, columns)}
