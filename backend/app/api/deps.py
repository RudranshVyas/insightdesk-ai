"""Shared API dependencies and the capability-disabled response contract."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import HTTPException, Query, Request

from backend.app.core.capability_loader import CapabilityDisabled, load_capabilities
from backend.app.core.config import get_settings
from backend.app.services import ticket_store

# A disabled subsystem is a stated fact about the dataset, not a client error and
# not a server fault. 409 keeps it distinguishable from both.
CAPABILITY_DISABLED_STATUS = 409


def capability_error(exc: CapabilityDisabled) -> HTTPException:
    return HTTPException(status_code=CAPABILITY_DISABLED_STATUS, detail=exc.payload())


def get_caps() -> dict[str, Any]:
    return load_capabilities()


def get_tickets() -> pd.DataFrame:
    try:
        return ticket_store.load_tickets()
    except ticket_store.TicketsUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "capability": "data",
                "enabled": False,
                "reason": str(exc),
            },
        ) from exc


class CommonFilters:
    """Filters shared by every analytics endpoint."""

    def __init__(
        self,
        date_from: str | None = Query(None, description="ISO date, inclusive"),
        date_to: str | None = Query(None, description="ISO date, inclusive"),
        product_area: list[str] | None = Query(None),
        issue_type: list[str] | None = Query(None),
        priority: list[str] | None = Query(None),
        status: list[str] | None = Query(None),
    ) -> None:
        self.date_from = date_from
        self.date_to = date_to
        self.product_area = product_area
        self.issue_type = issue_type
        self.priority = priority
        self.status = status

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        return ticket_store.apply_filters(
            df,
            date_from=self.date_from,
            date_to=self.date_to,
            product_area=self.product_area,
            issue_type=self.issue_type,
            priority=self.priority,
            status=self.status,
        )


async def enforce_body_size(request: Request) -> None:
    limit = get_settings().max_request_body_bytes
    declared = request.headers.get("content-length")
    if declared and int(declared) > limit:
        raise HTTPException(
            status_code=413, detail=f"request body exceeds {limit} bytes"
        )
