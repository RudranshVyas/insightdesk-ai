"""Lazy in-memory access to the processed ticket table.

The parquet is a static artifact produced by Phase 1. Loading it is a read, not
a build step, so it is allowed at request time — unlike embeddings, indexes, and
models, which are never built at API startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.core.config import get_settings


class TicketsUnavailable(Exception):
    """The processed parquet has not been produced yet."""


@dataclass
class _Cache:
    path: Path | None = None
    mtime: float | None = None
    df: pd.DataFrame | None = None


_cache = _Cache()


def load_tickets(path: Path | None = None) -> pd.DataFrame:
    p = Path(path or get_settings().tickets_parquet)
    if not p.exists():
        raise TicketsUnavailable(
            f"{p} does not exist. Run `python -m backend.scripts.ingest_tickets "
            f"--csv <file> --mapping <mapping.yaml>` first."
        )
    mtime = p.stat().st_mtime
    if _cache.df is None or _cache.path != p or _cache.mtime != mtime:
        _cache.df = pd.read_parquet(p)
        _cache.path = p
        _cache.mtime = mtime
    return _cache.df


def reset_cache() -> None:
    _cache.df = None
    _cache.path = None
    _cache.mtime = None


FILTERABLE = ("product_area", "issue_type", "priority", "status", "channel",
              "platform", "region", "customer_segment", "sla_plan")


def apply_filters(
    df: pd.DataFrame,
    date_from: str | None = None,
    date_to: str | None = None,
    **equals: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Filter a ticket frame. Returns the frame and a record of what was applied.

    A date filter on a dataset without ``created_at`` is reported as ignored
    rather than silently returning nothing.
    """
    applied: dict[str, Any] = {}
    ignored: dict[str, str] = {}
    out = df

    for key, value in equals.items():
        if value in (None, "", []):
            continue
        if key not in df.columns:
            ignored[key] = f"field '{key}' is not present in this dataset"
            continue
        values = value if isinstance(value, (list, tuple, set)) else [value]
        out = out[out[key].isin(list(values))]
        applied[key] = list(values)

    if date_from or date_to:
        if "created_at" not in df.columns or df["created_at"].notna().sum() == 0:
            ignored["date_range"] = (
                "created_at is unavailable in this dataset, so the date filter was "
                "not applied"
            )
        else:
            created = pd.to_datetime(out["created_at"], errors="coerce")
            if date_from:
                out = out[created >= pd.Timestamp(date_from)]
                created = pd.to_datetime(out["created_at"], errors="coerce")
                applied["date_from"] = str(date_from)
            if date_to:
                out = out[created <= pd.Timestamp(date_to)]
                applied["date_to"] = str(date_to)

    return out, {"applied": applied, "ignored": ignored, "rows_after_filter": len(out)}
