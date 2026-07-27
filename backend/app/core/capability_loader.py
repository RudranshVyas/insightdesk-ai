"""Runtime access to the capability manifest.

The API reads this; it never recomputes it. If the manifest is absent, every
capability is reported disabled with that as the reason — the app degrades, it
does not guess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings

SUBSYSTEMS = (
    "analytics",
    "retrieval",
    "resolution_generation",
    "analyst_agent",
    "clustering",
    "risk",
    "llm_provider",
    "ai_ops",
)

_MISSING_REASON = (
    "artifacts/capabilities.json has not been generated. Run "
    "`python -m backend.scripts.ingest_tickets` then "
    "`python -m backend.scripts.build_capabilities`."
)


class CapabilityDisabled(Exception):
    """Raised when a request touches a subsystem the dataset cannot support."""

    def __init__(self, subsystem: str, reason: str) -> None:
        super().__init__(f"{subsystem} disabled: {reason}")
        self.subsystem = subsystem
        self.reason = reason

    def payload(self) -> dict[str, Any]:
        return {
            "capability": self.subsystem,
            "enabled": False,
            "reason": self.reason,
            "detail": (
                "This subsystem is disabled because the dataset does not support it. "
                "No placeholder or zero-valued result is returned in its place."
            ),
        }


def _empty_manifest() -> dict[str, Any]:
    caps: dict[str, Any] = {
        "manifest_version": None,
        "generated_at": None,
        "data_hash": None,
        "row_count": 0,
        "available": False,
    }
    for s in SUBSYSTEMS:
        caps[s] = {"enabled": False, "reason": _MISSING_REASON}
    return caps


def load_capabilities(path: Path | None = None) -> dict[str, Any]:
    """Read the manifest from disk. Not cached — it is a small file and an
    operator may regenerate it between requests."""
    p = path or get_settings().capabilities_json
    if not Path(p).exists():
        return _empty_manifest()
    try:
        with open(p, encoding="utf-8") as fh:
            caps = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        caps = _empty_manifest()
        for s in SUBSYSTEMS:
            caps[s]["reason"] = f"capabilities.json could not be read: {exc}"
        return caps
    caps["available"] = True
    return caps


def is_enabled(subsystem: str, caps: dict[str, Any] | None = None) -> bool:
    caps = caps if caps is not None else load_capabilities()
    return bool(caps.get(subsystem, {}).get("enabled"))


def require(subsystem: str, caps: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the subsystem block, or raise :class:`CapabilityDisabled`."""
    caps = caps if caps is not None else load_capabilities()
    block = caps.get(subsystem, {"enabled": False, "reason": f"unknown subsystem {subsystem!r}"})
    if not block.get("enabled"):
        raise CapabilityDisabled(subsystem, block.get("reason") or "no reason recorded")
    return block


def require_metric(metric: str, caps: dict[str, Any] | None = None) -> None:
    """Analytics is metric-by-metric: the subsystem can be on while a metric is off."""
    caps = caps if caps is not None else load_capabilities()
    block = require("analytics", caps)
    if metric not in (block.get("available_metrics") or []):
        reason = (block.get("unavailable_metrics") or {}).get(
            metric, f"metric {metric!r} is not available for this dataset"
        )
        raise CapabilityDisabled(f"analytics.{metric}", reason)
