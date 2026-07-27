"""FastAPI application.

Startup reads artifacts. It never builds them: no embedding, no index build, no
clustering, no model training happens here.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import analytics as analytics_router
from backend.app.api import support_brief as support_brief_router
from backend.app.core.capability_loader import load_capabilities
from backend.app.core.config import get_settings

API_PREFIX = "/api/v1"

app = FastAPI(
    title="InsightDesk AI",
    version="0.1.0",
    description=(
        "Internal support intelligence. Analyzes historical tickets, retrieves "
        "similar resolved cases, clusters recurring issues, and — where the "
        "dataset supports it — scores escalation risk from creation-time features. "
        "Suggestions are historical evidence, never guaranteed resolutions."
    ),
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(analytics_router.router, prefix=API_PREFIX)
app.include_router(support_brief_router.router, prefix=API_PREFIX)


@app.get(f"{API_PREFIX}/health", tags=["meta"])
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get(f"{API_PREFIX}/capabilities", tags=["meta"])
def capabilities() -> dict[str, Any]:
    return load_capabilities()
