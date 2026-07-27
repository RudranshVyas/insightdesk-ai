"""Phase 6 API — the Support Brief.

Two routes, both read-only. Nothing here writes to a ticket, and nothing builds
an artifact: the retriever is loaded from disk at first use and reused.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from backend.app.core.capability_loader import (
    CapabilityDisabled,
    load_capabilities,
    require,
)
from backend.app.core.config import get_settings
from backend.app.orchestration import pipeline
from backend.app.schemas.brief import (
    CapabilityDisabledBrief,
    SupportBrief,
    SupportBriefRequest,
    TraceResponse,
)

router = APIRouter(tags=["support-brief"])

# Bounded in-memory trace store. Traces are operational summaries only — no raw
# ticket text, no resolution notes, no prompts, no provider responses — so this
# holds nothing that would be sensitive if the process were inspected.
_TRACES: dict[str, TraceResponse] = {}
_TRACE_ORDER: list[str] = []
_MAX_TRACES = 200

_retriever: Any = None


def get_retriever() -> Any:
    """Load retrieval artifacts once. Never builds them — that is offline work."""
    global _retriever
    if _retriever is None:
        from backend.app.services.retrieval import HybridRetriever

        _retriever = HybridRetriever.load(get_settings())
    return _retriever


def reset_retriever(instance: Any = None) -> None:
    """Test seam: inject a retriever without touching disk."""
    global _retriever
    _retriever = instance


def _remember(brief: SupportBrief, provider_calls: int) -> None:
    _TRACES[brief.request_id] = TraceResponse(
        request_id=brief.request_id,
        mode=brief.mode,
        retrieval_strength=brief.retrieval_strength,
        stage_trace=brief.stage_trace,
        versions=brief.versions,
        provider_calls=provider_calls,
    )
    _TRACE_ORDER.append(brief.request_id)
    while len(_TRACE_ORDER) > _MAX_TRACES:
        _TRACES.pop(_TRACE_ORDER.pop(0), None)


@router.post("/support-brief", response_model=None)
def create_support_brief(request: SupportBriefRequest) -> Any:
    caps = load_capabilities()
    try:
        require("retrieval", caps)
    except CapabilityDisabled as exc:
        # A structured refusal with a reason, never a fabricated empty brief.
        return JSONResponse(
            status_code=200,
            content=CapabilityDisabledBrief(
                request_id="req_disabled",
                capability=exc.subsystem,
                reason=exc.reason,
            ).model_dump(),
        )

    settings = get_settings()
    try:
        retriever = get_retriever()
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Retrieval artifacts are not loadable: {exc}. Run "
                f"`python -m backend.scripts.build_retrieval_index`."
            ),
        ) from exc

    brief = pipeline.run_pipeline(request, retriever, settings=settings)
    _remember(brief, provider_calls=_provider_calls_from(brief))
    return brief


def _provider_calls_from(brief: SupportBrief) -> int:
    return 1 if brief.mode == "llm" else 0


@router.get("/support-brief/{request_id}/trace", response_model=TraceResponse)
def get_trace(request_id: str) -> TraceResponse:
    trace = _TRACES.get(request_id)
    if trace is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No trace for request_id {request_id!r}. Traces are kept in memory "
                f"for the last {_MAX_TRACES} requests only."
            ),
        )
    return trace
