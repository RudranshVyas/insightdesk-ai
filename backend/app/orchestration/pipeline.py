"""Phase 6 — the Support Brief pipeline.

A hand-rolled typed orchestrator, deliberately. The flow is linear with a single
conditional (the strength gate) and no loops, so a graph library would add a
dependency and a layer of indirection without removing any control-flow
complexity. Phase 11's agent is where a framework earns its place; this is not.

    intake_and_redact -> retrieve -> gate -> curate_evidence
      -> suggest -> verify -> compose_brief

Every stage has a typed contract, a recorded latency, an explicit failure
policy, and no hidden side effects. **No stage writes to any record.**
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from backend.app.core import guardrails as G
from backend.app.core import redaction as R
from backend.app.core.config import Settings, get_settings
from backend.app.core.prompt_registry import Prompt, load_prompt
from backend.app.core.versions import version_stamp
from backend.app.orchestration import verifier as V
from backend.app.schemas.brief import (
    EvidenceTicket,
    GeneratedResolution,
    RetrievalStrength,
    StageTrace,
    SupportBrief,
    SupportBriefRequest,
    VersionStamp,
)
from backend.app.schemas.brief import (
    SuggestedStep as Step,
)
from backend.app.services import llm as LLM

# State machine labels, logged on every transition.
STATES = (
    "INTAKE",
    "RETRIEVING",
    "GATED",
    "CURATING",
    "SUGGESTING",
    "VERIFYING",
    "COMPOSED",
)

EVIDENCE_EXCERPT_CHARS = 700


@dataclass
class PipelineState:
    request_id: str
    request: SupportBriefRequest
    state: str = "INTAKE"
    query_text: str = ""
    retrieval: dict[str, Any] = field(default_factory=dict)
    strength: str = "weak"
    evidence: list[EvidenceTicket] = field(default_factory=list)
    generated: GeneratedResolution | None = None
    verification: V.VerificationResult | None = None
    mode: str = "deterministic"
    warnings: list[str] = field(default_factory=list)
    trace: list[StageTrace] = field(default_factory=list)
    provider_calls: int = 0
    prompt: Prompt | None = None
    provider_usage: dict[str, Any] = field(default_factory=dict)

    def transition(self, new_state: str) -> None:
        self.state = new_state


def _stage(state: PipelineState, name: str):
    """Time one stage and append its trace entry. Failures degrade, never crash."""

    class _Ctx:
        def __enter__(self_inner):
            self_inner.t0 = time.perf_counter()
            self_inner.warnings: list[str] = []
            self_inner.summary = ""
            self_inner.status = "ok"
            return self_inner

        def __exit__(self_inner, exc_type, exc, tb):
            latency = (time.perf_counter() - self_inner.t0) * 1000.0
            if exc is not None:
                self_inner.status = "failed"
                self_inner.summary = f"{type(exc).__name__}: {exc}"
            state.trace.append(
                StageTrace(
                    name=name,
                    status=self_inner.status,
                    latency_ms=round(latency, 2),
                    summary=self_inner.summary,
                    warnings=list(self_inner.warnings),
                )
            )
            return False

    return _Ctx()


# --- stage 1: intake ----------------------------------------------------------


def intake_and_redact(state: PipelineState, settings: Settings) -> None:
    state.transition("INTAKE")
    with _stage(state, "intake_and_redact") as ctx:
        raw = state.request.issue_description.strip()

        if len(raw) > settings.max_issue_text_chars:
            raw = raw[: settings.max_issue_text_chars]
            ctx.warnings.append(
                f"Query truncated to {settings.max_issue_text_chars} characters."
            )

        # Redaction runs before the text reaches retrieval, embedding, the
        # prompt, the trace, or the response.
        report = R.RedactionReport()
        redacted = R.redact_text(raw, report)
        if report.total():
            ctx.warnings.append(
                f"Redacted {report.total()} PII item(s) from the query before use: "
                f"{sorted(report.counts)}"
            )

        scan = G.scan_injection(redacted)
        if scan.flagged:
            ctx.warnings.append(
                f"Query contains instruction-like text ({', '.join(scan.labels)}). "
                f"It is treated as data describing a problem, never as instructions."
            )

        state.query_text = redacted
        ctx.summary = f"{len(redacted)} chars after redaction"
        state.warnings.extend(ctx.warnings)


# --- stage 2: retrieve --------------------------------------------------------


def retrieve(state: PipelineState, retriever: Any, settings: Settings) -> None:
    state.transition("RETRIEVING")
    with _stage(state, "retrieve") as ctx:
        result = retriever.search(
            state.query_text,
            product_area=state.request.product_area,
            issue_type=state.request.issue_type,
            top_k=state.request.top_k,
        )
        state.retrieval = result
        state.strength = result["strength"]["strength"]
        ctx.summary = (
            f"{len(result['results'])} candidates, "
            f"{result['fusion']['dense_candidates']} dense / "
            f"{result['fusion']['lexical_candidates']} lexical, strength={state.strength}"
        )


# --- stage 3: gate ------------------------------------------------------------


def gate(state: PipelineState) -> bool:
    """Deterministic policy, not an "agent". Returns whether generation may run.

    weak  -> skip generation entirely (and make zero provider calls)
    mixed -> generate, but force manual review
    strong-> generate
    """
    state.transition("GATED")
    with _stage(state, "gate") as ctx:
        allow = state.strength in ("strong", "mixed")
        if state.strength == "weak":
            ctx.warnings.append(
                "Retrieval strength is weak. Generation is skipped entirely and no "
                "provider call is made."
            )
        elif state.strength == "mixed":
            ctx.warnings.append(
                "Retrieval strength is mixed. Generation runs but the brief is "
                "flagged for mandatory human review."
            )
        ctx.summary = f"strength={state.strength}, generation_allowed={allow}"
        state.warnings.extend(ctx.warnings)
    return allow


# --- stage 4: curate evidence -------------------------------------------------


def curate_evidence(state: PipelineState, settings: Settings) -> None:
    """Select diverse, usable cases within a token budget.

    Five near-identical duplicates are worse evidence than three varied cases:
    they look like corroboration while being one case counted five times. Cases
    are therefore de-duplicated by MinHash template group before the budget is
    applied.
    """
    state.transition("CURATING")
    with _stage(state, "curate_evidence") as ctx:
        results = state.retrieval.get("results", [])
        seen_groups: set[int] = set()
        chosen: list[EvidenceTicket] = []
        dropped_siblings = 0
        dropped_unusable = 0
        budget = settings.llm_max_evidence_chars

        for row in results:
            attached = row.get("attached") or {}
            notes = str(attached.get("resolution_notes") or "").strip()
            if not notes:
                dropped_unusable += 1
                continue

            group = attached.get("template_group_id")
            if group is not None:
                gid = int(group)
                if gid in seen_groups:
                    dropped_siblings += 1
                    continue
                seen_groups.add(gid)

            excerpt = str(attached.get("issue_text") or attached.get("issue_subject") or "")
            ticket = EvidenceTicket(
                ticket_id=str(row["ticket_id"]),
                issue_subject=attached.get("issue_subject"),
                issue_excerpt=excerpt[:EVIDENCE_EXCERPT_CHARS],
                resolution_notes=notes[:EVIDENCE_EXCERPT_CHARS],
                product_area=attached.get("product_area"),
                issue_type=attached.get("issue_type"),
                dense_rank=row.get("dense_rank"),
                lexical_rank=row.get("lexical_rank"),
                dense_cosine=row.get("dense_cosine"),
                fusion_rank=row.get("fusion_rank"),
                matched_metadata=row.get("matched_metadata") or {},
                template_group_id=int(group) if group is not None else None,
                resolution_time_hours=_as_float(attached.get("resolution_time_hours")),
                escalated=_as_bool(attached.get("escalated")),
                sla_breached=_as_bool(attached.get("sla_breached")),
                csat_score=_as_float(attached.get("csat_score")),
                injection_flags=G.scan_injection(f"{excerpt}\n{notes}").labels,
            )

            cost = len(ticket.issue_excerpt) + len(ticket.resolution_notes or "")
            if budget - cost < 0 and chosen:
                break
            budget -= cost
            chosen.append(ticket)

            if len(chosen) >= settings.llm_max_evidence_cases:
                break

        state.evidence = chosen
        if dropped_siblings:
            ctx.warnings.append(
                f"Dropped {dropped_siblings} near-duplicate case(s) from the same "
                f"template group; duplicates inflate apparent corroboration."
            )
        if dropped_unusable:
            ctx.warnings.append(
                f"Dropped {dropped_unusable} case(s) with unusable resolution notes."
            )
        flagged = [t.ticket_id for t in chosen if t.injection_flags]
        if flagged:
            ctx.warnings.append(
                f"Evidence tickets {flagged} contain instruction-like text. They are "
                f"fenced as untrusted data in the prompt."
            )
        ctx.summary = f"{len(chosen)} diverse cases selected"
        state.warnings.extend(ctx.warnings)


def _as_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _as_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


# --- stage 5: suggest ---------------------------------------------------------


def render_evidence_block(evidence: list[EvidenceTicket]) -> str:
    """Fence each case so the model is told exactly where untrusted data starts."""
    parts: list[str] = []
    for t in evidence:
        body = G.neutralize_delimiters(t.issue_excerpt)
        notes = G.neutralize_delimiters(t.resolution_notes or "")
        parts.append(
            f"[EVIDENCE ticket_id={t.ticket_id}]\n"
            f"Product Area: {t.product_area or 'unknown'}\n"
            f"Issue Type: {t.issue_type or 'unknown'}\n"
            f"Reported issue (untrusted customer text): {body}\n"
            f"What support did (untrusted agent text): {notes}\n"
            f"[/EVIDENCE ticket_id={t.ticket_id}]"
        )
    return "\n\n".join(parts)


def suggest_deterministic(state: PipelineState) -> GeneratedResolution:
    """No provider needed. Renders the retrieved resolution notes with citations.

    This is not a degraded stub — it is the default mode, and the demo runs on
    it. Each step is one historical case, cited to itself, which is exactly as
    much as the evidence supports without a model to synthesize across cases.
    """
    steps = [
        Step(
            text=(
                f"In ticket {t.ticket_id} ({t.issue_type or 'unclassified'}"
                f"{', ' + t.product_area if t.product_area else ''}), support resolved a "
                f"similar report as follows: {t.resolution_notes}"
            ),
            citation_ticket_ids=[t.ticket_id],
        )
        for t in state.evidence
    ]
    explanation = (
        f"{len(state.evidence)} historical case(s) matched this description by hybrid "
        f"retrieval (dense embeddings fused with BM25). Retrieval strength is "
        f"{state.strength}. These are the resolutions recorded on those tickets, "
        f"reproduced verbatim and not synthesized — no language model was involved."
    )
    return GeneratedResolution(
        suggested_steps=steps,
        relevance_explanation=explanation,
        insufficient_evidence=not steps,
    )


def suggest_llm(
    state: PipelineState, provider: LLM.LLMProvider, settings: Settings
) -> GeneratedResolution:
    """One bounded retry, then the caller falls back to deterministic mode."""
    prompt = load_prompt("resolution", "v1")
    state.prompt = prompt
    gen = prompt.generation_settings()
    max_tokens = int(gen.get("max_output_tokens") or settings.llm_max_output_tokens)

    user = prompt.render_user(
        query=G.neutralize_delimiters(state.query_text),
        product_area=state.request.product_area or "not provided",
        issue_type=state.request.issue_type or "not provided",
        evidence=render_evidence_block(state.evidence),
        evidence_count=len(state.evidence),
    )

    schema = GeneratedResolution.model_json_schema()
    last_error: Exception | None = None

    for attempt in (1, 2):
        try:
            state.provider_calls += 1
            response = provider.complete_json(prompt.system, user, schema, max_tokens)
            state.provider_usage = response.usage()
            payload = LLM.extract_json_object(response.text)
            return GeneratedResolution.model_validate(payload)
        except LLM.LLMRefusal:
            # A refusal will not become a success on retry.
            raise
        except Exception as exc:  # noqa: BLE001 - any failure falls back
            last_error = exc
            if attempt == 2:
                break

    raise LLM.LLMError(f"generation failed after 2 attempts: {last_error}")


# --- stage 7: compose ---------------------------------------------------------


def compose_brief(state: PipelineState, settings: Settings) -> SupportBrief:
    state.transition("COMPOSED")
    v = state.verification
    raw_strength = state.retrieval.get("strength", {}) if state.retrieval else {}

    detail = None
    if raw_strength:
        detail = RetrievalStrength(
            strength=raw_strength.get("strength", state.strength),
            top_cosine=raw_strength.get("top_cosine"),
            margin=raw_strength.get("margin"),
            candidates_above_floor=raw_strength.get("candidates_above_floor", 0),
            rank_agreement=raw_strength.get("rank_agreement", 0),
            calibrated=bool(raw_strength.get("calibrated", False)),
            reasons=list(raw_strength.get("reasons") or []),
        )

    index = state.retrieval.get("index", {}) if state.retrieval else {}
    versions = VersionStamp(
        artifact=version_stamp(),
        index_version=index.get("version"),
        index_data_hash=index.get("data_hash"),
        embedding_model=index.get("embedding_model"),
        prompt_version=state.prompt.stamp if state.prompt else None,
        provider=settings.llm_provider if state.mode == "llm" else "none",
        provider_model=state.provider_usage.get("model"),
    )

    return SupportBrief(
        request_id=state.request_id,
        mode=state.mode,
        retrieval_strength=state.strength,
        strength_detail=detail,
        similar_cases=state.evidence,
        suggested_steps=v.steps if v else [],
        relevance_explanation=(v.relevance_explanation if v else None) or None,
        risk_signal=None,  # capability-gated; null rather than a fabricated zero
        manual_review_required=v.manual_review_required if v else True,
        insufficient_evidence=v.insufficient_evidence if v else True,
        warnings=state.warnings,
        stage_trace=state.trace,
        versions=versions,
    )


# --- orchestrator -------------------------------------------------------------


def run_pipeline(
    request: SupportBriefRequest,
    retriever: Any,
    provider_factory: Callable[[], LLM.LLMProvider] | None = None,
    settings: Settings | None = None,
    request_id: str | None = None,
) -> SupportBrief:
    s = settings or get_settings()
    state = PipelineState(
        request_id=request_id or f"req_{uuid.uuid4().hex[:16]}", request=request
    )

    intake_and_redact(state, s)
    retrieve(state, retriever, s)
    generation_allowed = gate(state)
    curate_evidence(state, s)

    # --- mode selection -------------------------------------------------------
    # `evidence_only` is not a failure: retrieval worked, but no case carried a
    # usable resolution note, so similar cases are returned with no steps.
    if not state.evidence:
        state.mode = "evidence_only"
        state.verification = V.VerificationResult(
            steps=[],
            relevance_explanation=(
                "Similar cases were retrieved, but none carried usable resolution "
                "notes, so no steps are suggested."
                if state.retrieval.get("results")
                else "No sufficiently similar historical case was found."
            ),
            insufficient_evidence=True,
            manual_review_required=True,
            warnings=[],
        )
        state.transition("VERIFYING")
        return compose_brief(state, s)

    # --- stage 5: suggest -----------------------------------------------------
    state.transition("SUGGESTING")
    generated: GeneratedResolution | None = None

    provider: LLM.LLMProvider = (provider_factory or (lambda: LLM.build_provider(s)))()
    want_llm = (
        generation_allowed
        and provider.enabled
        and request.force_mode != "deterministic"
    )

    with _stage(state, "suggest") as ctx:
        if not generation_allowed:
            state.mode = "deterministic"
            generated = suggest_deterministic(state)
            ctx.status = "skipped"
            ctx.summary = (
                "Weak retrieval: generation skipped, zero provider calls. "
                "Similar cases are returned as evidence."
            )
        elif want_llm:
            try:
                generated = suggest_llm(state, provider, s)
                state.mode = "llm"
                ctx.summary = (
                    f"{len(generated.suggested_steps)} step(s) generated from "
                    f"{len(state.evidence)} cases"
                )
            except LLM.LLMError as exc:
                state.mode = "deterministic"
                generated = suggest_deterministic(state)
                ctx.status = "degraded"
                ctx.summary = "provider failed; deterministic fallback used"
                ctx.warnings.append(f"Generation fell back to deterministic mode: {exc}")
        else:
            state.mode = "deterministic"
            generated = suggest_deterministic(state)
            ctx.summary = f"deterministic rendering of {len(state.evidence)} cases"
        state.warnings.extend(ctx.warnings)

    state.generated = generated

    # --- stage 6: verify ------------------------------------------------------
    state.transition("VERIFYING")
    with _stage(state, "verify") as ctx:
        result = V.verify(generated, [t.ticket_id for t in state.evidence], state.strength)
        state.verification = result
        ctx.warnings.extend(result.warnings)
        ctx.summary = (
            f"{len(result.steps)} step(s) survived; "
            f"{len(result.dropped_citations)} citation(s) and "
            f"{result.dropped_steps} step(s) dropped"
        )
        if result.rejected and state.mode == "llm":
            ctx.status = "degraded"
        state.warnings.extend(result.warnings)

    # A rejected LLM result falls back rather than returning nothing.
    if state.verification.rejected and state.mode == "llm":
        state.mode = "deterministic"
        fallback = suggest_deterministic(state)
        state.verification = V.verify(
            fallback, [t.ticket_id for t in state.evidence], state.strength
        )
        state.warnings.append(
            "The generated result failed verification; the deterministic rendering "
            "of the retrieved cases is returned instead."
        )

    return compose_brief(state, s)
