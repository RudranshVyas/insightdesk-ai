"""Phase 6 — Support Brief contracts.

The shape of these models is a product decision, not a serialization detail.

Three things are deliberately absent and must stay absent:

* **No confidence field on anything the model produces.** A language model
  asserting its own reliability is not evidence. `retrieval_strength` is
  computed by the backend from raw cosine and rank agreement, and it is a
  three-valued label, never a percentage.
* **No raw RRF score presented as similarity.** The fusion score is a rank
  artifact; showing it as "87% match" would be a fabricated measurement.
* **No chain-of-thought.** `stage_trace` carries operational summaries only.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Mode = Literal["deterministic", "llm", "evidence_only", "disabled"]
Strength = Literal["strong", "mixed", "weak"]
StageStatus = Literal["ok", "skipped", "failed", "degraded"]


class SupportBriefRequest(BaseModel):
    issue_description: str = Field(min_length=1)
    product_area: str | None = None
    issue_type: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    # Lets a demo show the deterministic path even when a key is configured.
    force_mode: Literal["deterministic", "llm"] | None = None

    @field_validator("issue_description")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("issue_description must not be blank")
        return v


class SuggestedStep(BaseModel):
    text: str
    # Every step must cite at least one evidence ticket. The verifier drops any
    # step that ends up with none, so an empty list here is a transient state
    # inside verification, never a value that reaches a client.
    citation_ticket_ids: list[str] = Field(default_factory=list)


class GeneratedResolution(BaseModel):
    """Exactly what the provider is asked to return. Note the absent field."""

    suggested_steps: list[SuggestedStep] = Field(default_factory=list)
    relevance_explanation: str = ""
    insufficient_evidence: bool = False


class EvidenceTicket(BaseModel):
    ticket_id: str
    issue_subject: str | None = None
    issue_excerpt: str
    resolution_notes: str | None = None
    product_area: str | None = None
    issue_type: str | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    dense_cosine: float | None = None
    fusion_rank: int | None = None
    matched_metadata: dict[str, bool] = Field(default_factory=dict)
    template_group_id: int | None = None
    # Outcome fields attached AFTER retrieval. They were never indexed.
    resolution_time_hours: float | None = None
    escalated: bool | None = None
    sla_breached: bool | None = None
    csat_score: float | None = None
    injection_flags: list[str] = Field(default_factory=list)


class RetrievalStrength(BaseModel):
    strength: Strength
    top_cosine: float | None = None
    margin: float | None = None
    candidates_above_floor: int = 0
    rank_agreement: int = 0
    calibrated: bool = False
    reasons: list[str] = Field(default_factory=list)
    note: str = (
        "Backend-computed gate over dense cosine and rank agreement. Not a "
        "probability, not a similarity percentage, not produced by a language model."
    )


class RiskSignal(BaseModel):
    """Null whenever the risk capability is disabled — never a zero."""

    score: float
    target: str
    target_kind: str
    threshold: float
    above_threshold: bool
    calibrated: bool
    caveat: str | None = None


class StageTrace(BaseModel):
    name: str
    status: StageStatus
    latency_ms: float
    summary: str
    warnings: list[str] = Field(default_factory=list)


class VersionStamp(BaseModel):
    artifact: dict[str, int]
    index_version: int | None = None
    index_data_hash: str | None = None
    embedding_model: str | None = None
    prompt_version: str | None = None
    provider: str | None = None
    provider_model: str | None = None


class SupportBrief(BaseModel):
    request_id: str
    mode: Mode
    retrieval_strength: Strength
    strength_detail: RetrievalStrength | None = None
    similar_cases: list[EvidenceTicket] = Field(default_factory=list)
    suggested_steps: list[SuggestedStep] = Field(default_factory=list)
    relevance_explanation: str | None = None
    risk_signal: RiskSignal | None = None
    manual_review_required: bool = True
    insufficient_evidence: bool = False
    warnings: list[str] = Field(default_factory=list)
    stage_trace: list[StageTrace] = Field(default_factory=list)
    versions: VersionStamp
    disclaimer: str = (
        "Historical evidence, not a guaranteed resolution. Human review required "
        "before any customer action."
    )
    capability_reason: str | None = None


class CapabilityDisabledBrief(BaseModel):
    """Returned instead of a brief when the subsystem is off. No fake payload."""

    request_id: str
    mode: Literal["disabled"] = "disabled"
    capability: str
    reason: str
    detail: str = (
        "This capability is disabled because the dataset or configuration does not "
        "support it. No placeholder result is returned in its place."
    )


class TraceResponse(BaseModel):
    request_id: str
    mode: Mode
    retrieval_strength: Strength
    stage_trace: list[StageTrace]
    versions: VersionStamp
    provider_calls: int
    note: str = (
        "Operational summaries only. Raw ticket text, resolution notes, customer "
        "identifiers, full prompts, and provider responses are never recorded here."
    )


def brief_extra(**kwargs: Any) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}
