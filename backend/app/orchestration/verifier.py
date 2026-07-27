"""Phase 6.4 — the deterministic verifier.

This layer always runs and is never optional. A semantic critic may be added
alongside it, but may never replace it: an LLM judging an LLM is advisory, and
the guarantees below have to hold whether or not a model is reachable.

Stated plainly, because it is the single most over-claimed property in this kind
of system:

    **ID validation proves the cited ticket was in the evidence set.
    It does not prove the step is semantically supported by that ticket.**

`assert` is deliberately not used anywhere here. Python strips assertions under
`-O`, which would silently delete the guardrails in exactly the deployment mode
where they matter most.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from backend.app.core import guardrails as G
from backend.app.core import redaction as R
from backend.app.schemas.brief import GeneratedResolution, SuggestedStep


@dataclass
class VerificationResult:
    steps: list[SuggestedStep]
    relevance_explanation: str
    insufficient_evidence: bool
    manual_review_required: bool
    warnings: list[str] = field(default_factory=list)
    dropped_citations: list[str] = field(default_factory=list)
    dropped_steps: int = 0
    rejected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_count": len(self.steps),
            "insufficient_evidence": self.insufficient_evidence,
            "manual_review_required": self.manual_review_required,
            "dropped_citations": self.dropped_citations,
            "dropped_steps": self.dropped_steps,
            "rejected": self.rejected,
            "warnings": self.warnings,
        }


def verify(
    generated: GeneratedResolution,
    evidence_ids: Sequence[str],
    strength: str,
) -> VerificationResult:
    """Apply every deterministic guarantee to a model-produced resolution."""
    valid_ids = {str(t) for t in evidence_ids}
    warnings: list[str] = []
    dropped_citations: list[str] = []

    # --- 1. weak retrieval can never produce generated steps -----------------
    # Checked first and unconditionally. The gate should already have prevented
    # the provider call; this is the backstop that makes the property testable
    # rather than merely intended.
    if strength == "weak":
        return VerificationResult(
            steps=[],
            relevance_explanation="",
            insufficient_evidence=True,
            manual_review_required=True,
            warnings=[
                "Retrieval strength was weak, so no generated steps are returned. "
                "This is enforced after generation as well as before it."
            ],
            rejected=True,
        )

    # --- 2. citation validation ----------------------------------------------
    kept_steps: list[SuggestedStep] = []
    dropped_steps = 0

    for step in generated.suggested_steps:
        text = (step.text or "").strip()
        if not text:
            dropped_steps += 1
            continue

        good: list[str] = []
        for cited in step.citation_ticket_ids:
            cid = str(cited).strip()
            if cid in valid_ids:
                if cid not in good:
                    good.append(cid)
            else:
                # A fabricated id is the failure mode this whole layer exists
                # for. Record the exact value: it is the evidence that the
                # guardrail fired.
                dropped_citations.append(cid)

        if not good:
            dropped_steps += 1
            continue

        kept_steps.append(SuggestedStep(text=text, citation_ticket_ids=good))

    if dropped_citations:
        warnings.append(
            f"Dropped {len(dropped_citations)} citation(s) referring to ticket ids "
            f"that were not in the evidence set: {sorted(set(dropped_citations))[:5]}"
        )
    if dropped_steps:
        warnings.append(
            f"Dropped {dropped_steps} suggested step(s) left with no valid citation."
        )

    # --- 3. empty result is rejected unless abstention was declared ----------
    insufficient = bool(generated.insufficient_evidence)
    rejected = False
    if not kept_steps and not insufficient:
        rejected = True
        warnings.append(
            "The model returned no usable steps but did not set insufficient_evidence. "
            "Treating this as a failed generation rather than as an abstention."
        )
        insufficient = True

    # --- 4. PII and overclaiming scans ---------------------------------------
    explanation = (generated.relevance_explanation or "").strip()
    blob = "\n".join([s.text for s in kept_steps] + [explanation])

    pii = R.scan_pii(blob)
    pii.pop("url", None)
    if pii:
        warnings.append(f"Redacted PII detected in generated text: {sorted(pii)}")
        kept_steps = [
            SuggestedStep(text=R.redact_text(s.text), citation_ticket_ids=s.citation_ticket_ids)
            for s in kept_steps
        ]
        explanation = R.redact_text(explanation)

    overclaim = G.scan_overclaiming(blob)
    if overclaim.flagged:
        warnings.append(
            f"Generated text contains certainty language ({', '.join(overclaim.labels)}). "
            f"A resolution suggestion may not promise an outcome; the analyst is told "
            f"to treat it as historical evidence."
        )

    injection = G.scan_injection(blob)
    if injection.flagged:
        warnings.append(
            f"Generated text echoes injection-like content from the evidence "
            f"({', '.join(injection.labels)}). Treat the output with suspicion."
        )

    # --- 5. mixed strength always forces human review ------------------------
    manual_review = strength != "strong" or rejected or bool(warnings)

    return VerificationResult(
        steps=kept_steps,
        relevance_explanation=explanation,
        insufficient_evidence=insufficient,
        manual_review_required=manual_review,
        warnings=warnings,
        dropped_citations=dropped_citations,
        dropped_steps=dropped_steps,
        rejected=rejected,
    )
