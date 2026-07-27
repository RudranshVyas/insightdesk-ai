## What changed

<!-- One paragraph. What does this do, and why now? -->

## Assumptions

<!-- State them explicitly. An unstated assumption is a defect waiting for a reader. -->

## Checklist

- [ ] No fabricated schema fields or metrics
- [ ] Capability gating preserved (disabled reports a reason, never a zero)
- [ ] No PII in logs or traces
- [ ] No resolution text or outcome field in the retrieval index
- [ ] No post-outcome features added to the T0 list
- [ ] No guardrail implemented with `assert`
- [ ] Tests added; quality gates pass
- [ ] Failure and fallback paths tested, not just the happy path
- [ ] AI-generated code manually reviewed line by line
