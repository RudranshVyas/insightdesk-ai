# Architecture

## The controlling idea

Every subsystem is gated by a **capability manifest** generated from an audit of
the real dataset. Nothing downstream decides for itself whether it can run.

```
raw CSV ──► schema adapter ──► audit + data card ──► capability manifest
                                                            │
                     ┌──────────────────────────────────────┤ gates everything
                     ▼                                      ▼
                analytics API                    hybrid retrieval index
                                              FAISS IndexFlatIP + BM25, RRF
                                                            │
                                                            ▼
                                              Support Brief pipeline
```

Consequence: adding a feature means proving the data supports it, and a feature
whose data is absent turns itself off with a stated reason rather than returning
a zero.

---

## Offline vs request path

**Artifacts are built by offline scripts. Never at API startup, never during a
request.**

| Script | Writes |
|---|---|
| `ingest_tickets` | `data_audit.json/.md`, `data_card.json`, `tickets.parquet` |
| `build_capabilities` | `capabilities.json` |
| `build_retrieval_index` | `index.faiss`, `id_map.json`, `bm25.pkl`, `corpus.parquet`, `manifest.json` |
| `eval_retrieval` | `retrieval/evaluation.json`, `retrieval/baseline.json` |
| `eval_guardrails` | `guardrails/evaluation.json` |

The API loads what these write. It reads `capabilities.json` per request (small
file, an operator may regenerate it between requests) and lazily loads the
retriever once.

---

## Layers

```
backend/app/
├── core/          config, capabilities loader, redaction, guardrails,
│                  prompt registry, versions
├── db/            (SQLite; unused so far)
├── schemas/       Pydantic contracts — brief.py is the product surface
├── services/      audit, schema_adapter, data_card, capabilities,
│                  analytics, retrieval, text_utils, llm
├── orchestration/ pipeline.py (7 typed stages), verifier.py
├── evaluation/    metrics, retrieval_eval, guardrail_eval, regression_gate
├── prompts/       versioned prompt registry
└── api/           analytics, support_brief
```

`core/` holds nothing dataset-specific. `services/` never references a source
column name — that lives only in a mapping YAML.

---

## Two invariants that shape the design

### 1. The index holds the problem side only

A retrieval document is:

```
Product Area: {product_area}
Issue Type: {issue_type}
Issue: {issue_description}
```

Resolution notes, durations, escalation flags, SLA outcomes, and CSAT are
**never indexed**. A brand-new ticket has none of them, so indexing them would
match on information the query cannot possess. They are attached *after*
retrieval by `HybridRetriever.attach`.

Enforced by `assert_document_is_problem_side_only`, which checks **per row**:
ticket X's document must not contain ticket X's own resolution note. An earlier
global version false-positived on stock phrasing shared between an agent's reply
and a different customer's question — ordinary English, not leakage.

FAISS stores vectors and integer positions only, so a `faiss_pos → ticket_id`
list is maintained alongside it and both are written atomically.

### 2. Strength is computed by the backend, never by a model

`assess_strength` derives `strong` / `mixed` / `weak` from raw dense cosine,
top-1-to-top-3 margin, candidate count above a floor, and dense/BM25 rank
agreement — **not** from the fused RRF score, which no longer contains a cosine
once fused by rank.

Fusion uses RRF at `k=60`. One RRF unit is `1/61 ≈ 0.0164`, so a naive `+0.05`
metadata bonus would exceed the entire score range and let a product-area match
outrank genuine relevance. Metadata is a filter by preference; the optional boost
is a deliberate quarter of one unit.

The fused score is never displayed as a percentage similarity.

---

## The Support Brief pipeline

```
intake_and_redact → retrieve → gate → curate_evidence
                  → suggest → verify → compose_brief
```

State transitions are explicit: `INTAKE → RETRIEVING → GATED → CURATING →
SUGGESTING → VERIFYING → COMPOSED`. Every stage records latency, status, a
summary, and warnings. **No stage writes to any record.**

**Hand-rolled, deliberately.** The flow is linear with one conditional and no
loops, so a graph library would add a dependency and indirection without removing
control-flow complexity. A framework belongs where it earns its place — a capped
tool-calling loop with conditional routing and human interrupt — not here.

### The gate is policy, not an agent

| Strength | Behaviour |
|---|---|
| `weak` | skip generation entirely — **zero provider calls** |
| `mixed` | generate, but force `manual_review_required` |
| `strong` | generate |

### Evidence curation

Template siblings are dropped before the token budget is applied. Five
near-identical duplicates are worse evidence than three varied cases: they look
like corroboration while being one case counted five times.

### The verifier

Always runs. Never optional. Uses **no `assert`** — Python strips assertions
under `-O`, which would delete the guardrails in exactly the deployment mode
where they matter most.

- drops citation ids not in the evidence set
- drops steps left with no valid citation
- `weak` can never produce steps (backstop, independent of the gate)
- empty result rejected unless `insufficient_evidence` was declared
- scans output for PII and certainty language
- JSON parse failure → one bounded retry → deterministic fallback

Stated plainly in its own docstring: **ID validation proves the cited ticket was
in the evidence set. It does not prove the step is semantically supported.**

### Modes

`deterministic` (default, no key needed) · `llm` · `evidence_only` (retrieval
worked, no usable notes) · `disabled` (capability off, structured reason).

---

## Untrusted input

Ticket text and resolution notes are **data, never instructions**.

- redaction runs before storage, embedding, logging, tracing, display, prompting
- evidence is fenced with explicit markers and labelled untrusted in both the
  system and user prompt
- delimiter-escape sequences inside evidence are neutralised
- injection detected anywhere — query or evidence — escalates the **whole brief**
  to human review, not just the stage that found it

That last one was added because the Phase 7 evaluation caught its absence:
injection was detected and warned about while the brief came back marked as
needing no review.

The defence is structural — evidence lives in a data section, never an
instruction section. Detection is advisory and reduces risk without eliminating
it. "Not flagged" is never proof that text is benign.

---

## Versioning

`core/versions.py` is the single place every artifact version lives. Every
evaluation record, data card, manifest, and Support Brief stamps them, because an
unreproducible metric is not evidence.

Prompts are versioned files with metadata and a content hash. **A shipped version
is never edited in place** — an in-place edit silently invalidates every number
measured under the old wording.

---

## Evaluation

**Retrieval (Phase 5)** — Tier 1 is a leave-one-out automatic diagnostic that
excludes the query ticket *and its whole MinHash template group*, reports a
random-corpus baseline alongside every number, and is labelled a weak diagnostic
because the proxy label saturates. Tier 2 is a human-graded set; until it exists
the manifest reads `manual_set_not_yet_labeled` and no Hit@K is reported.

**Guardrails (Phase 7)** — 14 categories run through the *real* pipeline with
doubles only at retrieval and the provider. One metric,
`weak_retrieval_generation_violations`, is a hard gate rather than a score: "we
generate from weak evidence 3% of the time" is not a quality level, it is a
broken invariant.

**Rates over an empty denominator return `None`, not 0.** A disabled capability
reports `not_applicable`. A zero implies a measurement that did not happen.
