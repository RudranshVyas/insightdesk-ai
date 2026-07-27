# InsightDesk AI

**An internal, human-supervised support intelligence platform.** It audits historical support tickets, builds a leakage-safe retrieval index over resolved cases, and composes an evidence-cited **Support Brief** for a human support analyst.

It is **not** a customer-facing chatbot, not an autonomous ticket resolver, and not permitted to modify any record. The language is always *resolution suggestion*, never *automatic resolution*.

---

## The one idea this project is built around

Most RAG demos answer the question *"can I retrieve something?"*. This one is built around a harder question: **what is this system allowed to claim?**

Every capability is gated by a **capability manifest** generated from an audit of the real dataset. If the data cannot support a feature, the feature disables itself and says why. It never returns a plausible-looking zero.

```jsonc
"risk": {
  "enabled": false,
  "target": null,
  "reason": "No reliable creation-time target in this dataset"
}
```

A disabled capability reads `not_applicable`, never `0` — a zero implies a measurement that did not happen.

---

## Status

Built and tested. Honest about what is not.

| Phase | State |
|---|---|
| 1 — Ingestion, schema adapter, audit, redaction | ✅ |
| 2 — Capability manifest | ✅ |
| 3 — Analytics (deterministic) | ✅ |
| 4 — Hybrid retrieval (dense + BM25 + RRF) | ✅ |
| 5 — Retrieval evaluation + CI regression gate | ✅ |
| 6 — Support Brief pipeline + verifier | ✅ |
| 7 — Guardrail evaluation suite | ⬜ not built |
| 8 — Clustering | ⬜ not built |
| 9 — Risk model | ⬜ not built |
| 10 — OpenTelemetry observability | ⬜ not built |
| 11 — LangGraph analyst agent | ⬜ not built |

**302 tests passing.** No metric in this README is asserted without a test or an artifact behind it.

---

## What is actually enforced

These are not aspirations in a design doc. Each one has a test that fails if it regresses.

**The retrieval index contains the problem side only.** Resolution notes, resolution time, escalation flags, SLA outcomes, and CSAT never enter the dense or lexical index — a brand-new ticket has none of them, so indexing them would match on information the query cannot possess. They are attached *after* retrieval. A guard raises if a resolution note appears in an indexed document.

**Retrieval strength is computed by the backend, never by a model.** It is a three-valued label (`strong` / `mixed` / `weak`) derived from raw dense cosine, top-1-to-top-3 margin, candidate count above a floor, and dense/BM25 rank agreement. It is not a probability, not a percentage, and the fused RRF score is never displayed as a similarity.

**Weak retrieval makes zero provider calls.** Not "returns a low score" — the pipeline never reaches a provider. Asserted by a test that counts calls on a fake provider.

**The model cannot report confidence.** The output schema has exactly three fields. A test asserts the words *confidence*, *probability*, *certainty*, and *score* appear nowhere in the generated JSON schema.

**Every suggested step must cite a real evidence ticket.** The deterministic verifier drops any citation id not in the evidence set, drops any step left with no valid citation, and falls back to the deterministic rendering if nothing survives. Documented plainly in the code:

> ID validation proves the cited ticket was in the evidence set. It does not prove the step is semantically supported by that ticket.

**The verifier uses no `assert`.** Python strips assertions under `-O` — which would delete the guardrails in exactly the deployment mode where they matter most.

**Ticket text is data, never instructions.** Evidence is fenced, labeled untrusted, and delimiter-escape sequences inside it are neutralized. A committed red-team fixture exercises all eight injection categories and every PII shape the redactor claims to handle.

---

## The measurement rule

> Every component must be measurable against a no-component baseline. If removing it changes no metric, delete it and write down the negative result.

A worked example from this repo. The Phase 5 leave-one-out diagnostic reports:

```
config               Hit@3   Hit@5   MRR@5   rand@3   lift@3
bm25                 1.000   1.000   1.000    0.750    1.333
dense                1.000   1.000   1.000    0.750    1.333
hybrid               1.000   1.000   1.000    0.750    1.333
```

`Hit@3 = 1.000` looks like a triumph. It is not. The proxy label is `issue_type`, which has only a handful of values, so a **random draw from the corpus scores 0.750**. Real lift is 1.33×. The harness reports the baseline and the lift alongside the raw number and labels the metric as saturating, because a metric that flatters the system is worse than no metric.

That is why Tier 1 is labeled a *weak automatic diagnostic* and only the Tier 2 human-graded set may be quoted as retrieval quality. Until that set is graded, the manifest reads `evaluation_status: manual_set_not_yet_labeled` and the UI prints exactly that instead of a number.

---

## Architecture

```
raw CSV ──> schema adapter ──> audit + data card ──> capability manifest
                                                            │ gates everything below
        ┌───────────────────────────────────────────────────┤
        ▼                                                   ▼
   analytics API                            hybrid retrieval index
                                          (FAISS IndexFlatIP + BM25, RRF)
                                                            │
                                                            ▼
                                            Support Brief pipeline
   intake_and_redact → retrieve → gate → curate_evidence
                                → suggest → verify → compose_brief
```

The Support Brief pipeline is a **hand-rolled typed orchestrator**, deliberately. The flow is linear with one conditional and no loops, so a graph library would add a dependency and a layer of indirection without removing any control-flow complexity. A framework belongs where it earns its place — a capped tool-calling loop with conditional routing — not here.

### Pipeline modes

| Mode | When |
|---|---|
| `deterministic` | No API key. Retrieved resolution notes rendered with citations. **This is the default and the demo runs on it.** |
| `llm` | Provider configured *and* the strength gate allows generation |
| `evidence_only` | Retrieval worked but no case carried a usable resolution note |
| `disabled` | Capability off — structured reason, no fabricated payload |

---

## Run it

No API key needed. The whole application works at `LLM_PROVIDER=none`.

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r backend/requirements.txt
```

Place a ticket CSV in `data/raw/`, then:

```bash
python -m backend.scripts.ingest_tickets --csv data/raw/<file>.csv --suggest-mapping
```

Review the generated mapping — every value in it is a name-based guess. Then:

```bash
python -m backend.scripts.ingest_tickets --csv data/raw/<file>.csv --mapping backend/config/<mapping>.yaml
python -m backend.scripts.build_capabilities
python -m backend.scripts.build_retrieval_index
python -m backend.scripts.eval_retrieval
uvicorn backend.app.main:app --reload
```

Artifacts are built by offline scripts only — never at API startup, never during a request.

```bash
python -m pytest backend/tests -q
```

---

## The schema-adapter trap this caught

The auto-suggested mapping for the dataset guessed `sla_plan: sla_breached`. That is an **outcome** column being mapped to a **plan tier** — it would have fed a post-outcome field straight into the creation-time feature list for the risk model, which is precisely the leakage the whole design exists to prevent.

Every correction made during mapping review is recorded in the mapping file with its reason. No dataset column name appears anywhere else in the codebase.

---

## Deliberately not built

| Cut | Reason |
|---|---|
| PostgreSQL, Alembic | Static dataset, single user. SQLite is correct. |
| Kubernetes, microservices | No measured need. |
| Hosted vector DB, observability SaaS | A console exporter is free and truthful. |
| "Intake Agent", "Evidence Agent", "Policy Agent" | These are functions. Naming them agents is keyword-stuffing. |
| Live drift monitoring | The dataset is static. A drift dashboard over frozen data would be fabricated. |
| Four LLM providers | One good integration beats four superficial ones. |

---

## Stack

Python 3.11+, FastAPI, Pydantic v2, pandas, scikit-learn, sentence-transformers, faiss-cpu, rank-bm25, datasketch (MinHash), pytest.

---

## License and data

Code is this repository's own. **No dataset, index, embedding, model, or secret is committed** — `.gitignore` and a CI guard job both enforce it. The data card records the source, download date, and `raw_file_sha256` of whatever file was actually processed, and marks the licence `unverified` unless it was independently confirmed.

> Historical evidence, not a guaranteed resolution. Human review is required before any customer action.
