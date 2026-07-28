# Claims register

Every claim that may be made about this project, with the artifact that proves
it. **Anything not in this table does not get said** — not on a résumé, not in a
review, not in an interview.

Measured on the merged Kaggle corpus (`tickets_merged.csv`,
`raw_file_sha256` recorded in `artifacts/data_card.json`) unless noted.

---

## Data and auditing

| Claim | Evidence | Metric | Caveat |
|---|---|---|---|
| Rejected a dataset on measured grounds before building on it | `backend/scripts/triage_dataset.py`; run it against the 200K set | 10 distinct issue descriptions / 50,000 rows, ratio **0.0002** | The rejection is the artifact. Keep the command runnable. |
| Selected a replacement that clears a uniqueness bar | `triage_dataset.py` on `tickets_merged.csv` | `body` ratio **0.9958**, `answer` ratio **0.9958**, mean 428 chars | Bar is a judgement call, recorded in the script. |
| Recovered 53% more corpus by auditing every file in the export | `merge_ticket_csvs.py`, `test_merge_csvs.py` | 16,334 → **25,052** English rows; 20k file overlapped 38.5%, 4k file 0% | Found only because the files were profiled individually. |
| Schema mapping is config-driven; no dataset column name appears in service code | `backend/config/schema_map.*.yaml`; grep the codebase | — | Enforced by convention and review, not by a test. |
| Every mapping correction is recorded with its reason | `schema_map.kaggle.yaml` header | `sla_plan: sla_breached` rejected — an outcome mapped to a plan tier | — |
| PII is redacted before storage, embedding, logging, tracing, display, and prompting | `test_redteam_fixture.py` (18 tests) | 0 raw PII markers survive the adapter | Regex-based; not a claim of completeness. |
| Adversarial fixture exercises 8 injection categories and every PII shape the redactor claims | `data/fixtures/redteam_tickets.csv` | 16 rows, all 8 categories asserted present | — |
| Corpus restrictions are declarative and auditable | `filters:` block; `test_row_filters.py` | 19,134 rows dropped by the language filter, recorded with reason | — |

## Retrieval

| Claim | Evidence | Metric | Caveat |
|---|---|---|---|
| Hybrid retrieval over 25,036 resolved cases | `artifacts/retrieval/manifest.json` | FAISS `IndexFlatIP` 25036×384 + BM25, fused by RRF | — |
| The index contains the problem side only | `test_index_purity.py` (8 tests); `assert_document_is_problem_side_only` | per-row guard; 11 corrupt rows excluded | Guard was rewritten after it false-positived on stock phrasing. |
| Retrieval strength is computed by the backend, never by a model | `assess_strength` in `retrieval.py`; `test_support_brief.py` | 3-valued label from raw cosine, margin, rank agreement | **Thresholds are uncalibrated.** Say so. |
| Retrieval improved measurably with corpus size | live query, before/after | dashboard query: `mixed` @ 0.626 → **`strong` @ 0.696** | Single query pair, not a benchmark. |
| Evaluation harness compares BM25 / dense / RRF / hybrid+metadata | `backend/app/evaluation/retrieval_eval.py` | 4 configs, Hit@K, Recall@K, MRR, nDCG, p50/p95 | — |
| The automatic diagnostic reports its own random baseline | `artifacts/retrieval/evaluation.json` | Hit@3 1.000 vs random **0.750** → lift **1.33×** | This is the honest reading. Never quote the 1.000 alone. |
| CI fails on a retrieval metric regression | `regression_gate.py`, `test_regression_gate.py` (14 tests) | drop beyond tolerance, or a metric that stops being measured | — |

**Not claimable:** any retrieval *quality* number. The Tier 2 human-graded set
does not exist; the manifest reads `manual_set_not_yet_labeled`.

## Support Brief pipeline

| Claim | Evidence | Metric | Caveat |
|---|---|---|---|
| Typed 7-stage orchestrator with per-stage latency, status, and failure policy | `orchestration/pipeline.py`; `stage_trace` on every response | — | Hand-rolled deliberately; the flow is linear. |
| Fully functional with `LLM_PROVIDER=none` | `test_support_brief.py`; CI boot gate | deterministic mode is the default | — |
| Weak retrieval makes **zero** provider calls | `test_support_brief.py::test_weak_retrieval_makes_zero_provider_calls` | `provider.calls == 0`, counted on a fake provider | — |
| Fabricated citations are dropped and warned | same file; guardrail case G006 | invented id removed, step dropped, warning raised | — |
| The model cannot emit a confidence value | `test_generated_schema_has_no_confidence_field` | schema has 3 fields; "confidence"/"probability"/"score" absent | — |
| No guardrail uses `assert` | `verifier.py` docstring and source | Python strips assertions under `-O` | — |
| Four modes reachable: deterministic, llm, evidence_only, disabled | `test_support_brief.py`, `test_api_support_brief.py` | — | — |

## Guardrail evaluation (Phase 7)

| Claim | Evidence | Metric | Caveat |
|---|---|---|---|
| Categorized guardrail suite covering all 14 spec categories | `data/evaluation/guardrail_cases.jsonl`; `artifacts/guardrails/evaluation.json` | 14/14 categories | Hand-authored cases, not sampled traffic. |
| Zero weak-retrieval generation violations | same report | **0** — hard gate, fails CI | — |
| Citation validity | same report | **1.000** | Proves ids were in context, **not** semantic support. |
| Step citation coverage | same report | **1.000** | — |
| Abstention accuracy | same report | **1.000** | — |
| Injection resistance | same report | **1.000** | Reduces risk; does not eliminate it. |
| Deterministic fallback success | same report | **1.000** | — |
| PII leakage rate | same report | **0.000** | — |
| The suite found a real defect | commit "Phase 7: guardrail evaluation suite, and the defect it found" | injection detected but `manual_review_required` stayed False | **This is the strongest single claim here.** |

**Token usage and cost are `not_applicable`, never 0** — the provider is
scripted. No live-provider measurement exists.

## Engineering discipline

| Claim | Evidence | Metric | Caveat |
|---|---|---|---|
| 342 tests, lint clean | `pytest backend/tests`; `ruff check backend` | 342 passing | — |
| CI enforces quality gates and blocks dataset/secret commits | `.github/workflows/ci.yml` | guard job + 8 named gates | — |
| Fixed an O(n²) degeneration found by running real data | commit; `text_utils.template_groups` | 2006 CPU-seconds → **1.22s** at 200K rows | LSH degenerates when every row is a near-duplicate. |
| AI-agent working rules and PR checklist | `AGENTS.md`, `.github/pull_request_template.md` | — | — |

---

## Rejected with evidence

Recorded because a measured negative result is an asset, not an omission.

| Rejected | Why | Evidence |
|---|---|---|
| Kaggle 200K support-ticket dataset | uniqueness ratio 0.0002 — 10 distinct texts | `triage_dataset.py` output |
| UCI ServiceNow incident log | genuine tickets, but text explicitly stripped | dataset documentation |
| Customer Support on Twitter / Stack Exchange | real text, but not ticketing-system data | provenance judgement |
| Global-scan index purity guard | false-positived on stock phrasing shared across tickets | `test_index_purity.py::test_shared_stock_phrases...` |

---

## Résumé language

Permitted, because each maps to a row above:

> Built a support intelligence platform whose capability manifest, generated from
> a dataset audit, disables any feature the data cannot support instead of
> fabricating results — six capabilities disable themselves with stated reasons
> on the shipped corpus.
>
> Implemented leakage-safe hybrid retrieval (FAISS + BM25 fused by RRF) over
> 25,036 resolved cases, indexing problem-side text only and attaching outcome
> fields after retrieval, with a per-row purity guard in CI.
>
> Designed a typed Support Brief orchestrator with backend-computed
> retrieval-strength gating, per-step citation validation, and a fully functional
> deterministic mode requiring no LLM key; weak retrieval provably makes zero
> provider calls.
>
> Built a 14-category guardrail evaluation suite that measured the guardrails
> rather than asserting them — and found a real defect, where prompt injection
> was detected but failed to escalate the brief for human review.
>
> Rejected the initial dataset on measured grounds (uniqueness ratio 0.0002) and
> built the triage tool that caught it.

**Not permitted:** any retrieval quality figure, any token or cost figure, any
business outcome, and any reference to clustering, risk modelling,
OpenTelemetry, or an agent — none of which are built.
