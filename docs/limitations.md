# Limitations

Everything this system cannot do, or can only do with a caveat. Written to be
read out loud in a review, not to be buried.

---

## The dataset

**It is synthetic.** Sourced from Kaggle (`tobiasbueck/multilingual-customer-support-tickets`),
generated rather than collected from a real helpdesk. This was a deliberate
choice after a first dataset failed audit, and the reasoning is worth stating:

Public genuine support-ticket corpora with usable free text essentially do not
exist, because in a real ticket the free text *is* the PII — customer names,
account numbers, what someone said when they were angry. Organisations release
the metadata and strip the text. The [UCI ServiceNow incident
log](https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log)
is a genuine ServiceNow export of 24,918 incidents whose documentation states
plainly: *"Attributes used to record textual information are not placed in this
log."* So the available options are real-but-textless, or text-rich-but-generated.

**A rejected dataset is documented, not hidden.** The first candidate
(`mirzayasirabdullah07/customer-support-tickets-dataset-200k-records`, 200K rows)
was rejected after `triage_dataset.py` measured **10 distinct issue descriptions
across 50,000 sampled rows — a uniqueness ratio of 0.0002**. Retrieval over ten
documents is not retrieval. The current dataset scores 0.9958 on the same
measure.

**`resolution_notes` is an approximation.** The source column is `answer`, the
*first agent response*. Some first responses resolve the ticket; others are an
acknowledgement — one retrieved step reads *"Further details are required to
provide assistance."* The boilerplate filter removes the emptiest, but the
approximation itself cannot be filtered away and is disclosed instead.

**The corpus is English-only by construction.** The source is DE/EN/ES/PT/FR;
a declarative filter in the mapping keeps English. `all-MiniLM-L6-v2` is
English-centric, so mixing languages into one index degrades retrieval and
invalidates gate thresholds calibrated on English cosine distributions. The
filter, its row count, and its reason are recorded in the audit.

**Ticket ids are synthesized.** The export has no id column. Ids are derived
from row position and are stable only for the exact file recorded in the data
card's `raw_file_sha256`. They match nothing in any originating system.

**The licence is unverified.** The data card says so. It was not independently
confirmed, so it is not claimed.

---

## What the data cannot support

Six capabilities disable themselves on this dataset. That is the manifest
working, not a shortfall — but it does mean the following are **unbuilt and
undemonstrated**, not merely untested:

| Capability | Why it is off |
|---|---|
| `timeseries` | no `created_at` — the export has no timestamps at all |
| `resolution_time` | no durations |
| `response_time` | no first-response timestamp |
| `csat` | no rating column |
| `escalation_rate` | no escalation outcome |
| `risk` (Phase 9) | no creation-time-valid target survived the ladder |

Analytics is therefore reduced to `ticket_volume`. The analytics API works; there
is very little for it to compute.

---

## Retrieval

**No retrieval quality number exists yet.** The Tier 2 human-graded query set has
not been built, so `retrieval.evaluation_status` reads
`manual_set_not_yet_labeled` and the system reports no Hit@K anywhere. Do not
quote one.

**The Tier 1 automatic diagnostic saturates and must not be read as quality.**
It uses same-`issue_type` as a proxy label. With only a handful of type values, a
*random* draw from the corpus scores ~0.75 — so a Hit@3 of 1.000 represents about
1.33× lift, not near-perfect retrieval. The harness reports the random baseline
and the lift alongside the raw number for exactly this reason.

**Gate thresholds are uncalibrated.** `strong` / `mixed` / `weak` boundaries are
conservative defaults labelled `uncalibrated` in `config.py` and in the index
manifest. They were not tuned against a labeled set, because no labeled set
exists. Changing the embedding model invalidates them entirely.

**11 source rows were excluded as corrupt.** Their `answer` text was duplicated
into the customer-message field, so the "question" was written in agent voice and
already contained the resolution. Indexing them would let retrieval match on
outcome text a real incoming ticket cannot possess.

---

## Generation and guardrails

**ID validation is not semantic validation.** The verifier proves a cited ticket
was in the evidence set. It does **not** prove the step is actually supported by
that ticket. This is the single most over-claimed property in systems of this
kind and it is stated in the verifier's own docstring.

**No LLM path has been exercised against a live provider.** Every generation
measurement in `artifacts/guardrails/evaluation.json` comes from a *scripted*
provider. Token usage and cost therefore report `not_applicable`, never `0`.
Latency figures from that suite measure the pipeline, not a model.

**Injection detection reduces risk; it does not eliminate it.** Evidence is
fenced, labelled untrusted, and delimiter-escape sequences are neutralised — but
the defence is structural (evidence lives in a data section, never an instruction
section), not a claim that the regex catches everything. A "not flagged" result
is not proof that text is benign.

**The guardrail suite is 14 hand-authored cases.** It covers every category the
spec names, and every case passes. It is not a statistical sample of real
adversarial traffic, and the rates should be read as "these specific known
failure modes are handled" rather than as a population estimate.

---

## Not built

Phases 8 (clustering), 9 (risk model), 10 (OpenTelemetry), and 11 (LangGraph
analyst agent) are **not implemented**. The README status table says so. No
metric, screenshot, or claim anywhere in this repository refers to them.

There is no frontend. The API is the interface.

---

## Operational

**Cold start is ~20 seconds.** The embedding model loads lazily on first query.
Warm the service before any live demo.

**Traces are in-memory and bounded to the last 200 requests.** They do not
survive a restart. They carry operational summaries only — no raw ticket text,
no resolution notes, no prompts, no provider responses.

**There is no authentication, rate limiting, or multi-tenancy.** This is an
internal single-user tool as specified, not a service.
