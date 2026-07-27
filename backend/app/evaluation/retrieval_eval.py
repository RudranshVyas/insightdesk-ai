"""Phase 5 — retrieval evaluation.

Two tiers, and the difference between them is the honest part.

**Tier 1 — automatic diagnostic.** Hold out a resolved ticket, query with its own
issue text, and check whether the top-k contains another ticket of the same
``issue_type``. This is a *weak* signal: same ``issue_type`` is a coarse category,
not ground truth, and it is labeled as such everywhere it is reported. Its value
is regression detection, not quality measurement.

The trap this module exists to avoid: on templated support data, a ticket's own
near-duplicate siblings sit at the top of every result list. Counting those as
hits produces a Hit@5 near 1.0 that measures nothing. Both the query ticket and
its whole MinHash template group are excluded from its own candidate pool.

**Tier 2 — manual labeled set.** Human-graded queries in
``data/evaluation/retrieval_queries.jsonl``. This is the only tier whose numbers
may be quoted as retrieval quality. When the file does not exist, the report says
``not_yet_labeled`` and carries no metrics at all.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.evaluation import metrics as M

# The four configurations the spec requires comparing. `hybrid` is the shipped
# default; the other three exist so a claim that fusion helps is falsifiable.
CONFIGS: tuple[str, ...] = ("bm25", "dense", "hybrid", "hybrid_metadata")

K_VALUES: tuple[int, ...] = (3, 5)


def config_kwargs(config: str) -> dict[str, Any]:
    if config == "bm25":
        return {"mode": "lexical", "metadata_boost": False}
    if config == "dense":
        return {"mode": "dense", "metadata_boost": False}
    if config == "hybrid":
        return {"mode": "hybrid", "metadata_boost": False}
    if config == "hybrid_metadata":
        return {"mode": "hybrid", "metadata_boost": True}
    raise ValueError(f"unknown retrieval config: {config!r}")


# --- Tier 2: the labeled set --------------------------------------------------


@dataclass
class LabeledQuery:
    query_id: str
    text: str
    product_area: str | None = None
    issue_type: str | None = None
    # ticket_id -> graded relevance. 0 means judged and not relevant, which is
    # information; an absent id means never judged, which is not.
    relevance: dict[str, float] = field(default_factory=dict)
    notes: str | None = None

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> LabeledQuery:
        rel_raw = obj.get("relevance") or {}
        return cls(
            query_id=str(obj["query_id"]),
            text=str(obj["text"]),
            product_area=obj.get("product_area"),
            issue_type=obj.get("issue_type"),
            relevance={str(k): float(v) for k, v in rel_raw.items()},
            notes=obj.get("notes"),
        )


def load_labeled_queries(path: Path) -> list[LabeledQuery]:
    if not path.exists():
        return []
    out: list[LabeledQuery] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                out.append(LabeledQuery.from_json(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: malformed labeled query: {exc}") from exc
    return out


def validate_labeled_set(
    queries: Sequence[LabeledQuery], corpus: pd.DataFrame
) -> dict[str, Any]:
    """Structural checks the spec demands before a number may be quoted.

    A query that retrieves itself, or whose relevant set straddles a template
    group with the query, is not measuring retrieval — it is measuring
    deduplication. Both are reported as errors, not silently tolerated.
    """
    corpus_ids = set(corpus["ticket_id"].astype(str))
    groups = (
        corpus.set_index(corpus["ticket_id"].astype(str))["template_group_id"].to_dict()
        if "template_group_id" in corpus.columns
        else {}
    )

    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    for q in queries:
        if q.query_id in seen_ids:
            errors.append(f"{q.query_id}: duplicate query_id")
        seen_ids.add(q.query_id)

        if not q.text.strip():
            errors.append(f"{q.query_id}: empty query text")
        if not any(v > 0 for v in q.relevance.values()):
            warnings.append(f"{q.query_id}: no positively relevant ticket; will be skipped")

        unknown = sorted(set(q.relevance) - corpus_ids)
        if unknown:
            errors.append(
                f"{q.query_id}: graded ticket ids absent from the corpus: {unknown[:5]}"
            )

        if q.query_id in corpus_ids:
            errors.append(
                f"{q.query_id}: query_id is also a corpus ticket_id; a query must not "
                f"be able to retrieve itself"
            )

        if groups:
            judged_groups = {
                groups.get(tid) for tid, v in q.relevance.items() if v > 0 and tid in groups
            }
            judged_groups.discard(None)
            if len(judged_groups) == 1 and sum(1 for v in q.relevance.values() if v > 0) > 2:
                warnings.append(
                    f"{q.query_id}: every relevant ticket sits in one template group; "
                    f"this query rewards duplicate retrieval rather than relevance"
                )

    return {
        "query_count": len(queries),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def evaluate_labeled_set(
    retriever: Any,
    queries: Sequence[LabeledQuery],
    config: str,
    top_k: int = 10,
) -> dict[str, Any]:
    kwargs = config_kwargs(config)
    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []

    for q in queries:
        t0 = time.perf_counter()
        res = retriever.search(
            q.text,
            product_area=q.product_area,
            issue_type=q.issue_type,
            top_k=top_k,
            **kwargs,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ranked = [r["ticket_id"] for r in res["results"]]
        per_query.append(
            {
                "query_id": q.query_id,
                "ranked": ranked,
                "strength": res["strength"]["strength"],
                "scores": _score_one(ranked, q.relevance),
            }
        )

    return _aggregate(config, per_query, latencies, tier="manual_labeled")


def _score_one(ranked: Sequence[str], relevance: dict[str, float]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for k in K_VALUES:
        scores[f"hit_at_{k}"] = M.hit_at_k(ranked, relevance, k)
        scores[f"recall_at_{k}"] = M.recall_at_k(ranked, relevance, k)
        scores[f"mrr_at_{k}"] = M.mrr_at_k(ranked, relevance, k)
        scores[f"ndcg_at_{k}"] = M.ndcg_at_k(ranked, relevance, k)
    return scores


def _aggregate(
    config: str,
    per_query: list[dict[str, Any]],
    latencies: list[float],
    tier: str,
) -> dict[str, Any]:
    metric_names = [
        f"{name}_at_{k}" for k in K_VALUES for name in ("hit", "recall", "mrr", "ndcg")
    ]
    aggregated = {
        name: M.mean([pq["scores"].get(name) for pq in per_query]) for name in metric_names
    }
    strengths = [pq.get("strength") for pq in per_query if pq.get("strength")]
    return {
        "config": config,
        "tier": tier,
        "queries_run": len(per_query),
        "queries_scored": sum(
            1 for pq in per_query if any(v is not None for v in pq["scores"].values())
        ),
        "metrics": {k: (round(v, 4) if v is not None else None) for k, v in aggregated.items()},
        "latency_ms": {
            "p50": _round(M.percentile(latencies, 50)),
            "p95": _round(M.percentile(latencies, 95)),
            "mean": _round(M.mean(latencies)),
        },
        "strength_distribution": {
            s: strengths.count(s) for s in ("strong", "mixed", "weak") if strengths.count(s)
        },
        "per_query": per_query,
    }


def _round(v: float | None) -> float | None:
    return round(v, 2) if v is not None else None


# --- Tier 1: leave-one-out diagnostic ----------------------------------------


def evaluate_leave_one_out(
    retriever: Any,
    corpus: pd.DataFrame,
    config: str,
    sample_size: int = 200,
    top_k: int = 10,
    seed: int = 20260728,
) -> dict[str, Any]:
    """Weak automatic diagnostic. Never quote these as retrieval quality.

    Self-matches and MinHash template siblings are removed from each query's own
    candidate pool before scoring. Without that exclusion the metric on templated
    data is a duplicate-detection score wearing a relevance label.
    """
    if "issue_type" not in corpus.columns:
        return {
            "config": config,
            "tier": "leave_one_out",
            "available": False,
            "reason": "no issue_type column; the proxy label does not exist",
        }

    usable = corpus[corpus["issue_type"].notna() & (corpus["issue_type"].astype(str) != "")]
    if usable.empty:
        return {
            "config": config,
            "tier": "leave_one_out",
            "available": False,
            "reason": "no ticket carries an issue_type value",
        }

    rng = random.Random(seed)
    idx = list(range(len(usable)))
    rng.shuffle(idx)
    chosen = usable.iloc[idx[: min(sample_size, len(idx))]]

    has_groups = "template_group_id" in corpus.columns
    kwargs = config_kwargs(config)
    type_by_id = corpus.set_index(corpus["ticket_id"].astype(str))["issue_type"].to_dict()

    all_ids = corpus["ticket_id"].astype(str).tolist()
    per_query: list[dict[str, Any]] = []
    random_per_query: list[dict[str, Any]] = []
    latencies: list[float] = []
    excluded_total = 0

    for row in chosen.to_dict("records"):
        tid = str(row["ticket_id"])
        query_text = str(row.get("issue_text") or row.get("retrieval_document") or "")
        if not query_text.strip():
            continue

        exclude_groups: set[int] = set()
        if has_groups and row.get("template_group_id") is not None:
            exclude_groups = {int(row["template_group_id"])}

        t0 = time.perf_counter()
        res = retriever.search(
            query_text,
            top_k=top_k,
            exclude_ticket_ids={tid},
            exclude_template_groups=exclude_groups or None,
            **kwargs,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

        ranked = [r["ticket_id"] for r in res["results"]]
        excluded_total += 1  # the ticket itself, always
        own_type = str(row["issue_type"])

        per_query.append(
            {
                "query_id": tid,
                "ranked": ranked,
                "strength": res["strength"]["strength"],
                "scores": _proxy_scores(ranked, own_type, type_by_id),
            }
        )

        # Baseline: the same metric computed over a random draw from the corpus.
        # Without it a saturated Hit@K looks like success when it is only the
        # base rate of a coarse category with a handful of values.
        shuffled = rng.sample(all_ids, min(top_k, len(all_ids)))
        random_per_query.append(
            {"query_id": tid, "scores": _proxy_scores(shuffled, own_type, type_by_id)}
        )

    out = _aggregate(config, per_query, latencies, tier="leave_one_out")
    baseline = _aggregate(f"{config}__random_baseline", random_per_query, [], "leave_one_out")

    out["available"] = True
    out["random_baseline"] = {
        "metrics": baseline["metrics"],
        "note": (
            "Same proxy label, but the top-k is drawn uniformly at random from the "
            "corpus. This is the no-retrieval floor the governing measurement rule "
            "requires. Retrieval must beat it or the component is not earning its keep."
        ),
    }
    out["lift_over_random"] = {
        name: _lift(out["metrics"].get(name), baseline["metrics"].get(name))
        for name in out["metrics"]
    }
    out["exclusions"] = {
        "self_excluded": excluded_total,
        "template_siblings_excluded": bool(has_groups),
        "note": (
            "Both the query ticket and its whole MinHash template group were removed "
            "from its candidate pool. Without this, near-duplicate retrieval inflates "
            "Hit@K into a meaningless number on templated data."
        ),
    }
    out["interpretation"] = (
        "WEAK AUTOMATIC DIAGNOSTIC. Same issue_type is a coarse proxy, not ground "
        "truth: a correct retrieval of a different issue_type scores zero here, and "
        "an irrelevant ticket of the same type scores one. With only a handful of "
        "issue_type values this metric SATURATES near 1.0 — read lift_over_random, "
        "not the raw number. Use for regression detection only. Retrieval quality "
        "claims must come from the Tier 2 human-labeled set."
    )
    # Per-query detail is large and adds nothing to a diagnostic; drop it.
    out.pop("per_query", None)
    return out


def _proxy_scores(
    ranked: Sequence[str], own_type: str, type_by_id: dict[str, Any]
) -> dict[str, float | None]:
    """Score a ranking against the coarse same-issue_type proxy.

    Recall is deliberately left ``None``. The relevant set here is "every ticket
    in the corpus with this issue_type" — thousands of them — so recall against
    a top-10 is bounded near zero and carries no information. Reporting a number
    there would invite it to be read as a quality score.
    """
    relevance = {rid: 1.0 for rid in ranked if str(type_by_id.get(rid, "")) == own_type}
    scores = _score_one(ranked, relevance) if relevance else _unscored()
    for k in K_VALUES:
        scores[f"recall_at_{k}"] = None
    return scores


def _lift(measured: float | None, baseline: float | None) -> float | None:
    if measured is None or baseline is None or baseline <= 0:
        return None
    return round(measured / baseline, 3)


def _unscored() -> dict[str, float | None]:
    return dict.fromkeys(_score_one([], {"x": 1.0}))


# --- report assembly ----------------------------------------------------------


def build_report(
    retriever: Any,
    corpus: pd.DataFrame,
    labeled_path: Path,
    configs: Iterable[str] = CONFIGS,
    loo_sample: int = 200,
    index_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from datetime import datetime

    from backend.app.core.versions import version_stamp

    configs = list(configs)
    queries = load_labeled_queries(labeled_path)
    validation = validate_labeled_set(queries, corpus) if queries else None

    tier1 = {c: evaluate_leave_one_out(retriever, corpus, c, loo_sample) for c in configs}

    if not queries:
        tier2: dict[str, Any] = {
            "status": "not_yet_labeled",
            "path": str(labeled_path),
            "detail": (
                "No human-graded query set exists yet, so no retrieval quality metric "
                "is reported. The UI displays 'not yet labeled'. A fabricated Hit@K "
                "is worse than an absent one."
            ),
            "results": {},
        }
    elif not validation["valid"]:
        tier2 = {
            "status": "invalid_labeled_set",
            "path": str(labeled_path),
            "validation": validation,
            "detail": "The labeled set failed validation; no metrics were computed.",
            "results": {},
        }
    else:
        tier2 = {
            "status": "evaluated",
            "path": str(labeled_path),
            "validation": validation,
            "results": {c: evaluate_labeled_set(retriever, queries, c) for c in configs},
        }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "versions": version_stamp(),
        "index": {
            "embedding_model": (index_manifest or {}).get("embedding_model"),
            "embedding_model_revision": (index_manifest or {}).get("embedding_model_revision"),
            "data_hash": (index_manifest or {}).get("data_hash"),
            "corpus_size": (index_manifest or {}).get("corpus_size"),
            "index_version": (index_manifest or {}).get("index_version"),
        },
        "configs_compared": configs,
        "tier1_leave_one_out": tier1,
        "tier2_manual_labeled": tier2,
        "reading_guide": (
            "Tier 1 is a regression tripwire. Tier 2 is the only tier whose numbers "
            "describe retrieval quality. If BM25 alone wins on Tier 2, that is the "
            "result and it is reported as such."
        ),
    }


def summarize_winner(report: dict[str, Any]) -> dict[str, Any]:
    """Which config actually won, by Tier 2 Recall@5. Honest about ties."""
    tier2 = report.get("tier2_manual_labeled", {})
    if tier2.get("status") != "evaluated":
        return {"decided": False, "reason": tier2.get("status", "unknown")}

    scored = {
        c: r["metrics"].get("recall_at_5")
        for c, r in tier2["results"].items()
        if r["metrics"].get("recall_at_5") is not None
    }
    if not scored:
        return {"decided": False, "reason": "no config produced a Recall@5"}

    best = max(scored, key=lambda c: scored[c])
    tied = [c for c, v in scored.items() if abs(v - scored[best]) < 1e-9]
    return {
        "decided": True,
        "metric": "recall_at_5",
        "scores": scored,
        "best": best,
        "tied_with": [c for c in tied if c != best],
        "note": (
            "A tie means the extra machinery bought nothing measurable. Per the "
            "governing measurement rule, prefer the simpler config and record the "
            "negative result."
        ),
    }
