"""Ranking metrics.

Kept deliberately small and dependency-free so the numbers can be checked by
hand against the definitions below. Every function takes a ranked list of
document ids and a relevance lookup, and every one returns ``None`` when the
query has no relevant document at all — a query with nothing to find scores
nothing, and averaging a zero into the report would understate the system
against a measurement that was never possible.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def hit_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float | None:
    """1.0 if any relevant document appears in the top k."""
    if not _has_relevant(relevant):
        return None
    return 1.0 if any(relevant.get(d, 0) > 0 for d in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float | None:
    """Fraction of all known relevant documents retrieved in the top k.

    Bounded by the size of the labeled relevant set, which is itself bounded by
    what a human graded. This is recall against the *labels*, not against the
    corpus, and the report says so.
    """
    total = sum(1 for v in relevant.values() if v > 0)
    if not total:
        return None
    found = sum(1 for d in ranked[:k] if relevant.get(d, 0) > 0)
    return found / total


def mrr_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float | None:
    """Reciprocal rank of the first relevant document, 0 if none in the top k."""
    if not _has_relevant(relevant):
        return None
    for i, d in enumerate(ranked[:k], start=1):
        if relevant.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, float], k: int) -> float | None:
    """Graded nDCG with the standard 2^rel - 1 gain and log2(i+1) discount."""
    if not _has_relevant(relevant):
        return None
    dcg = sum(
        (2 ** relevant.get(d, 0.0) - 1) / math.log2(i + 1)
        for i, d in enumerate(ranked[:k], start=1)
    )
    ideal = sorted((v for v in relevant.values() if v > 0), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return (dcg / idcg) if idcg > 0 else None


def _has_relevant(relevant: Mapping[str, float]) -> bool:
    return any(v > 0 for v in relevant.values())


def mean(values: Sequence[float | None]) -> float | None:
    """Average over the queries that were actually measurable.

    ``None`` results are dropped rather than counted as zero. If nothing was
    measurable the answer is ``None``, never ``0.0`` — the distinction between
    "scored badly" and "was not measured" is the whole point.
    """
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile. No interpolation, no numpy dependency."""
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[idx]
