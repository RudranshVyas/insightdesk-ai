"""Ranking metrics, checked against hand-computed values.

Every expected number here is derived from the definition by hand, not from a
library. If a metric silently changes, this file is what notices.
"""

from __future__ import annotations

import math

import pytest

from backend.app.evaluation import metrics as M

# Ranking: a b c d e. Relevant: b (rank 2) and d (rank 4).
RANKED = ["a", "b", "c", "d", "e"]
REL = {"b": 1.0, "d": 1.0}


# --- hit ---------------------------------------------------------------------


def test_hit_at_1_misses_when_top_is_irrelevant() -> None:
    assert M.hit_at_k(RANKED, REL, 1) == 0.0


def test_hit_at_2_catches_the_first_relevant() -> None:
    assert M.hit_at_k(RANKED, REL, 2) == 1.0


def test_hit_is_binary_not_a_count() -> None:
    assert M.hit_at_k(RANKED, REL, 5) == 1.0


# --- recall -------------------------------------------------------------------


def test_recall_at_2_finds_one_of_two() -> None:
    assert M.recall_at_k(RANKED, REL, 2) == pytest.approx(0.5)


def test_recall_at_5_finds_both() -> None:
    assert M.recall_at_k(RANKED, REL, 5) == pytest.approx(1.0)


def test_recall_denominator_ignores_zero_graded_documents() -> None:
    """A document graded 0 was judged and found irrelevant. It is not a target."""
    rel = {"b": 1.0, "d": 1.0, "a": 0.0, "c": 0.0}
    assert M.recall_at_k(RANKED, rel, 5) == pytest.approx(1.0)


# --- mrr ----------------------------------------------------------------------


def test_mrr_uses_the_first_relevant_rank() -> None:
    assert M.mrr_at_k(RANKED, REL, 5) == pytest.approx(0.5)  # first hit at rank 2


def test_mrr_is_zero_when_nothing_relevant_is_in_the_window() -> None:
    assert M.mrr_at_k(RANKED, REL, 1) == 0.0


# --- ndcg ---------------------------------------------------------------------


def test_ndcg_matches_hand_computation() -> None:
    # gains 2^1-1 = 1 at ranks 2 and 4 -> dcg = 1/log2(3) + 1/log2(5)
    # ideal places both at ranks 1 and 2 -> idcg = 1/log2(2) + 1/log2(3)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert M.ndcg_at_k(RANKED, REL, 5) == pytest.approx(dcg / idcg)


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    assert M.ndcg_at_k(["b", "d", "a"], REL, 3) == pytest.approx(1.0)


def test_ndcg_rewards_higher_grades_first() -> None:
    graded = {"b": 3.0, "d": 1.0}
    good = M.ndcg_at_k(["b", "d"], graded, 2)
    bad = M.ndcg_at_k(["d", "b"], graded, 2)
    assert good > bad


# --- the not-measured contract ------------------------------------------------


@pytest.mark.parametrize("fn", [M.hit_at_k, M.recall_at_k, M.mrr_at_k, M.ndcg_at_k])
def test_no_relevant_documents_returns_none_not_zero(fn) -> None:
    """A query with nothing to find was not measured. Zero would be a lie."""
    assert fn(RANKED, {}, 5) is None
    assert fn(RANKED, {"a": 0.0, "b": 0.0}, 5) is None


def test_mean_skips_unmeasured_queries() -> None:
    assert M.mean([1.0, None, 0.0]) == pytest.approx(0.5)


def test_mean_of_nothing_measurable_is_none() -> None:
    assert M.mean([None, None]) is None
    assert M.mean([]) is None


# --- percentile ---------------------------------------------------------------


def test_percentile_uses_nearest_rank() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert M.percentile(values, 50) == 20.0
    assert M.percentile(values, 95) == 40.0
    assert M.percentile(values, 100) == 40.0


def test_percentile_of_empty_is_none() -> None:
    assert M.percentile([], 50) is None


def test_percentile_of_one_value() -> None:
    assert M.percentile([7.0], 95) == 7.0
