"""Phase 5 — the evaluation harness itself.

A fake retriever is used deliberately. These tests are about whether the harness
scores, validates, and *refuses* correctly; whether MiniLM is good at finding
billing tickets is a different question and belongs to the real report.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from backend.app.evaluation import retrieval_eval as E

# --- a controllable stand-in for HybridRetriever ------------------------------


class FakeRetriever:
    """Returns a fixed ranking per config, and records what it was asked to exclude."""

    def __init__(self, rankings: dict[str, list[str]] | None = None) -> None:
        self.rankings = rankings or {}
        self.calls: list[dict] = []

    def search(self, text, product_area=None, issue_type=None, top_k=5, **kwargs):
        self.calls.append(
            {
                "text": text,
                "top_k": top_k,
                "mode": kwargs.get("mode"),
                "metadata_boost": kwargs.get("metadata_boost"),
                "exclude_ticket_ids": kwargs.get("exclude_ticket_ids"),
                "exclude_template_groups": kwargs.get("exclude_template_groups"),
            }
        )
        mode = kwargs.get("mode", "hybrid")
        ranked = self.rankings.get(mode, self.rankings.get("default", []))
        return {
            "results": [{"ticket_id": t} for t in ranked[:top_k]],
            "strength": {"strength": "strong"},
        }


@pytest.fixture
def corpus() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticket_id": ["T1", "T2", "T3", "T4", "T5", "T6"],
            "issue_text": [
                "payment failed but amount deducted",
                "charged twice for one invoice",
                "otp not received on login",
                "sso redirect loop after reset",
                "dashboard export times out",
                "report data is stale by a day",
            ],
            "issue_type": [
                "Billing", "Billing", "Technical", "Technical", "Technical", "Technical"
            ],
            # T1 and T2 are template siblings; the rest are singletons.
            "template_group_id": [0, 0, 1, 2, 3, 4],
        }
    )


# --- config wiring ------------------------------------------------------------


def test_every_spec_config_maps_to_search_arguments() -> None:
    assert set(E.CONFIGS) == {"bm25", "dense", "hybrid", "hybrid_metadata"}
    assert E.config_kwargs("bm25")["mode"] == "lexical"
    assert E.config_kwargs("dense")["mode"] == "dense"
    assert E.config_kwargs("hybrid")["metadata_boost"] is False
    assert E.config_kwargs("hybrid_metadata")["metadata_boost"] is True


def test_unknown_config_is_rejected_not_silently_defaulted() -> None:
    with pytest.raises(ValueError, match="unknown retrieval config"):
        E.config_kwargs("magic")


# --- Tier 1 exclusions: the whole point --------------------------------------


def test_leave_one_out_excludes_the_query_ticket_and_its_template_group(corpus) -> None:
    retriever = FakeRetriever({"default": ["T3", "T4"]})
    E.evaluate_leave_one_out(retriever, corpus, "hybrid", sample_size=6)

    # Every search must have asked to exclude the query itself.
    real_calls = [c for c in retriever.calls if c["exclude_ticket_ids"] is not None]
    assert real_calls, "no leave-one-out searches were issued"
    for call in real_calls:
        assert len(call["exclude_ticket_ids"]) == 1
        assert call["exclude_template_groups"], "template siblings were not excluded"


def test_template_sibling_exclusion_covers_the_whole_group(corpus) -> None:
    retriever = FakeRetriever({"default": ["T3"]})
    E.evaluate_leave_one_out(retriever, corpus, "hybrid", sample_size=6)

    by_query = {
        next(iter(c["exclude_ticket_ids"])): c["exclude_template_groups"]
        for c in retriever.calls
        if c["exclude_ticket_ids"]
    }
    # T1 and T2 share group 0, so querying with either must exclude that group,
    # which removes the other one too.
    assert by_query["T1"] == {0}
    assert by_query["T2"] == {0}


def test_leave_one_out_reports_a_random_baseline_and_lift(corpus) -> None:
    retriever = FakeRetriever({"default": ["T3", "T4", "T5"]})
    out = E.evaluate_leave_one_out(retriever, corpus, "hybrid", sample_size=6)
    assert out["available"] is True
    assert "random_baseline" in out
    assert "lift_over_random" in out
    assert "SATURATES" in out["interpretation"]


def test_leave_one_out_never_reports_recall(corpus) -> None:
    """Recall against a coarse category is bounded near zero and misleads."""
    retriever = FakeRetriever({"default": ["T3", "T4"]})
    out = E.evaluate_leave_one_out(retriever, corpus, "hybrid", sample_size=6)
    assert out["metrics"]["recall_at_5"] is None
    assert out["metrics"]["recall_at_3"] is None


def test_leave_one_out_unavailable_without_issue_type() -> None:
    corpus = pd.DataFrame({"ticket_id": ["T1"], "issue_text": ["x"]})
    out = E.evaluate_leave_one_out(FakeRetriever(), corpus, "hybrid")
    assert out["available"] is False
    assert "issue_type" in out["reason"]
    assert "metrics" not in out


def test_leave_one_out_is_deterministic_for_a_fixed_seed(corpus) -> None:
    a = E.evaluate_leave_one_out(
        FakeRetriever({"default": ["T3"]}), corpus, "hybrid", sample_size=3, seed=7
    )
    b = E.evaluate_leave_one_out(
        FakeRetriever({"default": ["T3"]}), corpus, "hybrid", sample_size=3, seed=7
    )
    assert a["metrics"] == b["metrics"]


# --- Tier 2 loading and validation --------------------------------------------


def _write(tmp_path, rows):
    p = tmp_path / "retrieval_queries.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_missing_labeled_file_loads_as_empty(tmp_path) -> None:
    assert E.load_labeled_queries(tmp_path / "nope.jsonl") == []


def test_comment_and_blank_lines_are_skipped(tmp_path) -> None:
    p = tmp_path / "q.jsonl"
    p.write_text(
        '// a comment\n\n{"query_id":"Q1","text":"x","relevance":{"T1":2}}\n',
        encoding="utf-8",
    )
    assert len(E.load_labeled_queries(p)) == 1


def test_malformed_line_names_the_line_number(tmp_path) -> None:
    p = tmp_path / "q.jsonl"
    p.write_text('{"query_id":"Q1","text":"x"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":2: malformed"):
        E.load_labeled_queries(p)


def test_validator_rejects_a_query_that_could_retrieve_itself(corpus, tmp_path) -> None:
    p = _write(tmp_path, [{"query_id": "T1", "text": "x", "relevance": {"T2": 2}}])
    result = E.validate_labeled_set(E.load_labeled_queries(p), corpus)
    assert not result["valid"]
    assert any("retrieve itself" in e for e in result["errors"])


def test_validator_rejects_grades_for_unknown_tickets(corpus, tmp_path) -> None:
    p = _write(tmp_path, [{"query_id": "Q1", "text": "x", "relevance": {"NOPE": 2}}])
    result = E.validate_labeled_set(E.load_labeled_queries(p), corpus)
    assert not result["valid"]
    assert any("absent from the corpus" in e for e in result["errors"])


def test_validator_rejects_duplicate_query_ids(corpus, tmp_path) -> None:
    p = _write(
        tmp_path,
        [
            {"query_id": "Q1", "text": "a", "relevance": {"T1": 2}},
            {"query_id": "Q1", "text": "b", "relevance": {"T2": 2}},
        ],
    )
    result = E.validate_labeled_set(E.load_labeled_queries(p), corpus)
    assert not result["valid"]
    assert any("duplicate query_id" in e for e in result["errors"])


def test_validator_warns_when_relevance_hides_inside_one_template_group(
    corpus, tmp_path
) -> None:
    """Such a query rewards duplicate retrieval, not relevance."""
    corpus2 = corpus.copy()
    corpus2.loc[corpus2["ticket_id"].isin(["T1", "T2", "T3"]), "template_group_id"] = 0
    p = _write(
        tmp_path,
        [{"query_id": "Q1", "text": "x", "relevance": {"T1": 2, "T2": 2, "T3": 1}}],
    )
    result = E.validate_labeled_set(E.load_labeled_queries(p), corpus2)
    assert result["valid"]  # a warning, not an error
    assert any("one template group" in w for w in result["warnings"])


def test_validator_warns_on_a_query_with_no_relevant_ticket(corpus, tmp_path) -> None:
    p = _write(tmp_path, [{"query_id": "Q1", "text": "x", "relevance": {"T1": 0}}])
    result = E.validate_labeled_set(E.load_labeled_queries(p), corpus)
    assert result["valid"]
    assert any("no positively relevant" in w for w in result["warnings"])


# --- report assembly ----------------------------------------------------------


def test_report_refuses_to_invent_tier2_metrics(corpus, tmp_path) -> None:
    report = E.build_report(
        FakeRetriever({"default": ["T3", "T4"]}),
        corpus,
        tmp_path / "absent.jsonl",
        configs=["hybrid"],
        loo_sample=3,
    )
    tier2 = report["tier2_manual_labeled"]
    assert tier2["status"] == "not_yet_labeled"
    assert tier2["results"] == {}
    # No key anywhere in Tier 2 may carry a number.
    assert "metrics" not in json.dumps(tier2)


def test_report_blocks_metrics_when_the_labeled_set_is_invalid(corpus, tmp_path) -> None:
    p = _write(tmp_path, [{"query_id": "T1", "text": "x", "relevance": {"T2": 2}}])
    report = E.build_report(
        FakeRetriever({"default": ["T2"]}), corpus, p, configs=["hybrid"], loo_sample=2
    )
    assert report["tier2_manual_labeled"]["status"] == "invalid_labeled_set"
    assert report["tier2_manual_labeled"]["results"] == {}


def test_report_scores_a_valid_labeled_set(corpus, tmp_path) -> None:
    p = _write(
        tmp_path,
        [
            {"query_id": "Q1", "text": "charged twice", "relevance": {"T2": 2, "T1": 1}},
            {"query_id": "Q2", "text": "otp missing", "relevance": {"T3": 2}},
        ],
    )
    retriever = FakeRetriever({"hybrid": ["T2", "T1", "T5"]})
    report = E.build_report(retriever, corpus, p, configs=["hybrid"], loo_sample=2)

    tier2 = report["tier2_manual_labeled"]
    assert tier2["status"] == "evaluated"
    result = tier2["results"]["hybrid"]
    assert result["queries_run"] == 2
    # Q1: both relevant tickets at ranks 1-2. Q2: T3 never returned.
    assert result["metrics"]["hit_at_3"] == pytest.approx(0.5)
    assert result["metrics"]["recall_at_5"] == pytest.approx(0.5)


def test_report_stamps_versions_and_index_identity(corpus, tmp_path) -> None:
    report = E.build_report(
        FakeRetriever({"default": ["T3"]}),
        corpus,
        tmp_path / "absent.jsonl",
        configs=["hybrid"],
        loo_sample=2,
        index_manifest={"embedding_model": "m", "data_hash": "h", "corpus_size": 6},
    )
    assert report["versions"]["index"]
    assert report["index"]["embedding_model"] == "m"
    assert report["index"]["data_hash"] == "h"


# --- winner selection ---------------------------------------------------------


def test_winner_is_undecided_without_a_labeled_set() -> None:
    report = {"tier2_manual_labeled": {"status": "not_yet_labeled"}}
    assert E.summarize_winner(report)["decided"] is False


def test_winner_reports_ties_and_recommends_the_simpler_config() -> None:
    report = {
        "tier2_manual_labeled": {
            "status": "evaluated",
            "results": {
                "bm25": {"metrics": {"recall_at_5": 0.8}},
                "hybrid": {"metrics": {"recall_at_5": 0.8}},
                "dense": {"metrics": {"recall_at_5": 0.6}},
            },
        }
    }
    winner = E.summarize_winner(report)
    assert winner["decided"] is True
    assert set([winner["best"]] + winner["tied_with"]) == {"bm25", "hybrid"}
    assert "negative result" in winner["note"]


def test_winner_can_be_bm25_and_the_harness_says_so_plainly() -> None:
    report = {
        "tier2_manual_labeled": {
            "status": "evaluated",
            "results": {
                "bm25": {"metrics": {"recall_at_5": 0.9}},
                "hybrid": {"metrics": {"recall_at_5": 0.7}},
            },
        }
    }
    assert E.summarize_winner(report)["best"] == "bm25"
