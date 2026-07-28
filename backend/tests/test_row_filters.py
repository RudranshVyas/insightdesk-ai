"""Declarative row filters — a corpus restriction must be recorded, not silent."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.app.services import schema_adapter as adapter


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "subject": ["a", "b", "c", "d"],
            "body": ["issue one", "issue two", "issue three", "issue four"],
            "language": ["en", "de", "EN", "fr"],
        }
    )


def _cfg(**extra) -> dict:
    return {
        "dataset": {"name": "t"},
        "columns": {"ticket_id": "id", "issue_subject": "subject", "issue_description": "body"},
        **extra,
    }


def test_no_filters_keeps_every_row() -> None:
    out, report = adapter.apply_mapping(_df(), _cfg())
    assert len(out) == 4
    assert "row_filters" not in report.normalizations


def test_filter_keeps_only_matching_rows() -> None:
    cfg = _cfg(filters=[{"column": "language", "keep": ["en"], "reason": "English-only index"}])
    out, report = adapter.apply_mapping(_df(), cfg)
    assert len(out) == 2  # "en" and "EN"
    assert report.rows_in == 4


def test_matching_is_case_insensitive() -> None:
    cfg = _cfg(filters=[{"column": "language", "keep": ["EN"]}])
    out, _ = adapter.apply_mapping(_df(), cfg)
    assert len(out) == 2


def test_the_drop_is_recorded_with_its_reason() -> None:
    reason = "all-MiniLM-L6-v2 is English-centric"
    cfg = _cfg(filters=[{"column": "language", "keep": ["en"], "reason": reason}])
    _, report = adapter.apply_mapping(_df(), cfg)

    recorded = report.normalizations["row_filters"][0]
    assert recorded["rows_before"] == 4
    assert recorded["rows_after"] == 2
    assert recorded["rows_dropped"] == 2
    assert recorded["reason"] == reason
    # Also surfaced where a reader of the audit will actually see it.
    assert any("row filter on language" in r for r in report.rejections)


def test_a_filter_without_a_reason_says_so_rather_than_omitting_it() -> None:
    cfg = _cfg(filters=[{"column": "language", "keep": ["en"]}])
    _, report = adapter.apply_mapping(_df(), cfg)
    assert report.normalizations["row_filters"][0]["reason"] == "no reason recorded"


def test_unknown_column_is_a_hard_error_not_a_silent_no_op() -> None:
    """A typo that quietly disables a filter would misrepresent the corpus."""
    cfg = _cfg(filters=[{"column": "langauge", "keep": ["en"]}])
    with pytest.raises(ValueError, match="not in the CSV"):
        adapter.apply_mapping(_df(), cfg)


@pytest.mark.parametrize("bad", [{"column": "language"}, {"keep": ["en"]}, {}])
def test_incomplete_filter_is_rejected(bad: dict) -> None:
    with pytest.raises(ValueError, match="needs both"):
        adapter.apply_mapping(_df(), _cfg(filters=[bad]))


def test_multiple_filters_compose() -> None:
    df = _df()
    df["queue"] = ["billing", "billing", "tech", "tech"]
    cfg = _cfg(
        filters=[
            {"column": "language", "keep": ["en"]},
            {"column": "queue", "keep": ["billing"]},
        ]
    )
    out, report = adapter.apply_mapping(df, cfg)
    assert len(out) == 1
    assert len(report.normalizations["row_filters"]) == 2


# --- synthesized ticket_id ----------------------------------------------------


def test_missing_ticket_id_without_opt_in_is_refused() -> None:
    df = _df().drop(columns=["id"])
    cfg = {"dataset": {"name": "t"},
           "columns": {"issue_subject": "subject", "issue_description": "body"}}
    with pytest.raises(ValueError, match="row_index"):
        adapter.apply_mapping(df, cfg)


def test_row_index_ids_are_synthesized_and_recorded() -> None:
    df = _df().drop(columns=["id"])
    cfg = {
        "dataset": {"name": "t", "id_prefix": "TB"},
        "columns": {"issue_subject": "subject", "issue_description": "body"},
        "derivations": {"ticket_id": "row_index"},
    }
    out, report = adapter.apply_mapping(df, cfg)

    assert list(out["ticket_id"]) == ["TB0000000", "TB0000001", "TB0000002", "TB0000003"]
    assert out["ticket_id"].is_unique
    # The caveat must reach the audit, not just the code.
    assert "row position" in report.derived["ticket_id"]
    assert any("synthesized from row position" in r for r in report.rejections)


def test_synthesized_ids_are_assigned_after_filtering() -> None:
    """Ids must be dense over the rows actually kept, not over the raw file."""
    cfg = {
        "dataset": {"name": "t"},
        "columns": {"issue_subject": "subject", "issue_description": "body"},
        "filters": [{"column": "language", "keep": ["en"]}],
        "derivations": {"ticket_id": "row_index"},
    }
    out, _ = adapter.apply_mapping(_df().drop(columns=["id"]), cfg)
    assert list(out["ticket_id"]) == ["T0000000", "T0000001"]
