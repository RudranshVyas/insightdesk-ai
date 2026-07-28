"""Merging partially-overlapping ticket exports."""

from __future__ import annotations

import pandas as pd

from backend.scripts.merge_ticket_csvs import main


def _write(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_inputs_missing_a_required_column_are_skipped(tmp_path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write(a, [{"body": "one", "answer": "fixed it"}])
    _write(b, [{"body": "two"}])  # no answer column
    out = tmp_path / "merged.csv"

    assert main(["--inputs", str(a), str(b), "--require-column", "answer",
                 "--out", str(out)]) == 0
    merged = pd.read_csv(out)
    assert len(merged) == 1
    assert merged["body"].tolist() == ["one"]


def test_overlapping_rows_are_deduplicated_on_body(tmp_path) -> None:
    """The same ticket twice is one case that looks like corroboration."""
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write(a, [{"body": "shared", "answer": "x"}, {"body": "only_a", "answer": "y"}])
    _write(b, [{"body": "shared", "answer": "x"}, {"body": "only_b", "answer": "z"}])
    out = tmp_path / "merged.csv"

    main(["--inputs", str(a), str(b), "--out", str(out)])
    merged = pd.read_csv(out)
    assert sorted(merged["body"]) == ["only_a", "only_b", "shared"]


def test_provenance_column_records_the_originating_file(tmp_path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    _write(a, [{"body": "from_a", "answer": "x"}])
    _write(b, [{"body": "from_b", "answer": "y"}])
    out = tmp_path / "merged.csv"

    main(["--inputs", str(a), str(b), "--out", str(out)])
    merged = pd.read_csv(out).set_index("body")
    assert merged.loc["from_a", "_source_file"] == "a.csv"
    assert merged.loc["from_b", "_source_file"] == "b.csv"


def test_missing_input_file_is_skipped_not_fatal(tmp_path) -> None:
    a = tmp_path / "a.csv"
    _write(a, [{"body": "one", "answer": "x"}])
    out = tmp_path / "merged.csv"

    assert main(["--inputs", str(a), str(tmp_path / "absent.csv"),
                 "--out", str(out)]) == 0
    assert len(pd.read_csv(out)) == 1


def test_no_usable_inputs_exits_nonzero(tmp_path) -> None:
    b = tmp_path / "b.csv"
    _write(b, [{"body": "two"}])
    assert main(["--inputs", str(b), "--require-column", "answer",
                 "--out", str(tmp_path / "m.csv")]) == 2
