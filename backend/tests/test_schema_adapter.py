from __future__ import annotations

import copy

import pandas as pd
import pytest

from backend.app.core import canonical as C
from backend.app.services import schema_adapter as adapter


def test_all_canonical_fields_present_in_output(canonical_df: pd.DataFrame) -> None:
    assert list(canonical_df.columns) == list(C.CANONICAL_FIELDS)


def test_every_canonical_field_is_mapped_derived_or_missing(adapter_report) -> None:
    accounted = (
        set(adapter_report.mapped)
        | set(adapter_report.derived)
        | set(adapter_report.missing)
    )
    assert accounted == set(C.CANONICAL_FIELDS)


def test_required_fields_are_populated(canonical_df: pd.DataFrame) -> None:
    for f in C.REQUIRED_FIELDS:
        assert canonical_df[f].notna().all()
        assert (canonical_df[f].astype(str).str.strip() != "").all()


def test_issue_text_is_subject_plus_description(canonical_df: pd.DataFrame) -> None:
    row = canonical_df.iloc[0]
    assert row["issue_subject"] in row["issue_text"]
    assert row["issue_description"][:30] in row["issue_text"]


def test_purchase_date_is_never_mapped_to_created_at(
    raw_fixture_df, fixture_mapping
) -> None:
    """The `Date of Purchase` trap: mapping it must be rejected, not honoured."""
    cfg = copy.deepcopy(fixture_mapping)
    cfg["columns"]["created_at"] = "Date of Purchase"
    df, report = adapter.apply_mapping(raw_fixture_df, cfg)

    assert "created_at" in report.missing
    assert df["created_at"].isna().all()
    assert any("created_at mapping rejected" in r for r in report.rejections)


def test_status_normalization_maps_to_canonical_vocabulary(
    canonical_df: pd.DataFrame,
) -> None:
    seen = set(canonical_df["status"].dropna().unique())
    assert seen <= set(C.STATUS_VALUES)
    assert "resolved" in seen and "open" in seen and "pending" in seen


def test_unmapped_status_values_become_null_and_are_reported(
    raw_fixture_df, fixture_mapping
) -> None:
    cfg = copy.deepcopy(fixture_mapping)
    cfg["normalization"]["status"] = {"open": ["Open"]}  # drop Closed / Pending
    df, report = adapter.apply_mapping(raw_fixture_df, cfg)

    assert df["status"].isna().sum() > 0
    assert "unmapped_values" in report.normalizations["status"]
    assert any("absent from the mapping" in r for r in report.rejections)


def test_boolean_normalization(canonical_df: pd.DataFrame) -> None:
    vals = set(canonical_df["escalated"].dropna().unique())
    assert vals <= {True, False}
    assert True in vals and False in vals


def test_unrecognised_boolean_value_becomes_null(raw_fixture_df, fixture_mapping) -> None:
    df_raw = raw_fixture_df.copy()
    df_raw.loc[0, "Escalated"] = "maybe"
    df, report = adapter.apply_mapping(df_raw, fixture_mapping)
    assert pd.isna(df.loc[0, "escalated"])
    assert "unmapped_values" in report.normalizations["escalated"]


def test_timestamps_are_parsed(canonical_df: pd.DataFrame) -> None:
    assert pd.api.types.is_datetime64_any_dtype(canonical_df["created_at"])
    assert canonical_df["created_at"].notna().all()


def test_resolution_time_is_derived_and_non_negative(
    canonical_df: pd.DataFrame, adapter_report
) -> None:
    assert "resolution_time_hours" in adapter_report.derived
    vals = canonical_df["resolution_time_hours"].dropna()
    assert len(vals) > 0
    assert (vals >= 0).all()


def test_negative_durations_are_nulled(raw_fixture_df, fixture_mapping) -> None:
    df_raw = raw_fixture_df.copy()
    df_raw.loc[0, "Time to Resolution"] = "2020-01-01 00:00:00"  # before creation
    df, report = adapter.apply_mapping(df_raw, fixture_mapping)
    assert pd.isna(df.loc[0, "resolution_time_hours"])
    assert any("resolved_at < created_at" in r for r in report.rejections)


def test_csat_zero_is_treated_as_no_response(canonical_df: pd.DataFrame, adapter_report) -> None:
    assert adapter_report.normalizations["csat_score"]["zero_means_no_response"] is True
    assert adapter_report.normalizations["csat_score"]["zeros_treated_as_no_response"] > 0
    assert (canonical_df["csat_score"].dropna() > 0).all()


def test_csat_out_of_range_is_nulled(raw_fixture_df, fixture_mapping) -> None:
    df_raw = raw_fixture_df.copy()
    df_raw.loc[0, "Customer Satisfaction Rating"] = "9"
    df, report = adapter.apply_mapping(df_raw, fixture_mapping)
    assert pd.isna(df.loc[0, "csat_score"])
    assert report.normalizations["csat_score"]["out_of_range_nulled"] == 1


def test_personal_columns_are_dropped_not_mapped(adapter_report) -> None:
    dropped = set(adapter_report.dropped_personal_columns)
    assert {"Customer Name", "Customer Age", "Customer Gender"} <= dropped
    assert "customer_name" not in adapter_report.mapped


def test_customer_id_is_hashed(canonical_df: pd.DataFrame, raw_fixture_df) -> None:
    hashes = canonical_df["customer_id_hash"].dropna()
    assert len(hashes) == len(canonical_df)
    joined = " ".join(hashes.tolist())
    for email in raw_fixture_df["Customer Email"].dropna():
        assert email not in joined


def test_stored_text_is_redacted(canonical_df: pd.DataFrame) -> None:
    blob = " ".join(canonical_df["issue_text"].tolist())
    assert "@example.com" not in blob
    assert "4111 1111 1111 1111" not in blob
    assert "[EMAIL]" in blob or "[PHONE]" in blob or "[CARD]" in blob


def test_empty_text_cells_become_empty_string_not_nan(canonical_df: pd.DataFrame) -> None:
    """A literal "nan" in resolution_notes would qualify a ticket as a source case."""
    notes = canonical_df["resolution_notes"].astype(str)
    assert not notes.str.strip().str.lower().isin({"nan", "nat", "none"}).any()
    assert (notes == "").sum() > 0  # the fixture has unresolved tickets


def test_boilerplate_resolution_notes_are_detectable(canonical_df: pd.DataFrame) -> None:
    notes = canonical_df["resolution_notes"].astype(str).str.strip().str.lower()
    usable = ~notes.isin(C.BOILERPLATE_RESOLUTIONS)
    assert usable.sum() < len(canonical_df)  # "resolved" / blanks filtered out
    assert usable.sum() > 0


def test_duplicate_ticket_ids_are_dropped(raw_fixture_df, fixture_mapping) -> None:
    doubled = pd.concat([raw_fixture_df, raw_fixture_df.head(3)], ignore_index=True)
    df, report = adapter.apply_mapping(doubled, fixture_mapping)
    assert report.duplicate_ticket_ids == 3
    assert df["ticket_id"].is_unique


def test_mapping_to_a_non_canonical_field_raises(raw_fixture_df, fixture_mapping) -> None:
    cfg = copy.deepcopy(fixture_mapping)
    cfg["columns"]["agent_mood"] = "Ticket Type"
    with pytest.raises(ValueError, match="not canonical fields"):
        adapter.apply_mapping(raw_fixture_df, cfg)


def test_mapping_to_a_missing_source_column_raises(raw_fixture_df, fixture_mapping) -> None:
    cfg = copy.deepcopy(fixture_mapping)
    cfg["columns"]["region"] = "Nonexistent Column"
    with pytest.raises(ValueError, match="absent from the CSV"):
        adapter.apply_mapping(raw_fixture_df, cfg)


def test_suggest_mapping_never_proposes_a_purchase_date_as_created_at(
    raw_fixture_df,
) -> None:
    proposal = adapter.suggest_mapping(raw_fixture_df)
    assert proposal["columns"]["created_at"] != "Date of Purchase"
    assert proposal["dataset"]["license_status"] == "unverified"
