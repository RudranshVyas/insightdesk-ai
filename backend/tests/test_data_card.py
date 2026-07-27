"""Phase 1 — data card provenance.

The card is the thing that makes a published metric traceable. Its most
load-bearing behaviour is the licence field: an unconfirmed licence must read
`unverified`, never the optimistic string somebody typed into a YAML file.
"""

from __future__ import annotations

import json

import pytest

from backend.app.core import versions as V
from backend.app.services import data_card as DC


@pytest.fixture
def file_meta() -> dict:
    return {
        "path": "data/raw/tickets.csv",
        "bytes": 12345,
        "sha256": "a" * 64,
        "rows_read": 100,
    }


def test_card_records_the_raw_file_hash(fixture_mapping, file_meta) -> None:
    card = DC.build_data_card(fixture_mapping, file_meta, {"rows_in": 100, "rows_out": 98})
    assert card["raw_file"]["raw_file_sha256"] == "a" * 64
    assert card["raw_file"]["rows_read"] == 100
    assert card["rows"] == {"rows_in": 100, "rows_out": 98}


def test_card_stamps_component_versions(fixture_mapping, file_meta) -> None:
    card = DC.build_data_card(fixture_mapping, file_meta, {})
    assert card["versions"] == {
        "adapter_version": V.ADAPTER_VERSION,
        "redaction_version": V.REDACTION_VERSION,
        "audit_version": V.AUDIT_VERSION,
    }


@pytest.mark.parametrize("declared", ["owned", "public_domain", "cc0"])
def test_confirmed_licence_states_pass_through(file_meta, declared: str) -> None:
    mapping = {"dataset": {"license_status": declared}, "columns": {}}
    card = DC.build_data_card(mapping, file_meta, {})
    assert card["license"]["status"] == declared


@pytest.mark.parametrize(
    "declared",
    ["MIT", "Apache-2.0", "cc-by-4.0", "free to use", "kaggle", "probably fine"],
)
def test_unconfirmed_licence_is_downgraded_to_unverified(file_meta, declared: str) -> None:
    mapping = {"dataset": {"license_status": declared}, "columns": {}}
    card = DC.build_data_card(mapping, file_meta, {})
    assert card["license"]["status"] == "unverified"
    assert card["license"]["declared"] == declared.lower()
    assert declared.lower() in card["license"]["note"]


def test_missing_licence_is_unverified(file_meta) -> None:
    card = DC.build_data_card({"dataset": {}, "columns": {}}, file_meta, {})
    assert card["license"]["status"] == "unverified"
    assert card["license"]["declared"] is None


def test_mapped_and_unmapped_fields_are_both_recorded(fixture_mapping, file_meta) -> None:
    card = DC.build_data_card(fixture_mapping, file_meta, {})
    assert "ticket_id" in card["mapped_fields"]
    assert "issue_description" in card["mapped_fields"]
    # The fixture deliberately leaves these unmapped.
    assert "platform" in card["unmapped_fields"]
    assert "sla_deadline" in card["unmapped_fields"]
    assert not set(card["mapped_fields"]) & set(card["unmapped_fields"])


def test_derivations_are_carried_from_the_adapter_report(
    fixture_mapping, file_meta, adapter_report
) -> None:
    card = DC.build_data_card(fixture_mapping, file_meta, {}, adapter_report.to_dict())
    assert "issue_text" in card["derivations"]
    assert "customer_id_hash" in card["derivations"]


def test_adversarial_flag_is_surfaced(redteam_mapping, file_meta) -> None:
    card = DC.build_data_card(redteam_mapping, file_meta, {})
    assert card["dataset"]["adversarial"] is True


def test_card_is_json_serializable(fixture_mapping, file_meta, tmp_path) -> None:
    card = DC.build_data_card(fixture_mapping, file_meta, {"rows_in": 1, "rows_out": 1})
    out = tmp_path / "nested" / "data_card.json"
    DC.write_data_card(card, out)
    assert json.loads(out.read_text(encoding="utf-8"))["data_card_version"] == 1
