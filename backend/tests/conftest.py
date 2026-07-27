from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backend.app.services import schema_adapter as adapter

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CSV = REPO_ROOT / "data" / "fixtures" / "sample_tickets.csv"
FIXTURE_MAPPING = REPO_ROOT / "data" / "fixtures" / "fixture_mapping.yaml"
REDTEAM_CSV = REPO_ROOT / "data" / "fixtures" / "redteam_tickets.csv"
REDTEAM_MAPPING = REPO_ROOT / "data" / "fixtures" / "redteam_mapping.yaml"


@pytest.fixture(scope="session")
def raw_fixture_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_CSV, dtype=str, keep_default_na=True)


@pytest.fixture(scope="session")
def fixture_mapping() -> dict:
    return adapter.load_mapping(FIXTURE_MAPPING)


@pytest.fixture(scope="session")
def canonical_pair(raw_fixture_df, fixture_mapping):
    return adapter.apply_mapping(raw_fixture_df, fixture_mapping)


@pytest.fixture(scope="session")
def canonical_df(canonical_pair) -> pd.DataFrame:
    return canonical_pair[0]


@pytest.fixture(scope="session")
def adapter_report(canonical_pair):
    return canonical_pair[1]


# --- red-team fixture --------------------------------------------------------


@pytest.fixture(scope="session")
def raw_redteam_df() -> pd.DataFrame:
    return pd.read_csv(REDTEAM_CSV, dtype=str, keep_default_na=True)


@pytest.fixture(scope="session")
def redteam_mapping() -> dict:
    return adapter.load_mapping(REDTEAM_MAPPING)


@pytest.fixture(scope="session")
def redteam_pair(raw_redteam_df, redteam_mapping):
    return adapter.apply_mapping(raw_redteam_df, redteam_mapping)


@pytest.fixture(scope="session")
def redteam_df(redteam_pair) -> pd.DataFrame:
    return redteam_pair[0]
