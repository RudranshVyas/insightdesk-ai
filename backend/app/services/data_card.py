"""Phase 1 — the data card.

Provenance for the dataset the whole app is built on. Written next to the audit
so that any published number can be traced back to a specific file, on a
specific date, processed by specific component versions.

The licence field is the honest part. If nobody confirmed the licence, it says
``unverified``. It never says "MIT" because a README somewhere said so.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.app.core import versions as V

# Values that are claims, not confirmations. Anything not in this set is
# recorded verbatim but still marked as operator-asserted.
CONFIRMED_LICENSE_STATES: frozenset[str] = frozenset({"owned", "public_domain", "cc0"})


def build_data_card(
    mapping: dict[str, Any],
    file_meta: dict[str, Any],
    row_counts: dict[str, int],
    adapter_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ds = dict(mapping.get("dataset") or {})

    declared = str(ds.get("license_status") or "").strip().lower()
    if not declared:
        license_status = "unverified"
        license_note = "No licence was declared in the mapping file."
    elif declared in CONFIRMED_LICENSE_STATES:
        license_status = declared
        license_note = "Declared by the operator in the mapping file."
    else:
        license_status = "unverified"
        license_note = (
            f"Mapping declares '{declared}', which has not been independently "
            f"confirmed. Treated as unverified."
        )

    return {
        "data_card_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": ds.get("name"),
            "source_url": ds.get("source_url"),
            "download_date": ds.get("download_date"),
            "synthetic": bool(ds.get("synthetic", False)),
            "adversarial": bool(ds.get("adversarial", False)),
        },
        "raw_file": {
            "path": file_meta.get("path"),
            "bytes": file_meta.get("bytes"),
            "raw_file_sha256": file_meta.get("sha256"),
            "rows_read": file_meta.get("rows_read"),
        },
        "license": {"status": license_status, "declared": declared or None, "note": license_note},
        "rows": row_counts,
        "versions": {
            "adapter_version": V.ADAPTER_VERSION,
            "redaction_version": V.REDACTION_VERSION,
            "audit_version": V.AUDIT_VERSION,
        },
        "mapped_fields": sorted(
            k for k, v in (mapping.get("columns") or {}).items() if v is not None
        ),
        "unmapped_fields": sorted(
            k for k, v in (mapping.get("columns") or {}).items() if v is None
        ),
        "derivations": dict((adapter_report or {}).get("derived_fields") or {}),
        "disclaimer": (
            "Every metric produced by this application is computed over the file "
            "identified by raw_file_sha256. A different file produces different "
            "numbers and requires a new data card."
        ),
    }


def write_data_card(card: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2, default=str)
