"""Config-driven source -> canonical schema adapter.

This is the ONLY module allowed to know dataset-specific column names, and even
here they arrive from a YAML mapping file rather than from code.

Nothing is inferred silently. Every normalization, derivation, and rejection is
recorded in the returned :class:`AdapterReport` and lands in the audit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from backend.app.core import canonical as C
from backend.app.core.redaction import hash_identifier, looks_personal, redact_text

DEFAULT_NULL_TOKENS = ["", "na", "n/a", "null", "none", "nan", "-", "--", "unknown"]

# Columns whose names indicate they are NOT ticket creation time, no matter what
# the mapping file says. `Date of Purchase` is the well-known trap.
CREATED_AT_BLOCKLIST = re.compile(
    r"purchase|order[_ ]?date|signup|registration|birth|renewal|subscription[_ ]?start",
    re.IGNORECASE,
)

# Heuristics used only by --suggest-mapping. A human reviews the output.
_SUGGEST_HINTS: dict[str, tuple[str, ...]] = {
    "ticket_id": ("ticket id", "ticket_id", "ticketid", "id", "case id", "incident id"),
    "created_at": (
        "date of ticket",
        "ticket date",
        "created",
        "created at",
        "opened",
        "open date",
        "reported",
        "submitted",
    ),
    "resolved_at": ("resolved", "closed at", "close date", "resolution date", "completion"),
    "first_response_at": ("first response", "first reply", "responded at"),
    "sla_deadline": ("sla due", "sla deadline", "due date", "target resolution"),
    "customer_segment": ("segment", "tier", "plan tier", "account type"),
    "product_area": ("product", "product purchased", "product area", "component", "module"),
    "issue_type": ("ticket type", "issue type", "category", "request type"),
    "priority": ("priority", "severity", "urgency"),
    "status": ("status", "state", "ticket status"),
    "channel": ("channel", "source", "ticket channel", "contact method"),
    "platform": ("platform", "os", "device"),
    "region": ("region", "country", "geo", "location"),
    "sla_plan": ("sla plan", "sla", "support plan", "service level"),
    "issue_subject": ("subject", "title", "summary", "short description"),
    "issue_description": ("description", "body", "ticket description", "issue"),
    "resolution_notes": ("resolution", "resolution notes", "solution", "close notes"),
    "response_time_hours": ("first response time", "response time"),
    "resolution_time_hours": ("time to resolution", "resolution time", "handling time"),
    "reopened": ("reopened", "re-opened"),
    "escalated": ("escalat",),
    "sla_breached": ("sla breach", "breached", "sla met", "missed sla"),
    "csat_score": ("satisfaction", "csat", "rating", "score"),
}


@dataclass
class AdapterReport:
    dataset: dict[str, Any] = field(default_factory=dict)
    source_columns: list[str] = field(default_factory=list)
    source_dtypes: dict[str, str] = field(default_factory=dict)
    mapped: dict[str, str] = field(default_factory=dict)
    derived: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    dropped_personal_columns: list[str] = field(default_factory=list)
    parse_failures: dict[str, int] = field(default_factory=dict)
    normalizations: dict[str, Any] = field(default_factory=dict)
    rejections: list[str] = field(default_factory=list)
    duplicate_ticket_ids: int = 0
    rows_in: int = 0
    rows_out: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_columns": self.source_columns,
            "source_dtypes": self.source_dtypes,
            "canonical_mapping": self.mapped,
            "derived_fields": self.derived,
            "missing_canonical_fields": self.missing,
            "unexpected_source_columns": self.unexpected_columns,
            "dropped_personal_columns": self.dropped_personal_columns,
            "parse_failures": self.parse_failures,
            "normalizations": self.normalizations,
            "rejections": self.rejections,
            "duplicate_ticket_ids": self.duplicate_ticket_ids,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
        }


# --- mapping file -----------------------------------------------------------


def load_mapping(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if "columns" not in cfg:
        raise ValueError(f"{path}: mapping file must contain a 'columns' section")
    return cfg


def suggest_mapping(df: pd.DataFrame) -> dict[str, Any]:
    """Propose a mapping for HUMAN REVIEW. Never used automatically."""
    cols = list(df.columns)
    low = {c: c.strip().lower() for c in cols}
    proposal: dict[str, str | None] = {}
    used: set[str] = set()

    for canon, hints in _SUGGEST_HINTS.items():
        best: str | None = None
        for hint in hints:
            for col in cols:
                if col in used:
                    continue
                if hint in low[col]:
                    if canon == "created_at" and CREATED_AT_BLOCKLIST.search(col):
                        continue
                    best = col
                    break
            if best:
                break
        if best:
            used.add(best)
        proposal[canon] = best

    return {
        "dataset": {
            "name": "TODO",
            "source_url": "TODO",
            "download_date": "TODO",
            "license_status": "unverified",
        },
        "columns": proposal,
        "_review_notes": [
            "Every value above is a NAME-BASED GUESS. Verify each against the data.",
            "created_at must be a genuine ticket creation timestamp. A purchase or "
            "signup date is NOT ticket creation time — leave it null instead.",
            "Set any field you cannot verify to null. Null disables the dependent "
            "feature, which is the correct outcome.",
        ],
    }


# --- normalization helpers --------------------------------------------------


def _null_out(series: pd.Series, null_tokens: list[str]) -> pd.Series:
    toks = {str(t).strip().lower() for t in null_tokens}
    s = series.astype("object")
    mask = s.map(lambda v: isinstance(v, str) and v.strip().lower() in toks)
    return s.mask(mask, other=None)


def normalize_status(
    series: pd.Series, status_map: dict[str, list[str]], report: AdapterReport
) -> pd.Series:
    lookup: dict[str, str] = {}
    for canon_status, raws in (status_map or {}).items():
        if canon_status not in C.STATUS_VALUES:
            report.rejections.append(
                f"status mapping target '{canon_status}' is not in STATUS_VALUES; kept as-is"
            )
        for raw in raws or []:
            lookup[str(raw).strip().lower()] = canon_status

    unmapped: dict[str, int] = {}

    def _map(v: object) -> object:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        key = str(v).strip().lower()
        if key in lookup:
            return lookup[key]
        unmapped[key] = unmapped.get(key, 0) + 1
        return None

    out = series.map(_map)
    if unmapped:
        report.normalizations.setdefault("status", {})["unmapped_values"] = unmapped
        report.rejections.append(
            f"{sum(unmapped.values())} rows had a status value absent from the mapping "
            f"and were set to null: {sorted(unmapped)[:10]}"
        )
    return out


def normalize_boolean(
    series: pd.Series,
    true_values: list[str],
    false_values: list[str],
    report: AdapterReport,
    field_name: str,
) -> pd.Series:
    tset = {str(v).strip().lower() for v in true_values}
    fset = {str(v).strip().lower() for v in false_values}
    unmapped: dict[str, int] = {}

    def _map(v: object) -> object:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        key = str(v).strip().lower()
        if key in tset:
            return True
        if key in fset:
            return False
        unmapped[key] = unmapped.get(key, 0) + 1
        return None

    out = series.map(_map)
    if unmapped:
        report.normalizations.setdefault(field_name, {})["unmapped_values"] = unmapped
    return out


def parse_timestamps(
    series: pd.Series, fmt: str | None, report: AdapterReport, field_name: str
) -> pd.Series:
    raw_non_null = series.notna().sum()
    if fmt:
        out = pd.to_datetime(series, format=fmt, errors="coerce")
    else:
        out = pd.to_datetime(series, errors="coerce", format="mixed")
    failures = int(raw_non_null - out.notna().sum())
    if failures:
        report.parse_failures[field_name] = failures
    return out


# --- main entry point -------------------------------------------------------


def _apply_row_filters(
    df: pd.DataFrame, cfg: dict[str, Any], report: AdapterReport
) -> pd.DataFrame:
    """Restrict rows per the mapping's ``filters:`` block, recording every drop.

    Shape:

        filters:
          - column: language
            keep: ["en"]
            reason: "all-MiniLM-L6-v2 is English-centric; mixing languages into
                     one index degrades retrieval and invalidates the gate
                     thresholds."

    Matching is case-insensitive on the stringified value. A filter naming a
    column that does not exist is a hard error, not a silent no-op — a typo that
    quietly disables a filter would misrepresent the corpus.
    """
    filters = cfg.get("filters") or []
    if not filters:
        return df

    applied: list[dict[str, Any]] = []
    for spec in filters:
        column = spec.get("column")
        keep = spec.get("keep")
        if not column or not keep:
            raise ValueError(f"filter needs both 'column' and 'keep': {spec!r}")
        if column not in df.columns:
            raise ValueError(
                f"filter references column {column!r}, which is not in the CSV. "
                f"Available: {sorted(map(str, df.columns))[:20]}"
            )

        wanted = {str(v).strip().lower() for v in keep}
        before = len(df)
        df = df[df[column].astype(str).str.strip().str.lower().isin(wanted)]
        applied.append(
            {
                "column": str(column),
                "kept_values": sorted(wanted),
                "rows_before": before,
                "rows_after": len(df),
                "rows_dropped": before - len(df),
                "reason": spec.get("reason") or "no reason recorded",
            }
        )

    report.normalizations["row_filters"] = applied
    for a in applied:
        report.rejections.append(
            f"row filter on {a['column']}: kept {a['kept_values']}, dropped "
            f"{a['rows_dropped']} of {a['rows_before']} rows. {a['reason']}"
        )
    return df


def apply_mapping(
    df: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, AdapterReport]:
    report = AdapterReport()
    report.dataset = dict(cfg.get("dataset") or {})
    report.dataset.setdefault("license_status", "unverified")
    report.source_columns = [str(c) for c in df.columns]
    report.source_dtypes = {str(c): str(t) for c, t in df.dtypes.items()}
    report.rows_in = len(df)

    # --- declarative row filters --------------------------------------------
    # A corpus restriction is a claim about what the system was built on, so it
    # belongs in the mapping file and the audit rather than in a preprocessing
    # step somebody ran once and forgot. The motivating case: a multilingual
    # ticket export indexed with an English-centric embedding model. Restricting
    # to one language is defensible; doing it silently is not.
    df = _apply_row_filters(df, cfg, report)

    columns: dict[str, str | None] = {
        k: v for k, v in (cfg.get("columns") or {}).items() if v
    }
    norm = cfg.get("normalization") or {}
    null_tokens = list(norm.get("nulls") or DEFAULT_NULL_TOKENS)

    unknown_targets = set(columns) - set(C.CANONICAL_FIELDS)
    if unknown_targets:
        raise ValueError(
            f"mapping targets are not canonical fields: {sorted(unknown_targets)}"
        )

    # --- created_at trap guard ---------------------------------------------
    ca_source = columns.get("created_at")
    forbidden = [
        str(x).strip().lower()
        for x in (norm.get("timestamps") or {}).get("forbidden_created_at_columns", [])
    ]
    if ca_source and (
        CREATED_AT_BLOCKLIST.search(str(ca_source))
        or str(ca_source).strip().lower() in forbidden
    ):
        report.rejections.append(
            f"created_at mapping rejected: source column '{ca_source}' is not a ticket "
            f"creation timestamp. created_at left null; all time-based analytics and "
            f"temporal splitting disabled."
        )
        columns.pop("created_at")

    missing_sources = {k: v for k, v in columns.items() if v not in df.columns}
    if missing_sources:
        raise ValueError(
            f"mapping references columns absent from the CSV: {missing_sources}"
        )

    out = pd.DataFrame(index=df.index)

    # --- straight copies + null normalization -------------------------------
    for canon, src in columns.items():
        out[canon] = _null_out(df[src], null_tokens)
        report.mapped[canon] = str(src)

    # --- ticket_id ----------------------------------------------------------
    if "ticket_id" not in out.columns:
        # Some exports carry no id column at all. Refusing outright would be
        # unhelpful; synthesizing one silently would be worse, because a
        # positional id is only stable for one exact file. So it is synthesized,
        # recorded as a derivation, and tied to the file whose sha256 the data
        # card records. Re-export the source and the ids change.
        synth = (cfg.get("derivations") or {}).get("ticket_id")
        if synth != "row_index":
            raise ValueError(
                "mapping must provide ticket_id. If the CSV genuinely has no id "
                "column, set `derivations: {ticket_id: row_index}` to synthesize "
                "one from row position — positional ids are stable only for this "
                "exact file, and that caveat is recorded in the audit."
            )
        prefix = str((cfg.get("dataset") or {}).get("id_prefix") or "T")
        out["ticket_id"] = [f"{prefix}{i:07d}" for i in range(len(out))]
        report.derived["ticket_id"] = (
            f"synthesized from row position as {prefix}<7-digit index>; the CSV "
            f"had no id column. Stable only for the exact file recorded in the "
            f"data card's raw_file_sha256."
        )
        report.rejections.append(
            "no ticket_id column in the source; ids were synthesized from row "
            "position. They will not match any id in the originating system."
        )
    out["ticket_id"] = out["ticket_id"].map(
        lambda v: None if v is None else str(v).strip()
    )
    before = len(out)
    out = out[out["ticket_id"].notna() & (out["ticket_id"] != "")]
    if len(out) < before:
        report.rejections.append(f"{before - len(out)} rows dropped: null/empty ticket_id")
    dupes = int(out["ticket_id"].duplicated().sum())
    report.duplicate_ticket_ids = dupes
    if dupes:
        out = out[~out["ticket_id"].duplicated(keep="first")]
        report.rejections.append(f"{dupes} duplicate ticket_id rows dropped (kept first)")

    # --- customer id -> hash -------------------------------------------------
    cust_src = (cfg.get("privacy") or {}).get("customer_id_column")
    if cust_src and cust_src in df.columns:
        out["customer_id_hash"] = df.loc[out.index, cust_src].map(hash_identifier)
        report.derived["customer_id_hash"] = f"sha256(salt + {cust_src})[:16]"
    elif "customer_id_hash" in out.columns:
        out["customer_id_hash"] = out["customer_id_hash"].map(hash_identifier)
        report.derived["customer_id_hash"] = "sha256(salt + mapped value)[:16]"

    # --- timestamps ----------------------------------------------------------
    ts_fmt = (norm.get("timestamps") or {}).get("format")
    for f in C.TIMESTAMP_FIELDS:
        if f in out.columns:
            out[f] = parse_timestamps(out[f], ts_fmt, report, f)

    # --- status --------------------------------------------------------------
    if "status" in out.columns:
        raw_counts = out["status"].astype("object").value_counts(dropna=True).to_dict()
        report.normalizations.setdefault("status", {})["raw_value_counts"] = {
            str(k): int(v) for k, v in list(raw_counts.items())[:50]
        }
        out["status"] = normalize_status(out["status"], norm.get("status") or {}, report)

    # --- booleans ------------------------------------------------------------
    bools = norm.get("booleans") or {}
    tvals = bools.get("true_values") or ["yes", "true", "1", "y", "t"]
    fvals = bools.get("false_values") or ["no", "false", "0", "n", "f"]
    for f in C.BOOLEAN_FIELDS:
        if f in out.columns:
            out[f] = normalize_boolean(out[f], tvals, fvals, report, f)

    # --- numerics ------------------------------------------------------------
    for f in C.NUMERIC_FIELDS:
        if f in out.columns:
            coerced = pd.to_numeric(out[f], errors="coerce")
            failures = int(out[f].notna().sum() - coerced.notna().sum())
            if failures:
                report.parse_failures[f] = failures
            out[f] = coerced

    # --- CSAT semantics ------------------------------------------------------
    csat_cfg = norm.get("csat") or {}
    if "csat_score" in out.columns:
        lo = csat_cfg.get("scale_min")
        hi = csat_cfg.get("scale_max")
        zero_means_no_response = bool(csat_cfg.get("zero_means_no_response", False))
        info: dict[str, Any] = {
            "scale_min": lo,
            "scale_max": hi,
            "zero_means_no_response": zero_means_no_response,
        }
        if zero_means_no_response:
            n_zero = int((out["csat_score"] == 0).sum())
            out.loc[out["csat_score"] == 0, "csat_score"] = np.nan
            info["zeros_treated_as_no_response"] = n_zero
        if lo is not None and hi is not None:
            oob = int(((out["csat_score"] < lo) | (out["csat_score"] > hi)).sum())
            if oob:
                out.loc[
                    (out["csat_score"] < lo) | (out["csat_score"] > hi), "csat_score"
                ] = np.nan
                info["out_of_range_nulled"] = oob
        report.normalizations["csat_score"] = info

    # --- text ----------------------------------------------------------------
    def _as_text(v: object) -> str:
        # NaN must become "", not the literal string "nan". A stray "nan" in
        # resolution_notes would silently qualify a ticket as a retrieval source.
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        s = str(v).strip()
        return "" if s.lower() in ("nan", "nat", "none") else s

    for f in ("issue_subject", "issue_description", "resolution_notes"):
        if f in out.columns:
            out[f] = out[f].map(_as_text)

    subj = out["issue_subject"] if "issue_subject" in out.columns else None
    desc = out["issue_description"] if "issue_description" in out.columns else None
    if subj is not None and desc is not None:
        out["issue_text"] = (subj.fillna("") + "\n\n" + desc.fillna("")).str.strip()
        report.derived["issue_text"] = "issue_subject + '\\n\\n' + issue_description"
    elif desc is not None:
        out["issue_text"] = desc.fillna("")
        report.derived["issue_text"] = "issue_description (no subject column mapped)"
    elif subj is not None:
        out["issue_text"] = subj.fillna("")
        report.derived["issue_text"] = "issue_subject (no description column mapped)"
    else:
        raise ValueError(
            "mapping must provide issue_subject and/or issue_description; "
            "issue_text cannot be constructed"
        )

    # --- derivations ---------------------------------------------------------
    derivations = cfg.get("derivations") or {}

    if derivations.get("resolution_time_hours") == "from_timestamps":
        if "created_at" in out.columns and "resolved_at" in out.columns:
            delta = (out["resolved_at"] - out["created_at"]).dt.total_seconds() / 3600.0
            negatives = int((delta < 0).sum())
            delta = delta.where(delta >= 0)
            out["resolution_time_hours"] = delta
            report.derived["resolution_time_hours"] = (
                "(resolved_at - created_at) in hours; negative durations nulled"
            )
            if negatives:
                report.rejections.append(
                    f"{negatives} rows had resolved_at < created_at; "
                    f"resolution_time_hours nulled for those rows"
                )
        else:
            report.rejections.append(
                "resolution_time_hours derivation requested but created_at and/or "
                "resolved_at are unavailable; field left missing"
            )

    if derivations.get("response_time_hours") == "from_timestamps":
        if "created_at" in out.columns and "first_response_at" in out.columns:
            delta = (
                out["first_response_at"] - out["created_at"]
            ).dt.total_seconds() / 3600.0
            out["response_time_hours"] = delta.where(delta >= 0)
            report.derived["response_time_hours"] = (
                "(first_response_at - created_at) in hours; negatives nulled"
            )
        else:
            report.rejections.append(
                "response_time_hours derivation requested but the required timestamps "
                "are unavailable; field left missing"
            )

    # --- redaction (applied before anything is persisted) --------------------
    for f in ("issue_subject", "issue_description", "issue_text", "resolution_notes"):
        if f in out.columns:
            out[f] = out[f].map(redact_text)
    report.normalizations["redaction"] = {
        "applied_to": [
            f
            for f in ("issue_subject", "issue_description", "issue_text", "resolution_notes")
            if f in out.columns
        ],
        "note": "redaction runs before parquet write, embedding, logging, UI, and prompts",
    }

    # --- personal columns explicitly dropped ---------------------------------
    mapped_sources = set(columns.values())
    for col in df.columns:
        if str(col) in mapped_sources:
            continue
        if looks_personal(str(col)):
            report.dropped_personal_columns.append(str(col))
        else:
            report.unexpected_columns.append(str(col))

    # --- finalize ------------------------------------------------------------
    for f in C.CANONICAL_FIELDS:
        if f not in out.columns:
            report.missing.append(f)
            out[f] = pd.Series([None] * len(out), index=out.index, dtype="object")

    out = out[list(C.CANONICAL_FIELDS)]
    out = out.reset_index(drop=True)
    report.rows_out = len(out)
    return out, report
