"""Dataset audit.

Produces the evidence that every later phase is gated on. It measures; it never
repairs. Anything it cannot verify is reported as unavailable rather than
guessed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core import canonical as C
from backend.app.core.redaction import find_placeholders, redact_text, scan_pii
from backend.app.services import text_utils as T

# Phrases in issue text that suggest the outcome leaked backwards into the
# problem description. Reported, never auto-rejected.
LEAKAGE_PHRASES = (
    "escalated",
    "escalation",
    "sla breach",
    "breached sla",
    "missed sla",
    "sla was missed",
    "reopened",
    "re-opened",
    "ticket was closed",
    "resolved by",
    "refund issued",
)

SAMPLE_CAP = 20_000  # rows sampled for expensive text statistics


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.isoformat()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if pd.isna(v):  # type: ignore[arg-type]
        return None
    return str(v)


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0


# --- field-level ------------------------------------------------------------


def audit_fields(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    out: dict[str, Any] = {}
    for f in C.CANONICAL_FIELDS:
        if f not in df.columns:
            out[f] = {"present": False}
            continue
        s = df[f]
        if f in C.TEXT_FIELDS:
            non_null = int((s.notna() & (s.astype("object") != "")).sum())
        else:
            non_null = int(s.notna().sum())
        entry: dict[str, Any] = {
            "present": True,
            "non_null_count": non_null,
            "null_pct": _pct(n - non_null, n),
            "unique_count": int(s.nunique(dropna=True)),
        }
        if f in C.CATEGORICAL_FIELDS or f in C.BOOLEAN_FIELDS:
            vc = s.value_counts(dropna=True).head(15)
            entry["top_values"] = {str(k): int(v) for k, v in vc.items()}
        elif f in C.NUMERIC_FIELDS:
            vals = pd.to_numeric(s, errors="coerce").dropna()
            if len(vals):
                entry["stats"] = {
                    "min": float(vals.min()),
                    "p25": float(vals.quantile(0.25)),
                    "p50": float(vals.quantile(0.50)),
                    "p75": float(vals.quantile(0.75)),
                    "p95": float(vals.quantile(0.95)),
                    "max": float(vals.max()),
                    "mean": float(vals.mean()),
                    "negative_count": int((vals < 0).sum()),
                    "zero_count": int((vals == 0).sum()),
                }
        elif f in C.TIMESTAMP_FIELDS:
            vals = pd.to_datetime(s, errors="coerce").dropna()
            if len(vals):
                entry["range"] = {
                    "min": vals.min().isoformat(),
                    "max": vals.max().isoformat(),
                }
        elif f in C.TEXT_FIELDS:
            examples = [
                redact_text(x)[:200]
                for x in s.dropna().astype(str).head(3).tolist()
                if x.strip()
            ]
            entry["redacted_examples"] = examples
        out[f] = entry
    return out


# --- text -------------------------------------------------------------------


def audit_text_field(
    series: pd.Series, name: str, group_ids: Sequence[int] | None = None
) -> dict[str, Any]:
    raw = [str(x) for x in series.fillna("").astype(str).tolist()]
    non_empty = [x for x in raw if x.strip()]
    n = len(non_empty)
    if n == 0:
        return {"field": name, "non_empty_count": 0, "usable": False}

    normed = [T.normalize_for_dedup(x) for x in non_empty]
    exact_ratio = T.unique_ratio(non_empty)
    norm_ratio = T.unique_ratio(normed)

    lengths = np.array([len(x) for x in non_empty])
    # Rough token proxy; the true count comes from the tokenizer in Phase 4.
    approx_tokens = np.array([max(1, len(x.split())) for x in non_empty])

    placeholder_rows = 0
    placeholder_tokens: dict[str, int] = {}
    for x in non_empty:
        hits = find_placeholders(x)
        if hits:
            placeholder_rows += 1
            for h in hits:
                key = h.strip().lower()
                placeholder_tokens[key] = placeholder_tokens.get(key, 0) + 1

    entry: dict[str, Any] = {
        "field": name,
        "non_empty_count": n,
        "empty_count": len(raw) - n,
        "exact_unique_ratio": exact_ratio,
        "exact_unique_label": T.repetition_label(exact_ratio),
        "normalized_unique_ratio": norm_ratio,
        "normalized_unique_label": T.repetition_label(norm_ratio),
        "duplicate_count": n - len(set(non_empty)),
        "normalized_duplicate_count": n - len(set(normed)),
        "char_length_percentiles": {
            "p05": int(np.percentile(lengths, 5)),
            "p50": int(np.percentile(lengths, 50)),
            "p95": int(np.percentile(lengths, 95)),
            "max": int(lengths.max()),
        },
        "approx_word_count_percentiles": {
            "p50": int(np.percentile(approx_tokens, 50)),
            "p95": int(np.percentile(approx_tokens, 95)),
            "max": int(approx_tokens.max()),
        },
        "pct_over_256_words": _pct(int((approx_tokens > 256).sum()), n),
        "placeholder_rows": placeholder_rows,
        "placeholder_pct": _pct(placeholder_rows, n),
        "placeholder_tokens": dict(
            sorted(placeholder_tokens.items(), key=lambda kv: -kv[1])[:20]
        ),
        "ratio_note": (
            "Uniqueness ratios are diagnostic labels only. They are not a pass/fail "
            "threshold and do not prove the data is or is not usable."
        ),
    }
    if group_ids is not None:
        entry["template_groups"] = T.group_size_stats(group_ids)
    return entry


def audit_text(df: pd.DataFrame, group_ids: Sequence[int] | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "issue_text": audit_text_field(df["issue_text"], "issue_text", group_ids)
    }
    if "resolution_notes" in df.columns:
        out["resolution_notes"] = audit_text_field(
            df["resolution_notes"], "resolution_notes"
        )
        pairs = (
            df["issue_text"].fillna("").astype(str)
            + "||"
            + df["resolution_notes"].fillna("").astype(str)
        )
        mask = (df["issue_text"].fillna("") != "") & (
            df["resolution_notes"].fillna("") != ""
        )
        sub = pairs[mask]
        out["issue_resolution_pairs"] = {
            "count": len(sub),
            "duplicate_pair_count": int(len(sub) - sub.nunique()) if len(sub) else 0,
        }
        notes = df["resolution_notes"].fillna("").astype(str)
        boiler = notes.map(lambda v: v.strip().lower() in C.BOILERPLATE_RESOLUTIONS)
        out["resolution_notes"]["boilerplate_count"] = int(boiler.sum())
        out["resolution_notes"]["usable_count"] = int((~boiler & (notes != "")).sum())
    return out


# --- outcomes ---------------------------------------------------------------


def audit_outcomes(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    out: dict[str, Any] = {}

    for target in C.OUTCOME_CANDIDATES:
        if target not in df.columns or df[target].notna().sum() == 0:
            out[target] = {
                "available": False,
                "reason": "column absent from the dataset or entirely null",
            }
            continue

        s = df[target]
        entry: dict[str, Any] = {
            "available": True,
            "non_null_count": int(s.notna().sum()),
            "missing_pct": _pct(int(s.isna().sum()), n),
        }

        if target in C.BOOLEAN_FIELDS:
            # `== True` is deliberate and must not be rewritten to a truth check:
            # this is a nullable-boolean Series where None is a third state, so
            # the comparison is elementwise. `if s:` raises "truth value of a
            # Series is ambiguous", and `s.sum()` would silently count None.
            pos = int((s == True).sum())  # noqa: E712
            neg = int((s == False).sum())  # noqa: E712
            entry.update(
                {
                    "kind": "binary",
                    "definition": f"canonical boolean field '{target}' as mapped",
                    "positive_count": pos,
                    "negative_count": neg,
                    "prevalence": round(pos / (pos + neg), 4) if (pos + neg) else None,
                    "both_classes_present": pos > 0 and neg > 0,
                }
            )
            if "status" in df.columns and df["status"].notna().any():
                ct = pd.crosstab(df["status"], s.astype("object"), dropna=True)
                entry["by_status"] = {
                    str(k): {str(c): int(v) for c, v in row.items()}
                    for k, row in ct.to_dict("index").items()
                }
                # A perfect status->target mapping means the column is derived,
                # not observed.
                deterministic = all(
                    sum(1 for v in row.values() if v > 0) == 1
                    for row in ct.to_dict("index").values()
                )
                entry["deterministic_from_status"] = bool(deterministic)
                if deterministic:
                    entry["warning"] = (
                        f"'{target}' is perfectly determined by 'status'. It is almost "
                        f"certainly generated from it and carries no independent signal."
                    )
        else:
            vals = pd.to_numeric(s, errors="coerce").dropna()
            entry.update(
                {
                    "kind": "numeric",
                    "definition": f"canonical numeric field '{target}' as mapped/derived",
                    "valid_count": len(vals),
                    "p50": float(vals.median()) if len(vals) else None,
                    "p75": float(vals.quantile(0.75)) if len(vals) else None,
                    "nonpositive_count": int((vals <= 0).sum()) if len(vals) else 0,
                    "distinct_values": int(vals.nunique()) if len(vals) else 0,
                }
            )
            if len(vals) and vals.nunique() <= 2:
                entry["warning"] = (
                    f"'{target}' has {vals.nunique()} distinct values; it does not "
                    f"behave like a continuous outcome."
                )

        out[target] = entry

    # --- leakage scan on issue text ----------------------------------------
    sample = df["issue_text"].fillna("").astype(str)
    if len(sample) > SAMPLE_CAP:
        sample = sample.sample(SAMPLE_CAP, random_state=0)
    low = sample.str.lower()
    hits = {p: int(low.str.contains(p, regex=False).sum()) for p in LEAKAGE_PHRASES}
    out["_issue_text_leakage_scan"] = {
        "sampled_rows": len(sample),
        "phrase_hits": {k: v for k, v in hits.items() if v},
        "note": (
            "Reported, not auto-rejected. A ticket may legitimately say 'please "
            "escalate'. Review before using any target these phrases could reveal."
        ),
    }
    return out


# --- timestamps -------------------------------------------------------------


def audit_timestamps(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    has_created = "created_at" in df.columns and df["created_at"].notna().any()
    out["created_at_available"] = bool(has_created)

    if not has_created:
        out["temporal_split_feasible"] = False
        out["reason"] = (
            "No genuine ticket-creation timestamp is available. Time-based analytics "
            "and temporal splitting are disabled."
        )
        return out

    created = pd.to_datetime(df["created_at"], errors="coerce")
    out["range"] = {
        "min": created.min().isoformat(),
        "max": created.max().isoformat(),
        "span_days": int((created.max() - created.min()).days),
    }
    by_month = created.dt.to_period("M").value_counts().sort_index()
    out["density_by_month"] = {str(k): int(v) for k, v in by_month.items()}
    out["distinct_days"] = int(created.dt.date.nunique())

    if "resolved_at" in df.columns and df["resolved_at"].notna().any():
        resolved = pd.to_datetime(df["resolved_at"], errors="coerce")
        both = created.notna() & resolved.notna()
        violations = int((resolved[both] < created[both]).sum())
        out["resolved_before_created_violations"] = violations
        out["resolved_before_created_pct"] = _pct(violations, int(both.sum()))
        derived = (resolved - created).dt.total_seconds() / 3600.0
        out["derived_resolution_hours"] = {
            "valid_count": int((derived >= 0).sum()),
            "negative_count": int((derived < 0).sum()),
            "p50": float(derived[derived >= 0].median()) if (derived >= 0).any() else None,
        }
        if "resolution_time_hours" in df.columns:
            given = pd.to_numeric(df["resolution_time_hours"], errors="coerce")
            cmp_mask = given.notna() & derived.notna() & (derived >= 0)
            if cmp_mask.any():
                diff = (given[cmp_mask] - derived[cmp_mask]).abs()
                out["duration_column_vs_derived"] = {
                    "compared_rows": int(cmp_mask.sum()),
                    "median_abs_diff_hours": float(diff.median()),
                    "pct_within_1h": _pct(int((diff <= 1).sum()), int(cmp_mask.sum())),
                }

    span_ok = out["range"]["span_days"] >= 30 and out["distinct_days"] >= 10
    out["temporal_split_feasible"] = bool(span_ok)
    if not span_ok:
        out["reason"] = (
            "created_at exists but the date range is too narrow or too sparse for a "
            "meaningful temporal split."
        )
    return out


# --- PII --------------------------------------------------------------------


def audit_pii(df: pd.DataFrame, adapter_report: dict[str, Any]) -> dict[str, Any]:
    fields = [f for f in C.TEXT_FIELDS if f in df.columns]
    totals: dict[str, dict[str, int]] = {}
    for f in fields:
        s = df[f].fillna("").astype(str)
        if len(s) > SAMPLE_CAP:
            s = s.sample(SAMPLE_CAP, random_state=0)
        agg: dict[str, int] = {}
        for txt in s:
            for k, v in scan_pii(txt).items():
                agg[k] = agg.get(k, 0) + v
        totals[f] = {
            "sampled_rows": len(s),
            **dict(sorted(agg.items(), key=lambda kv: -kv[1])),
        }
    return {
        "post_redaction_scan": totals,
        "note": (
            "Counts are measured AFTER redaction on the stored text. Non-zero values "
            "indicate patterns the redactor did not catch and must be investigated."
        ),
        "dropped_personal_source_columns": adapter_report.get(
            "dropped_personal_columns", []
        ),
        "customer_id_handling": (
            "hashed with sha256(salt + id)[:16]"
            if "customer_id_hash" in adapter_report.get("derived_fields", {})
            else "no customer id column mapped"
        ),
        "protected_attributes_note": (
            "name, email, gender, and age are never mapped into canonical fields and "
            "are listed in FORBIDDEN_RISK_FEATURES."
        ),
    }


# --- assembly ---------------------------------------------------------------


def build_audit(
    df: pd.DataFrame,
    adapter_report: dict[str, Any],
    group_ids: Sequence[int] | None,
    file_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_file": file_meta,
        "schema": adapter_report,
        "fields": audit_fields(df),
        "text": audit_text(df, group_ids),
        "outcomes": audit_outcomes(df),
        "timestamps": audit_timestamps(df),
        "pii": audit_pii(df, adapter_report),
    }


def write_audit_json(audit: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(audit), fh, indent=2)


# --- markdown ---------------------------------------------------------------


def _md_table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def render_audit_markdown(audit: dict[str, Any]) -> str:
    s = audit["schema"]
    ds = s.get("dataset", {})
    parts: list[str] = []
    parts.append("# InsightDesk AI — Data Audit\n")
    parts.append(f"Generated: {audit['generated_at']}\n")

    parts.append("## Source\n")
    fm = audit["source_file"]
    parts.append(
        _md_table(
            [
                ("name", ds.get("name", "unspecified")),
                ("source_url", ds.get("source_url", "unspecified")),
                ("download_date", ds.get("download_date", "unspecified")),
                ("license_status", ds.get("license_status", "unverified")),
                ("file", fm.get("path", "")),
                ("sha256", fm.get("sha256", "")),
                ("bytes", fm.get("bytes", "")),
                ("rows_in", s.get("rows_in", "")),
                ("rows_out", s.get("rows_out", "")),
            ],
            ("key", "value"),
        )
        + "\n"
    )

    parts.append("## Canonical mapping\n")
    rows = []
    for f in C.CANONICAL_FIELDS:
        if f in s.get("canonical_mapping", {}):
            rows.append((f, "mapped", s["canonical_mapping"][f]))
        elif f in s.get("derived_fields", {}):
            rows.append((f, "derived", s["derived_fields"][f]))
        else:
            rows.append((f, "MISSING", "feature disabled"))
    parts.append(_md_table(rows, ("canonical field", "status", "source / definition")) + "\n")

    if s.get("rejections"):
        parts.append("## Rejections and warnings\n")
        for r in s["rejections"]:
            parts.append(f"- {r}")
        parts.append("")

    parts.append("## Text\n")
    for key, t in audit["text"].items():
        if key == "issue_resolution_pairs":
            parts.append(
                f"- **issue/resolution pairs**: {t['count']} pairs, "
                f"{t['duplicate_pair_count']} duplicated\n"
            )
            continue
        if not t.get("non_empty_count"):
            parts.append(f"- **{key}**: empty\n")
            continue
        parts.append(
            f"### {key}\n\n"
            f"- non-empty: {t['non_empty_count']}\n"
            f"- exact unique ratio: {t['exact_unique_ratio']} ({t['exact_unique_label']})\n"
            f"- normalized unique ratio: {t['normalized_unique_ratio']} "
            f"({t['normalized_unique_label']})\n"
            f"- placeholder rows: {t['placeholder_rows']} ({t['placeholder_pct']}%)\n"
            f"- char length p50/p95/max: {t['char_length_percentiles']['p50']}/"
            f"{t['char_length_percentiles']['p95']}/{t['char_length_percentiles']['max']}\n"
        )
        if t.get("template_groups"):
            g = t["template_groups"]
            parts.append(
                f"- near-duplicate template groups: {g['n_multi_member_groups']} groups "
                f"covering {g['pct_rows_in_multi_member_groups']}% of rows; top-10 groups "
                f"cover {g['pct_rows_in_top10_groups']}%\n"
            )

    parts.append("## Outcome candidates\n")
    rows = []
    for target in C.OUTCOME_CANDIDATES:
        e = audit["outcomes"].get(target, {})
        if not e.get("available"):
            rows.append((target, "unavailable", e.get("reason", ""), ""))
        elif e.get("kind") == "binary":
            rows.append(
                (
                    target,
                    "available",
                    f"prevalence {e.get('prevalence')}",
                    e.get("warning", ""),
                )
            )
        else:
            rows.append(
                (target, "available", f"p50 {e.get('p50')}, p75 {e.get('p75')}", e.get("warning", ""))
            )
    parts.append(_md_table(rows, ("target", "status", "summary", "warning")) + "\n")

    leak = audit["outcomes"].get("_issue_text_leakage_scan", {})
    if leak.get("phrase_hits"):
        parts.append("### Leakage phrases found in issue text\n")
        parts.append(
            _md_table(
                [(k, v) for k, v in leak["phrase_hits"].items()], ("phrase", "rows")
            )
            + f"\n\n{leak['note']}\n"
        )

    parts.append("## Timestamps\n")
    ts = audit["timestamps"]
    if not ts.get("created_at_available"):
        parts.append(f"- created_at unavailable. {ts.get('reason', '')}\n")
    else:
        parts.append(
            f"- range: {ts['range']['min']} → {ts['range']['max']} "
            f"({ts['range']['span_days']} days, {ts['distinct_days']} distinct days)\n"
            f"- temporal split feasible: {ts['temporal_split_feasible']}\n"
        )
        if "resolved_before_created_violations" in ts:
            parts.append(
                f"- resolved_at < created_at violations: "
                f"{ts['resolved_before_created_violations']} "
                f"({ts['resolved_before_created_pct']}%)\n"
            )

    parts.append("## PII\n")
    pii = audit["pii"]
    parts.append(f"- {pii['note']}\n")
    for f, counts in pii["post_redaction_scan"].items():
        residual = {k: v for k, v in counts.items() if k != "sampled_rows"}
        parts.append(
            f"- **{f}**: sampled {counts['sampled_rows']} rows, residual hits: "
            f"{residual or 'none'}\n"
        )
    if pii["dropped_personal_source_columns"]:
        parts.append(
            f"- source columns dropped as personal: "
            f"{', '.join(pii['dropped_personal_source_columns'])}\n"
        )
    parts.append(f"- customer id: {pii['customer_id_handling']}\n")

    return "\n".join(parts)
