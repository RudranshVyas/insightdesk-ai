"""Phase 1 — ingest a raw ticket CSV, audit it, write the canonical parquet.

Two modes:

  --suggest-mapping   profile the CSV and emit a mapping YAML for human review
  --mapping PATH      apply a reviewed mapping, audit, and write artifacts

Never runs at API startup.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
import yaml

from backend.app.core.config import get_settings
from backend.app.services import audit as audit_mod
from backend.app.services import data_card
from backend.app.services import schema_adapter as adapter
from backend.app.services import text_utils as T


def file_meta(path: Path) -> dict[str, object]:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": h.hexdigest(),
    }


def read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, nrows=nrows, encoding=enc, dtype=str, keep_default_na=True)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"could not decode {path} as utf-8, utf-8-sig, or latin-1")


def cmd_suggest(csv_path: Path, out_path: Path) -> int:
    df = read_csv(csv_path, nrows=5000)
    proposal = adapter.suggest_mapping(df)
    proposal["dataset"]["name"] = csv_path.stem

    profile = {
        col: {
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "unique": int(df[col].nunique(dropna=True)),
            "examples": [str(x)[:80] for x in df[col].dropna().head(3).tolist()],
        }
        for col in df.columns
    }
    proposal["_source_profile"] = profile

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(proposal, fh, sort_keys=False, allow_unicode=True, width=100)

    print(f"Wrote suggested mapping to {out_path}")
    print("\nREVIEW IT BEFORE USE. Every value is a name-based guess.")
    print("Pay particular attention to created_at: a purchase or signup date is NOT")
    print("ticket creation time. Set anything you cannot verify to null.\n")
    print(f"Source columns ({len(df.columns)}): {', '.join(map(str, df.columns))}")
    return 0


def cmd_ingest(
    csv_path: Path, mapping_path: Path, sample: int | None, no_dedup: bool
) -> int:
    settings = get_settings()
    settings.ensure_dirs()

    cfg = adapter.load_mapping(mapping_path)
    meta = file_meta(csv_path)
    print(f"Reading {csv_path} ...")
    df_raw = read_csv(csv_path, nrows=sample)
    meta["rows_read"] = len(df_raw)
    print(f"  {len(df_raw)} rows, {len(df_raw.columns)} columns")

    print("Applying schema adapter ...")
    df, report = adapter.apply_mapping(df_raw, cfg)
    print(f"  {report.rows_out} canonical rows")
    for r in report.rejections:
        print(f"  ! {r}")

    # --- documented derived helper columns ---------------------------------
    resolved_statuses = {
        str(s).strip().lower()
        for s in ((cfg.get("normalization") or {}).get("resolved_definition") or {}).get(
            "statuses", ["resolved", "closed"]
        )
    }
    if df["status"].notna().any():
        df["is_resolved"] = df["status"].map(
            lambda v: None if v is None else str(v).strip().lower() in resolved_statuses
        )
        resolved_def = f"status in {sorted(resolved_statuses)}"
    elif df["resolved_at"].notna().any():
        df["is_resolved"] = df["resolved_at"].notna()
        resolved_def = "resolved_at is not null (no usable status column)"
    else:
        df["is_resolved"] = None
        resolved_def = "UNAVAILABLE — no status and no resolved_at"
    report.derived["is_resolved"] = resolved_def
    print(f"  is_resolved: {resolved_def}")

    group_ids = None
    if no_dedup:
        df["template_group_id"] = range(len(df))
        report.derived["template_group_id"] = "SKIPPED (--no-dedup); every row its own group"
    else:
        print("Grouping near-duplicate templates (MinHash LSH) ...")
        group_ids = T.template_groups(df["issue_text"].fillna("").astype(str).tolist())
        df["template_group_id"] = group_ids
        report.derived["template_group_id"] = (
            "MinHash-LSH over char-5-grams of dedup-normalized issue_text, "
            "Jaccard threshold 0.8, num_perm 64, connected components"
        )
        stats = T.group_size_stats(group_ids)
        print(
            f"  {stats['n_groups']} groups; "
            f"{stats['pct_rows_in_multi_member_groups']}% of rows sit in a "
            f"multi-member template group"
        )

    print("Building audit ...")
    audit = audit_mod.build_audit(df, report.to_dict(), group_ids, meta)
    audit_mod.write_audit_json(audit, str(settings.audit_json))
    with open(settings.audit_md, "w", encoding="utf-8") as fh:
        fh.write(audit_mod.render_audit_markdown(audit))

    card = data_card.build_data_card(
        cfg,
        meta,
        {"rows_in": report.rows_in, "rows_out": report.rows_out},
        report.to_dict(),
    )
    data_card.write_data_card(card, settings.data_card_json)

    df.to_parquet(settings.tickets_parquet, index=False)

    print(f"\nWrote {settings.audit_json}")
    print(f"Wrote {settings.audit_md}")
    print(f"Wrote {settings.data_card_json}  (licence: {card['license']['status']})")
    print(f"Wrote {settings.tickets_parquet}")

    missing = report.missing
    print(f"\nMissing canonical fields ({len(missing)}): {', '.join(missing) or 'none'}")
    print("Next: python -m backend.scripts.build_capabilities")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest and audit a raw ticket CSV.")
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--mapping", type=Path, help="reviewed mapping YAML")
    p.add_argument(
        "--suggest-mapping",
        type=Path,
        nargs="?",
        const=Path("backend/config/schema_map.suggested.yaml"),
        help="profile the CSV and write a mapping proposal for review",
    )
    p.add_argument("--sample", type=int, default=None, help="read only N rows (dev)")
    p.add_argument(
        "--no-dedup", action="store_true", help="skip MinHash template grouping (dev)"
    )
    args = p.parse_args(argv)

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        print(
            "Place the dataset in data/raw/ manually. This app never downloads it.",
            file=sys.stderr,
        )
        return 2

    if args.suggest_mapping:
        return cmd_suggest(args.csv, args.suggest_mapping)
    if not args.mapping:
        print("Provide --mapping PATH, or --suggest-mapping to generate one.", file=sys.stderr)
        return 2
    if not args.mapping.exists():
        print(f"Mapping not found: {args.mapping}", file=sys.stderr)
        return 2
    return cmd_ingest(args.csv, args.mapping, args.sample, args.no_dedup)


if __name__ == "__main__":
    raise SystemExit(main())
