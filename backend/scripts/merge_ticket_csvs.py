"""Merge several ticket CSVs that share a schema into one ingestible file.

Written for a Kaggle export shipped as five partially-overlapping CSVs. Two
carried no answer column and were dropped; the remaining three turned out not to
be subsets of one another — the 20k file overlapped the largest by 38.5% and the
4k file not at all — so merging recovered 53% more usable rows than any single
file.

Deduplicates on the issue body, because the same ticket appearing twice in a
retrieval corpus is one case that looks like corroboration.

    python -m backend.scripts.merge_ticket_csvs \\
        --inputs "data/raw/dir/a.csv" "data/raw/dir/b.csv" \\
        --out data/raw/merged.csv --require-column answer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Merge ticket CSVs sharing a schema.")
    p.add_argument("--inputs", nargs="+", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--require-column",
        action="append",
        default=[],
        help="skip any input lacking this column (repeatable)",
    )
    p.add_argument(
        "--dedup-on",
        default="body",
        help="column to deduplicate on (default: body)",
    )
    args = p.parse_args(argv)

    frames: list[pd.DataFrame] = []
    total_in = 0

    for path in args.inputs:
        if not path.exists():
            print(f"  SKIP {path.name}: not found", file=sys.stderr)
            continue
        df = pd.read_csv(path, dtype=str)
        missing = [c for c in args.require_column if c not in df.columns]
        if missing:
            print(f"  SKIP {path.name}: missing required column(s) {missing}")
            continue
        total_in += len(df)
        df["_source_file"] = path.name
        frames.append(df)
        print(f"  KEEP {path.name}: {len(df)} rows")

    if not frames:
        print("no usable inputs", file=sys.stderr)
        return 2

    merged = pd.concat(frames, ignore_index=True, sort=False)

    if args.dedup_on in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=[args.dedup_on], keep="first")
        dropped = before - len(merged)
        print(f"\nDeduplicated on {args.dedup_on!r}: dropped {dropped} duplicate rows")
    else:
        print(f"\n! column {args.dedup_on!r} not present; no deduplication applied")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    print(f"\n{total_in} rows in -> {len(merged)} rows out")
    print(f"Wrote {args.out}")
    print("\nProvenance is preserved in the _source_file column, so the audit can")
    print("still attribute any row to the file it came from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
