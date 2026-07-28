"""Score a candidate dataset BEFORE committing to it.

The failure this exists to catch: a dataset whose description says "200,000 real
customer support tickets" and whose `issue_description` column holds ten distinct
strings repeated 20,000 times each. Retrieval over ten unique documents is not
retrieval, and no amount of downstream engineering fixes it.

Reads only the first N rows, so it is fast and cheap on a multi-GB file.

    python -m backend.scripts.triage_dataset --csv data/raw/candidate.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# A column is a plausible free-text issue or resolution field if its name hints
# at it. Name-based, so it is a starting point for a human, never a conclusion.
ISSUE_HINTS = ("description", "issue", "body", "text", "message", "subject",
               "question", "complaint", "inbound", "content")
RESOLUTION_HINTS = ("resolution", "answer", "reply", "response", "solution",
                    "fix", "outbound", "agent")

VERDICTS = {
    "good": "GOOD",
    "weak": "WEAK",
    "unusable": "UNUSABLE",
}


def uniqueness(series: pd.Series) -> dict[str, Any]:
    vals = series.dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    n = len(vals)
    if n == 0:
        return {"rows": 0, "unique": 0, "ratio": 0.0, "mean_chars": 0.0}
    uniq = vals.nunique()
    return {
        "rows": n,
        "unique": uniq,
        "ratio": round(uniq / n, 4),
        "mean_chars": round(vals.str.len().mean(), 1),
        "top_repeat_pct": round(100 * vals.value_counts().iloc[0] / n, 1),
    }


def judge_text_column(stats: dict[str, Any]) -> tuple[str, str]:
    """Verdict for one candidate free-text column."""
    if stats["rows"] == 0:
        return "unusable", "column is empty"
    if stats["unique"] < 50:
        return (
            "unusable",
            f"only {stats['unique']} distinct values across {stats['rows']} rows — "
            f"this is a template, not free text. Retrieval over it is meaningless.",
        )
    if stats["ratio"] < 0.05:
        return (
            "unusable",
            f"uniqueness ratio {stats['ratio']} — over 95% of rows repeat an "
            f"existing value.",
        )
    if stats["ratio"] < 0.3:
        return "weak", f"uniqueness ratio {stats['ratio']} — heavily templated."
    if stats["mean_chars"] < 40:
        return "weak", f"mean length {stats['mean_chars']} chars — very short for a ticket."
    return "good", f"{stats['unique']} distinct values, ratio {stats['ratio']}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score a dataset's fitness for this project.")
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--rows", type=int, default=50_000, help="rows to sample (default 50k)")
    args = p.parse_args(argv)

    if not args.csv.exists():
        print(f"not found: {args.csv}", file=sys.stderr)
        return 2

    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(args.csv, nrows=args.rows, encoding=enc, dtype=str)
            break
        except UnicodeDecodeError:
            continue
    else:
        print("could not decode the file", file=sys.stderr)
        return 2

    size_mb = args.csv.stat().st_size / (1024 * 1024)
    print(f"\n{args.csv.name}  —  {size_mb:.1f} MB, sampled {len(df)} rows, "
          f"{len(df.columns)} columns\n")

    issue_cols = [c for c in df.columns if any(h in str(c).lower() for h in ISSUE_HINTS)]
    res_cols = [c for c in df.columns if any(h in str(c).lower() for h in RESOLUTION_HINTS)]

    def _report(title: str, cols: list[str]) -> list[tuple[str, str, str]]:
        print(f"--- {title} ---")
        if not cols:
            print("  none found by name\n")
            return []
        out: list[tuple[str, str, str]] = []
        for c in cols:
            s = uniqueness(df[c])
            verdict, why = judge_text_column(s)
            out.append((c, verdict, why))
            print(f"  {VERDICTS[verdict]:<9} {c}")
            print(f"            {s['unique']} unique / {s['rows']} rows "
                  f"(ratio {s['ratio']}), mean {s['mean_chars']} chars, "
                  f"most common value is {s.get('top_repeat_pct', 0)}% of rows")
            print(f"            {why}")
        print()
        return out

    issue_verdicts = _report("Candidate ISSUE text columns", issue_cols)
    res_verdicts = _report("Candidate RESOLUTION text columns", res_cols)

    # --- supporting structure -------------------------------------------------
    print("--- Supporting fields ---")
    status_cols = [c for c in df.columns if "status" in str(c).lower()]
    date_cols = [c for c in df.columns if any(h in str(c).lower()
                                              for h in ("date", "time", "created", "resolved"))]
    print(f"  status-like columns:    {status_cols or 'none'}")
    print(f"  timestamp-like columns: {date_cols or 'none'}")
    if not date_cols:
        print("    -> no timestamp: time-based analytics and temporal splitting disable.")
    print()

    # --- overall --------------------------------------------------------------
    best_issue = _best(issue_verdicts)
    best_res = _best(res_verdicts)

    print("=== VERDICT ===")
    if best_issue == "good" and best_res == "good":
        print("  USABLE. Real free-text on both the problem and resolution side.")
        print("  Run --suggest-mapping next, review every guess, then ingest.")
        code = 0
    elif best_issue in ("good", "weak") and best_res in ("good", "weak"):
        print("  MARGINAL. Text exists on both sides but is templated.")
        print("  Retrieval will work but the evaluation numbers will saturate —")
        print("  expect Hit@K near 1.0 that measures deduplication, not relevance.")
        code = 1
    else:
        print("  NOT USABLE for retrieval.")
        if best_issue not in ("good", "weak"):
            print("  The issue-text side has too few distinct values.")
        if best_res not in ("good", "weak"):
            print("  The resolution side has too few distinct values, or is absent.")
        print("  The capability manifest would disable retrieval on this data,")
        print("  which is the correct outcome — but it means no demo.")
        code = 3
    print()
    return code


def _best(verdicts: list[tuple[str, str, str]]) -> str:
    order = ["good", "weak", "unusable"]
    found = [v for _, v, _ in verdicts]
    for level in order:
        if level in found:
            return level
    return "unusable"


if __name__ == "__main__":
    raise SystemExit(main())
