"""Phase 2 — generate artifacts/capabilities.json from the audit and parquet.

Explicit command. Never runs at API startup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from backend.app.core.config import get_settings
from backend.app.services import capabilities as cap


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    p = argparse.ArgumentParser(description="Build the capability manifest.")
    p.add_argument("--parquet", type=Path, default=settings.tickets_parquet)
    p.add_argument("--audit", type=Path, default=settings.audit_json)
    p.add_argument("--out", type=Path, default=settings.capabilities_json)
    args = p.parse_args(argv)

    if not args.parquet.exists() or not args.audit.exists():
        print(
            "Run `python -m backend.scripts.ingest_tickets` first — "
            f"missing {args.parquet if not args.parquet.exists() else args.audit}",
            file=sys.stderr,
        )
        return 2

    df = pd.read_parquet(args.parquet)
    with open(args.audit, encoding="utf-8") as fh:
        audit = json.load(fh)

    caps = cap.build_capabilities(df, audit)
    cap.write_capabilities(caps, args.out)

    print(f"Wrote {args.out}\n")
    for name in (
        "analytics",
        "retrieval",
        "resolution_generation",
        "analyst_agent",
        "clustering",
        "risk",
        "llm_provider",
        "ai_ops",
    ):
        block = caps[name]
        state = "ENABLED " if block["enabled"] else "disabled"
        print(f"  {state}  {name}")
        if not block["enabled"]:
            print(f"            reason: {block['reason']}")

    r = caps["retrieval"]
    if r["enabled"] and r.get("note"):
        print(f"\n  corpus: serving {r['corpus_size_served']} of "
              f"{r['corpus_size_audited']} audited source cases")
    if r["enabled"]:
        print(f"  retrieval evaluation status: {r['evaluation_status']}")

    a = caps["analytics"]
    print(f"\n  analytics metrics available: {', '.join(a['available_metrics'])}")
    if a["unavailable_metrics"]:
        for m, why in a["unavailable_metrics"].items():
            print(f"    - {m}: {why}")
    if caps["risk"]["enabled"]:
        r = caps["risk"]
        print(f"\n  risk target: {r['target']} (target_kind={r['target_kind']}, "
              f"prevalence={r['prevalence']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
