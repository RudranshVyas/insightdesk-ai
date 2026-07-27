"""Phase 5 — evaluate retrieval and write artifacts/retrieval/evaluation.json.

Offline command. Compares BM25-only, dense-only, RRF hybrid, and hybrid with the
metadata boost, on both tiers, and prints the honest winner.

  python -m backend.scripts.eval_retrieval
  python -m backend.scripts.eval_retrieval --loo-sample 500
  python -m backend.scripts.eval_retrieval --configs bm25 hybrid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.app.core.capability_loader import load_capabilities
from backend.app.core.config import get_settings
from backend.app.evaluation import regression_gate as RG
from backend.app.evaluation import retrieval_eval as E
from backend.app.services import retrieval as R


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    p = argparse.ArgumentParser(description="Evaluate hybrid retrieval.")
    p.add_argument("--configs", nargs="+", default=list(E.CONFIGS), choices=list(E.CONFIGS))
    p.add_argument("--loo-sample", type=int, default=200)
    p.add_argument(
        "--labeled",
        type=Path,
        default=None,
        help="path to the graded query set (default: data/evaluation/retrieval_queries.jsonl)",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--ignore-capabilities", action="store_true")
    p.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="regression baseline (default: artifacts/retrieval/baseline.json)",
    )
    p.add_argument(
        "--record-baseline",
        action="store_true",
        help="overwrite the baseline with this run's numbers (a human decision)",
    )
    p.add_argument(
        "--check-regression",
        action="store_true",
        help="exit non-zero if any metric dropped beyond tolerance (CI gate)",
    )
    p.add_argument(
        "--gate-tier",
        default="tier1_leave_one_out",
        choices=["tier1_leave_one_out", "tier2_manual_labeled"],
    )
    args = p.parse_args(argv)

    caps = load_capabilities()
    if not args.ignore_capabilities and not caps.get("retrieval", {}).get("enabled"):
        print(
            "retrieval is disabled by the capability manifest:\n  "
            f"{caps.get('retrieval', {}).get('reason')}",
            file=sys.stderr,
        )
        return 3

    if not (settings.retrieval_dir / "manifest.json").exists():
        print(
            "no retrieval index found; run `python -m backend.scripts.build_retrieval_index`",
            file=sys.stderr,
        )
        return 2

    labeled = args.labeled or (settings.evaluation_dir / "retrieval_queries.jsonl")
    out = args.out or (settings.retrieval_dir / "evaluation.json")

    print("Loading retrieval artifacts ...")
    retriever = R.HybridRetriever.load(settings)
    print(f"  corpus: {len(retriever.corpus)} source cases")

    report = E.build_report(
        retriever,
        retriever.corpus,
        labeled,
        configs=args.configs,
        loo_sample=args.loo_sample,
        index_manifest=retriever.manifest,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote {out}\n")

    # --- Tier 1 ---------------------------------------------------------------
    print("Tier 1 — leave-one-out (WEAK DIAGNOSTIC, self + template siblings excluded)")
    print(
        f"  {'config':<18}{'Hit@3':>8}{'Hit@5':>8}{'MRR@5':>8}"
        f"{'rand@3':>9}{'lift@3':>9}{'p50 ms':>9}{'p95 ms':>9}"
    )
    for config, r in report["tier1_leave_one_out"].items():
        if not r.get("available"):
            print(f"  {config:<18}unavailable: {r.get('reason')}")
            continue
        m, lat = r["metrics"], r["latency_ms"]
        base = r["random_baseline"]["metrics"]
        lift = r["lift_over_random"]
        print(
            f"  {config:<18}{_fmt(m['hit_at_3']):>8}{_fmt(m['hit_at_5']):>8}"
            f"{_fmt(m['mrr_at_5']):>8}{_fmt(base['hit_at_3']):>9}"
            f"{_fmt(lift['hit_at_3']):>9}{_fmt(lat['p50']):>9}{_fmt(lat['p95']):>9}"
        )
    print(
        "  Hit@K here saturates on a coarse proxy label. Read lift over the random "
        "baseline, not the raw number."
    )

    # --- Tier 2 ---------------------------------------------------------------
    tier2 = report["tier2_manual_labeled"]
    print(f"\nTier 2 — manual labeled set: {tier2['status']}")
    if tier2["status"] == "not_yet_labeled":
        print(f"  {tier2['detail']}")
        print(f"  Create {labeled} with 30-50 graded queries, then re-run.")
    elif tier2["status"] == "invalid_labeled_set":
        for err in tier2["validation"]["errors"][:20]:
            print(f"  ERROR {err}")
        return 1
    else:
        for w in tier2["validation"]["warnings"][:20]:
            print(f"  warning: {w}")
        print(
            f"\n  {'config':<18}{'Hit@3':>8}{'Hit@5':>8}{'Rec@5':>8}"
            f"{'MRR@5':>8}{'nDCG@5':>9}{'p50 ms':>9}{'p95 ms':>9}"
        )
        for config, r in tier2["results"].items():
            m, lat = r["metrics"], r["latency_ms"]
            print(
                f"  {config:<18}{_fmt(m['hit_at_3']):>8}{_fmt(m['hit_at_5']):>8}"
                f"{_fmt(m['recall_at_5']):>8}{_fmt(m['mrr_at_5']):>8}"
                f"{_fmt(m['ndcg_at_5']):>9}{_fmt(lat['p50']):>9}{_fmt(lat['p95']):>9}"
            )

        winner = E.summarize_winner(report)
        if winner["decided"]:
            print(f"\n  Best by Recall@5: {winner['best']}")
            if winner["tied_with"]:
                print(f"  Tied with: {', '.join(winner['tied_with'])}")
                print(f"  {winner['note']}")

    # --- regression gate ------------------------------------------------------
    baseline_path = args.baseline or (settings.retrieval_dir / "baseline.json")
    if args.record_baseline or args.check_regression:
        measured = RG.extract_metrics(report, tier=args.gate_tier)
        if not measured:
            print(f"\nNo comparable metrics on {args.gate_tier}; gate skipped.")
            return 0

        if args.record_baseline:
            RG.write_baseline(
                measured,
                baseline_path,
                tier=args.gate_tier,
                index_manifest=retriever.manifest,
            )
            print(f"\nRecorded regression baseline -> {baseline_path}")
            return 0

        baseline = RG.load_baseline(baseline_path)
        if baseline is None:
            print(
                f"\nNo baseline at {baseline_path}. Record one with "
                f"`--record-baseline` once you accept these numbers.",
                file=sys.stderr,
            )
            return 4
        if baseline.get("tier") != args.gate_tier:
            print(
                f"\nBaseline was recorded on tier '{baseline.get('tier')}' but the gate "
                f"is checking '{args.gate_tier}'. Refusing to compare.",
                file=sys.stderr,
            )
            return 4

        gate = RG.gate_result(RG.compare(baseline, measured))
        print(f"\nRegression gate ({args.gate_tier}) vs {baseline_path.name}")
        for line in gate["improvements"]:
            print(f"  improved: {line}")
        for line in gate["new_metrics"]:
            print(f"  new:      {line}")
        if not gate["passed"]:
            for line in gate["failures"]:
                print(f"  FAIL:     {line}", file=sys.stderr)
            return 1
        print("  passed")

    print(
        "\nRemember: rerun `python -m backend.scripts.build_capabilities` so the "
        "manifest picks up the new evaluation_status."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
