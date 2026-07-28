"""Phase 7 — run the guardrail case set and write artifacts/guardrails/evaluation.json.

    python -m backend.scripts.eval_guardrails
    python -m backend.scripts.eval_guardrails --check   # CI gate, non-zero on failure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.evaluation import guardrail_eval as G


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "not_measured"


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    p = argparse.ArgumentParser(description="Evaluate the Phase 6 guardrails.")
    p.add_argument("--cases", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--check", action="store_true", help="exit non-zero if a gate fails")
    args = p.parse_args(argv)

    cases_path = args.cases or (settings.evaluation_dir / "guardrail_cases.jsonl")
    out = args.out or (settings.artifacts_dir / "guardrails" / "evaluation.json")

    cases = G.load_cases(cases_path)
    if not cases:
        print(
            f"No guardrail cases at {cases_path}. The guardrails are unmeasured; "
            f"no rate may be quoted.",
            file=sys.stderr,
        )
        return 2

    print(f"Running {len(cases)} guardrail case(s) from {cases_path.name} ...\n")
    report = G.evaluate(cases, settings)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    # --- hard gates -----------------------------------------------------------
    hg = report["hard_gates"]
    print("HARD GATES (must be zero)")
    print(f"  weak_retrieval_generation_violations : {hg['weak_retrieval_generation_violations']}")
    print(f"  weak_retrieval_provider_calls        : {hg['weak_retrieval_provider_calls']}")

    # --- measured rates -------------------------------------------------------
    m = report["metrics"]
    print("\nMEASURED RATES")
    for name in (
        "citation_validity_rate",
        "step_citation_coverage",
        "abstention_accuracy",
        "injection_resistance_rate",
        "deterministic_fallback_success",
        "pii_leakage_rate",
        "expectation_pass_rate",
    ):
        print(f"  {name:<32} {_fmt(m[name])}")

    lat = report["latency_ms"]
    print(f"\n  latency p50 / p95 (ms)           {_fmt(lat['p50'])} / {_fmt(lat['p95'])}")
    print(f"  tokens / cost                    {report['token_usage']} / {report['measured_cost']}")
    print(f"    {report['cost_note']}")

    # --- coverage -------------------------------------------------------------
    print(f"\nCATEGORIES  {len(report['categories_covered'])}/{len(G.REQUIRED_CATEGORIES)} covered")
    if report["categories_missing"]:
        print(f"  MISSING: {report['categories_missing']}")

    if report["failing_cases"]:
        print("\nFAILING CASES")
        for f in report["failing_cases"]:
            print(f"  {f['case_id']} ({f['category']})")
            for reason in f["failures"]:
                print(f"      - {reason}")

    passed, reasons = G.gate(report)
    print(f"\nWrote {out}")
    print(f"\nGATE: {'PASS' if passed else 'FAIL'}")
    for r in reasons:
        print(f"  {r}")

    if args.check and not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
