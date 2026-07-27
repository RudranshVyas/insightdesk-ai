"""Phase 5 / Phase 12 — the retrieval metric regression gate.

A committed baseline records what the fixture corpus scored when the numbers
were last accepted. CI recomputes them and fails if any metric has dropped
further than its documented tolerance.

Two deliberate asymmetries:

* **Only drops fail.** An improvement is reported loudly and the baseline is
  left alone, because a metric that moved up still needs a human to look at why
  before it becomes the new floor.
* **A metric that was `None` and is now a number is not a regression**, and a
  metric that was a number and is now `None` is a *hard* failure — the
  measurement stopped happening, which is worse than scoring badly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

# Default tolerance per metric. Retrieval on a small fixture is noisy; these are
# wide enough to survive a library patch release and tight enough to catch a
# broken tokenizer, a swapped embedding backend, or resolution text leaking into
# the index.
DEFAULT_TOLERANCES: dict[str, float] = {
    "hit_at_3": 0.05,
    "hit_at_5": 0.05,
    "recall_at_3": 0.05,
    "recall_at_5": 0.05,
    "mrr_at_3": 0.05,
    "mrr_at_5": 0.05,
    "ndcg_at_3": 0.05,
    "ndcg_at_5": 0.05,
}


@dataclass
class Finding:
    config: str
    metric: str
    baseline: float | None
    measured: float | None
    tolerance: float
    kind: str  # "regression" | "measurement_lost" | "improvement" | "new"

    @property
    def failed(self) -> bool:
        return self.kind in ("regression", "measurement_lost")

    def describe(self) -> str:
        if self.kind == "measurement_lost":
            return (
                f"{self.config}.{self.metric}: baseline {self.baseline} but nothing was "
                f"measured this run. A metric that stopped being computed is a harder "
                f"failure than one that got worse."
            )
        if self.kind == "regression":
            drop = self.baseline - self.measured
            return (
                f"{self.config}.{self.metric}: {self.baseline:.4f} -> {self.measured:.4f} "
                f"(down {drop:.4f}, tolerance {self.tolerance:.4f})"
            )
        if self.kind == "improvement":
            return (
                f"{self.config}.{self.metric}: {self.baseline:.4f} -> {self.measured:.4f} "
                f"(up {self.measured - self.baseline:.4f}) — review, then update the baseline"
            )
        return f"{self.config}.{self.metric}: new metric, measured {self.measured}"


def extract_metrics(report: dict[str, Any], tier: str = "tier1_leave_one_out") -> dict[str, dict]:
    """Pull the comparable numbers out of an evaluation report."""
    if tier == "tier1_leave_one_out":
        return {
            config: dict(r.get("metrics") or {})
            for config, r in (report.get(tier) or {}).items()
            if r.get("available")
        }
    tier2 = report.get("tier2_manual_labeled") or {}
    if tier2.get("status") != "evaluated":
        return {}
    return {config: dict(r.get("metrics") or {}) for config, r in tier2["results"].items()}


def compare(
    baseline: dict[str, Any],
    measured: dict[str, dict],
    tolerances: dict[str, float] | None = None,
) -> list[Finding]:
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {}), **(baseline.get("tolerances") or {})}
    base_metrics: dict[str, dict] = baseline.get("metrics") or {}
    findings: list[Finding] = []

    for config, base in base_metrics.items():
        got = measured.get(config)
        if got is None:
            # An entire config vanished from the report. Every one of its
            # measured metrics counts as lost.
            for metric, b in base.items():
                if b is not None:
                    findings.append(
                        Finding(config, metric, b, None, tol.get(metric, 0.05), "measurement_lost")
                    )
            continue

        for metric, b in base.items():
            m = got.get(metric)
            t = tol.get(metric, 0.05)
            if b is None:
                if m is not None:
                    findings.append(Finding(config, metric, None, m, t, "new"))
                continue
            if m is None:
                findings.append(Finding(config, metric, b, None, t, "measurement_lost"))
            elif m < b - t:
                findings.append(Finding(config, metric, b, m, t, "regression"))
            elif m > b + t:
                findings.append(Finding(config, metric, b, m, t, "improvement"))

    return findings


def load_baseline(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_baseline(
    measured: dict[str, dict],
    path: Path,
    *,
    tier: str,
    index_manifest: dict[str, Any] | None = None,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    from datetime import datetime

    from backend.app.core.versions import version_stamp

    baseline = {
        "baseline_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "tier": tier,
        "versions": version_stamp(),
        "index": {
            "embedding_model": (index_manifest or {}).get("embedding_model"),
            "embedding_model_revision": (index_manifest or {}).get("embedding_model_revision"),
            "corpus_size": (index_manifest or {}).get("corpus_size"),
        },
        "tolerances": tolerances or {},
        "metrics": measured,
        "note": (
            "Recorded from a run a human accepted. CI fails when a metric drops "
            "below baseline minus tolerance, or stops being measured at all. "
            "Changing the embedding model invalidates this file: re-record it."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(baseline, fh, indent=2)
    return baseline


def gate_result(findings: list[Finding]) -> dict[str, Any]:
    failures = [f for f in findings if f.failed]
    return {
        "passed": not failures,
        "failures": [f.describe() for f in failures],
        "improvements": [f.describe() for f in findings if f.kind == "improvement"],
        "new_metrics": [f.describe() for f in findings if f.kind == "new"],
    }
