"""Compare two runs metric by metric with paired bootstrap confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llmeval.stats import mean, paired_delta_ci


@dataclass
class Row:
    metric: str
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    n: int
    verdict: str  # regression | improvement | no clear change


def compare_runs(a: dict[str, Any], b: dict[str, Any], seed: int = 0) -> list[Row]:
    """Rows describe B relative to A (A = baseline). Cases are paired by id."""
    by_id_a = {c["id"]: c for c in a["cases"]}
    by_id_b = {c["id"]: c for c in b["cases"]}
    shared = [cid for cid in by_id_a if cid in by_id_b]
    if not shared:
        raise ValueError("the two runs share no case ids")
    rows = []
    metrics = [m for m in a["summary"]["metrics"] if m in b["summary"]["metrics"]]
    for metric in metrics:
        xs, ys = [], []
        for cid in shared:
            sa, sb = by_id_a[cid]["scores"].get(metric), by_id_b[cid]["scores"].get(metric)
            if sa is not None and sb is not None:
                xs.append(sa)
                ys.append(sb)
        if not xs:
            continue
        delta, lo, hi = paired_delta_ci(xs, ys, seed=seed)
        verdict = "regression" if hi < 0 else "improvement" if lo > 0 else "no clear change"
        rows.append(Row(metric, mean(xs), mean(ys), delta, lo, hi, len(xs), verdict))
    return rows


def format_table(rows: list[Row]) -> str:
    head = f"{'metric':<14}{'A':>8}{'B':>8}{'delta':>9}  {'95% CI':<18}{'n':>4}  verdict"
    lines = [head, "-" * len(head)]
    for r in rows:
        ci = f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]"
        lines.append(f"{r.metric:<14}{r.mean_a:>8.3f}{r.mean_b:>8.3f}{r.delta:>+9.3f}  {ci:<18}{r.n:>4}  {r.verdict}")
    return "\n".join(lines)
