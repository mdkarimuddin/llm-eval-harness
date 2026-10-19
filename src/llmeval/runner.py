"""Run a case set against a target and aggregate the scores."""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from typing import Any, Callable

from llmeval.judge import Judge, JudgeError, JudgeParseError
from llmeval.schema import Case, CaseSet
from llmeval.scorers import SCORERS, Context
from llmeval.stats import bootstrap_ci, mean, percentile
from llmeval.targets import Output, coerce


class RunError(RuntimeError):
    """The run could not start (bad configuration)."""


def select_cases(
    caseset: CaseSet, tags: list[str] | None = None, subset: int | None = None, seed: int = 0
) -> list[Case]:
    """Filter by tag, then take a seeded random subset (order preserved). Deterministic for a seed."""
    cases = caseset.cases
    if tags:
        cases = [c for c in cases if set(tags) & set(c.tags)]
    if subset is not None and subset < len(cases):
        keep = set(random.Random(seed).sample(range(len(cases)), subset))
        cases = [c for i, c in enumerate(cases) if i in keep]
    if not cases:
        raise RunError("no cases selected (check --tags / --subset)")
    return cases


def run_cases(
    caseset: CaseSet,
    cases: list[Case],
    target: Callable[[str], Any],
    *,
    target_name: str = "target",
    judge: Judge | None = None,
    seed: int = 0,
    now: datetime | None = None,
) -> dict[str, Any]:
    scorers = [SCORERS[name] for name in caseset.scorers]
    if any(s.needs_judge for s in scorers) and judge is None:
        raise RunError("this test set uses judge scorers (faithfulness/relevance); pass --judge anthropic")
    ctx = Context(k=caseset.k, judge=judge)
    results = []
    for case in cases:
        entry: dict[str, Any] = {"id": case.id, "tags": case.tags, "input": case.input, "errors": []}
        started = time.perf_counter()
        out: Output | None
        try:
            out = coerce(target(case.input))
        except Exception as exc:  # noqa: BLE001 - any target failure counts as a failed case
            out = None
            entry["errors"].append(f"target: {type(exc).__name__}: {exc}")
        entry["latency_s"] = round(time.perf_counter() - started, 4)
        entry["answer"] = out.answer if out else ""
        entry["sources"] = out.sources if out else []
        entry["contexts"] = out.contexts if out else []
        entry["cited"] = out.cited if out else []
        entry["tool_calls"] = out.tool_calls if out else []
        entry["tokens_in"] = out.tokens_in if out else 0
        entry["tokens_out"] = out.tokens_out if out else 0
        scores: dict[str, float | None] = {}
        for sc in scorers:
            if not sc.applies(case):
                scores[sc.name] = None
            elif out is None:
                scores[sc.name] = 0.0  # a crashed target fails every applicable check
            else:
                try:
                    scores[sc.name] = sc.score(case, out, ctx)
                except (JudgeParseError, JudgeError) as exc:
                    scores[sc.name] = None
                    entry["errors"].append(f"{sc.name}: {exc}")
        entry["scores"] = scores
        results.append(entry)
    summary = summarise(results, caseset.scorers, seed)
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    judge_tokens = {
        "tokens_in": getattr(judge, "tokens_in", 0),
        "tokens_out": getattr(judge, "tokens_out", 0),
        "cache_hits": getattr(judge, "hits", 0),
        "calls": getattr(judge, "calls", 0),
    }
    return {
        "name": caseset.name,
        "created": stamp,
        "target": target_name,
        "judge": getattr(judge, "name", None),
        "k": caseset.k,
        "seed": seed,
        "scorers": caseset.scorers,
        "cases": results,
        "summary": summary,
        "judge_usage": judge_tokens,
    }


def summarise(results: list[dict[str, Any]], scorer_names: list[str], seed: int = 0) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name in scorer_names:
        values = [r["scores"][name] for r in results if r["scores"].get(name) is not None]
        if not values:
            continue
        lo, hi = bootstrap_ci(values, seed=seed)
        metrics[name] = {"mean": mean(values), "n": len(values), "ci_low": lo, "ci_high": hi}
    latencies = [r["latency_s"] for r in results]
    return {
        "metrics": metrics,
        "totals": {
            "cases": len(results),
            "error_cases": sum(1 for r in results if r["errors"]),
            "target_tokens_in": sum(r["tokens_in"] for r in results),
            "target_tokens_out": sum(r["tokens_out"] for r in results),
            "latency_mean_s": round(mean(latencies), 4),
            "latency_p95_s": round(percentile(latencies, 0.95), 4),
        },
    }
