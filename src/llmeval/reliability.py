"""Judge reliability: compare a judge with your own hand labels, across repeated runs and judge models."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from llmeval.judge import Judge, JudgeParseError, judge_score
from llmeval.stats import agreement, cohen_kappa, mean, std

PASS_THRESHOLD = 0.75  # normalised score >= 0.75 means a judge score of 4 or 5


class LabelError(ValueError):
    pass


def export_for_labeling(results: dict[str, Any], metric: str, n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Pick up to n cases with an answer and write them with an empty human_label to fill in (0 or 1)."""
    pool = [c for c in results["cases"] if c["answer"] and (metric != "faithfulness" or c["contexts"])]
    if not pool:
        raise LabelError("no cases with answers to label")
    chosen = random.Random(seed).sample(pool, min(n, len(pool)))
    return [
        {
            "id": c["id"],
            "input": c["input"],
            "answer": c["answer"],
            "contexts": c["contexts"],
            "human_label": None,
        }
        for c in chosen
    ]


def load_labels(path: str | Path) -> list[dict[str, Any]]:
    items = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LabelError(f"labels line {n}: invalid JSON") from exc
        if row.get("human_label") not in (0, 1, True, False):
            raise LabelError(f"labels line {n} (id={row.get('id')}): human_label must be 0 or 1")
        row["human_label"] = int(row["human_label"])
        items.append(row)
    if not items:
        raise LabelError("labels file is empty")
    return items


def _judge_run(judge: Judge, kind: str, items: list[dict[str, Any]]) -> list[float | None]:
    scores: list[float | None] = []
    for it in items:
        try:
            scores.append(judge_score(judge, kind, it["input"], it["answer"], it.get("contexts", [])))
        except JudgeParseError:
            scores.append(None)
    return scores


def evaluate_judge(
    items: list[dict[str, Any]], judge: Judge, kind: str, runs: int = 3, other_judge: Judge | None = None
) -> dict[str, Any]:
    """Judge vs human labels over repeated runs, plus an optional second judge model."""
    human = [it["human_label"] for it in items]
    per_run = []
    all_scores: list[list[float | None]] = []
    for _ in range(runs):
        scores = _judge_run(judge, kind, items)
        all_scores.append(scores)
        per_run.append(_vs_human(human, scores))
    item_std = []
    unanimous = 0
    comparable = 0
    for i in range(len(items)):
        col = [r[i] for r in all_scores if r[i] is not None]
        if len(col) == runs:
            comparable += 1
            item_std.append(std(col))
            labels = {int(v >= PASS_THRESHOLD) for v in col}
            unanimous += len(labels) == 1
    report: dict[str, Any] = {
        "kind": kind,
        "judge": judge.name,
        "n_items": len(items),
        "human_pass_rate": mean(human),
        "runs": per_run,
        "run_to_run": {
            "mean_item_std": mean(item_std) if item_std else None,
            "pass_label_unanimous": unanimous / comparable if comparable else None,
            "items_compared": comparable,
        },
    }
    if other_judge is not None:
        scores2 = _judge_run(other_judge, kind, items)
        first = all_scores[0]
        shift = [b - a for a, b in zip(first, scores2, strict=True) if a is not None and b is not None]
        both = [(a, b) for a, b in zip(first, scores2, strict=True) if a is not None and b is not None]
        report["other_judge"] = {
            "judge": other_judge.name,
            "vs_human": _vs_human(human, scores2),
            "mean_score_shift": mean(shift) if shift else None,
            "pass_label_agreement_with_first": (
                agreement([int(a >= PASS_THRESHOLD) for a, _ in both], [int(b >= PASS_THRESHOLD) for _, b in both])
                if both
                else None
            ),
        }
    return report


def _vs_human(human: list[int], scores: list[float | None]) -> dict[str, Any]:
    pairs = [(h, int(s >= PASS_THRESHOLD)) for h, s in zip(human, scores, strict=True) if s is not None]
    if not pairs:
        return {"n": 0, "agreement": None, "kappa": None, "unparsed": len(scores)}
    hs, js = [p[0] for p in pairs], [p[1] for p in pairs]
    return {
        "n": len(pairs),
        "agreement": agreement(hs, js),
        "kappa": cohen_kappa(hs, js),
        "unparsed": len(scores) - len(pairs),
    }


def format_report(rep: dict[str, Any]) -> str:
    def f(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    lines = [
        f"Judge reliability: {rep['kind']} with {rep['judge']} on {rep['n_items']} hand-labelled items "
        f"(human pass rate {rep['human_pass_rate']:.2f}; pass = judge score >= 4 of 5)"
    ]
    for i, r in enumerate(rep["runs"], start=1):
        lines.append(f"  run {i}: agreement {f(r['agreement'])}, kappa {f(r['kappa'])}, unparsed {r['unparsed']}")
    rr = rep["run_to_run"]
    lines.append(
        f"  run-to-run: mean per-item std {f(rr['mean_item_std'])}, same pass/fail in all runs "
        f"{f(rr['pass_label_unanimous'])} ({rr['items_compared']} items)"
    )
    if "other_judge" in rep:
        o = rep["other_judge"]
        lines.append(
            f"  second judge {o['judge']}: agreement {f(o['vs_human']['agreement'])}, "
            f"kappa {f(o['vs_human']['kappa'])}, "
            f"mean score shift {f(o['mean_score_shift'])}, pass/fail agreement with first judge "
            f"{f(o['pass_label_agreement_with_first'])}"
        )
    return "\n".join(lines)
