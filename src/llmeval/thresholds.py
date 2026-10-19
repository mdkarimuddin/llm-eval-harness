"""CI gates: thresholds such as ``recall_at_k: ">= 0.8"`` checked against a run summary."""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq}
_RULE = re.compile(r"^\s*(>=|<=|==|>|<)\s*(-?\d+(?:\.\d+)?)\s*$")


class ThresholdError(ValueError):
    pass


@dataclass
class Violation:
    metric: str
    rule: str
    actual: float | None

    def __str__(self) -> str:
        shown = "missing" if self.actual is None else f"{self.actual:.3f}"
        return f"{self.metric}: required {self.rule}, got {shown}"


def parse_rule(rule: Any) -> tuple[str, float]:
    if isinstance(rule, bool):
        raise ThresholdError(f"invalid rule: {rule!r}")
    if isinstance(rule, (int, float)):
        return ">=", float(rule)  # a bare number means "at least"
    m = _RULE.match(str(rule))
    if not m:
        raise ThresholdError(f"invalid rule {rule!r}; use e.g. '>= 0.8'")
    return m.group(1), float(m.group(2))


def load_thresholds(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ThresholdError("thresholds file must be a mapping")
    rules = data.get("thresholds", data)
    if not isinstance(rules, dict) or not rules:
        raise ThresholdError("no thresholds defined")
    for value in rules.values():
        parse_rule(value)  # validate early
    return rules


def check(summary: dict[str, Any], rules: dict[str, Any]) -> list[Violation]:
    """A metric named in the rules but absent from the run is a violation, not a silent pass."""
    violations = []
    for metric, rule in rules.items():
        op, bound = parse_rule(rule)
        if metric in summary["metrics"]:
            actual: float | None = summary["metrics"][metric]["mean"]
        elif metric in summary["totals"]:
            actual = float(summary["totals"][metric])
        else:
            actual = None
        if actual is None or not _OPS[op](round(actual, 9), bound):
            violations.append(Violation(metric, f"{op} {bound:g}", actual))
    return violations
