"""Small statistics helpers: bootstrap CIs, paired deltas, agreement and Cohen's kappa."""

from __future__ import annotations

import math
import random
from typing import Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def std(values: Sequence[float]) -> float:
    """Population standard deviation (0.0 for fewer than two values)."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. With one value the interval collapses to it."""
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return percentile(means, alpha / 2), percentile(means, 1 - alpha / 2)


def paired_delta_ci(
    a: Sequence[float], b: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (b - a) with a paired bootstrap CI. Returns (delta, lo, hi)."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be equal-length and non-empty")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    lo, hi = bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)
    return mean(diffs), lo, hi


def agreement(x: Sequence[int], y: Sequence[int]) -> float:
    if len(x) != len(y) or not x:
        raise ValueError("label lists must be equal-length and non-empty")
    return sum(1 for p, q in zip(x, y, strict=True) if p == q) / len(x)


def cohen_kappa(x: Sequence[int], y: Sequence[int]) -> float | None:
    """Cohen's kappa for two binary raters. None when it is undefined (both raters constant and equal)."""
    po = agreement(x, y)
    n = len(x)
    px1, py1 = sum(x) / n, sum(y) / n
    pe = px1 * py1 + (1 - px1) * (1 - py1)
    if math.isclose(pe, 1.0):
        return None
    return (po - pe) / (1 - pe)
