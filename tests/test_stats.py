import math

import pytest

from llmeval.stats import agreement, bootstrap_ci, cohen_kappa, mean, paired_delta_ci, percentile, std


def test_mean_and_std():
    assert mean([1, 2, 3]) == 2
    assert std([2, 2, 2]) == 0
    assert std([1.0]) == 0
    assert math.isclose(std([0, 2]), 1.0)


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([5], 0.95) == 5
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    values = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    lo, hi = bootstrap_ci(values, seed=1)
    assert (lo, hi) == bootstrap_ci(values, seed=1)
    assert lo <= mean(values) <= hi
    assert 0 <= lo and hi <= 1


def test_bootstrap_ci_degenerate_cases():
    assert bootstrap_ci([0.7]) == (0.7, 0.7)
    assert bootstrap_ci([1, 1, 1, 1]) == (1.0, 1.0)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_ci_narrows_with_more_data():
    small = [0, 1] * 5
    large = [0, 1] * 100
    w_small = bootstrap_ci(small)[1] - bootstrap_ci(small)[0]
    w_large = bootstrap_ci(large)[1] - bootstrap_ci(large)[0]
    assert w_large < w_small


def test_paired_delta_detects_consistent_drop():
    a = [1.0] * 20
    b = [0.0] * 5 + [1.0] * 15
    delta, lo, hi = paired_delta_ci(a, b)
    assert delta == -0.25 and hi < 0


def test_paired_delta_requires_equal_lengths():
    with pytest.raises(ValueError):
        paired_delta_ci([1, 0], [1])


def test_agreement():
    assert agreement([1, 0, 1, 1], [1, 0, 0, 1]) == 0.75
    with pytest.raises(ValueError):
        agreement([], [])


def test_kappa_known_values():
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    # perfect disagreement on balanced labels
    assert cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == -1.0
    # 80% agreement, balanced raters: kappa = (0.8 - 0.5) / 0.5 = 0.6
    assert math.isclose(cohen_kappa([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]), 0.6)


def test_kappa_undefined_when_both_constant():
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) is None
