import pytest

from llmeval.compare import compare_runs, format_table
from llmeval.thresholds import ThresholdError, check, load_thresholds, parse_rule


def summary(**means):
    return {
        "metrics": {k: {"mean": v, "n": 10, "ci_low": v, "ci_high": v} for k, v in means.items()},
        "totals": {"error_cases": 0, "latency_p95_s": 0.2},
    }


def test_parse_rule_variants():
    assert parse_rule(">= 0.8") == (">=", 0.8)
    assert parse_rule("==1") == ("==", 1.0)
    assert parse_rule(0.9) == (">=", 0.9)
    for bad in ("high", ">>1", "", True):
        with pytest.raises(ThresholdError):
            parse_rule(bad)


def test_check_passes_and_fails():
    rules = {"recall_at_k": ">= 0.8", "safety": "== 1", "error_cases": "== 0"}
    assert check(summary(recall_at_k=0.85, safety=1.0), rules) == []
    v = check(summary(recall_at_k=0.79, safety=0.9), rules)
    assert [x.metric for x in v] == ["recall_at_k", "safety"]
    assert "required >= 0.8" in str(v[0])


def test_missing_metric_is_a_violation_not_a_pass():
    v = check(summary(safety=1.0), {"recall_at_k": ">= 0.8"})
    assert len(v) == 1 and v[0].actual is None and "missing" in str(v[0])


def test_totals_can_be_gated():
    s = summary()
    s["totals"]["error_cases"] = 2
    assert len(check(s, {"error_cases": "== 0"})) == 1
    assert check(s, {"latency_p95_s": "<= 0.5"}) == []


def test_float_noise_does_not_fail_equality():
    assert check(summary(safety=0.9999999999), {"safety": "== 1"}) == []


def test_load_thresholds_from_yaml(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("thresholds:\n  safety: '== 1'\n")
    assert load_thresholds(p) == {"safety": "== 1"}
    p.write_text("safety: 0.9\n")
    assert load_thresholds(p) == {"safety": 0.9}
    p.write_text("- nope\n")
    with pytest.raises(ThresholdError):
        load_thresholds(p)
    p.write_text("thresholds:\n  safety: ???\n")
    with pytest.raises(ThresholdError):
        load_thresholds(p)


def run_with(scores, metric="m"):
    return {
        "cases": [{"id": f"c{i}", "scores": {metric: s}} for i, s in enumerate(scores)],
        "summary": {"metrics": {metric: {}}},
    }


def test_compare_flags_regression_improvement_and_noise():
    a = run_with([1.0] * 30)
    worse = run_with([0.0] * 10 + [1.0] * 20)
    rows = compare_runs(a, worse)
    assert rows[0].verdict == "regression" and rows[0].delta < 0 and rows[0].n == 30
    better = compare_runs(worse, a)
    assert better[0].verdict == "improvement"
    noisy_a = run_with([1, 0, 1, 0, 1, 0])
    noisy_b = run_with([0, 1, 1, 0, 0, 1])
    assert compare_runs(noisy_a, noisy_b)[0].verdict == "no clear change"


def test_compare_pairs_by_id_and_ignores_missing_scores():
    a = run_with([1.0, 1.0, None, 1.0])
    b = run_with([0.0, 1.0, 1.0, None])
    rows = compare_runs(a, b)
    assert rows[0].n == 2 and rows[0].delta == -0.5


def test_compare_requires_shared_cases():
    a = run_with([1.0])
    b = {"cases": [{"id": "other", "scores": {"m": 1.0}}], "summary": {"metrics": {"m": {}}}}
    with pytest.raises(ValueError):
        compare_runs(a, b)


def test_format_table_contains_verdicts():
    text = format_table(compare_runs(run_with([1.0] * 30), run_with([0.0] * 30)))
    assert "regression" in text and "metric" in text
