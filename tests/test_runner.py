from datetime import datetime, timezone

import pytest
from conftest import FakeJudge

from llmeval.runner import RunError, run_cases, select_cases
from llmeval.schema import Case, CaseSet
from llmeval.targets import Output

NOW = datetime(2026, 10, 19, tzinfo=timezone.utc)


def make_set(scorers=("exact_match", "safety"), n=10):
    cases = [Case(f"c{i}", f"question {i}", reference=f"answer {i}", tags=["even" if i % 2 == 0 else "odd"])
             for i in range(n)]
    return CaseSet("t", cases, list(scorers))


def echo(question):
    return Output(answer=f"answer {question.split()[-1]}", tokens_in=3, tokens_out=2)


def test_select_cases_tags_and_subset_are_deterministic():
    cs = make_set()
    assert {c.id for c in select_cases(cs, tags=["even"])} == {"c0", "c2", "c4", "c6", "c8"}
    a = select_cases(cs, subset=4, seed=7)
    b = select_cases(cs, subset=4, seed=7)
    assert [c.id for c in a] == [c.id for c in b] and len(a) == 4
    assert [c.id for c in a] == sorted(c.id for c in a)  # original order kept
    assert [c.id for c in select_cases(cs, subset=4, seed=8)] != [c.id for c in a]


def test_select_cases_empty_selection_is_an_error():
    with pytest.raises(RunError):
        select_cases(make_set(), tags=["nope"])


def test_subset_larger_than_set_returns_everything():
    assert len(select_cases(make_set(n=3), subset=50)) == 3


def test_run_scores_and_summary():
    cs = make_set()
    res = run_cases(cs, cs.cases, echo, target_name="echo", now=NOW)
    assert res["summary"]["metrics"]["exact_match"]["mean"] == 1.0
    assert res["summary"]["totals"]["cases"] == 10 and res["summary"]["totals"]["error_cases"] == 0
    assert res["summary"]["totals"]["target_tokens_in"] == 30
    assert "safety" not in res["summary"]["metrics"]  # not applicable to any case
    assert res["created"].startswith("2026-10-19")


def test_partial_failures_lower_the_mean_and_widen_the_ci():
    cs = make_set()

    def half_right(q):
        n = int(q.split()[-1])
        return Output(answer=f"answer {n}" if n % 2 == 0 else "wrong")

    m = run_cases(cs, cs.cases, half_right)["summary"]["metrics"]["exact_match"]
    assert m["mean"] == 0.5 and m["ci_low"] < 0.5 < m["ci_high"] and m["n"] == 10


def test_a_crashing_target_fails_the_case_instead_of_hiding_it():
    cs = make_set(n=4)

    def flaky(q):
        if q.endswith("2"):
            raise RuntimeError("boom")
        return echo(q)

    res = run_cases(cs, cs.cases, flaky)
    bad = next(c for c in res["cases"] if c["id"] == "c2")
    assert bad["scores"]["exact_match"] == 0.0 and "boom" in bad["errors"][0]
    assert res["summary"]["totals"]["error_cases"] == 1
    assert res["summary"]["metrics"]["exact_match"]["mean"] == 0.75


def test_target_may_return_str_or_dict():
    cs = make_set(n=2)
    res = run_cases(cs, cs.cases, lambda q: {"answer": f"answer {q.split()[-1]}", "unknown_key": 1})
    assert res["summary"]["metrics"]["exact_match"]["mean"] == 1.0
    res = run_cases(cs, cs.cases, lambda q: f"answer {q.split()[-1]}")
    assert res["summary"]["metrics"]["exact_match"]["mean"] == 1.0
    res = run_cases(cs, cs.cases, lambda q: 42)  # wrong type: counted as a failed case
    assert res["summary"]["totals"]["error_cases"] == 2


def test_judge_scorers_need_a_judge():
    cs = make_set(scorers=("faithfulness",))
    with pytest.raises(RunError, match="--judge"):
        run_cases(cs, cs.cases, echo)


def test_judge_scoring_and_unparseable_replies():
    cs = make_set(scorers=("faithfulness", "relevance"), n=3)
    replies = iter(['{"score": 5}', '{"score": 1}', "I refuse to answer", '{"score": 3}', '{"score": 5}', '{"score": 5}'])
    judge = FakeJudge(lambda s, u: next(replies))
    res = run_cases(cs, cs.cases, echo, judge=judge)
    c0, c1, c2 = res["cases"]
    assert c0["scores"] == {"faithfulness": 1.0, "relevance": 0.0}
    assert c1["scores"]["faithfulness"] is None and c1["errors"]  # unparseable: excluded and reported
    assert res["summary"]["totals"]["error_cases"] == 1
    assert res["summary"]["metrics"]["faithfulness"]["n"] == 2
    assert c2["scores"] == {"faithfulness": 1.0, "relevance": 1.0}


def test_latency_and_token_totals_present():
    cs = make_set(n=3)
    totals = run_cases(cs, cs.cases, echo)["summary"]["totals"]
    assert totals["latency_p95_s"] >= 0 and totals["target_tokens_out"] == 6
