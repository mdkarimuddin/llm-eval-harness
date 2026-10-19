import json

import pytest
from conftest import FakeJudge

from llmeval.reliability import (
    LabelError,
    evaluate_judge,
    export_for_labeling,
    format_report,
    load_labels,
)


def items(labels):
    return [{"id": f"i{n}", "input": f"q{n}", "answer": f"a{n}", "contexts": ["c"], "human_label": h}
            for n, h in enumerate(labels)]


def judge_by_answer(passes):
    """Judge that gives 5 to answers listed in `passes`, else 1."""
    return FakeJudge(lambda s, u: '{"score": 5}' if any(f"ANSWER: {a}" in u for a in passes) else '{"score": 1}')


def test_perfect_judge_has_full_agreement_and_kappa():
    its = items([1, 1, 0, 0])
    rep = evaluate_judge(its, judge_by_answer({"a0", "a1"}), "faithfulness", runs=3)
    assert all(r["agreement"] == 1.0 and r["kappa"] == 1.0 for r in rep["runs"])
    assert rep["run_to_run"]["pass_label_unanimous"] == 1.0 and rep["run_to_run"]["mean_item_std"] == 0


def test_lenient_judge_has_low_kappa():
    its = items([1, 0, 0, 0])
    rep = evaluate_judge(its, judge_by_answer({"a0", "a1", "a2", "a3"}), "faithfulness", runs=1)
    r = rep["runs"][0]
    assert r["agreement"] == 0.25 and r["kappa"] == 0.0


def test_noisy_judge_shows_run_to_run_variance():
    state = {"n": 0}

    def alternating(system, user):
        state["n"] += 1
        return '{"score": 5}' if state["n"] % 2 else '{"score": 1}'

    rep = evaluate_judge(items([1, 1, 1]), FakeJudge(alternating), "relevance", runs=2)
    assert rep["run_to_run"]["mean_item_std"] > 0
    assert rep["run_to_run"]["pass_label_unanimous"] < 1.0


def test_unparseable_replies_are_counted_not_hidden():
    rep = evaluate_judge(items([1, 0]), FakeJudge(lambda s, u: "no idea"), "relevance", runs=1)
    assert rep["runs"][0]["n"] == 0 and rep["runs"][0]["unparsed"] == 2


def test_second_judge_comparison():
    its = items([1, 0, 1, 0])
    first = judge_by_answer({"a0", "a2"})
    second = FakeJudge(lambda s, u: '{"score": 5}', name="fake:lenient")
    rep = evaluate_judge(its, first, "faithfulness", runs=1, other_judge=second)
    other = rep["other_judge"]
    assert other["judge"] == "fake:lenient" and other["mean_score_shift"] > 0
    assert other["pass_label_agreement_with_first"] == 0.5
    text = format_report(rep)
    assert "second judge" in text and "kappa" in text


def test_load_labels_validates(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text(json.dumps({"id": "a", "human_label": None}) + "\n")
    with pytest.raises(LabelError, match="human_label"):
        load_labels(p)
    p.write_text(json.dumps({"id": "a", "human_label": 1}) + "\n\n" + json.dumps({"id": "b", "human_label": False}))
    assert [it["human_label"] for it in load_labels(p)] == [1, 0]
    p.write_text("")
    with pytest.raises(LabelError):
        load_labels(p)
    p.write_text("{bad")
    with pytest.raises(LabelError, match="invalid JSON"):
        load_labels(p)


def test_export_for_labeling_samples_and_blanks_labels():
    results = {"cases": [{"id": f"c{i}", "input": "q", "answer": "a" if i else "", "contexts": ["x"]} for i in range(10)]}
    out = export_for_labeling(results, "faithfulness", n=4, seed=1)
    assert len(out) == 4 and all(o["human_label"] is None and o["answer"] for o in out)
    assert export_for_labeling(results, "faithfulness", n=4, seed=1) == out
    assert len(export_for_labeling(results, "faithfulness", n=100)) == 9
    with pytest.raises(LabelError):
        export_for_labeling({"cases": [{"id": "c", "input": "q", "answer": "", "contexts": []}]}, "relevance", n=5)
