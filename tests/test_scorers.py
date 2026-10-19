from llmeval.schema import Case
from llmeval.scorers import SCORERS, Context
from llmeval.targets import Output

CTX = Context(k=3)


def score(name, case, out, ctx=CTX):
    return SCORERS[name].score(case, out, ctx)


def test_exact_match_modes():
    out = Output(answer="  The refund   takes 5 Business Days. ")
    assert score("exact_match", Case("c", "q", reference="5 business days"), out) == 1
    assert score("exact_match", Case("c", "q", reference="7 business days"), out) == 0
    assert score("exact_match", Case("c", "q", reference="the refund takes 5 business days.", match="exact"), out) == 1
    assert score("exact_match", Case("c", "q", reference="refund", match="exact"), out) == 0
    assert score("exact_match", Case("c", "q", reference=r"\b5\b.*days", match="regex"), out) == 1
    assert score("exact_match", Case("c", "q", reference=r"^\d+$", match="regex"), out) == 0


def test_recall_at_k_respects_k():
    case = Case("c", "q", gold_sources=["a", "b"])
    out = Output(sources=["x", "a", "y", "b"])
    assert score("recall_at_k", case, out, Context(k=2)) == 0.5
    assert score("recall_at_k", case, out, Context(k=4)) == 1.0
    assert score("recall_at_k", case, Output(sources=[]), CTX) == 0.0


def test_mrr():
    case = Case("c", "q", gold_sources=["a", "b"])
    assert score("mrr", case, Output(sources=["a"])) == 1.0
    assert score("mrr", case, Output(sources=["x", "y", "b"])) == 1 / 3
    assert score("mrr", case, Output(sources=["x"])) == 0.0


def test_tool_trace_order_and_extras():
    case = Case("c", "q", expected_tools=["search", "summarise"])
    ok = Output(tool_calls=[{"name": "search"}, "log", {"name": "summarise", "args": {}}])
    wrong_order = Output(tool_calls=["summarise", "search"])
    missing = Output(tool_calls=["search"])
    assert score("tool_trace", case, ok) == 1
    assert score("tool_trace", case, wrong_order) == 0
    assert score("tool_trace", case, missing) == 0


def test_tool_trace_forbidden_wins():
    case = Case("c", "q", expected_tools=["search"], forbidden_tools=["delete"])
    assert score("tool_trace", case, Output(tool_calls=["search", "delete"])) == 0


def test_tool_trace_no_tools_expected():
    case = Case("c", "q", expected_tools=[], forbidden_tools=["delete"])
    assert score("tool_trace", case, Output(tool_calls=[])) == 1
    assert score("tool_trace", case, Output(tool_calls=["delete"])) == 0


def test_safety_forbidden_strings_case_insensitive():
    case = Case("c", "q", forbidden_strings=["secret"])
    assert score("safety", case, Output(answer="The SECRET is 42")) == 0
    assert score("safety", case, Output(answer="Nothing to share")) == 1


def test_safety_requires_refusal_when_expected():
    case = Case("c", "q", expect_refusal=True)
    assert score("safety", case, Output(answer="I do not have information on that.")) == 1
    assert score("safety", case, Output(answer="I can't help with that")) == 1
    assert score("safety", case, Output(answer="Sure, the CEO is Alex.")) == 0


def test_applies_only_when_case_has_the_field():
    plain = Case("c", "q")
    assert not SCORERS["exact_match"].applies(plain)
    assert not SCORERS["recall_at_k"].applies(plain)
    assert not SCORERS["tool_trace"].applies(plain)
    assert not SCORERS["safety"].applies(plain)
    assert SCORERS["exact_match"].applies(Case("c", "q", reference="x"))
    assert SCORERS["recall_at_k"].applies(Case("c", "q", gold_sources=["a"]))
    assert SCORERS["tool_trace"].applies(Case("c", "q", forbidden_tools=["x"]))
    assert SCORERS["safety"].applies(Case("c", "q", expect_refusal=True))


def test_judge_scorers_are_flagged_as_needing_a_judge():
    assert SCORERS["faithfulness"].needs_judge and SCORERS["relevance"].needs_judge
    assert not SCORERS["safety"].needs_judge


def test_citation_precision():
    case = Case("c", "q", gold_sources=["a", "b"])
    assert score("citation", case, Output(cited=["a", "b"])) == 1.0
    assert score("citation", case, Output(cited=["a", "x"])) == 0.5
    assert score("citation", case, Output(cited=["a", "a"])) == 1.0  # duplicates counted once
    assert score("citation", case, Output(cited=["x"])) == 0.0
    assert score("citation", case, Output(cited=[])) == 0.0  # an uncited answer fails
    assert not SCORERS["citation"].applies(Case("c", "q"))
