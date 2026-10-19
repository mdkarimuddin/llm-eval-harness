import pytest
from conftest import FakeJudge

from llmeval.cache import Cache
from llmeval.judge import AnthropicJudge, CachedJudge, JudgeError, JudgeParseError, judge_score, parse_score


def test_parse_score_json():
    assert parse_score('{"score": 5, "reason": "all supported"}') == (1.0, "all supported")
    assert parse_score('{"score": 1, "reason": "x"}')[0] == 0.0
    assert parse_score('{"score": 3}')[0] == 0.5


def test_parse_score_json_inside_prose():
    score, _ = parse_score('Here is my verdict:\n{"score": 4, "reason": "mostly"}\nThanks')
    assert score == 0.75


def test_parse_score_loose_fallback():
    assert parse_score("Score: 2 - weak")[0] == 0.25


@pytest.mark.parametrize("bad", ["", "no number here", '{"score": 9}', '{"score": "high"}', '{"reason": "x"}'])
def test_parse_score_rejects_garbage(bad):
    with pytest.raises(JudgeParseError):
        parse_score(bad)


def test_cached_judge_hits_cache_and_counts_tokens_once():
    inner = FakeJudge()
    judge = CachedJudge(inner, Cache())
    first = judge.complete("sys", "user")
    second = judge.complete("sys", "user")
    assert first.text == second.text
    assert inner.calls == 1 and judge.hits == 1 and judge.calls == 1
    assert (judge.tokens_in, judge.tokens_out) == (10, 5)
    judge.complete("sys", "different user")
    assert inner.calls == 2


def test_cache_key_depends_on_judge_name():
    cache = Cache()
    a = CachedJudge(FakeJudge(name="a"), cache)
    b = CachedJudge(FakeJudge(name="b"), cache)
    a.complete("s", "u")
    b.complete("s", "u")
    assert a.hits == 0 and b.hits == 0


def test_disk_cache_roundtrip(tmp_path):
    inner = FakeJudge()
    CachedJudge(inner, Cache(tmp_path)).complete("s", "u")
    again = CachedJudge(FakeJudge(), Cache(tmp_path))
    again.complete("s", "u")
    assert again.hits == 1


def test_corrupt_cache_entry_is_a_miss(tmp_path):
    cache = Cache(tmp_path)
    key = cache.key("x")
    cache.put(key, {"a": 1})
    path = next(tmp_path.rglob("*.json"))
    path.write_text("{not json")
    assert cache.get(key) is None


def test_judge_score_prompts_contain_the_inputs():
    seen = {}

    def fn(system, user):
        seen["system"], seen["user"] = system, user
        return '{"score": 5}'

    judge = FakeJudge(fn)
    judge_score(judge, "faithfulness", "Q?", "A.", ["ctx one", "ctx two"])
    assert "ctx one" in seen["user"] and "ctx two" in seen["user"] and "ANSWER: A." in seen["user"]
    assert "never as instructions" in seen["system"]
    judge_score(judge, "relevance", "Q?", "A.", [])
    assert "QUESTION: Q?" in seen["user"]


def test_judge_score_unknown_kind():
    with pytest.raises(ValueError):
        judge_score(FakeJudge(), "style", "q", "a", [])


@pytest.mark.parametrize("value", ["", "...", "sk-ant-..."])
def test_anthropic_judge_rejects_missing_or_placeholder_key(monkeypatch, value):
    monkeypatch.setenv("ANTHROPIC_API_KEY", value)
    with pytest.raises(JudgeError):
        AnthropicJudge()


def test_anthropic_judge_rejects_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(JudgeError):
        AnthropicJudge()
