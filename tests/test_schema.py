import json

import pytest

from llmeval.schema import TestSetError, load_caseset


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_load_yaml(tmp_path):
    p = write(tmp_path, "t.yaml", "name: demo\nk: 3\nscorers: [exact_match]\ncases:\n  - id: a\n    input: hi\n    reference: yo\n")
    cs = load_caseset(p)
    assert cs.name == "demo" and cs.k == 3 and cs.scorers == ["exact_match"]
    assert cs.cases[0].reference == "yo" and cs.cases[0].match == "contains"


def test_load_jsonl_uses_defaults(tmp_path):
    rows = [{"id": "a", "input": "q1"}, {"id": "b", "input": "q2", "tags": ["x"]}]
    p = write(tmp_path, "t.jsonl", "\n".join(json.dumps(r) for r in rows) + "\n\n")
    cs = load_caseset(p)
    assert [c.id for c in cs.cases] == ["a", "b"] and cs.cases[1].tags == ["x"] and cs.k == 5


@pytest.mark.parametrize(
    "body,fragment",
    [
        ("cases: []\n", "no cases"),
        ("cases:\n  - input: x\n", "'id'"),
        ("cases:\n  - id: a\n", "'input'"),
        ("cases:\n  - id: a\n    input: x\n  - id: a\n    input: y\n", "duplicate"),
        ("cases:\n  - id: a\n    input: x\n    match: fuzzy\n", "match must be"),
        ("cases:\n  - id: a\n    input: x\n    oops: 1\n", "unknown fields"),
        ("scorers: [vibes]\ncases:\n  - id: a\n    input: x\n", "unknown scorers"),
        ("k: 0\ncases:\n  - id: a\n    input: x\n", "'k'"),
        ("cases:\n  - id: a\n    input: x\n    tags: foo\n", "list of strings"),
        ("just a string", "mapping"),
        ("cases: [unclosed", "invalid YAML"),
    ],
)
def test_invalid_yaml_is_rejected_with_a_clear_message(tmp_path, body, fragment):
    with pytest.raises(TestSetError, match=fragment):
        load_caseset(write(tmp_path, "bad.yaml", body))


def test_bad_jsonl_line(tmp_path):
    with pytest.raises(TestSetError, match="line 2"):
        load_caseset(write(tmp_path, "t.jsonl", '{"id": "a", "input": "x"}\n{oops\n'))


def test_missing_file_and_wrong_extension(tmp_path):
    with pytest.raises(TestSetError, match="not found"):
        load_caseset(tmp_path / "nope.yaml")
    with pytest.raises(TestSetError, match="unsupported"):
        load_caseset(write(tmp_path, "t.txt", "x"))
