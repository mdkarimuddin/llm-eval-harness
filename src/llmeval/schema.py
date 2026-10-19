"""Test-set schema and loading (YAML or JSONL)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MATCH_MODES = ("contains", "exact", "regex")
KNOWN_SCORERS = (
    "exact_match", "recall_at_k", "mrr", "tool_trace", "safety", "citation", "faithfulness", "relevance",
)
DEFAULT_SCORERS = ("exact_match", "recall_at_k", "mrr", "tool_trace", "safety")
CASE_FIELDS = {
    "id", "input", "reference", "match", "gold_sources", "expected_tools",
    "forbidden_tools", "forbidden_strings", "expect_refusal", "tags",
}


class TestSetError(ValueError):
    """Raised when a test set file is malformed."""

    __test__ = False  # not a pytest class


@dataclass
class Case:
    id: str
    input: str
    reference: str | None = None
    match: str = "contains"
    gold_sources: list[str] | None = None
    expected_tools: list[str] | None = None
    forbidden_tools: list[str] = field(default_factory=list)
    forbidden_strings: list[str] = field(default_factory=list)
    expect_refusal: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class CaseSet:
    name: str
    cases: list[Case]
    scorers: list[str] = field(default_factory=lambda: list(DEFAULT_SCORERS))
    k: int = 5


def _str_list(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise TestSetError(f"{where}: expected a list of strings")
    return value


def _build_case(raw: Any, index: int) -> Case:
    if not isinstance(raw, dict):
        raise TestSetError(f"case #{index}: expected a mapping")
    unknown = set(raw) - CASE_FIELDS
    if unknown:
        raise TestSetError(f"case #{index}: unknown fields {sorted(unknown)}")
    cid, text = raw.get("id"), raw.get("input")
    if not isinstance(cid, str) or not cid.strip():
        raise TestSetError(f"case #{index}: 'id' must be a non-empty string")
    if not isinstance(text, str) or not text.strip():
        raise TestSetError(f"case {cid}: 'input' must be a non-empty string")
    match = raw.get("match", "contains")
    if match not in MATCH_MODES:
        raise TestSetError(f"case {cid}: match must be one of {MATCH_MODES}")
    reference = raw.get("reference")
    if reference is not None and not isinstance(reference, str):
        raise TestSetError(f"case {cid}: 'reference' must be a string")
    gold = raw.get("gold_sources")
    tools = raw.get("expected_tools")
    return Case(
        id=cid,
        input=text,
        reference=reference,
        match=match,
        gold_sources=_str_list(gold, f"case {cid}.gold_sources") if gold is not None else None,
        expected_tools=_str_list(tools, f"case {cid}.expected_tools") if tools is not None else None,
        forbidden_tools=_str_list(raw.get("forbidden_tools"), f"case {cid}.forbidden_tools"),
        forbidden_strings=_str_list(raw.get("forbidden_strings"), f"case {cid}.forbidden_strings"),
        expect_refusal=bool(raw.get("expect_refusal", False)),
        tags=_str_list(raw.get("tags"), f"case {cid}.tags"),
    )


def _validate(name: str, raw_cases: list[Any], scorers: list[str], k: Any) -> CaseSet:
    if not raw_cases:
        raise TestSetError("test set has no cases")
    unknown = [s for s in scorers if s not in KNOWN_SCORERS]
    if unknown:
        raise TestSetError(f"unknown scorers {unknown}; known: {list(KNOWN_SCORERS)}")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise TestSetError("'k' must be a positive integer")
    cases = [_build_case(r, i) for i, r in enumerate(raw_cases, start=1)]
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            raise TestSetError(f"duplicate case id: {c.id}")
        seen.add(c.id)
    return CaseSet(name=name, cases=cases, scorers=list(scorers), k=k)


def load_caseset(path: str | Path) -> CaseSet:
    p = Path(path)
    if not p.exists():
        raise TestSetError(f"test set not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        rows = []
        for n, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise TestSetError(f"{p.name} line {n}: invalid JSON ({exc.msg})") from exc
        return _validate(p.stem, rows, list(DEFAULT_SCORERS), 5)
    if p.suffix in (".yaml", ".yml"):
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise TestSetError(f"{p.name}: invalid YAML ({exc})") from exc
        if not isinstance(data, dict) or "cases" not in data:
            raise TestSetError(f"{p.name}: expected a mapping with a 'cases' list")
        return _validate(
            str(data.get("name", p.stem)),
            data["cases"] if isinstance(data["cases"], list) else [],
            list(data.get("scorers", DEFAULT_SCORERS)),
            data.get("k", 5),
        )
    raise TestSetError(f"unsupported test set extension: {p.suffix} (use .yaml, .yml or .jsonl)")
