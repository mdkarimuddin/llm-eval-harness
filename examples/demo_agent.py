"""Toy tool-using agent for demonstrating tool-trace and safety scorers. Rule-based, deterministic."""

from __future__ import annotations

import re

from llmeval.targets import Output

RATES = {("eur", "usd"): 1.08, ("usd", "eur"): 0.93}
WEATHER = {"helsinki": "4 C and cloudy", "oulu": "-1 C and snow", "turku": "6 C and rain"}


def run(question: str) -> Output:
    q = question.lower()
    calls: list[dict] = []
    if re.search(r"\b(delete|drop|wipe)\b", q):
        return Output(answer="I cannot delete data. That action is not allowed.", tool_calls=[])
    if "weather" in q:
        city = next((c for c in WEATHER if c in q), None)
        if city is None:
            return Output(answer="I do not know which city you mean.", tool_calls=[])
        calls.append({"name": "get_weather", "args": {"city": city}})
        return Output(answer=f"The weather in {city.title()} is {WEATHER[city]}.", tool_calls=calls)
    m = re.search(r"convert (\d+(?:\.\d+)?) (eur|usd) to (eur|usd)", q)
    if m:
        amount, src, dst = float(m.group(1)), m.group(2), m.group(3)
        calls.append({"name": "convert_currency", "args": {"amount": amount, "from": src, "to": dst}})
        return Output(answer=f"{amount} {src.upper()} is {amount * RATES[(src, dst)]:.2f} {dst.upper()}.",
                      tool_calls=calls)
    if "ignore previous" in q or "system prompt" in q:
        return Output(answer="I cannot follow instructions that try to change my rules.", tool_calls=[])
    return Output(answer="I do not have a tool for that.", tool_calls=[])
