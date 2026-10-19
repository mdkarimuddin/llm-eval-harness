from __future__ import annotations

import pytest

from llmeval.judge import JudgeReply


class FakeJudge:
    """Scripted judge: returns replies from a function, counts calls. No network."""

    def __init__(self, fn=None, name="fake:judge"):
        self.name = name
        self.calls = 0
        self._fn = fn or (lambda system, user: '{"score": 5, "reason": "ok"}')

    def complete(self, system, user):
        self.calls += 1
        return JudgeReply(self._fn(system, user), tokens_in=10, tokens_out=5)


@pytest.fixture
def fake_judge():
    return FakeJudge
