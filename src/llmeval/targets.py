"""Targets: anything that turns a question into an Output (callable, HTTP endpoint)."""

from __future__ import annotations

import importlib
import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from llmeval.cache import Cache


@dataclass
class Output:
    answer: str = ""
    sources: list[str] = field(default_factory=list)  # retrieved source ids, best first
    contexts: list[str] = field(default_factory=list)  # retrieved text passages
    tool_calls: list[Any] = field(default_factory=list)  # names or {"name": ..., "args": ...}
    cited: list[str] = field(default_factory=list)  # source ids the answer cites
    tokens_in: int = 0
    tokens_out: int = 0


class TargetError(RuntimeError):
    """The target could not produce an output."""


def coerce(obj: Any) -> Output:
    if isinstance(obj, Output):
        return obj
    if isinstance(obj, str):
        return Output(answer=obj)
    if isinstance(obj, dict):
        known = {k: obj[k] for k in Output.__dataclass_fields__ if k in obj}
        out = Output(**known)
        if not isinstance(out.answer, str):
            raise TargetError("'answer' must be a string")
        return out
    raise TargetError(f"target returned {type(obj).__name__}; expected str, dict or Output")


def load_callable(spec: str) -> Callable[[str], Any]:
    if ":" not in spec:
        raise TargetError("target must look like 'package.module:function'")
    module_name, func_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise TargetError(f"cannot import {module_name!r}: {exc}") from exc
    func = getattr(module, func_name, None)
    if not callable(func):
        raise TargetError(f"{module_name}.{func_name} is not callable")
    return func


class HttpTarget:
    """POSTs {"input": text} and expects JSON with at least {"answer": ...}."""

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        if not url.startswith(("http://", "https://")):
            raise TargetError("url must start with http:// or https://")
        self.url = url
        self.timeout = timeout

    def __call__(self, text: str) -> Output:
        body = json.dumps({"input": text}).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise TargetError(f"HTTP target failed: {exc}") from exc
        return coerce(payload)


class CachedTarget:
    """Caches target outputs by input, so repeated runs do not repeat model calls."""

    def __init__(self, target: Callable[[str], Any], cache: Cache, identity: str) -> None:
        self.target, self.cache, self.identity = target, cache, identity
        self.hits = 0
        self.calls = 0

    def __call__(self, text: str) -> Output:
        key = self.cache.key("target", self.identity, text)
        found = self.cache.get(key)
        if found is not None:
            self.hits += 1
            return Output(**found)
        self.calls += 1
        out = coerce(self.target(text))
        self.cache.put(key, asdict(out))
        return out
