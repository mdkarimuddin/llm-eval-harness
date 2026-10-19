"""Tiny content-addressed JSON cache (in memory, or a directory on disk)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class Cache:
    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory else None
        self._memory: dict[str, Any] = {}

    @staticmethod
    def key(*parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        assert self.directory is not None
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        if self.directory is None:
            return self._memory.get(key)
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None  # a corrupt entry is treated as a miss

    def put(self, key: str, value: Any) -> None:
        if self.directory is None:
            self._memory[key] = value
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
