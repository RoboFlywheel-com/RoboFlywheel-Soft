# SPDX-License-Identifier: Apache-2.0
"""Atomic, JSONL and logical-root IO primitives."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .canonical import canonical_dumps, sha256_file


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, canonical_dumps(value) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> str:
    text = "".join(canonical_dumps(dict(row)) + "\n" for row in rows)
    atomic_write_text(path, text)
    return sha256_file(path)


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(canonical_dumps(dict(row)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            yield value


def hash_ref(path: str | Path) -> dict[str, str]:
    path = Path(path)
    return {"path": path.as_posix(), "sha256": sha256_file(path)}


def resolve_ref(ref: Mapping[str, Any], roots: Mapping[str, str | Path], base: str | Path) -> Path:
    """Resolve an alias/relative artifact reference, rejecting absolute paths."""
    alias = ref.get("root_alias")
    relative = ref.get("relative_path")
    if not isinstance(alias, str) or not isinstance(relative, str):
        raise ValueError("artifact ref requires root_alias and relative_path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("artifact relative_path must stay below its logical root")
    root = Path(roots[alias]) if alias in roots else Path(base)
    return root / relative_path
