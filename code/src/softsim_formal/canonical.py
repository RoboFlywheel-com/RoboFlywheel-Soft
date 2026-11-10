# SPDX-License-Identifier: Apache-2.0
"""Canonical JSON and content fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _normalise(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalise(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON cannot contain NaN or infinity")
        # JSON's encoder is deterministic for finite floats under a fixed Python.
        return value
    return value


def canonical_dumps(value: Any) -> str:
    return json.dumps(_normalise(value), ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def canonical_bytes(value: Any) -> bytes:
    return canonical_dumps(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(value) + "\n", encoding="utf-8")
    return sha256_file(path)
