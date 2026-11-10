# SPDX-License-Identifier: Apache-2.0
"""Stable hashing and path helpers used by every emitted record."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    """Return a byte-stable JSON representation suitable for hashing."""

    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def normalise_relative_path(value: str | os.PathLike[str], root: Path | None = None) -> str:
    """Convert an old absolute/reference path into a portable logical path.

    P0 records contain several machine-specific prefixes.  We retain the
    meaningful suffix and never emit an absolute path in a benchmark contract.
    """

    raw = str(value).replace("\\", "/")
    if root is not None:
        try:
            candidate = Path(raw)
            if candidate.is_absolute():
                raw = candidate.relative_to(root.resolve()).as_posix()
        except (ValueError, OSError):
            pass
    raw = re.sub(r"^[A-Za-z]:/", "", raw)
    raw = re.sub(r"^/+", "", raw)
    # Known workstation prefixes are provenance, not portable artifact paths.
    markers = (
        "/experiments/",
        "/output/",
        "/assets/",
        "/results/",
        "/videos/",
        "/keyframes/",
    )
    for marker in markers:
        if marker in raw:
            raw = raw.split(marker, 1)[1]
            raw = marker.lstrip("/") + raw
            break
    return raw or "unknown"


def reference_digest(value: str | os.PathLike[str], root: Path | None = None) -> dict[str, Any]:
    """Hash a source artifact when present, otherwise hash its logical reference."""

    path = Path(value)
    if root is not None and not path.is_absolute():
        rooted = root / path
        if rooted.is_file():
            path = rooted
    exists = path.is_file()
    logical = normalise_relative_path(value, root=root)
    digest = sha256_file(path) if exists else sha256_text("missing:" + logical)
    return {"path": logical, "sha256": digest, "exists": exists}


def combined_source_hash(records: Iterable[dict[str, Any]]) -> str:
    compact = [
        {"path": item.get("path"), "sha256": item.get("sha256"), "exists": item.get("exists")}
        for item in records
    ]
    return sha256_json(sorted(compact, key=lambda item: str(item["path"])))


def stable_seed(*parts: Any) -> int:
    """Derive a reproducible 32-bit seed without Python hash randomisation."""

    digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)
