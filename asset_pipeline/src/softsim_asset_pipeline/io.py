# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, TextIO


class JsonlError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_id(value: Any) -> str:
    return f"sha256:{sha256_bytes(canonical_json_bytes(value))}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise JsonlError(f"{path}:{line_number}: expected a JSON object")
            yield value


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def atomic_text_writer(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    with atomic_text_writer(path) as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with atomic_text_writer(path) as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row).decode("utf-8"))
            handle.write("\n")
            count += 1
    return count


def validate_relative_path(value: str) -> str:
    if not value or "\\" in value:
        raise ValueError(f"Invalid POSIX relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path.as_posix()


def safe_join(root: Path, relative: str, *, require_exists: bool = False) -> Path:
    normalized = validate_relative_path(relative)
    root_resolved = root.resolve()
    candidate = root / Path(normalized)
    resolved = candidate.resolve(strict=require_exists)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Path escapes logical root: {relative!r}") from exc
    return resolved


def parse_root_aliases(values: Iterable[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        alias, separator, raw_path = value.partition("=")
        if not separator or not alias or not raw_path:
            raise ValueError(f"Expected ROOT_ALIAS=PATH, got {value!r}")
        if alias in roots:
            raise ValueError(f"Duplicate root alias: {alias}")
        if not alias.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid root alias: {alias!r}")
        roots[alias] = Path(raw_path).expanduser().resolve()
    return roots


def artifact_descriptor(role: str, root_alias: str, path: str, absolute_path: Path) -> dict[str, Any]:
    return {
        "role": role,
        "root_alias": root_alias,
        "path": validate_relative_path(path),
        "sha256": sha256_file(absolute_path),
        "size_bytes": absolute_path.stat().st_size,
        "media_type": {
            ".glb": "model/gltf-binary",
            ".png": "image/png",
            ".json": "application/json",
            ".jsonl": "application/x-ndjson",
        }.get(absolute_path.suffix.lower(), "application/octet-stream"),
    }
