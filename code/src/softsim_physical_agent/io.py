# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator


def canonical_bytes(value: Any) -> bytes:
    """Return the byte representation used for all content identifiers."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


# Compatibility alias used by the earlier local prototype.
content_id = content_hash


def portable_value(value: Any) -> Any:
    """Recursively remove machine-specific absolute paths from contracts."""

    if isinstance(value, dict):
        return {str(key): portable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_value(item) for item in value]
    if isinstance(value, tuple):
        return [portable_value(item) for item in value]
    if isinstance(value, str):
        raw = value.replace("\\", "/")
        if raw.startswith("/") or (len(raw) > 2 and raw[1] == ":" and raw[2:3] == "/"):
            for marker in ("/assets/", "/output/", "/samples/", "/results/", "/experiments/"):
                if marker in raw:
                    return marker.lstrip("/") + raw.split(marker, 1)[1]
            return Path(raw).name
        return raw
    return value


def file_sha256(path: Path, *, prefixed: bool = True) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    return "sha256:" + value if prefixed else value


@lru_cache(maxsize=1)
def package_source_hash() -> str:
    """Hash the shipped Python/config source without including runtime outputs."""

    package_root = Path(__file__).resolve().parents[2]
    installed_package = Path(__file__).resolve().parent
    source_layout = (package_root / "src" / "softsim_physical_agent").is_dir()
    if not source_layout:
        package_root = installed_package
    candidates = [
        *((package_root / "src" / "softsim_physical_agent").rglob("*.py") if source_layout else installed_package.rglob("*.py")),
        *(([package_root / "physical_agent" / "pyproject.toml"]) if (package_root / "physical_agent" / "pyproject.toml").is_file() else []),
        *( (package_root / "contracts" / "v1").rglob("*.json") if (package_root / "contracts" / "v1").is_dir() else []),
        *( (package_root / "scripts" / "genesis_probe.py", package_root / "scripts" / "build_golden_reference_audit.py") if (package_root / "scripts").is_dir() else ()),
    ]
    files = sorted(path for path in candidates if path.is_file() and "__pycache__" not in path.parts)
    entries = [
        {"path": path.relative_to(package_root).as_posix(), "sha256": file_sha256(path)}
        for path in files
    ]
    return content_hash(entries)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_json_value(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_bytes(row).decode("utf-8"))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def relative_path(path: Path | str, root: Path) -> str:
    """Return a POSIX relative path without leaking an absolute path."""

    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix().lstrip("/")


def copy_file_reference(source: Path, destination: Path) -> str | None:
    if not source.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return file_sha256(destination)


def resolve_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        alias, separator, raw_path = value.partition("=")
        if not separator or not alias or not raw_path:
            raise ValueError(f"Expected ROOT_ALIAS=PATH, got {value!r}")
        roots[alias] = Path(raw_path).expanduser().resolve()
    return roots


def is_hash(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) == 71
