# SPDX-License-Identifier: Apache-2.0
"""Portable deployment archive builder for the formal package."""

from __future__ import annotations

import gzip
import io
import re
import tarfile
from pathlib import Path
from typing import Any

from .canonical import canonical_dumps, sha256_bytes


_FORBIDDEN = re.compile(r"(?:^|[\s\"'])/(?:tmp|var/tmp|home|mnt|Node10_nvme)/|(?:^|[\s\"'])[A-Za-z]:[\\/]")
_SKIP_PARTS = {"__pycache__", ".pytest_cache", ".git", "dist", "build", "*.egg-info", "runs", "outputs", "rendered", "results"}
_SKIP_SUFFIXES = {".pyc", ".pyo", ".mp4", ".gif", ".png", ".jpg", ".jpeg", ".npz", ".pt", ".pth", ".ckpt", ".safetensors"}


def _files(root: Path) -> list[Path]:
    selected: list[Path] = []
    for top_level in ("README.md", "RUNBOOK.md"):
        path = root / top_level
        if path.is_file():
            selected.append(path)
    for base_name in ("code", "configs", "docs", "tests", "deployment"):
        base = root / base_name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel_parts = path.relative_to(root).parts
            rel = path.relative_to(root).as_posix()
            # deployment/experiments/** is study working data (renders, runner
            # input snapshots with host paths), never deployment material.
            if base_name == "deployment" and len(rel_parts) > 1 and rel_parts[1] == "experiments":
                continue
            if any(part in _SKIP_PARTS or part.endswith(".egg-info") for part in rel_parts):
                continue
            if path.suffix.lower() in _SKIP_SUFFIXES:
                continue
            # Do not recursively include previously built archives/checksums.
            if rel.endswith(".tar.gz") or rel.endswith(".sha256"):
                continue
            if path.name == "SHA256SUMS":
                continue
            selected.append(path)
    return sorted(set(selected))


def create_archive(root: str | Path, output: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = _files(root_path)
    entries = []
    for path in files:
        data = path.read_bytes()
        text = data.decode("utf-8", errors="ignore")
        # Documentation may mention forbidden locations as policy examples;
        # executable/configuration inputs may not depend on them.
        check_path = path.suffix.lower() in {".json", ".toml", ".sh", ".yaml", ".yml"}
        if check_path and _FORBIDDEN.search(text):
            raise ValueError(f"absolute or temporary path in archive input: {path.relative_to(root_path)}")
        entries.append({"path": path.relative_to(root_path).as_posix(), "sha256": sha256_bytes(data), "bytes": len(data)})
    manifest = {"schema_name": "softsim.deployment_manifest", "schema_version": 1, "archive_kind": "code_fixture_docs_only", "files": entries}
    manifest_data = (canonical_dumps(manifest) + "\n").encode("utf-8")
    with output_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for path in files:
                    data = path.read_bytes()
                    info = tarfile.TarInfo(path.relative_to(root_path).as_posix())
                    info.size = len(data)
                    info.mode = 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))
                info = tarfile.TarInfo("deployment_manifest.json")
                info.size = len(manifest_data)
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                tar.addfile(info, io.BytesIO(manifest_data))
    return {"path": output_path.as_posix(), "sha256": sha256_bytes(output_path.read_bytes()), "file_count": len(files), "manifest": manifest}


def verify_archive(path: str | Path) -> dict[str, Any]:
    archive_path = Path(path)
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError("unsafe archive member")
            if member.isfile():
                data = tar.extractfile(member).read()  # type: ignore[union-attr]
                member_suffix = Path(member.name).suffix.lower()
                if member_suffix in {".json", ".toml", ".sh", ".yaml", ".yml"} and _FORBIDDEN.search(data.decode("utf-8", errors="ignore")):
                    raise ValueError(f"forbidden path in archive member {member.name}")
    return {"path": archive_path.as_posix(), "sha256": sha256_bytes(archive_path.read_bytes()), "member_count": len(members), "verified": True}
