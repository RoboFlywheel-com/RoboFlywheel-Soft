# SPDX-License-Identifier: Apache-2.0
"""Create a portable deployment archive without data-heavy artifacts."""

from __future__ import annotations

import io
import gzip
import json
import re
import tarfile
from pathlib import Path
from typing import Iterable

from .hashing import canonical_json, sha256_bytes


EXCLUDED_SUFFIXES = {".obj", ".glb", ".ply", ".mp4", ".avi", ".mov", ".png", ".jpg", ".jpeg", ".npz", ".pt", ".pth", ".ckpt", ".safetensors", ".pem", ".key"}
EXCLUDED_PARTS = {"cache", "__pycache__", ".pytest_cache", "results", "outputs", "videos", "keyframes", "meshes", "checkpoints", "build", "dist"}
ABSOLUTE_PATTERN = re.compile(r"(?:^|[\s\"'])/(?:mnt|home|tmp|var|opt|Node10_nvme)/|(?:^|[\s\"'])[A-Za-z]:[\\/]")


def _included_files(root: Path) -> list[Path]:
    selected: list[Path] = []
    # Whitelist this workstream.  The paper workspace also contains an
    # unrelated physical-agent package whose absolute reference paths must not
    # leak into this deployment archive.
    explicit = [
        root / "code" / "fragile_soft_benchmark" / "pyproject.toml",
        root / "code" / "fragile_soft_benchmark" / "setup.py",
        root / "code" / "README.md",
        root / "code" / "scripts" / "run_fragile_soft_smoke.sh",
    ]
    for path in explicit:
        if path.is_file():
            selected.append(path)
    roots = [root / "code" / "fragile_soft_benchmark", root / "configs", root / "docs", root / "tests"]
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            allowed = (
                relative.startswith("code/fragile_soft_benchmark/")
                or relative in {"code/README.md", "code/pyproject.toml"}
                or relative == "code/scripts/run_fragile_soft_smoke.sh"
                or relative in {"code/src/fragile_soft_benchmark/__init__.py", "code/src/fragile_soft_benchmark/__main__.py"}
                or relative.startswith("configs/schema/")
                or relative in {"configs/benchmark_config.json", "configs/qualification_config.json", "configs/difficulty_states.json", "configs/server_shards.json"}
                or relative in {"docs/FRAGILE_SOFT_BUILDER_ZH.md", "docs/REFERENCES_ZH.md", "docs/SERVER_RUNBOOK_ZH.md", "docs/OUTPUT_CONTRACT_ZH.md", "docs/SCHEMA_REFERENCE_ZH.md"}
                or relative.startswith("tests/fixtures/p0_legacy/")
                or relative.startswith("tests/fixtures/physical_join/")
                or relative in {"tests/__init__.py", "tests/conftest.py", "tests/test_adapter.py", "tests/test_materials.py", "tests/test_scenario_difficulty.py", "tests/test_qualification.py", "tests/test_physical_join.py", "tests/test_sharding_archive.py", "tests/test_unittest_fragile_soft.py", "tests/test_unittest_benchmark_join.py"}
            )
            if not allowed:
                continue
            if path.suffix.lower() in EXCLUDED_SUFFIXES or any(part in EXCLUDED_PARTS for part in path.parts):
                continue
            selected.append(path)
    return sorted(set(selected))


def create_deployment_archive(root: str | Path, output_path: str | Path) -> dict[str, object]:
    root_path = Path(root).resolve()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = _included_files(root_path)
    entries = []
    for path in files:
        relative = path.relative_to(root_path).as_posix()
        data = path.read_bytes()
        text = data.decode("utf-8", errors="ignore")
        if ABSOLUTE_PATTERN.search(text):
            raise ValueError(f"absolute path in deployment input: {relative}")
        entries.append({"path": relative, "sha256": sha256_bytes(data), "bytes": len(data)})
    manifest = {
        "schema_version": "fragile-soft.deployment.v1",
        "archive_kind": "code_and_contracts_only",
        "excluded": sorted(EXCLUDED_SUFFIXES | EXCLUDED_PARTS),
        "files": entries,
    }
    # Fixed gzip/tar metadata makes the deployment hash reproducible.
    with output.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w") as archive:
                for path, entry in zip(files, entries):
                    info = tarfile.TarInfo(entry["path"])
                    data = path.read_bytes()
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    archive.addfile(info, io.BytesIO(data))
                manifest_data = (canonical_json(manifest) + "\n").encode("utf-8")
                info = tarfile.TarInfo("deployment_manifest.json")
                info.size = len(manifest_data)
                info.mode = 0o644
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(manifest_data))
    return {"archive": output.as_posix(), "sha256": sha256_bytes(output.read_bytes()), "file_count": len(files), "manifest": manifest}


def verify_deployment_archive(path: str | Path) -> dict[str, object]:
    archive_path = Path(path)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("archive contains unsafe path")
        forbidden = [name for name in names if Path(name).suffix.lower() in EXCLUDED_SUFFIXES]
        if forbidden:
            raise ValueError(f"archive contains excluded artifacts: {forbidden}")
        for member in members:
            if not member.isfile():
                continue
            data = archive.extractfile(member).read()  # type: ignore[union-attr]
            if ABSOLUTE_PATTERN.search(data.decode("utf-8", errors="ignore")):
                raise ValueError(f"archive contains absolute path: {member.name}")
    return {"path": archive_path.as_posix(), "sha256": sha256_bytes(archive_path.read_bytes()), "member_count": len(names), "verified": True}
