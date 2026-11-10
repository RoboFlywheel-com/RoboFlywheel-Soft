#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare a frozen manifest with precomputed flat mesh-root indexes.

This avoids one FUSE metadata lookup per manifest row while preserving an
auditable record of exactly which index files were used. It does not validate
GLB bytes; run asset-audit on the present subset for content validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def relative_mesh(row: dict) -> str:
    value = row.get("mesh_relpath") or row.get("mesh")
    if value:
        value = str(value).replace("\\", "/")
        if value.startswith("/") or any(part in {"", ".", ".."} for part in value.split("/")):
            raise ValueError(f"unsafe mesh path: {value!r}")
        return value
    sample_id = str(row.get("sample_id") or "")
    if not sample_id:
        raise ValueError("sample_id missing")
    return f"samples/{sample_id}/mesh/{sample_id}.glb"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--mesh-index", action="append", required=True, help="ROOT_ALIAS=flat index file")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    indexes: dict[str, set[str]] = {}
    index_paths: dict[str, Path] = {}
    index_hashes: dict[str, str] = {}
    for value in args.mesh_index:
        alias, separator, raw_path = value.partition("=")
        if not separator or not alias or not raw_path or alias in indexes:
            raise ValueError(f"expected unique ROOT_ALIAS=INDEX_FILE, got {value!r}")
        index_path = Path(raw_path).expanduser().resolve()
        with index_path.open("r", encoding="utf-8") as handle:
            indexes[alias] = {line.strip() for line in handle if line.strip()}
        index_paths[alias] = index_path
        index_hashes[alias] = sha256_file(index_path)

    total = 0
    unique_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    invalid_rows: list[dict] = []
    by_family: dict[str, Counter[str]] = {}
    missing_noncloth: list[dict] = []
    missing_cloth: list[str] = []
    duplicate_placements: list[str] = []
    for line_number, row in enumerate(rows(args.manifest), 1):
        total += 1
        sample_id = str(row.get("sample_id") or "")
        family = str(row.get("geometry_family") or "unknown")
        counts = by_family.setdefault(family, Counter())
        if not sample_id:
            invalid_rows.append({"line": line_number, "failure": "sample_id_missing"})
            counts["invalid"] += 1
            continue
        if sample_id in unique_ids:
            duplicate_ids.add(sample_id)
        unique_ids.add(sample_id)
        try:
            relative = relative_mesh(row)
        except ValueError as exc:
            invalid_rows.append({"line": line_number, "sample_id": sample_id, "failure": str(exc)})
            counts["invalid"] += 1
            continue
        placements = [alias for alias, values in indexes.items() if relative in values]
        if len(placements) == 1:
            counts["present"] += 1
        elif len(placements) > 1:
            counts["duplicate"] += 1
            duplicate_placements.append(sample_id)
        else:
            counts["missing"] += 1
            if family == "cloth_sheet" or row.get("generation_route") == "procedural_cloth_grid":
                missing_cloth.append(sample_id)
            else:
                missing_noncloth.append({"sample_id": sample_id, "object_type": row.get("object_type"), "geometry_family": family})

    report = {
        "schema_name": "softsim.asset_mesh_coverage_report",
        "schema_version": 1,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.manifest),
        "mesh_indexes": {
            alias: {
                "path": str(index_paths[alias]),
                "sha256": index_hashes[alias],
                "entries": len(values),
            }
            for alias, values in indexes.items()
        },
        "total_rows": total,
        "unique_sample_ids": len(unique_ids),
        "duplicate_sample_ids": sorted(duplicate_ids),
        "invalid_rows": invalid_rows,
        "duplicate_placements": sorted(duplicate_placements),
        "by_family": {family: dict(sorted(counts.items())) for family, counts in sorted(by_family.items())},
        "missing_noncloth": sorted(missing_noncloth, key=lambda value: value["sample_id"]),
        "missing_cloth_count": len(missing_cloth),
        "missing_cloth_sample_ids": sorted(missing_cloth),
        "content_validation": "deferred_to_asset-audit",
    }
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "total_rows": total, "missing_noncloth": len(missing_noncloth), "missing_cloth": len(missing_cloth)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
