#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Add a flat mesh path to each frozen manifest row without changing its data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_mesh_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if not normalized or normalized.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe mesh template result: {value!r}")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mesh-template", default="{sample_id}.glb")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent)
    temporary = Path(temporary_name)
    rows = 0
    try:
        with args.input.open("r", encoding="utf-8") as source, os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
            for line_number, raw in enumerate(source, 1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError(f"{args.input}:{line_number}: expected an object")
                sample_id = str(row.get("sample_id") or "")
                if not sample_id:
                    raise ValueError(f"{args.input}:{line_number}: sample_id missing")
                row["mesh_relpath"] = safe_mesh_path(args.mesh_template.format(sample_id=sample_id))
                target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                target.write("\n")
                rows += 1
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps({
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "rows": rows,
        "mesh_template": args.mesh_template,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
