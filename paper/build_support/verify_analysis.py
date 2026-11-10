# SPDX-License-Identifier: Apache-2.0
"""Verify the portable manuscript CSV snapshot before regenerating tables."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> int:
    data = Path(__file__).resolve().parents[1] / "analysis_data"
    manifest = json.loads((data / "source_registry.json").read_text())
    failures = []
    for name, expected in manifest["files"].items():
        path = data / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(name)
    actual = {path.name for path in data.glob("*.csv")}
    failures.extend(sorted(actual - manifest["files"].keys()))
    if failures:
        raise SystemExit("Analysis snapshot mismatch: " + ", ".join(failures))
    print(f"Verified {len(manifest['files'])} analysis CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
