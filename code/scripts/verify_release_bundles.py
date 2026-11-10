#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify every file referenced by a physical_bundle_index.jsonl against its
recorded sha256 (bundle / trace / calibration / heldout refs), plus the index
file's own sidecar hash.  Use after any move/copy of a release directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, help="release dir containing physical_bundle_index.jsonl")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    root = Path(args.release)
    index = root / "physical_bundle_index.jsonl"

    sidecar = (root / "physical_bundle_index.sha256").read_text().split()[0]
    actual_index = sha256_path(index)
    print(f"index sha256: {actual_index} (sidecar: {sidecar}) match={actual_index == sidecar}")
    if actual_index != sidecar:
        return 1

    jobs = []
    for line in index.open():
        row = json.loads(line)
        for field, key in (("bundle", None),):
            b = row.get("bundle")
            if b:
                jobs.append((row["sample_id"], "bundle", b["path"], str(b["sha256"]).split(":")[-1]))
        for field in ("agent_trace_ref", "calibration_evidence_ref", "heldout_evidence_ref"):
            ref = row.get(field)
            if ref:
                jobs.append((row["sample_id"], field, ref["relative_path"], str(ref["sha256"]).split(":")[-1]))
    print(f"files to verify: {len(jobs)}")

    bad = []
    done = 0

    def check(job):
        sid, field, rel, expect = job
        p = root / rel
        if not p.is_file():
            return (sid, field, "missing")
        if sha256_path(p) != expect:
            return (sid, field, "hash_mismatch")
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(check, jobs):
            done += 1
            if res:
                bad.append(res)
            if done % 20000 == 0:
                print(f"progress {done}/{len(jobs)} bad={len(bad)}", flush=True)
    print(json.dumps({"verified": len(jobs) - len(bad), "bad": len(bad), "total": len(jobs)}, indent=1))
    for item in bad[:20]:
        print("BAD:", item, file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
