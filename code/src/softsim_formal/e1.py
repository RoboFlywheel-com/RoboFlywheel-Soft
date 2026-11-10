# SPDX-License-Identifier: Apache-2.0
"""E1 simulator-ready contract/runtime/task probes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .canonical import fingerprint
from .release import ReleaseBundle
from .runner import run_all_shards
from .planner import plan_run
from .aggregate import aggregate_run


E1_PROTOCOL_ID = "softsim.e1.runtime-contract.v1"
E1_FAILURE_CODES = ("physical_bundle_not_verified", "source_task_failure", "nan_or_instability", "unsupported_adapter", "timeout")


def contract_probe(bundle: ReleaseBundle | None) -> dict[str, Any]:
    """Return a contract audit without launching a simulator."""
    if bundle is None:
        return {"status": "blocked", "failure_code": "blocked_missing_physical_release", "denominator": 0, "passed": False}
    rows = list(bundle.physical)
    verified = [row for row in rows if row.get("status") == "verified"]
    rejected = [row for row in rows if row.get("status") == "rejected"]
    failures = [row for row in rows if row.get("status") == "task_failure"]
    return {
        "status": "pass" if rows and len(verified) == len(rows) else "pass_with_rejections",
        "passed": bool(rows),
        "denominator": len(rows),
        "physical_execution_population": len(rows),
        "verified_count": len(verified),
        "rejected_count": len(rejected),
        "task_failure_count": len(failures),
        "failure_codes": sorted({str(row.get("failure_code")) for row in rows if row.get("failure_code")}),
        "source_hash": fingerprint(bundle.source_hashes),
    }


def run_e1_smoke(output_dir: str, *, stage: str = "fixture", count: int = 10, shard_count: int = 1, release_root: str | None = None) -> dict[str, Any]:
    cfg = {"promotion_stage": stage, "count": int(count), "methods": ["contract_probe"], "seeds": [0]}
    plan = plan_run("e1-runtime", output_dir=output_dir, config=cfg, release_root=release_root, fixture=release_root is None, count=count, shard_count=shard_count)
    if plan.status == "blocked":
        return plan.as_dict()
    run_all_shards(output_dir)
    return aggregate_run(output_dir)
