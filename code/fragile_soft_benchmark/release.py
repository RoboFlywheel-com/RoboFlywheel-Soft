# SPDX-License-Identifier: Apache-2.0
"""Promotion guard for a formal E3-A manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hashing import sha256_json


def promote_e3_a_manifest(
    instances: Sequence[Mapping[str, Any]],
    qualification_report: Mapping[str, Any],
    *,
    output_path: str | Path,
    allow_mock: bool = False,
) -> dict[str, Any]:
    """Write an E3-A manifest only after every instance passes qualification.

    Local mock qualification is useful for regression, but promotion is denied
    unless the caller explicitly opts in (the production runbook never does).
    """

    if qualification_report.get("status") != "qualified" or not qualification_report.get("e3_a_eligible"):
        raise ValueError("qualification report is not eligible for E3-A promotion")
    if (
        qualification_report.get("backend_id") == "mock.deterministic.v1"
        or str(qualification_report.get("release_scope", "")).startswith("local_mock")
    ) and not allow_mock:
        raise ValueError("mock qualification cannot be promoted to formal E3-A")
    rows = []
    for instance in instances:
        if instance.get("qualification_status") != "qualified" or not instance.get("e3_a_eligible"):
            raise ValueError(f"unqualified instance cannot enter E3-A: {instance.get('benchmark_instance_id')}")
        if (
            instance.get("physical_join_status") != "joined_verified"
            or not instance.get("formal_join_ready")
            or not instance.get("physical_bundle_id")
            or not instance.get("sample_id")
            or not instance.get("asset_revision_id")
        ):
            raise ValueError(f"instance has no verified physical bundle join: {instance.get('benchmark_instance_id')}")
        row = dict(instance)
        row["formal_e3_a_manifest"] = True
        rows.append(row)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return {"path": target.as_posix(), "instance_count": len(rows), "manifest_hash": sha256_json(rows), "formal_e3_a_manifest": True}
