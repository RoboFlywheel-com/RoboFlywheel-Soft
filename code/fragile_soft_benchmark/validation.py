# SPDX-License-Identifier: Apache-2.0
"""Contract validation without requiring the optional jsonschema package."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .hashing import sha256_json
from .scenario import PHASES, validate_scenario


REQUIRED_OUTPUTS = (
    "instance_manifest.jsonl",
    "material_records.jsonl",
    "material_config_records.jsonl",
    "physical_join_report.json",
    "config_records.jsonl",
    "episode_records.jsonl",
    "metric_records.jsonl",
    "video_keyframe_manifest.jsonl",
    "qualification_report.json",
    "failed_repair_replace.jsonl",
)
ABSOLUTE_PATH_RE = re.compile(r"(?:^|[\s\"'])/(?:mnt|home|tmp|var|opt|Node10_nvme)/|(?:^|[\s\"'])[A-Za-z]:[\\/]")


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield value


def validate_instance(instance: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ("benchmark_instance_id", "geometry_revision_id", "physical_bundle_id", "provisional_physical_bundle_id", "physical_join_status", "difficulty_id", "damage_model_id", "scenario", "source", "source_hash", "status")
    errors.extend(f"missing:{key}" for key in required if key not in instance)
    scenario = instance.get("scenario")
    if isinstance(scenario, Mapping):
        try:
            validate_scenario(scenario)
        except ValueError as exc:
            errors.append(f"scenario:{exc}")
        if tuple(item.get("name") for item in scenario.get("phases", [])) != PHASES:
            errors.append("scenario:phase_order")
    else:
        errors.append("scenario:not_object")
    source = instance.get("source")
    if not isinstance(source, Mapping) or not str(source.get("p0_asset_source_sha256", "")).startswith("sha256:"):
        errors.append("source:missing_hash")
    if not str(instance.get("source_hash", "")).startswith("sha256:"):
        errors.append("source_hash:missing")
    join_status = str(instance.get("physical_join_status", ""))
    if join_status == "joined_verified":
        if not instance.get("physical_bundle_id") or not instance.get("sample_id") or not instance.get("asset_revision_id"):
            errors.append("physical_join:incomplete_verified_join")
        if instance.get("formal_join_ready") is not True:
            errors.append("physical_join:formal_ready_false")
    elif instance.get("formal_join_ready") is True:
        errors.append("physical_join:unjoined_marked_ready")
    difficulty = instance.get("difficulty")
    if isinstance(difficulty, Mapping):
        immutable = difficulty.get("immutable_contract") or {}
        if isinstance(scenario, Mapping):
            expected = {
                "scenario_hash": scenario.get("contract_hash"),
            }
            if expected["scenario_hash"] and immutable.get("scenario_hash") != expected["scenario_hash"]:
                errors.append("difficulty:scenario_mutated")
    return errors


def validate_output_dir(path: str | Path, *, expected_count: int | None = None) -> dict[str, Any]:
    root = Path(path)
    missing = [name for name in REQUIRED_OUTPUTS if not (root / name).is_file()]
    if missing:
        raise ValueError(f"missing output files: {missing}")
    instances = list(iter_jsonl(root / "instance_manifest.jsonl"))
    try:
        join_report = json.loads((root / "physical_join_report.json").read_text(encoding="utf-8"))
        for key in ("schema_version", "counts", "all_joined_verified"):
            if key not in join_report:
                errors = [f"physical_join_report:missing:{key}"]
                return {"valid": False, "instance_count": len(instances), "errors": errors}
    except (OSError, ValueError) as exc:
        return {"valid": False, "instance_count": len(instances), "errors": [f"physical_join_report:{exc}"]}
    errors: list[str] = []
    for row in instances:
        row_errors = validate_instance(row)
        if row_errors:
            errors.append(f"{row.get('benchmark_instance_id')}:" + ";".join(row_errors))
        if row.get("instance_hash"):
            copy = dict(row)
            expected = copy.pop("instance_hash")
            if expected != sha256_json(copy):
                errors.append(f"{row.get('benchmark_instance_id')}:instance_hash")
    ids = [str(row.get("benchmark_instance_id")) for row in instances]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_instance_id")
    if expected_count is not None and len(instances) != expected_count:
        errors.append(f"count:{len(instances)}!={expected_count}")
    for file_path in root.iterdir():
        if not file_path.is_file():
            continue
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        if ABSOLUTE_PATH_RE.search(text):
            errors.append(f"absolute_path:{file_path.name}")
    for name, key in (("episode_records.jsonl", "episode_hash"), ("metric_records.jsonl", "metric_record_hash")):
        for row in iter_jsonl(root / name):
            if key in row:
                copy = dict(row)
                expected = copy.pop(key)
                if expected != sha256_json(copy):
                    errors.append(f"{name}:{row.get('episode_id') or row.get('instance_id')}:hash")
    return {"valid": not errors, "instance_count": len(instances), "errors": errors}
