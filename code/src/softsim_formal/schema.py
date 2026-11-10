# SPDX-License-Identifier: Apache-2.0
"""Shared schema validators and status vocabulary.

The validators are intentionally explicit rather than permissive dataclass
coercions: a malformed record must fail before it can reach aggregation.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


TERMINAL_EXECUTION = {"completed", "failed", "invalid"}
OUTCOME_STATUSES = {"success", "failure", "rejected", "not_applicable", "blocked"}
PHYSICAL_STATUSES = {"verified", "rejected", "task_failure"}
FAILURE_DOMAINS = {"none", "infrastructure", "numerical", "contract", "policy", "task", "damage", "release"}


class SchemaError(ValueError):
    pass


def require(mapping: Mapping[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise SchemaError("missing required fields: " + ", ".join(missing))


def require_hash(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise SchemaError(f"{field} must be sha256:<64 hex characters>")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise SchemaError(f"{field} has invalid sha256") from exc


def require_id(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SchemaError(f"{field} must be a non-empty stable string")


def validate_artifact_ref(ref: Mapping[str, Any], field: str = "artifact_ref") -> None:
    if not isinstance(ref, Mapping):
        raise SchemaError(f"{field} must be an object")
    require(ref, "root_alias", "relative_path", "sha256")
    require_id(ref["root_alias"], field + ".root_alias")
    relative = ref["relative_path"]
    if not isinstance(relative, str) or not relative or relative.startswith("/") or ".." in relative.split("/"):
        raise SchemaError(f"{field}.relative_path must be a POSIX relative path")
    require_hash(ref["sha256"], field + ".sha256")


def validate_asset_row(row: Mapping[str, Any]) -> None:
    require(row, "sample_id", "asset_revision_id", "dataset_release_id", "status", "geometry_ref", "metric_scale", "record_sha256")
    for key in ("sample_id", "asset_revision_id", "dataset_release_id", "status"):
        require_id(row[key], key)
    if row["status"] != "released":
        raise SchemaError("asset rows must have status=released for formal input")
    validate_artifact_ref(row["geometry_ref"], "geometry_ref")
    scale = row["metric_scale"]
    if not isinstance(scale, Mapping):
        raise SchemaError("metric_scale must be an object")
    raw_scale = scale.get("meters_per_unit")
    if raw_scale is None:
        raw_scale = scale.get("scale_to_meter")
    if raw_scale is None:
        raw_scale = scale.get("scale_m")
    try:
        scale_value = float(raw_scale)
    except (TypeError, ValueError) as exc:
        raise SchemaError("metric_scale must declare meters_per_unit/scale_to_meter") from exc
    if not math.isfinite(scale_value) or scale_value <= 0:
        raise SchemaError("metric_scale must be a finite positive meter conversion")
    require_hash(row["record_sha256"], "record_sha256")


def validate_physical_row(row: Mapping[str, Any], allow_nonverified: bool = True) -> None:
    require(row, "sample_id", "asset_revision_id", "dataset_release_id", "physical_bundle_id", "status", "record_sha256")
    for key in ("sample_id", "asset_revision_id", "dataset_release_id", "physical_bundle_id"):
        require_id(row[key], key)
    status = row["status"]
    if status not in PHYSICAL_STATUSES:
        raise SchemaError(f"unknown physical status: {status}")
    if not allow_nonverified and status != "verified":
        raise SchemaError(f"non-verified bundle {row['physical_bundle_id']} cannot enter formal input")
    require_hash(row["record_sha256"], "record_sha256")
    trace_ref = row.get("agent_trace_ref") or row.get("trace_ref")
    if trace_ref is None:
        raise SchemaError("physical bundle requires agent_trace_ref/trace_ref, including rejected/failure rows")
    validate_artifact_ref(trace_ref, "agent_trace_ref")
    if status == "verified":
        require(row, "calibration_evidence_ref", "heldout_evidence_ref", "method_contract_hash")
        validate_artifact_ref(row["calibration_evidence_ref"], "calibration_evidence_ref")
        validate_artifact_ref(row["heldout_evidence_ref"], "heldout_evidence_ref")
        require_hash(row["method_contract_hash"], "method_contract_hash")
    else:
        require(row, "failure_code")
        require_id(row["failure_code"], "failure_code")


def validate_episode_record(record: Mapping[str, Any]) -> None:
    require(record, "schema_name", "schema_version", "run_id", "task_id", "episode_id",
            "experiment_id", "protocol_hash", "data_role", "method_id", "execution_status",
            "outcome_status", "failure_domain", "record_sha256")
    if record["schema_name"] not in {"softsim.episode_record", "softsim.case_record", "softsim.call_record"}:
        raise SchemaError("unknown record schema_name")
    if record["schema_version"] != 1:
        raise SchemaError("unsupported record schema_version")
    for key in ("run_id", "task_id", "episode_id", "experiment_id", "protocol_hash", "data_role", "method_id"):
        require_id(record[key], key)
    if record["execution_status"] not in TERMINAL_EXECUTION:
        raise SchemaError("record must be terminal before aggregation")
    if record["outcome_status"] not in OUTCOME_STATUSES:
        raise SchemaError("invalid outcome_status")
    if record["failure_domain"] not in FAILURE_DOMAINS:
        raise SchemaError("invalid failure_domain")
    require_hash(record["protocol_hash"], "protocol_hash")
    require_hash(record["record_sha256"], "record_sha256")
    if record["outcome_status"] == "failure":
        require(record, "failure_code")
        require_id(record["failure_code"], "failure_code")
    if record.get("experiment_id") == "e3a":
        outcomes = record.get("outcomes")
        measurements = record.get("measurements")
        if not isinstance(outcomes, Mapping) or not isinstance(measurements, Mapping):
            raise SchemaError("E3-A records require outcomes and measurements")
        for field in ("task_success", "damage_failure", "intact_success"):
            if field not in outcomes:
                raise SchemaError("E3-A outcomes missing " + field)
        for field in ("damage_fraction", "integrity", "safe_force_window", "slip", "contact_count", "convergence"):
            if field not in measurements:
                raise SchemaError("E3-A measurements missing " + field)


def validate_run_manifest(manifest: Mapping[str, Any]) -> None:
    require(manifest, "schema_name", "schema_version", "run_id", "experiment_id", "protocol_hash",
            "expected_task_count", "task_manifest_sha256", "config_hash", "code_hash")
    if manifest["schema_name"] != "softsim.run_manifest" or manifest["schema_version"] != 1:
        raise SchemaError("unsupported run manifest")
    require_id(manifest["run_id"], "run_id")
    require_id(manifest["experiment_id"], "experiment_id")
    for field in ("protocol_hash", "task_manifest_sha256", "config_hash", "code_hash"):
        require_hash(manifest[field], field)
    if not isinstance(manifest["expected_task_count"], int) or manifest["expected_task_count"] < 0:
        raise SchemaError("expected_task_count must be non-negative integer")


def validate_unique(rows: Iterable[Mapping[str, Any]], field: str) -> None:
    seen = set()
    for row in rows:
        value = row.get(field)
        if value in seen:
            raise SchemaError(f"duplicate {field}: {value}")
        seen.add(value)
