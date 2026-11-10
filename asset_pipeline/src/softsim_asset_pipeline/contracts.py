# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from typing import Any

from .io import content_id


ASSET_STATUSES = {
    "ready",
    "ready_with_warnings",
    "repairable",
    "invalid",
    "missing_artifact",
    "deferred_cloth_generation",
    "superseded",
}

PHYSICAL_STATUSES = {
    "unassigned",
    "predicted",
    "calibrated_semantic",
    "calibrated_behavior",
    "verified",
    "rejected",
    "unsupported",
}

SOURCE_TYPES = {
    "vlm_prior",
    "reviewed_family_prior",
    "measured",
    "behavior_calibration",
    "synthetic_fixture",
}


def with_record_hash(record: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    return {**payload, "record_sha256": content_id(payload)}


def verify_record_hash(record: dict[str, Any]) -> None:
    expected = with_record_hash(record)["record_sha256"]
    if record.get("record_sha256") != expected:
        raise ValueError(f"record_sha256 mismatch for {record.get('sample_id')}")


def _require(record: dict[str, Any], fields: set[str], schema: str) -> None:
    missing = sorted(field for field in fields if field not in record)
    if missing:
        raise ValueError(f"{schema}: missing fields: {', '.join(missing)}")


def validate_asset_ledger_record(record: dict[str, Any]) -> None:
    _require(
        record,
        {
            "schema_name",
            "schema_version",
            "sample_id",
            "asset_revision_id",
            "dataset_release_id",
            "geometry_family",
            "generation_route",
            "asset_status",
            "failure_codes",
            "artifacts",
            "maintenance",
            "record_sha256",
        },
        "asset_ledger",
    )
    if record["schema_name"] != "softsim.asset_ledger" or record["schema_version"] != 1:
        raise ValueError("asset_ledger: unsupported schema")
    if record["asset_status"] not in ASSET_STATUSES:
        raise ValueError(f"asset_ledger: invalid status {record['asset_status']!r}")
    if not isinstance(record["artifacts"], list):
        raise ValueError("asset_ledger: artifacts must be a list")
    verify_record_hash(record)


def validate_physical_bundle(record: dict[str, Any]) -> None:
    _require(
        record,
        {
            "schema_name",
            "schema_version",
            "sample_id",
            "asset_revision_id",
            "physical_bundle_id",
            "status",
            "simulation_domain",
            "representation",
            "model_family",
            "parameters",
            "solver_adapter",
            "solver_numerics",
            "estimator",
            "agent",
            "verification",
            "limitations",
            "warnings",
            "provenance",
            "record_sha256",
        },
        "physical_bundle",
    )
    if record["schema_name"] != "softsim.physical_bundle" or record["schema_version"] != 1:
        raise ValueError("physical_bundle: unsupported schema")
    if record["status"] not in PHYSICAL_STATUSES:
        raise ValueError(f"physical_bundle: invalid status {record['status']!r}")
    if record["status"] in {"calibrated_semantic", "calibrated_behavior", "verified"}:
        verification = record["verification"]
        if not verification.get("calibration_probe_suite_hash"):
            raise ValueError("physical_bundle: calibrated/verified status lacks calibration evidence")
    if record["status"] == "verified" and not record["verification"].get("heldout_probe_suite_hash"):
        raise ValueError("physical_bundle: verified status lacks held-out evidence")
    for parameter in record["parameters"]:
        _require(
            parameter,
            {"name", "value", "unit", "valid_range", "confidence", "source_type"},
            "parameter",
        )
        value = parameter["value"]
        confidence = parameter["confidence"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"physical_bundle: non-finite parameter {parameter.get('name')}")
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"physical_bundle: invalid confidence for {parameter.get('name')}")
        if parameter["source_type"] not in SOURCE_TYPES:
            raise ValueError(f"physical_bundle: invalid source_type {parameter['source_type']!r}")
    verify_record_hash(record)


def physical_bundle_id(record: dict[str, Any]) -> str:
    identity = {
        "sample_id": record["sample_id"],
        "asset_revision_id": record["asset_revision_id"],
        "simulation_domain": record["simulation_domain"],
        "representation": record["representation"],
        "model_family": record["model_family"],
        "parameters": record["parameters"],
        "solver_adapter": record["solver_adapter"],
        "solver_numerics": record["solver_numerics"],
        "verification": record["verification"],
    }
    return content_id(identity)
