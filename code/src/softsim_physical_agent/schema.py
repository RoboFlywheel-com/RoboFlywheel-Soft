# SPDX-License-Identifier: Apache-2.0
"""Small compatibility/schema helpers for release readers."""

from __future__ import annotations

from typing import Any, Mapping

from .io import content_hash
from .routing import physical_values


def record_hash(record: Mapping[str, Any]) -> str:
    return content_hash({key: value for key, value in record.items() if key not in {"record_sha256"}})


def bundle_id(record: Mapping[str, Any]) -> str:
    return content_hash({
        "sample_id": record.get("sample_id"),
        "asset_revision_id": record.get("asset_revision_id"),
        "route": record.get("route"),
        "candidate": record.get("candidate"),
        "solver_numerics": record.get("solver_numerics"),
        "verification": record.get("verification"),
    })


def parameter_map(bundle: Mapping[str, Any]) -> dict[str, float]:
    return physical_values(bundle)


def validate_probe_pair(calibration: Mapping[str, Any], heldout: Mapping[str, Any]) -> None:
    for key in ("probe_id", "probe_hash", "stimulus_hash", "action_hash", "config_hash"):
        if not calibration.get(key) or not heldout.get(key):
            raise ValueError(f"probe pair missing {key}")
        if calibration.get(key) == heldout.get(key):
            raise ValueError(f"calibration and heldout {key} must differ")


def validate_task_contract(contract: Mapping[str, Any]) -> None:
    required = ("sample_id", "asset_revision_id", "route", "fixed_solver_numerics", "fixed_solver_numerics_hash", "calibration_probe", "heldout_probe")
    missing = [key for key in required if key not in contract]
    if missing:
        raise ValueError("task contract missing: " + ", ".join(missing))
    validate_probe_pair(contract["calibration_probe"], contract["heldout_probe"])
    fixed = contract["fixed_solver_numerics"]
    for key in ("particle_size_m", "dt_s", "substeps", "grid_density", "scale_m", "seed", "proxy_mode"):
        if key not in fixed:
            raise ValueError(f"fixed solver numeric missing: {key}")


def validate_physics_bundle(bundle: Mapping[str, Any]) -> None:
    required = ("sample_id", "asset_revision_id", "status", "route", "solver_numerics", "verification", "agent")
    missing = [key for key in required if key not in bundle]
    if missing:
        raise ValueError("physics bundle missing: " + ", ".join(missing))
    if bundle["status"] not in {"accepted_first_pass", "corrected_then_accepted", "rejected_model_limit", "rejected_numerical_failure", "uncertain_human_escalation"}:
        raise ValueError(f"invalid physics bundle status: {bundle['status']}")
    validate_probe_pair(bundle["verification"]["calibration"], bundle["verification"]["heldout"])
    if bundle.get("correction_applied") and bundle["status"] == "accepted_first_pass":
        raise ValueError("accepted_first_pass cannot claim correction_applied")
    if bundle.get("status") == "corrected_then_accepted" and not bundle.get("correction_applied"):
        raise ValueError("corrected_then_accepted requires correction_applied")
