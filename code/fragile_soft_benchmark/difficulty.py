# SPDX-License-Identifier: Apache-2.0
"""Deterministic, bounded difficulty-state constructor."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from .hashing import sha256_json
from .scenario import immutable_contract_hashes, make_scenario


ALLOWED_DIFFICULTY_AXES = frozenset(
    {
        "force_scale",
        "damage_onset_scale",
        "damage_complete_scale",
        "contact_sensitivity",
        "residual_stiffness_scale",
        "yield_scale",
        "gel_relaxation_scale",
    }
)

DIFFICULTY_STATES: dict[str, dict[str, float]] = {
    "d00": {"force_scale": 0.80, "damage_onset_scale": 1.20, "damage_complete_scale": 1.20, "contact_sensitivity": 0.90},
    "d01": {"force_scale": 0.90, "damage_onset_scale": 1.10, "damage_complete_scale": 1.10, "contact_sensitivity": 0.95},
    "d02": {"force_scale": 1.00, "damage_onset_scale": 1.00, "damage_complete_scale": 1.00, "contact_sensitivity": 1.00},
    "d03": {"force_scale": 1.10, "damage_onset_scale": 0.95, "damage_complete_scale": 0.98, "contact_sensitivity": 1.00},
    "d04": {"force_scale": 1.20, "damage_onset_scale": 0.90, "damage_complete_scale": 0.95, "contact_sensitivity": 1.05},
    "d05": {"force_scale": 0.95, "damage_onset_scale": 1.00, "damage_complete_scale": 1.00, "contact_sensitivity": 0.75},
    "d06": {"force_scale": 1.05, "damage_onset_scale": 1.00, "damage_complete_scale": 1.00, "contact_sensitivity": 1.25},
    "d07": {"force_scale": 1.00, "damage_onset_scale": 0.85, "damage_complete_scale": 0.90, "contact_sensitivity": 1.00, "residual_stiffness_scale": 0.75},
    "d08": {"force_scale": 1.15, "damage_onset_scale": 1.05, "damage_complete_scale": 1.05, "contact_sensitivity": 1.10, "yield_scale": 0.90},
    "d09": {"force_scale": 1.25, "damage_onset_scale": 0.80, "damage_complete_scale": 0.88, "contact_sensitivity": 1.20, "residual_stiffness_scale": 0.50},
}


def _state(state_id: str, registry: Mapping[str, Mapping[str, float]] | None) -> dict[str, float]:
    states = registry or DIFFICULTY_STATES
    if isinstance(states, Mapping) and isinstance(states.get("states"), Mapping):
        states = states["states"]  # type: ignore[assignment]
    if state_id not in states:
        raise KeyError(f"unknown difficulty_id: {state_id}")
    values = {str(key): float(value) for key, value in states[state_id].items()}
    unknown = set(values) - ALLOWED_DIFFICULTY_AXES
    if unknown:
        raise ValueError(f"unregistered difficulty axes: {sorted(unknown)}")
    if any(value <= 0 for value in values.values()):
        raise ValueError("difficulty multipliers must be positive")
    return values


def construct_difficulty(
    material: Mapping[str, Any],
    base_force_schedule: Mapping[str, float] | list[float] | tuple[float, ...],
    *,
    difficulty_id: str,
    geometry_revision_id: str,
    scenario_hash: str,
    action_protocol_hash: str,
    evaluator_hash: str,
    registry: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Apply one registered state while preserving immutable contracts."""

    values = _state(difficulty_id, registry)
    fixed = immutable_contract_hashes(make_scenario())
    supplied = {
        "scenario_hash": scenario_hash,
        "action_protocol_hash": action_protocol_hash,
        "evaluator_hash": evaluator_hash,
    }
    if supplied != fixed:
        raise ValueError("difficulty cannot change the frozen scenario/action/evaluator contract")
    material_out = copy.deepcopy(dict(material))
    params = copy.deepcopy(dict(material_out.get("parameters", material_out)))
    model_id = str(material_out.get("material_model_id") or material_out.get("model_id") or material_out.get("model"))
    changed: dict[str, Any] = {}
    if "damage_onset_scale" in values and "damage_start_compressive_strain" in params:
        old = float(params["damage_start_compressive_strain"])
        params["damage_start_compressive_strain"] = old * values["damage_onset_scale"]
        changed["damage_start_compressive_strain"] = [old, params["damage_start_compressive_strain"]]
    if "damage_complete_scale" in values and "damage_complete_compressive_strain" in params:
        old = float(params["damage_complete_compressive_strain"])
        params["damage_complete_compressive_strain"] = old * values["damage_complete_scale"]
        changed["damage_complete_compressive_strain"] = [old, params["damage_complete_compressive_strain"]]
    if "residual_stiffness_scale" in values and "residual_stiffness_fraction" in params:
        old = float(params["residual_stiffness_fraction"])
        params["residual_stiffness_fraction"] = min(1.0, old * values["residual_stiffness_scale"])
        changed["residual_stiffness_fraction"] = [old, params["residual_stiffness_fraction"]]
    if "yield_scale" in values:
        for key in ("yield_lower", "yield_higher"):
            if key in params:
                old = float(params[key])
                params[key] = old * values["yield_scale"]
                changed[key] = [old, params[key]]
    if "gel_relaxation_scale" in values and "relaxation_time_s" in params:
        old = float(params["relaxation_time_s"])
        params["relaxation_time_s"] = old * values["gel_relaxation_scale"]
        changed["relaxation_time_s"] = [old, params["relaxation_time_s"]]
    if (
        "damage_start_compressive_strain" in params
        and "damage_complete_compressive_strain" in params
        and params["damage_complete_compressive_strain"] <= params["damage_start_compressive_strain"]
    ):
        raise ValueError("difficulty scaling must preserve damage onset ordering")
    if isinstance(base_force_schedule, Mapping):
        forces = {str(key): float(value) * values.get("force_scale", 1.0) for key, value in base_force_schedule.items()}
    else:
        forces = [float(value) * values.get("force_scale", 1.0) for value in base_force_schedule]
    result = {
        "difficulty_id": difficulty_id,
        "difficulty_axes": values,
        "material_model_id": model_id,
        "material_parameters": params,
        "force_schedule": forces,
        "contact_sensitivity": values.get("contact_sensitivity", 1.0),
        "geometry_revision_id": geometry_revision_id,
        "immutable_contract": {
            "scenario_hash": scenario_hash,
            "action_protocol_hash": action_protocol_hash,
            "evaluator_hash": evaluator_hash,
        },
        "changed_parameters": changed,
    }
    result["difficulty_hash"] = sha256_json(result)
    return result
