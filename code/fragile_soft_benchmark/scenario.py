# SPDX-License-Identifier: Apache-2.0
"""Frozen grasp-lift-hold scenario contract."""

from __future__ import annotations

from typing import Any, Mapping

from .hashing import sha256_json


SCENARIO_SCHEMA_VERSION = "fragile-soft.scenario.v1"
PHASES = ("approach", "contact", "squeeze", "lift", "hold")
EVALUATOR_ID = "fragile-soft.evaluator.v1"
ACTION_PROTOCOL_ID = "grasp-lift-hold.v1"

DEFAULT_SCENARIO: dict[str, Any] = {
    "schema_version": SCENARIO_SCHEMA_VERSION,
    "scenario_id": ACTION_PROTOCOL_ID,
    "phases": [
        {"name": "approach", "steps": 8, "hand_velocity_mps": 0.04, "contact_force_n": 0.0},
        {"name": "contact", "steps": 8, "hand_velocity_mps": 0.01, "contact_force_n": 0.0},
        {"name": "squeeze", "steps": 12, "hand_velocity_mps": 0.0, "contact_force_n": "scheduled"},
        {"name": "lift", "steps": 16, "hand_velocity_mps": 0.05, "contact_force_n": "held"},
        {"name": "hold", "steps": 12, "hand_velocity_mps": 0.0, "contact_force_n": "held"},
    ],
    "action_protocol": {
        "gripper": "two_finger_force_control",
        "approach_axis": "world_z",
        "lift_axis": "world_z",
        "lift_height_m": 0.05,
        "hold_duration_s": 0.12,
        "force_ramp": "smoothstep",
        "force_schedule_source": "difficulty_only",
    },
    "evaluator_id": EVALUATOR_ID,
    "evaluator_contract": {
        "fixed": True,
        "metrics": [
            "task_success",
            "damage_onset",
            "terminal_damage_fraction",
            "integrity_score",
            "safe_force_window",
            "contact_fraction",
            "slip_distance_m",
            "max_penetration_m",
            "finite_fraction",
            "converged",
        ],
    },
}


def make_scenario(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a validated copy; only force schedule is overrideable downstream."""

    import copy

    scenario = copy.deepcopy(DEFAULT_SCENARIO)
    if overrides:
        for key, value in overrides.items():
            scenario[key] = copy.deepcopy(value)
    validate_scenario(scenario)
    scenario["contract_hash"] = sha256_json(
        {key: value for key, value in scenario.items() if key != "contract_hash"}
    )
    return scenario


def validate_scenario(scenario: Mapping[str, Any]) -> None:
    if tuple(item.get("name") for item in scenario.get("phases", [])) != PHASES:
        raise ValueError(f"scenario phases must be exactly {' -> '.join(PHASES)}")
    if scenario.get("evaluator_id") != EVALUATOR_ID:
        raise ValueError("benchmark evaluator is frozen")
    if scenario.get("scenario_id") != ACTION_PROTOCOL_ID:
        raise ValueError("benchmark action protocol is frozen")
    if any(int(item.get("steps", 0)) <= 0 for item in scenario.get("phases", [])):
        raise ValueError("scenario phase steps must be positive")


def immutable_contract_hashes(scenario: Mapping[str, Any]) -> dict[str, str]:
    validate_scenario(scenario)
    return {
        "scenario_hash": str(scenario.get("contract_hash") or sha256_json(scenario)),
        "action_protocol_hash": sha256_json(scenario["action_protocol"]),
        "evaluator_hash": sha256_json(
            {"evaluator_id": scenario["evaluator_id"], "contract": scenario["evaluator_contract"]}
        ),
    }
