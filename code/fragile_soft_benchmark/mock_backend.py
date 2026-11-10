# SPDX-License-Identifier: Apache-2.0
"""Small deterministic backend for local fixtures and qualification smoke."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .hashing import sha256_json, stable_seed


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _material_model(instance: Mapping[str, Any]) -> str:
    material = instance.get("material") or instance.get("material_config") or {}
    return str(material.get("material_model_id") or material.get("model_id") or material.get("model") or "unknown")


def _parameters(instance: Mapping[str, Any]) -> Mapping[str, Any]:
    difficulty = instance.get("difficulty") or {}
    if isinstance(difficulty.get("material_parameters"), Mapping):
        return difficulty["material_parameters"]
    material = instance.get("material") or instance.get("material_config") or {}
    return material.get("parameters", material.get("material_parameters", material))


def _target_force(instance: Mapping[str, Any], force_n: float | None) -> float:
    if force_n is not None:
        return max(0.0, float(force_n))
    difficulty = instance.get("difficulty") or {}
    schedule = difficulty.get("force_schedule") or instance.get("force_schedule") or {}
    if isinstance(schedule, Mapping):
        values = list(schedule.values())
        return float(values[len(values) // 2]) if values else 10.0
    if isinstance(schedule, (list, tuple)) and schedule:
        return float(schedule[len(schedule) // 2])
    return 10.0


def _sample_at_force(
    instance: Mapping[str, Any],
    force_n: float,
    *,
    seed: int,
    no_contact: bool = False,
    backend_failure: bool = False,
) -> dict[str, Any]:
    model = _material_model(instance)
    params = _parameters(instance)
    difficulty = instance.get("difficulty") or {}
    contact_sensitivity = float(difficulty.get("contact_sensitivity", 1.0))
    # Bounded deterministic jitter models solver noise without making replay
    # nondeterministic.  Reusing a seed gives byte-identical records.
    jitter = ((stable_seed(instance.get("benchmark_instance_id"), seed, force_n) % 1001) - 500) / 1_000_000.0
    if no_contact:
        effective_force = 0.0
        contact = 0.0
    else:
        effective_force = max(0.0, force_n * contact_sensitivity)
        contact = 0.0 if force_n <= 0 else _clamp(0.80 + jitter)
    if model == "tofu_damage":
        onset = float(params.get("damage_start_compressive_strain", 0.05))
        complete = float(params.get("damage_complete_compressive_strain", onset + 0.02))
        # The normalized load law is deliberately simple and monotone for the
        # qualification backend; a real simulator can consume the same contract.
        # Calibrated only as a deterministic fixture law: P0's low force stays
        # below onset while the high force enters the damage band.
        strain = effective_force / 150.0
        damage = _clamp((strain - onset) / max(complete - onset, 1e-9))
    elif model in {"plastic", "foam", "gel", "elastic"}:
        yield_value = float(params.get("yield_lower", 0.03))
        damage = 0.0
        strain = effective_force / max(float(params.get("youngs_modulus_pa", 30000.0)) / 1000.0, 1.0)
        # Plastic/foam retain integrity but exhibit reversible/irreversible
        # deformation in the separate deformation proxy.
        if model == "foam":
            yield_value = max(yield_value, 0.02)
    else:
        return {
            "backend_status": "unsupported",
            "failure_code": "unsupported_material_model",
            "phase": "hold",
            "time_s": 0.0,
            "force_n": force_n,
            "damage_fraction": 0.0,
            "integrity_score": 0.0,
            "contact_fraction": contact,
            "slip_distance_m": 0.0,
            "max_penetration_m": 0.0,
            "lift_height_m": 0.0,
            "finite_fraction": 1.0,
            "converged": False,
        }
    integrity = _clamp(1.0 - damage)
    # A low force can slip; once contact force is sufficient, the object follows
    # the fixed lift trajectory.  This produces a non-empty safe window.
    grip_threshold = float((instance.get("geometry") or {}).get("grip_threshold_n", 4.0))
    slip = max(0.0, (grip_threshold - effective_force) / max(grip_threshold, 1.0)) * 0.02
    required_lift = float(instance.get("scenario", {}).get("action_protocol", {}).get("lift_height_m", 0.05))
    lift = required_lift * _clamp((effective_force - 0.5 * grip_threshold) / max(grip_threshold, 1.0))
    penetration = max(0.0, strain * 0.001 - 0.004)
    return {
        "backend_status": "failed" if backend_failure else "completed",
        "phase": "hold",
        "time_s": 0.44,
        "force_n": force_n,
        "damage_fraction": _clamp(damage),
        "integrity_score": integrity,
        "contact_fraction": contact,
        "slip_distance_m": slip,
        "max_penetration_m": penetration,
        "lift_height_m": lift,
        "finite_fraction": 0.0 if backend_failure else 1.0,
        "converged": not backend_failure,
    }


def run_mock_episode(
    instance: Mapping[str, Any],
    *,
    seed: int = 0,
    force_n: float | None = None,
    no_contact: bool = False,
    backend_failure: bool = False,
    force_sweep: list[float] | None = None,
) -> dict[str, Any]:
    """Run a compact deterministic episode with all five protocol phases."""

    target = _target_force(instance, force_n)
    instance_id = str(instance.get("benchmark_instance_id", "unknown"))
    episode_id = f"ep-{instance_id}-seed{int(seed):04d}"
    phases = ("approach", "contact", "squeeze", "lift", "hold")
    samples: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for index, phase in enumerate(phases):
        phase_force = 0.0 if phase in {"approach", "contact"} else target
        sample = _sample_at_force(
            instance,
            phase_force,
            seed=seed,
            no_contact=no_contact,
            backend_failure=backend_failure,
        )
        sample["phase"] = phase
        sample["time_s"] = round(0.08 * index, 6)
        samples.append(sample)
        actions.append({"phase": phase, "force_n": phase_force, "step": index})
    sweep_values = force_sweep or sorted({round(target * factor, 6) for factor in (0.5, 0.75, 1.0, 1.25, 1.5) if target * factor > 0})
    sweep_samples = [
        _sample_at_force(instance, value, seed=seed, no_contact=no_contact, backend_failure=backend_failure)
        for value in sweep_values
    ]
    episode = {
        "schema_version": "fragile-soft.episode.v1",
        "episode_id": episode_id,
        "instance_id": instance_id,
        "seed": int(seed),
        "backend_id": "mock.deterministic.v1",
        "backend_status": "failed" if backend_failure else ("unsupported" if any(s.get("backend_status") == "unsupported" for s in samples) else "completed"),
        "required_lift_height_m": float(instance.get("scenario", {}).get("action_protocol", {}).get("lift_height_m", 0.05)),
        "actions": actions,
        "samples": samples,
        "force_sweep_samples": sweep_samples,
    }
    episode["action_replay_hash"] = sha256_json(actions)
    episode["episode_hash"] = sha256_json(episode)
    return episode
