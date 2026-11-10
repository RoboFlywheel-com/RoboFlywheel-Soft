# SPDX-License-Identifier: Apache-2.0
"""Frozen evaluator for grasp-lift-hold episodes."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from .hashing import sha256_json


METRIC_NAMES = (
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
)


def evaluate_force_sweep(
    samples: Iterable[Mapping[str, Any]],
    *,
    damage_threshold: float = 0.10,
    integrity_threshold: float = 0.80,
    slip_threshold_m: float = 0.010,
    penetration_threshold_m: float = 0.004,
) -> dict[str, Any]:
    """Summarize force-indexed terminal records using the frozen evaluator."""

    rows = sorted((dict(row) for row in samples), key=lambda row: float(row.get("force_n", 0.0)))
    damage = [(_finite(row.get("force_n")), _finite(row.get("damage_fraction"))) for row in rows]
    violations = [
        {"lower_force_n": left[0], "higher_force_n": right[0], "decrease": left[1] - right[1]}
        for left, right in zip(damage, damage[1:])
        if right[1] + 0.03 < left[1]
    ]
    onset = next((force for force, value in damage if value > damage_threshold), None)
    window = _safe_window(
        rows,
        damage_threshold=damage_threshold,
        integrity_threshold=integrity_threshold,
        slip_threshold_m=slip_threshold_m,
        penetration_threshold_m=penetration_threshold_m,
    )
    return {
        "damage_by_force": [{"force_n": force, "damage_fraction": value} for force, value in damage],
        "damage_onset_force_n": onset,
        "monotonic_violations": violations,
        "monotonic": not violations,
        "safe_force_window": window,
    }


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _bool(value: Any) -> bool:
    return bool(value) if isinstance(value, (bool, int, float)) else False


def _safe_window(
    samples: Iterable[Mapping[str, Any]],
    *,
    damage_threshold: float,
    integrity_threshold: float,
    slip_threshold_m: float,
    penetration_threshold_m: float,
) -> dict[str, Any]:
    safe_forces: list[float] = []
    for sample in samples:
        force = _finite(sample.get("force_n"))
        damage = _finite(sample.get("damage_fraction"))
        integrity = _finite(sample.get("integrity_score"), 0.0)
        slip = _finite(sample.get("slip_distance_m"))
        penetration = _finite(sample.get("max_penetration_m"))
        finite = _finite(sample.get("finite_fraction"), 0.0) >= 1.0 - 1e-12
        if (
            finite
            and damage <= damage_threshold
            and integrity >= integrity_threshold
            and slip <= slip_threshold_m
            and penetration <= penetration_threshold_m
            and force > 0.0
        ):
            safe_forces.append(force)
    if not safe_forces:
        return {
            "non_empty": False,
            "min_force_n": 0.0,
            "max_force_n": 0.0,
            "width_n": 0.0,
            "sample_count": 0,
        }
    low, high = min(safe_forces), max(safe_forces)
    return {
        "non_empty": True,
        "min_force_n": low,
        "max_force_n": high,
        "width_n": max(0.0, high - low),
        "sample_count": len(safe_forces),
    }


def evaluate_episode(
    episode: Mapping[str, Any],
    *,
    evaluator_id: str = "fragile-soft.evaluator.v1",
    damage_threshold: float = 0.10,
    integrity_threshold: float = 0.80,
    slip_threshold_m: float = 0.010,
    penetration_threshold_m: float = 0.004,
) -> dict[str, Any]:
    """Compute all metrics, including for backend failures or empty traces.

    The evaluator never drops a failed episode.  Missing/non-finite values are
    represented by conservative finite defaults and accompanied by a failure
    code in the returned record.
    """

    raw_samples = episode.get("samples") or episode.get("states") or episode.get("trajectory")
    samples = [dict(sample) for sample in raw_samples] if isinstance(raw_samples, list) else []
    normalized: list[dict[str, Any]] = []
    for sample in samples:
        normalized.append(
            {
                "phase": str(sample.get("phase", "unknown")),
                "time_s": _finite(sample.get("time_s")),
                "force_n": _finite(sample.get("force_n")),
                "damage_fraction": min(
                    1.0,
                    max(0.0, _finite(sample.get("damage_fraction", sample.get("damage", sample.get("damaged_fraction"))))),
                ),
                "integrity_score": min(
                    1.0,
                    max(0.0, _finite(sample.get("integrity_score", sample.get("integrity")), default=1.0)),
                ),
                "contact_fraction": min(
                    1.0,
                    max(0.0, _finite(sample.get("contact_fraction", sample.get("contact")))),
                ),
                "slip_distance_m": max(
                    0.0, _finite(sample.get("slip_distance_m", sample.get("slip_m", sample.get("slip"))))
                ),
                "max_penetration_m": max(
                    0.0, _finite(sample.get("max_penetration_m", sample.get("penetration_m", sample.get("penetration"))))
                ),
                "lift_height_m": _finite(sample.get("lift_height_m", sample.get("lift_m", sample.get("lift_height")))),
                "finite_fraction": min(
                    1.0,
                    max(0.0, _finite(sample.get("finite_fraction", sample.get("finite")), default=0.0)),
                ),
                "converged": _bool(sample.get("converged", False)),
            }
        )
    final = normalized[-1] if normalized else {
        "phase": "unknown",
        "time_s": 0.0,
        "force_n": 0.0,
        "damage_fraction": 0.0,
        "integrity_score": 0.0,
        "contact_fraction": 0.0,
        "slip_distance_m": 0.0,
        "max_penetration_m": 0.0,
        "lift_height_m": 0.0,
        "finite_fraction": 0.0,
        "converged": False,
    }
    onset = next((sample for sample in normalized if sample["damage_fraction"] > damage_threshold), None)
    finite_fraction = min((sample["finite_fraction"] for sample in normalized), default=0.0)
    converged = bool(normalized) and all(sample["converged"] for sample in normalized)
    max_contact = max((sample["contact_fraction"] for sample in normalized), default=0.0)
    max_slip = max((sample["slip_distance_m"] for sample in normalized), default=0.0)
    max_penetration = max((sample["max_penetration_m"] for sample in normalized), default=0.0)
    samples_for_window = episode.get("force_sweep_samples", normalized)
    if not isinstance(samples_for_window, list):
        samples_for_window = normalized
    safe_window = _safe_window(
        samples_for_window,
        damage_threshold=damage_threshold,
        integrity_threshold=integrity_threshold,
        slip_threshold_m=slip_threshold_m,
        penetration_threshold_m=penetration_threshold_m,
    )
    task_success = bool(
        final["lift_height_m"] >= _finite(episode.get("required_lift_height_m"), 0.010)
        and final["integrity_score"] >= integrity_threshold
        and final["slip_distance_m"] <= slip_threshold_m
        and final["finite_fraction"] >= 1.0 - 1e-12
        and converged
    )
    failure_codes: list[str] = []
    backend_status = str(episode.get("backend_status", "completed"))
    if backend_status != "completed":
        failure_codes.append("backend_failure")
    if not normalized:
        failure_codes.append("missing_samples")
    if finite_fraction < 1.0 - 1e-12:
        failure_codes.append("non_finite_state")
    if not converged:
        failure_codes.append("not_converged")
    metrics = {
        "task_success": task_success,
        "damage_onset": {
            "observed": onset is not None,
            "time_s": onset["time_s"] if onset else -1.0,
            "force_n": onset["force_n"] if onset else 0.0,
        },
        "terminal_damage_fraction": final["damage_fraction"],
        "integrity_score": final["integrity_score"],
        "safe_force_window": safe_window,
        "contact_fraction": max_contact,
        "slip_distance_m": max_slip,
        "max_penetration_m": max_penetration,
        "finite_fraction": finite_fraction,
        "converged": converged,
        # Short aliases keep readers written against either the P0 names or
        # the new verbose contract interoperable.
        "damage": {"onset": onset["force_n"] if onset else None, "terminal_fraction": final["damage_fraction"]},
        "integrity": final["integrity_score"],
        "slip": max_slip,
        "contact": max_contact,
        "penetration": max_penetration,
        "finite": finite_fraction,
        "convergence": converged,
    }
    record = {
        "schema_version": "fragile-soft.metric.v1",
        "evaluator_id": evaluator_id,
        "episode_id": episode.get("episode_id"),
        "instance_id": episode.get("instance_id"),
        "seed": episode.get("seed"),
        "backend_status": backend_status,
        "failure_codes": sorted(set(failure_codes)),
        "metrics": metrics,
        "metric_names": list(METRIC_NAMES),
        "action_replay_hash": episode.get("action_replay_hash") or sha256_json(episode.get("actions", [])),
    }
    record["metric_record_hash"] = sha256_json(record)
    return record
