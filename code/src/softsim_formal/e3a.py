# SPDX-License-Identifier: Apache-2.0
"""E3-A failure-in-the-loop scripted grasp evaluator facade."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .canonical import fingerprint
from .planner import plan_run
from .runner import run_all_shards
from .aggregate import aggregate_run


PRIVILEGED_FIELDS = {"particles", "stress", "damage_state", "Jp", "evaluator_threshold"}
REQUIRED_MEASUREMENTS = ("damage_fraction", "integrity", "safe_force_window", "slip", "contact_count", "convergence")


def validate_observation(observation: Mapping[str, Any]) -> None:
    leaked = PRIVILEGED_FIELDS.intersection(observation)
    if leaked:
        raise ValueError("privileged evaluator state leaked into policy observation: " + ",".join(sorted(leaked)))


def validate_episode_metrics(record: Mapping[str, Any]) -> None:
    outcomes = record.get("outcomes")
    measurements = record.get("measurements")
    if not isinstance(outcomes, Mapping) or not isinstance(measurements, Mapping):
        raise ValueError("E3-A episode requires outcomes and measurements")
    for field in REQUIRED_MEASUREMENTS:
        if field not in measurements:
            raise ValueError("E3-A measurement missing " + field)
    if not isinstance(outcomes.get("task_success"), bool):
        raise ValueError("task_success must be boolean")
    # Damage/integrity are mandatory even when the task fails.
    if not outcomes.get("task_success") and (
        measurements.get("damage_fraction") is None or measurements.get("integrity") is None
    ):
        raise ValueError("failed E3-A episode must retain damage and integrity")


def run_e3a_smoke(output_dir: str, *, count: int = 10, force_schedule: Sequence[float] = (6.0, 15.0, 30.0)) -> dict[str, Any]:
    cfg = {"promotion_stage": "fixture", "count": int(count), "methods": ["scripted_contact_regulation"], "seeds": [0], "force_schedule": list(force_schedule)}
    plan = plan_run("e3a", output_dir=output_dir, config=cfg, fixture=True, count=count, shard_count=1)
    run_all_shards(output_dir)
    result = aggregate_run(output_dir)
    return result
