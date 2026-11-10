# SPDX-License-Identifier: Apache-2.0
"""E3-B sim-real push data interfaces (no hardware control in Phase I)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .canonical import fingerprint


@dataclass(frozen=True)
class SimRealPairSpec:
    pair_id: str
    object_id: str
    action_id: str
    sim_bundle_id: str
    real_trial_id: str
    protocol_hash: str


@dataclass(frozen=True)
class TrajectoryAlignment:
    pair_id: str
    sim_times: tuple[float, ...]
    real_times: tuple[float, ...]
    index_pairs: tuple[tuple[int, int], ...]
    method: str = "nearest_timestamp"


class RealTrialImporter:
    """Validate, but do not mutate, a timestamped real-trial manifest."""

    def import_trial(self, manifest: Mapping[str, Any], trajectory: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        required = ("real_trial_id", "object_id", "trajectory_sha256", "source_hash")
        missing = [key for key in required if not manifest.get(key)]
        if missing:
            raise ValueError("real trial manifest missing: " + ", ".join(missing))
        if not trajectory:
            raise ValueError("real trial trajectory is empty")
        times = []
        for row in trajectory:
            try:
                value = float(row["time_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("trajectory rows require finite time_s") from exc
            if not math.isfinite(value):
                raise ValueError("trajectory times must be finite")
            times.append(value)
        if times != sorted(times) or len(set(times)) != len(times):
            raise ValueError("trajectory timestamps must be strictly increasing")
        return {"schema_name": "softsim.real_trial", "schema_version": 1, "manifest": dict(manifest), "trajectory": [dict(row) for row in trajectory], "trajectory_hash": fingerprint(list(trajectory)), "imported": True, "hardware_control_executed": False}


def align_trajectories(pair_id: str, sim: Sequence[Mapping[str, Any]], real: Sequence[Mapping[str, Any]]) -> TrajectoryAlignment:
    if not sim or not real:
        raise ValueError("both trajectories are required")
    sim_times = tuple(float(row["time_s"]) for row in sim)
    real_times = tuple(float(row["time_s"]) for row in real)
    pairs = []
    for i, time_s in enumerate(sim_times):
        j = min(range(len(real_times)), key=lambda index: abs(real_times[index] - time_s))
        pairs.append((i, j))
    return TrajectoryAlignment(pair_id, sim_times, real_times, tuple(pairs))


class PairedPushEvaluator:
    def evaluate(self, pair: SimRealPairSpec, sim: Sequence[Mapping[str, Any]], real: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        alignment = align_trajectories(pair.pair_id, sim, real)
        errors = []
        for i, j in alignment.index_pairs:
            sim_row, real_row = sim[i], real[j]
            sim_xyz = [float(sim_row.get(axis, 0.0)) for axis in ("x", "y", "z")]
            real_xyz = [float(real_row.get(axis, 0.0)) for axis in ("x", "y", "z")]
            errors.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(sim_xyz, real_xyz))))
        rmse = math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else None
        return {"schema_name": "softsim.paired_push_result", "schema_version": 1, "pair_id": pair.pair_id, "sim_bundle_id": pair.sim_bundle_id, "real_trial_id": pair.real_trial_id, "alignment_method": alignment.method, "trajectory_rmse": rmse, "n_aligned": len(errors), "hardware_control_executed": False}
