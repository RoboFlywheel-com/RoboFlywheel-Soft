# SPDX-License-Identifier: Apache-2.0
"""Controlled hidden-target recovery contracts and local method adapters."""

from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass
from typing import Any, Mapping

from .canonical import fingerprint
from .planner import plan_run
from .runner import run_all_shards
from .aggregate import aggregate_run


METHODS = (
    "fixed_profile",
    "one_shot_prior",
    "ruletune",
    "seeded_sobol",
    "vid2sid_style_reimplementation",
    "swim2real_style_reimplementation",
    "full_numeric_agent",
)
DISABLED_METHODS = {
    "gp_bo_disabled_optional": "optional dependency/checkpoint not installed",
    "external_policy_baseline": "checkpoint/data/source hash unavailable",
}
EXTERNAL_BASELINE_DISABLED = dict(DISABLED_METHODS)


class HeldoutAccessError(RuntimeError):
    pass


class HiddenTargetProviderError(ValueError):
    pass


def load_target_provider(reference: str):
    """Load a server-side ``module:function`` target provider."""
    if ":" not in reference:
        raise HiddenTargetProviderError("target provider must use module:function syntax")
    module_name, function_name = reference.split(":", 1)
    module = importlib.import_module(module_name)
    provider = getattr(module, function_name, None)
    if not callable(provider):
        raise HiddenTargetProviderError(f"target provider is not callable: {reference}")
    return provider


@dataclass
class HiddenTargetGuard:
    """In-memory guard used by tests and by the mock orchestrator.

    The held-out payload can be opened only after ``freeze``.  Reading it
    before that point raises, making calibration/held-out leakage testable.
    """

    calibration: dict[str, Any]
    heldout: dict[str, Any]
    frozen: bool = False
    heldout_reads: int = 0

    def read_calibration(self) -> dict[str, Any]:
        return copy.deepcopy(self.calibration)

    def freeze(self) -> None:
        self.frozen = True

    def read_heldout(self) -> dict[str, Any]:
        if not self.frozen:
            raise HeldoutAccessError("held-out target is inaccessible before candidate freeze")
        self.heldout_reads += 1
        return copy.deepcopy(self.heldout)


def make_problem(index: int = 0, *, method_id: str = "ruletune") -> dict[str, Any]:
    return {
        "problem_id": f"e2-problem-{int(index):04d}",
        "method_id": method_id,
        "representation": "watertight_volume",
        "model_family": "MPM.ElastoPlastic",
        "mutable_parameters": ["youngs_modulus_pa", "density_kg_m3", "poisson_ratio"],
        "prior": {"youngs_modulus_pa": 50000.0, "density_kg_m3": 600.0, "poisson_ratio": 0.30},
        "hidden_target": {"youngs_modulus_pa": 60000.0, "density_kg_m3": 650.0, "poisson_ratio": 0.31},
        "calibration_probe": {"probe_id": "calibration.compression.v1", "stimulus_hash": fingerprint({"kind": "compression", "axis": "x"})},
        "heldout_probe": {"probe_id": "heldout.push.v1", "stimulus_hash": fingerprint({"kind": "push", "axis": "y"})},
        "budget": {"candidate_evaluations": [1, 2, 4, 8]},
    }


def validate_trace_isolation(steps: list[Mapping[str, Any]]) -> None:
    seen_freeze = False
    for step in steps:
        phase = str(step.get("phase"))
        if phase == "candidate_frozen":
            seen_freeze = True
        if phase == "heldout_verification" and not seen_freeze:
            raise ValueError("heldout verification occurred before freeze")
        if phase == "calibration" and bool(step.get("heldout_feedback_used")):
            raise ValueError("heldout feedback leaked into calibration")
        if phase == "heldout_verification" and step.get("feedback_allowed", False):
            raise ValueError("heldout feedback must be disabled")


def run_e2_smoke(output_dir: str, *, count: int = 10, methods: list[str] | None = None) -> dict[str, Any]:
    cfg = {"promotion_stage": "fixture", "count": int(count), "methods": methods or list(METHODS[:4]), "seeds": [0], "candidate_budget": 8}
    plan = plan_run("e2-recovery", output_dir=output_dir, config=cfg, fixture=True, count=count, shard_count=1)
    run_all_shards(output_dir)
    return aggregate_run(output_dir)
