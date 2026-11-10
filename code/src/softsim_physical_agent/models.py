# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ACCEPTED_FIRST_PASS = "accepted_first_pass"
CORRECTED_THEN_ACCEPTED = "corrected_then_accepted"
REJECTED_MODEL_LIMIT = "rejected_model_limit"
REJECTED_NUMERICAL_FAILURE = "rejected_numerical_failure"
UNCERTAIN_HUMAN_ESCALATION = "uncertain_human_escalation"
SIM_EXECUTED_UNVERIFIED = "sim_executed_unverified"

TERMINAL_STATUSES = {
    ACCEPTED_FIRST_PASS,
    CORRECTED_THEN_ACCEPTED,
    REJECTED_MODEL_LIMIT,
    REJECTED_NUMERICAL_FAILURE,
    UNCERTAIN_HUMAN_ESCALATION,
}


@dataclass(frozen=True)
class ProbeSpec:
    probe_id: str
    stimulus: dict[str, Any]
    action: dict[str, Any]
    config: dict[str, Any]
    tolerance: float = 0.05
    required_metrics: tuple[str, ...] = ("loss", "contact_fraction", "settled")

    @classmethod
    def from_dict(cls, value: dict[str, Any], default_id: str) -> "ProbeSpec":
        if not isinstance(value, dict):
            raise ValueError(f"probe {default_id} must be an object")
        probe_id = str(value.get("probe_id") or value.get("id") or default_id)
        stimulus = value.get("stimulus")
        action = value.get("action")
        config = value.get("config")
        if not isinstance(stimulus, dict) or not stimulus:
            raise ValueError(f"probe {probe_id} needs a non-empty stimulus")
        if not isinstance(action, dict) or not action:
            raise ValueError(f"probe {probe_id} needs a non-empty action")
        if not isinstance(config, dict) or not config:
            raise ValueError(f"probe {probe_id} needs a non-empty config")
        try:
            tolerance = float(value.get("tolerance", 0.05))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"probe {probe_id} has invalid tolerance") from exc
        metrics = value.get("required_metrics", ("loss", "contact_fraction", "settled"))
        if not isinstance(metrics, (list, tuple)) or not metrics:
            raise ValueError(f"probe {probe_id} needs required_metrics")
        return cls(probe_id, dict(stimulus), dict(action), dict(config), tolerance, tuple(str(x) for x in metrics))

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "stimulus": self.stimulus,
            "action": self.action,
            "config": self.config,
            "tolerance": self.tolerance,
            "required_metrics": list(self.required_metrics),
        }


@dataclass
class ProbeResult:
    passed: bool
    metrics: dict[str, Any]
    diagnosis: dict[str, Any]
    report_path: str | None = None
    report_hash: str | None = None
    execution_status: str = "completed"
    explicit_behavior_evidence: bool = True
    outputs: dict[str, Any] = field(default_factory=dict)
    raw_report: dict[str, Any] = field(default_factory=dict)


@dataclass
class Proposal:
    status: str
    patch: dict[str, float] = field(default_factory=dict)
    rule_id: str | None = None
    rationale: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.status == "proposed"


@dataclass
class TaskContext:
    sample_id: str
    asset_revision_id: str
    dataset_release_id: str
    route: dict[str, Any]
    artifact_refs: dict[str, Any]
    prior: dict[str, Any]
    fixed_solver_numerics: dict[str, Any]
    task_fingerprint: str
    config_hash: str


def validate_status(status: str) -> str:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"unknown terminal status: {status}")
    return status

