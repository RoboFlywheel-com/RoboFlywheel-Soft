# SPDX-License-Identifier: Apache-2.0
"""Shard-local execution, resume, and deterministic mock experiment backends."""

from __future__ import annotations

import copy
import importlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical import fingerprint, read_json, sha256_file
from .io import atomic_write_json, write_jsonl, read_jsonl
from .records import make_call_record, make_episode_record, record_hash, validate_record, validate_task_spec
from .schema import SchemaError
from .subprocess_runner import run_isolated


class RunnerError(ValueError):
    pass


def load_executor(reference: str):
    """Load a server-side formal episode executor without importing it locally."""
    if ":" not in reference:
        raise RunnerError("episode executor must use module:function syntax")
    module_name, function_name = reference.split(":", 1)
    module = importlib.import_module(module_name)
    executor = getattr(module, function_name, None)
    if not callable(executor):
        raise RunnerError(f"episode executor is not callable: {reference}")
    return executor


def _task_dir(run_dir: Path, task_id: str, attempt: int) -> Path:
    return run_dir / "tasks" / task_id / f"attempt-{attempt:03d}"


def _bind_attempt(record: Mapping[str, Any], attempt: int) -> dict[str, Any]:
    """Bind a canonical record to its immutable task attempt."""
    bound = dict(record)
    bound["attempt_id"] = f"attempt-{int(attempt):03d}"
    bound.pop("record_sha256", None)
    bound["record_sha256"] = record_hash(bound)
    return bound


def _bind_auxiliary_attempt(rows: Sequence[Mapping[str, Any]], attempt: int) -> list[dict[str, Any]]:
    bound_rows: list[dict[str, Any]] = []
    for row in rows:
        bound = dict(row)
        bound["attempt_id"] = f"attempt-{int(attempt):03d}"
        if "record_sha256" in bound:
            bound.pop("record_sha256", None)
            bound["record_sha256"] = record_hash(bound)
        bound_rows.append(bound)
    return bound_rows


def _next_attempt(run_dir: Path, task_id: str) -> int:
    parent = run_dir / "tasks" / task_id
    if not parent.is_dir():
        return 0
    values = []
    for path in parent.glob("attempt-*"):
        try:
            values.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(values, default=-1) + 1


def _terminal_marker(attempt_dir: Path) -> Path:
    return attempt_dir / "TERMINAL.json"


def _valid_terminal(run_manifest: Mapping[str, Any], task: Mapping[str, Any], attempt_dir: Path) -> dict[str, Any] | None:
    marker_path = _terminal_marker(attempt_dir)
    record_path = attempt_dir / "episode.json"
    if not marker_path.is_file() or not record_path.is_file():
        return None
    try:
        marker = read_json(marker_path)
        record = read_json(record_path)
        validate_record(record)
    except (OSError, ValueError, SchemaError):
        return None
    expected_task_id = str(task.get("task_id"))
    expected_task_fingerprint = str(task.get("task_fingerprint") or expected_task_id)
    if marker.get("task_id") != expected_task_id or marker.get("task_fingerprint") != expected_task_fingerprint:
        return None
    if marker.get("attempt_id") != attempt_dir.name:
        return None
    if marker.get("protocol_hash") != run_manifest.get("protocol_hash"):
        return None
    if marker.get("config_hash") != run_manifest.get("config_hash") or marker.get("config_hash") != task.get("config_hash"):
        return None
    if marker.get("code_hash") != run_manifest.get("code_hash"):
        return None
    if marker.get("simulator_fingerprint") != run_manifest.get("simulator_fingerprint"):
        return None
    if record.get("run_id") != run_manifest.get("run_id") or record.get("task_id") != expected_task_id:
        return None
    if record.get("protocol_hash") != run_manifest.get("protocol_hash") or record.get("attempt_id") != attempt_dir.name:
        return None
    if marker.get("record_sha256") != record.get("record_sha256"):
        return None
    if marker.get("record_file_sha256") != sha256_file(record_path):
        return None
    return marker


def _validate_record_binding(record: Mapping[str, Any], run_manifest: Mapping[str, Any], task: Mapping[str, Any], attempt: int) -> None:
    expected_attempt = f"attempt-{int(attempt):03d}"
    checks = {
        "run_id": run_manifest.get("run_id"),
        "task_id": task.get("task_id"),
        "experiment_id": run_manifest.get("experiment_id"),
        "protocol_hash": run_manifest.get("protocol_hash"),
        "attempt_id": expected_attempt,
    }
    for field, expected in checks.items():
        if expected is not None and record.get(field) != expected:
            raise RunnerError(f"episode record {field} does not match frozen task binding")


def _isolated_worker_failure(*args: Any, **kwargs: Any) -> Any:
    raise RunnerError("isolated worker did not return a terminal marker")


def _stable_number(task: Mapping[str, Any], salt: str) -> float:
    value = int(fingerprint({"task": task.get("task_id"), "salt": salt})[7:23], 16)
    return (value % 1000000) / 1000000.0


def _e1_result(task: Mapping[str, Any]) -> tuple[str, str, str, bool, dict[str, Any], dict[str, Any], str | None]:
    source_status = str(task.get("source_status", "verified"))
    u = _stable_number(task, "e1")
    if source_status in {"task_failure", "failure"}:
        return "failed", "failure", "infrastructure", False, {"finite_fraction": 0.0, "runtime_s": 0.0}, {"contract_pass": False, "runtime_pass": False}, "source_task_failure"
    if source_status in {"rejected", "rejected_model_limit", "predicted", "predicted_prior"}:
        return "completed", "rejected", "contract", False, {"finite_fraction": 1.0, "runtime_s": 0.0}, {"contract_pass": False, "runtime_pass": False}, "physical_bundle_not_verified"
    # A small deterministic failure rate exercises preservation of failures.
    if u < 0.06:
        return "completed", "failure", "numerical", False, {"finite_fraction": 0.98, "runtime_s": round(0.01 + u, 6)}, {"contract_pass": True, "runtime_pass": False}, "nan_or_instability"
    return "completed", "success", "none", False, {"finite_fraction": 1.0, "runtime_s": round(0.01 + u, 6), "gpu_seconds": 0.0, "penetration_m": round(u * 0.001, 8)}, {"contract_pass": True, "runtime_pass": True}, None


def _e2_blocked_result(task: Mapping[str, Any], run_manifest: Mapping[str, Any], code: str, detail: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    step = {
        "schema_name": "softsim.agent_step",
        "schema_version": 1,
        "task_id": task["task_id"],
        "step_index": 0,
        "phase": "blocked",
        "feedback_source": "none",
        "heldout_feedback_used": False,
        "target_access_policy": task.get("target_access_policy", "evaluator_sidecar_required"),
        "target_handle": task.get("hidden_target_handle"),
        "reason": code,
    }
    record = make_episode_record(
        run_manifest=run_manifest,
        task=task,
        execution_status="invalid",
        outcome_status="blocked",
        failure_domain="contract",
        failure_code=code,
        retriable=False,
        metrics={"candidate_count": 0},
        measurements={"target_access_policy": task.get("target_access_policy", "evaluator_sidecar_required"), "target_handle": task.get("hidden_target_handle"), "detail": detail},
        extra={"heldout_feedback_used": False, "hidden_target_exposed": False},
    )
    return record, [step]


_HIDDEN_PAYLOAD_KEYS = {
    "hidden_target", "target", "ground_truth", "groundtruth", "oracle_target",
    "oracle_parameters", "true_parameters", "heldout_target", "secret_target",
}


def _contains_hidden_payload(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in {"target_handle", "hidden_target_handle", "target_access_policy", "target_provider", "target_provider_required"}:
                if _contains_hidden_payload(item):
                    return True
                continue
            if normalized in _HIDDEN_PAYLOAD_KEYS or normalized.startswith("target_") or normalized.endswith("_target") or any(token in normalized for token in ("hidden_target", "ground_truth", "groundtruth", "oracle_", "oracle-", "true_parameters", "secret_target")):
                return True
            if _contains_hidden_payload(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_hidden_payload(item) for item in value)
    return False


def _validate_external_e2_trace(task: Mapping[str, Any], record: Mapping[str, Any], steps: Sequence[Mapping[str, Any]], calls: Sequence[Mapping[str, Any]], *, require_trace: bool = True) -> tuple[str, str] | None:
    """Enforce the hidden-target boundary on injected production executors.

    Payload checks always run.  Trace-phase and summary-field completeness is
    required only for completed episodes: an infrastructure failure record
    (Genesis crash/timeout) must keep its infrastructure taxonomy instead of
    being relabelled as a contract violation.
    """
    if _contains_hidden_payload(task) or _contains_hidden_payload(record):
        return "hidden_target_leak_detected", "task or episode record contains evaluator-only target payload"
    if bool(record.get("hidden_target_exposed")):
        return "hidden_target_leak_detected", "episode executor marked hidden target as exposed"
    if not require_trace:
        for index, step in enumerate(steps):
            if _contains_hidden_payload(step):
                return "hidden_target_leak_detected", f"agent step {index} contains evaluator-only target payload"
        for index, call in enumerate(calls):
            if _contains_hidden_payload(call):
                return "hidden_target_leak_detected", f"simulator call {index} contains evaluator-only target payload"
        return None
    phases = {str(step.get("phase", "")) for step in steps}
    abstained = "abstain" in phases
    if abstained:
        # Abstention (uncertain_human_escalation) legitimately ends before
        # candidate freeze and never touches held-out; all payload checks below
        # still apply.
        if not {"calibration"}.issubset(phases):
            return "e2_trace_incomplete", "abstained E2 episode must still record its calibration steps"
    elif not {"calibration", "candidate_frozen", "heldout_verification"}.issubset(phases):
        return "e2_trace_incomplete", "production E2 executor must return calibration, candidate_frozen, and heldout_verification steps"
    measurements = record.get("measurements") if isinstance(record.get("measurements"), Mapping) else {}
    has_prior = "prior_parameters" in record or "prior_parameters" in measurements
    has_final = "final_parameters" in record or "final_parameters" in measurements
    has_count = "candidate_count" in record or "candidate_count" in measurements or "candidate_count" in (record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {})
    if not (has_prior and has_final and has_count):
        return "e2_trace_incomplete", "production E2 record must retain prior_parameters, final_parameters, and candidate_count"
    seen_freeze = False
    seen_heldout = False
    phase_rank = {"calibration": 0, "candidate_frozen": 1, "heldout_verification": 2}
    last_rank = -1
    for index, step in enumerate(steps):
        if _contains_hidden_payload(step):
            return "hidden_target_leak_detected", f"agent step {index} contains evaluator-only target payload"
        phase = str(step.get("phase", ""))
        rank = phase_rank.get(phase)
        if rank is not None and rank < last_rank:
            return "heldout_isolation_violation", "E2 trace phases are out of order"
        if rank is not None:
            last_rank = rank
        if phase == "candidate_frozen":
            seen_freeze = True
        if phase == "heldout_verification" and not seen_freeze:
            return "heldout_isolation_violation", "held-out verification occurred before candidate freeze"
        if phase == "heldout_verification":
            seen_heldout = True
        if phase == "calibration" and seen_heldout:
            return "heldout_isolation_violation", "calibration resumed after held-out verification"
        if phase == "calibration" and bool(step.get("heldout_feedback_used")):
            return "heldout_isolation_violation", "held-out feedback entered calibration trace"
        if phase == "calibration" and "heldout" in str(step.get("probe_id", "")).lower():
            return "heldout_isolation_violation", "held-out probe entered calibration trace"
        if phase == "heldout_verification" and bool(step.get("feedback_allowed")):
            return "heldout_isolation_violation", "held-out feedback was enabled"
        if bool(step.get("heldout_feedback_used")):
            return "heldout_isolation_violation", "held-out feedback was recorded as used"
    for index, call in enumerate(calls):
        if _contains_hidden_payload(call):
            return "hidden_target_leak_detected", f"simulator call {index} contains evaluator-only target payload"
        if bool(call.get("heldout_feedback_used")):
            return "heldout_isolation_violation", "held-out feedback entered simulator call trace"
    return None


def _validate_external_e3a_record(record: Mapping[str, Any]) -> tuple[str, str] | None:
    outcomes = record.get("outcomes")
    measurements = record.get("measurements")
    if not isinstance(outcomes, Mapping) or not isinstance(measurements, Mapping):
        return "e3a_metrics_incomplete", "production E3-A episode requires outcomes and measurements"
    required_outcomes = ("task_success", "damage_failure", "intact_success")
    required_measurements = ("damage_fraction", "integrity", "safe_force_window", "slip", "contact_count", "convergence")
    if any(key not in outcomes for key in required_outcomes) or any(key not in measurements for key in required_measurements):
        return "e3a_metrics_incomplete", "production E3-A episode is missing required failure-loop metrics"
    if not isinstance(outcomes.get("task_success"), bool):
        return "e3a_metrics_incomplete", "E3-A task_success must be boolean"
    if not outcomes.get("task_success") and (measurements.get("damage_fraction") is None or measurements.get("integrity") is None):
        return "e3a_metrics_incomplete", "failed E3-A episode must retain damage and integrity"
    if not isinstance(measurements.get("safe_force_window"), Mapping):
        return "e3a_metrics_incomplete", "E3-A safe_force_window must be an object"
    return None


def terminal_failure_fields(experiment_id: str, code: str, detail: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Public contract: outcome/measurement fields for a terminal failure record.

    Any writer of terminal failure records (runner fallbacks, cloud launcher
    prefetch failures, scheduler crash handlers) must use this so E3-A's
    failure-episode evidence (damage/integrity/contact/convergence) is never
    dropped — the schema gate rejects E3-A records without these fields.
    """
    if str(experiment_id) == "e3a":
        return _e3a_blocked_fields(code, detail)
    return {}, {"error_detail": detail, "synthetic_formula_used": False}


def _e3a_blocked_fields(code: str, detail: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return metric-bearing fields for a blocked/failed E3-A executor call."""
    return (
        {
            "task_success": False,
            "grasp_success": False,
            "lift_success": False,
            "hold_success": False,
            "intact_success": False,
            "damage_failure": False,
        },
        {
            "damage_fraction": 0.0,
            "integrity": 1.0,
            "safe_force_window": {"low_n": None, "high_n": None, "width_n": 0.0, "non_empty": False},
            "slip": None,
            "contact_count": 0,
            "convergence": False,
            "finite_fraction": 0.0,
            "executor_failure_code": code,
            "executor_failure_detail": detail,
        },
    )


def _e2_result(task: Mapping[str, Any], run_manifest: Mapping[str, Any], task_dir: Path, target_provider: Any | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    method = str(task.get("method_id", "fixed_profile"))
    prior = dict(task.get("prior_parameters") or {"youngs_modulus_pa": 50000.0, "density_kg_m3": 600.0, "poisson_ratio": 0.3})
    # A release may itself be synthetic for schema qualification while still
    # exercising the production evaluator-side target boundary.  Only the
    # explicit task flag permits an inline hidden target.
    synthetic_fixture = bool(task.get("synthetic_fixture"))
    if synthetic_fixture:
        target = dict(task.get("hidden_target") or {"youngs_modulus_pa": 60000.0, "density_kg_m3": 650.0, "poisson_ratio": 0.31})
    else:
        if "hidden_target" in task:
            return _e2_blocked_result(task, run_manifest, "hidden_target_leak_detected", "public task manifest contains evaluator-only target")
        if target_provider is None:
            return _e2_blocked_result(task, run_manifest, "hidden_target_provider_missing", "formal E2 requires an evaluator-side target provider")
        try:
            provided = target_provider(dict(task))
            if not isinstance(provided, Mapping):
                raise TypeError("target provider must return a mapping")
            target = {str(key): float(value) for key, value in provided.items()}
        except Exception as exc:
            # Do not serialize provider exception text: it may accidentally
            # contain evaluator-only target values.
            return _e2_blocked_result(task, run_manifest, "hidden_target_provider_error", type(exc).__name__)
        if not target:
            return _e2_blocked_result(task, run_manifest, "hidden_target_empty", "evaluator-side target provider returned no parameters")
    budget = max(1, int(task.get("candidate_budget", 8)))
    steps: list[dict[str, Any]] = []
    candidate = dict(prior)
    candidate_count = 0
    prior_pass_hint = method in {"fixed_profile", "family/default", "one_shot_prior"} and _stable_number(task, "prior-pass") > 0.35
    max_steps = 1 if prior_pass_hint else min(budget, 3)
    for index in range(max_steps):
        candidate_count += 1
        if index > 0:
            # RuleTune/Sobol/style adapters make bounded, deterministic updates.
            for name, value in target.items():
                if method in {"ruletune", "RuleTune", "full_numeric_agent", "vid2sid_style_reimplementation", "swim2real_style_reimplementation"}:
                    candidate[name] = round(float(candidate.get(name, value)) + 0.6 * (float(value) - float(candidate.get(name, value))), 8)
                elif method in {"sobol", "seeded_sobol"}:
                    candidate[name] = round(float(value) * (0.9 + 0.02 * index), 8)
                else:
                    candidate[name] = float(candidate.get(name, value))
        loss = sum(abs(float(candidate.get(name, value)) - float(value)) / max(abs(float(value)), 1.0) for name, value in target.items()) / max(1, len(target))
        passed = loss <= (0.03 if index == max_steps - 1 else 0.08)
        steps.append({
            "schema_name": "softsim.agent_step",
            "schema_version": 1,
            "task_id": task["task_id"],
            "step_index": index,
            "phase": "calibration",
            "candidate_hash": fingerprint(candidate),
            "candidate_parameters": dict(candidate),
            "probe_id": str(task.get("calibration_probe_id", "calibration.synthetic.compression.v1")),
            "feedback_source": "calibration_only",
            "heldout_feedback_used": False,
            "simulator_evaluation": True,
            "loss": round(loss, 8),
            "passed": passed,
        })
        if passed:
            break
    final_loss = float(steps[-1]["loss"])
    heldout_loss = min(1.0, final_loss + (0.01 if _stable_number(task, "heldout") > 0.2 else 0.08))
    heldout_pass = heldout_loss <= 0.08
    first_candidate_passed = bool(steps and steps[0].get("passed"))
    accepted_first_pass = bool(first_candidate_passed and candidate_count == 1 and heldout_pass)
    correction_success = bool(candidate_count > 1 and heldout_pass)
    # Explicitly record the freeze boundary before held-out evaluation.  The
    # held-out event never appears as a calibration feedback source.
    steps.append({
        "schema_name": "softsim.agent_step",
        "schema_version": 1,
        "task_id": task["task_id"],
        "step_index": len(steps),
        "phase": "candidate_frozen",
        "candidate_hash": fingerprint(candidate),
        "candidate_parameters": dict(candidate),
        "feedback_source": "none",
        "heldout_feedback_used": False,
        "feedback_allowed": False,
    })
    steps.append({
        "schema_name": "softsim.agent_step",
        "schema_version": 1,
        "task_id": task["task_id"],
        "step_index": len(steps),
        "phase": "heldout_verification",
        "candidate_hash": fingerprint(candidate),
        "candidate_parameters": dict(candidate),
        "probe_id": str(task.get("heldout_probe_id", "heldout.synthetic.push.v1")),
        "feedback_source": "heldout_no_feedback",
        "heldout_feedback_used": False,
        "simulator_evaluation": True,
        "loss": round(heldout_loss, 8),
        "passed": heldout_pass,
        "feedback_allowed": False,
    })
    outcome = "success" if heldout_pass else "failure"
    failure_domain = "none" if heldout_pass else "task"
    failure_code = None if heldout_pass else "heldout_behavior_mismatch"
    measurements = {
        "heldout_behavior_error": round(heldout_loss, 8),
        "calibration_behavior_error": round(final_loss, 8),
        "candidate_count": candidate_count,
        "prior_parameters": prior,
        "final_parameters": candidate,
        "accepted_first_pass": accepted_first_pass,
        "correction_success": correction_success,
        "target_handle": task.get("hidden_target_handle"),
        "target_access_policy": task.get("target_access_policy", "inline_synthetic_fixture"),
    }
    record = make_episode_record(
        run_manifest=run_manifest,
        task=task,
        execution_status="completed",
        outcome_status=outcome,
        failure_domain=failure_domain,
        failure_code=failure_code,
        retriable=False,
        metrics={"heldout_behavior_error": round(heldout_loss, 8), "calibration_behavior_error": round(final_loss, 8), "candidate_count": candidate_count},
        measurements=measurements,
        extra={
            "calibration_probe_id": task.get("calibration_probe_id"),
            "heldout_probe_id": task.get("heldout_probe_id"),
            "heldout_feedback_used": False,
            "prior_parameters": prior,
            "final_parameters": candidate,
            "candidate_count": candidate_count,
            "target_handle": task.get("hidden_target_handle"),
            "target_access_policy": task.get("target_access_policy", "inline_synthetic_fixture"),
            "agent_status": "accepted_first_pass" if accepted_first_pass else ("corrected_then_accepted" if correction_success else "rejected_model_limit"),
        },
    )
    return record, steps


def _e3a_result(task: Mapping[str, Any], run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    force = float(task.get("force_n", 15.0))
    regime = str(task.get("fragility_regime", "medium"))
    u = _stable_number(task, "e3a")
    damage = max(0.0, min(1.0, (force - 8.0) / 28.0 + (u - 0.5) * 0.04))
    if regime == "low":
        damage *= 0.55
    elif regime == "high":
        damage = min(1.0, damage * 1.35)
    slip = max(0.0, (10.0 - force) / 20.0 + (u - 0.5) * 0.02)
    task_success = bool(force >= 10.0 and slip < 0.18 and damage < 0.78)
    integrity = max(0.0, 1.0 - damage)
    intact = bool(task_success and integrity >= 0.75)
    failure_code = None
    failure_domain = "none"
    outcome = "success" if task_success else "failure"
    if not task_success:
        failure_domain = "damage" if damage >= 0.78 else "task"
        failure_code = "damage_failure" if damage >= 0.78 else "slip_before_hold"
    measurements = {
        "damage_fraction": round(damage, 8),
        "integrity": round(integrity, 8),
        "safe_force_window": {"low_n": 10.0, "high_n": 22.0, "width_n": 12.0, "non_empty": True},
        "slip": round(slip, 8),
        "contact_count": int(4 + (u * 8)),
        "convergence": True,
        "finite_fraction": 1.0,
        "peak_contact_force_n": force,
        "retained_lifted_mass_fraction": round(max(0.0, 1.0 - damage * 0.8), 8),
        "residual_set": round(damage * 0.02, 8),
        "drop_or_slip": bool(slip >= 0.18),
        "contact_force_proxy_n": force,
    }
    outcomes = {
        "task_success": task_success,
        "grasp_success": force >= 8.0,
        "lift_success": task_success,
        "hold_success": task_success,
        "intact_success": intact,
        "damage_failure": bool(damage >= 0.78),
    }
    return make_episode_record(
        run_manifest=run_manifest,
        task=task,
        execution_status="completed",
        outcome_status=outcome,
        failure_domain=failure_domain,
        failure_code=failure_code,
        retriable=False,
        metrics={"task_success": task_success, "damage_fraction": round(damage, 8), "integrity": round(integrity, 8), "safe_force_window_width_n": 12.0, "slip": round(slip, 8), "convergence": True},
        outcomes=outcomes,
        measurements=measurements,
        extra={"damage_model_id": task.get("damage_model_id", "TofuDamage.continuum_damage_stiffness_loss_surrogate"), "commanded_force_n": force, "measured_contact_force_n": None, "contact_force_status": "not_available_in_local_mock"},
    )


def _blocked_real_executor(task: Mapping[str, Any], run_manifest: Mapping[str, Any], code: str, detail: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    outcomes: dict[str, Any] = {}
    measurements: dict[str, Any] = {"detail": detail, "synthetic_formula_used": False}
    if str(task.get("experiment_id") or run_manifest.get("experiment_id")) == "e3a":
        outcomes, measurements = _e3a_blocked_fields(code, detail)
    record = make_episode_record(
        run_manifest=run_manifest,
        task=task,
        execution_status="invalid",
        outcome_status="blocked",
        failure_domain="infrastructure",
        failure_code=code,
        retriable=False,
        metrics={"real_executor_used": False},
        outcomes=outcomes,
        measurements=measurements,
        extra={"real_executor_required": True, "synthetic_formula_used": False},
    )
    return record, [], []


def _execute_task(task: Mapping[str, Any], run_manifest: Mapping[str, Any], attempt_dir: Path, target_provider: Any | None = None, episode_executor: Any | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    experiment_id = str(task.get("experiment_id") or run_manifest.get("experiment_id"))
    if bool(run_manifest.get("real_executor_required", not bool(run_manifest.get("synthetic", False)))):
        if experiment_id.startswith("e2") and target_provider is None:
            record, steps = _e2_blocked_result(task, run_manifest, "hidden_target_provider_missing", "formal E2 requires an evaluator-side target provider")
            return record, steps, []
        if episode_executor is None:
            return _blocked_real_executor(task, run_manifest, "real_executor_missing", "formal production run requires an injected episode executor")
        try:
            result = episode_executor(dict(task), dict(run_manifest), attempt_dir, target_provider=target_provider)
        except TypeError:
            result = episode_executor(dict(task), dict(run_manifest), attempt_dir)
        if isinstance(result, Mapping):
            record = dict(result)
            if experiment_id.startswith("e2"):
                violation = _validate_external_e2_trace(task, record, (), (), require_trace=record.get("execution_status") == "completed")
                if violation is not None:
                    blocked, steps = _e2_blocked_result(task, run_manifest, *violation)
                    return blocked, steps, []
            if experiment_id == "e3a":
                violation = _validate_external_e3a_record(record)
                if violation is not None:
                    blocked, steps, calls = _blocked_real_executor(task, run_manifest, *violation)
                    return blocked, steps, calls
            return record, [], []
        if isinstance(result, (tuple, list)) and result and isinstance(result[0], Mapping):
            record = dict(result[0])
            steps = list(result[1]) if len(result) > 1 else []
            calls = list(result[2]) if len(result) > 2 else []
            if experiment_id.startswith("e2"):
                violation = _validate_external_e2_trace(task, record, steps, calls, require_trace=record.get("execution_status") == "completed")
                if violation is not None:
                    blocked, blocked_steps = _e2_blocked_result(task, run_manifest, *violation)
                    return blocked, blocked_steps, []
            if experiment_id == "e3a":
                violation = _validate_external_e3a_record(record)
                if violation is not None:
                    blocked, blocked_steps, blocked_calls = _blocked_real_executor(task, run_manifest, *violation)
                    return blocked, blocked_steps, blocked_calls
            return record, steps, calls
        raise RunnerError("episode executor must return a record or (record, steps, calls)")
    if experiment_id.startswith("e1"):
        execution, outcome, domain, retriable, metrics, outcomes, failure_code = _e1_result(task)
        record = make_episode_record(run_manifest=run_manifest, task=task, execution_status=execution, outcome_status=outcome, failure_domain=domain, failure_code=failure_code, retriable=retriable, metrics=metrics, outcomes=outcomes, measurements={"finite_fraction": metrics.get("finite_fraction", 0.0), "runtime_s": metrics.get("runtime_s", 0.0)}, extra={"audit_phase": experiment_id})
        return record, [], []
    if experiment_id.startswith("e2"):
        record, steps = _e2_result(task, run_manifest, attempt_dir, target_provider=target_provider)
        calls = []
        for index, step in enumerate(steps):
            calls.append(make_call_record(run_manifest=run_manifest, task={**task, "call_id": f"{task['task_id']}-call-{index:02d}", "episode_id": task["task_id"]}, execution_status="completed", outcome_status="success" if step.get("passed") else "failure", failure_domain="none" if step.get("passed") else "task", failure_code=None if step.get("passed") else "probe_mismatch", metrics={"loss": step.get("loss")}, measurements={"phase": step.get("phase"), "feedback_allowed": step.get("feedback_allowed", step.get("phase") == "calibration")}, extra={"call_index": index, "probe_id": step.get("probe_id"), "heldout_feedback_used": False}))
        return record, steps, calls
    if experiment_id == "e3a":
        return _e3a_result(task, run_manifest), [], []
    raise RunnerError(f"unsupported experiment: {experiment_id}")


def _run_one(run_dir: Path, run_manifest: Mapping[str, Any], task: Mapping[str, Any], *, force: bool = False, target_provider: Any | None = None, episode_executor: Any | None = None) -> dict[str, Any]:
    task_id_value = str(task["task_id"])
    experiment_id = str(task.get("experiment_id") or run_manifest.get("experiment_id"))
    parent = run_dir / "tasks" / task_id_value
    parent.mkdir(parents=True, exist_ok=True)
    if not force:
        for existing in sorted(parent.glob("attempt-*/")):
            marker = _valid_terminal(run_manifest, task, existing)
            if marker is not None:
                return marker
    attempt = _next_attempt(run_dir, task_id_value)
    attempt_dir = _task_dir(run_dir, task_id_value, attempt)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(attempt_dir / "request.json", {"run_id": run_manifest["run_id"], "task": dict(task), "attempt": attempt})
    started = time.time()
    try:
        record, steps, calls = _execute_task(task, run_manifest, attempt_dir, target_provider=target_provider, episode_executor=episode_executor)
        record = _bind_attempt(record, attempt)
        _validate_record_binding(record, run_manifest, task, attempt)
        steps = _bind_auxiliary_attempt(steps, attempt)
        calls = _bind_auxiliary_attempt(calls, attempt)
        validate_record(record)
        atomic_write_json(attempt_dir / "episode.json", record)
        if steps:
            write_jsonl(attempt_dir / "agent_steps.jsonl", steps)
        if calls:
            write_jsonl(attempt_dir / "calls.jsonl", calls)
        marker = {
            "schema_name": "softsim.terminal_marker",
            "schema_version": 1,
            "run_id": run_manifest["run_id"],
            "task_id": task_id_value,
            "task_fingerprint": task.get("task_fingerprint", task_id_value),
            "protocol_hash": run_manifest["protocol_hash"],
            "config_hash": run_manifest.get("config_hash"),
            "code_hash": run_manifest.get("code_hash"),
            "simulator_fingerprint": run_manifest.get("simulator_fingerprint"),
            "attempt_id": f"attempt-{attempt:03d}",
            "execution_status": record["execution_status"],
            "outcome_status": record["outcome_status"],
            "record_sha256": record["record_sha256"],
            "record_file_sha256": sha256_file(attempt_dir / "episode.json"),
            "started_at_epoch": started,
            "finished_at_epoch": time.time(),
            "resumable": True,
        }
        atomic_write_json(attempt_dir / "TERMINAL.json", marker)
        return marker
    except Exception as exc:
        # Infrastructure failures are terminal records too, so aggregate can
        # account for every planned task and resume does not hide them.
        fallback_outcomes: dict[str, Any] = {}
        fallback_measurements: dict[str, Any] = {"error_type": type(exc).__name__}
        if experiment_id == "e3a":
            fallback_outcomes, fallback_measurements = _e3a_blocked_fields("worker_exception", type(exc).__name__)
        fallback = make_episode_record(run_manifest=run_manifest, task=task, execution_status="failed", outcome_status="failure", failure_domain="infrastructure", failure_code="worker_exception", retriable=False, metrics={"finite_fraction": 0.0}, outcomes=fallback_outcomes, measurements=fallback_measurements)
        fallback = _bind_attempt(fallback, attempt)
        atomic_write_json(attempt_dir / "episode.json", fallback)
        marker = {
            "schema_name": "softsim.terminal_marker",
            "schema_version": 1,
            "run_id": run_manifest["run_id"],
            "task_id": task_id_value,
            "task_fingerprint": task.get("task_fingerprint", task_id_value),
            "protocol_hash": run_manifest["protocol_hash"],
            "config_hash": run_manifest.get("config_hash"),
            "code_hash": run_manifest.get("code_hash"),
            "simulator_fingerprint": run_manifest.get("simulator_fingerprint"),
            "attempt_id": f"attempt-{attempt:03d}",
            "execution_status": "failed",
            "outcome_status": "failure",
            "record_sha256": fallback["record_sha256"],
            "record_file_sha256": sha256_file(attempt_dir / "episode.json"),
            "error_type": type(exc).__name__,
            "resumable": True,
        }
        atomic_write_json(attempt_dir / "TERMINAL.json", marker)
        return marker


def _run_one_isolated(
    run_dir: Path,
    run_manifest: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    force: bool = False,
    episode_executor_reference: str | None = None,
    target_provider_reference: str | None = None,
    python_executable: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Execute one task in a fresh worker process and return its marker."""
    if not force:
        parent = run_dir / "tasks" / str(task["task_id"])
        for existing in sorted(parent.glob("attempt-*/")) if parent.is_dir() else ():
            marker = _valid_terminal(run_manifest, task, existing)
            if marker is not None:
                return marker
    payload = run_isolated(
        run_dir,
        str(task["task_id"]),
        python=python_executable,
        timeout_s=timeout_s,
        episode_executor=episode_executor_reference,
        target_provider=target_provider_reference,
        force=force,
    )
    marker = payload.get("result")
    if isinstance(marker, Mapping) and marker.get("attempt_id"):
        attempt_dir = run_dir / "tasks" / str(task["task_id"]) / str(marker["attempt_id"])
        if _valid_terminal(run_manifest, task, attempt_dir) is not None:
            return dict(marker)
    # A process crash/timeout must still become a terminal record so the
    # denominator remains closed and a later explicit force/resume can inspect it.
    failure_manifest = dict(run_manifest)
    failure_manifest["real_executor_required"] = True
    return _run_one(run_dir, failure_manifest, task, force=force, episode_executor=_isolated_worker_failure)


def _refresh_run_status(run_root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Refresh coordinator status after either a local or scheduler shard."""
    shard_count = max(1, int(manifest.get("shard_count", 1)))
    summaries: list[dict[str, Any]] = []
    missing = 0
    for index in range(shard_count):
        path = run_root / "shard_records" / f"shard-{index:04d}.json"
        if not path.is_file():
            missing += 1
            continue
        try:
            value = read_json(path)
        except (OSError, ValueError):
            missing += 1
            continue
        if isinstance(value, Mapping):
            summaries.append(dict(value))
    expected = int(manifest.get("expected_task_count", 0))
    completed = sum(int(item.get("completed_count", 0)) for item in summaries)
    planned = sum(int(item.get("planned_count", 0)) for item in summaries)
    status = "completed" if missing == 0 and planned == expected and completed == expected else ("running" if completed or summaries else "planned")
    result = {
        "status": status,
        "run_id": manifest.get("run_id"),
        "expected_task_count": expected,
        "completed_task_count": completed,
        "shard_count": shard_count,
        "shard_summaries": sorted(summaries, key=lambda item: int(item.get("shard_index", 0))),
    }
    atomic_write_json(run_root / "run_status.json", result)
    return result


def run_shard(run_dir: str | Path, *, shard_index: int = 0, workers: int = 1, force: bool = False, target_provider: Any | None = None, episode_executor: Any | None = None, process_isolation: bool = False, episode_executor_reference: str | None = None, target_provider_reference: str | None = None, python_executable: str | None = None, timeout_s: float | None = None) -> dict[str, Any]:
    run_root = Path(run_dir).resolve()
    manifest = read_json(run_root / "run_manifest.json")
    if manifest.get("status") == "blocked":
        return {"status": "blocked", "blocked_code": manifest.get("blocked_code"), "run_id": manifest.get("run_id"), "completed": 0}
    task_path = run_root / "task_manifest.jsonl"
    if not task_path.is_file():
        raise RunnerError("task_manifest.jsonl is missing")
    if manifest.get("task_manifest_sha256") and sha256_file(task_path) != manifest.get("task_manifest_sha256"):
        raise RunnerError("task manifest hash does not match frozen run manifest")
    all_tasks = list(read_jsonl(task_path))
    for task in all_tasks:
        try:
            validate_task_spec(task)
        except (SchemaError, ValueError) as exc:
            raise RunnerError(f"invalid frozen task specification: {exc}") from exc
    planned_shards = max(1, int(manifest.get("shard_count", 1)))
    shard_tasks = [
        task for task in all_tasks
        if int(task.get("shard_index", int(fingerprint(task.get("task_id", ""))[7:15], 16) % planned_shards)) == int(shard_index)
    ]
    if not shard_tasks and int(manifest.get("expected_task_count", 0)) > 0 and int(manifest.get("shard_count", 1)) <= int(shard_index):
        raise RunnerError("shard index outside planned range")
    records: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []
    if process_isolation:
        def worker_call(task: Mapping[str, Any]) -> dict[str, Any]:
            return _run_one_isolated(
                run_root,
                manifest,
                task,
                force=force,
                episode_executor_reference=episode_executor_reference,
                target_provider_reference=target_provider_reference,
                python_executable=python_executable,
                timeout_s=timeout_s,
            )
    else:
        def worker_call(task: Mapping[str, Any]) -> dict[str, Any]:
            return _run_one(run_root, manifest, task, force=force, target_provider=target_provider, episode_executor=episode_executor)
    if workers <= 1:
        markers = [worker_call(task) for task in shard_tasks]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            futures = [pool.submit(worker_call, task) for task in shard_tasks]
            markers = [future.result() for future in futures]
    markers.sort(key=lambda row: str(row.get("task_id")))
    for marker in markers:
        task_id_value = str(marker["task_id"])
        attempt_dir = run_root / "tasks" / task_id_value / str(marker["attempt_id"])
        record_path = attempt_dir / "episode.json"
        if record_path.is_file():
            records.append(read_json(record_path))
    records.sort(key=lambda row: (str(row.get("task_id")), str(row.get("record_sha256"))))
    shard_path = run_root / "shard_records" / f"shard-{int(shard_index):04d}.jsonl"
    write_jsonl(shard_path, records)
    summary = {
        "schema_name": "softsim.shard_summary",
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "shard_index": int(shard_index),
        "planned_count": len(shard_tasks),
        "completed_count": len(records),
        "status_counts": {status: sum(row.get("outcome_status") == status for row in records) for status in sorted({str(row.get("outcome_status")) for row in records})},
        "record_file_sha256": sha256_file(shard_path),
    }
    atomic_write_json(run_root / "shard_records" / f"shard-{int(shard_index):04d}.json", summary)
    _refresh_run_status(run_root, manifest)
    return summary


def run_all_shards(run_dir: str | Path, *, workers: int = 1, force: bool = False, target_provider: Any | None = None, episode_executor: Any | None = None, process_isolation: bool = False, episode_executor_reference: str | None = None, target_provider_reference: str | None = None, python_executable: str | None = None, timeout_s: float | None = None) -> dict[str, Any]:
    run_root = Path(run_dir).resolve()
    manifest = read_json(run_root / "run_manifest.json")
    if manifest.get("status") == "blocked":
        return {"status": "blocked", "blocked_code": manifest.get("blocked_code"), "run_id": manifest.get("run_id")}
    shard_count = max(1, int(manifest.get("shard_count", 1)))
    summaries = [run_shard(run_root, shard_index=index, workers=workers, force=force, target_provider=target_provider, episode_executor=episode_executor, process_isolation=process_isolation, episode_executor_reference=episode_executor_reference, target_provider_reference=target_provider_reference, python_executable=python_executable, timeout_s=timeout_s) for index in range(shard_count)]
    status_value = _refresh_run_status(run_root, manifest)
    return {"status": status_value["status"], "run_id": manifest["run_id"], "summaries": summaries}


def status_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    status_path = root / "run_status.json"
    if status_path.is_file():
        return read_json(status_path)
    return {"status": "unknown", "run_dir": root.as_posix()}
