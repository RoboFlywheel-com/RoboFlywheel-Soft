# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any, Mapping

from .io import atomic_write_json, atomic_write_jsonl, content_hash, file_sha256, package_source_hash, read_json, iter_jsonl, portable_value
from .input_adapter import merge_rows
from .models import (
    ACCEPTED_FIRST_PASS,
    CORRECTED_THEN_ACCEPTED,
    REJECTED_MODEL_LIMIT,
    REJECTED_NUMERICAL_FAILURE,
    SIM_EXECUTED_UNVERIFIED,
    UNCERTAIN_HUMAN_ESCALATION,
    ProbeResult,
    ProbeSpec,
    TaskContext,
)
from .probes import build_probe_pair, probe_record
from .proposers import build_proposer
from .routing import ALLOWED_PHYSICAL_PARAMETERS, DEFAULT_PARAMETER_BOUNDS, canonical_name, canonical_prior, build_route, fixed_numerics, physical_values, validate_patch
from .schema import validate_physics_bundle, validate_task_contract
from .input_adapter import load_asset_release, load_legacy_assets, load_previous_staging, load_stage3_manifest

# Compatibility aliases for callers of the earlier prototype.
load_release_assets = load_asset_release


POLICY_VERSION = "softsim-physical-agent.rebuild.v0.1"


def _safe_task_name(sample_id: str) -> str:
    return str(sample_id).replace("/", "_").replace("\\", "_").replace("..", "_")


def _normalise_parameter_bounds(prior: Mapping[str, Any], candidate: Mapping[str, float]) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for item in prior.get("parameters") or []:
        if not isinstance(item, Mapping) or "name" not in item:
            continue
        name = canonical_name(str(item["name"]))
        if name not in candidate:
            continue
        raw = item.get("valid_range")
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            try:
                lower, upper = float(raw[0]), float(raw[1])
                if lower < upper:
                    result[name] = (lower, upper)
            except (TypeError, ValueError):
                pass
    for name, value in candidate.items():
        result.setdefault(name, (max(1.0e-12, abs(value) * 0.05), max(abs(value) * 20.0, abs(value) + 1.0)))
    return result


def _clamp_candidate(candidate: Mapping[str, float], bounds: Mapping[str, tuple[float, float]]) -> dict[str, float]:
    return {name: min(float(bounds[name][1]), max(float(bounds[name][0]), float(value))) for name, value in candidate.items() if name in bounds}


def _prior_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    prior = row.get("prior")
    if not isinstance(prior, Mapping):
        raise ValueError(f"{row.get('sample_id')}: input row has no prior")
    route = build_route(row, prior)
    return canonical_prior(row, prior, route)


def build_tasks(rows: list[dict[str, Any]], config: Mapping[str, Any] | str | Path | None = None, prior_root: Path | str | None = None, prior_index: Path | str | None = None) -> list[dict[str, Any]]:
    """Convert any supported input source into immutable task contexts."""

    # Compatibility with the previous prototype's
    # build_tasks(rows, prior_root, prior_index) signature.
    if config is not None and not isinstance(config, Mapping):
        legacy_root = Path(config)
        legacy_index = Path(prior_root) if prior_root is not None else (Path(prior_index) if prior_index is not None else None)
        config = {}
        prior_root = legacy_root
        prior_index = legacy_index
    config = dict(config or {})
    prior_paths: dict[str, Path] = {}
    if prior_index:
        index_path = Path(prior_index)
        if index_path.is_file():
            for item in iter_jsonl(index_path):
                info = item.get("bundle") if isinstance(item.get("bundle"), Mapping) else {}
                raw = info.get("path") or item.get("physics_path") or item.get("path")
                if raw:
                    candidate_path = index_path.parent / str(raw)
                    if candidate_path.is_file():
                        prior_paths[str(item.get("sample_id"))] = candidate_path
    if prior_root:
        root = Path(prior_root)
        for candidate_path in [*root.glob("bundles/*/physics.json"), *root.glob("*/physics.json")]:
            prior_paths.setdefault(candidate_path.parent.name, candidate_path)
    config_hash = content_hash(config)
    tasks: list[dict[str, Any]] = []
    for raw in merge_rows(rows):
        row = copy.deepcopy(raw)
        if str(row.get("sample_id")) in prior_paths:
            row["prior"] = read_json(prior_paths[str(row.get("sample_id"))])
        prior = _prior_from_row(row)
        route = dict(prior.get("route") or build_route(row, prior))
        fixed = fixed_numerics(prior, route)
        sample_id = str(row.get("sample_id") or prior.get("sample_id"))
        revision = str(row.get("asset_revision_id") or prior.get("asset_revision_id") or content_hash(row))
        task_core = {
            "sample_id": sample_id,
            "asset_revision_id": revision,
            "dataset_release_id": str(row.get("dataset_release_id") or "unknown"),
            "route": route,
            "artifact_refs": row.get("artifact_refs") or {},
            "prior": prior,
            "fixed_solver_numerics": fixed,
            "config_hash": config_hash,
        }
        task_fingerprint = content_hash(task_core)
        task = {
            **task_core,
            "task_fingerprint": task_fingerprint,
            "object_type": row.get("object_type"),
            "geometry_family": row.get("geometry_family") or route.get("geometry_family"),
            "material_family": row.get("material_family") or route.get("material_family"),
            "canonical_id": row.get("canonical_id"),
            "metadata": portable_value(row.get("metadata") or {}),
            "source_kind": row.get("source_kind"),
            "source_root": row.get("source_root"),
            "resolved_artifacts": portable_value(row.get("resolved_artifacts") or {}),
            "mock_target": row.get("mock_target"),
            "mock_behavior": row.get("mock_behavior"),
        }
        tasks.append(task)
    return tasks


def task_context(task: Mapping[str, Any]) -> TaskContext:
    return TaskContext(
        sample_id=str(task["sample_id"]),
        asset_revision_id=str(task["asset_revision_id"]),
        dataset_release_id=str(task.get("dataset_release_id") or "unknown"),
        route=dict(task.get("route") or {}),
        artifact_refs=dict(task.get("artifact_refs") or {}),
        prior=dict(task.get("prior") or {}),
        fixed_solver_numerics=dict(task.get("fixed_solver_numerics") or {}),
        task_fingerprint=str(task.get("task_fingerprint") or content_hash(dict(task))),
        config_hash=str(task.get("config_hash") or content_hash({})),
    )


def _call_simulator(simulator: Any, task: Mapping[str, Any], candidate: Mapping[str, float], probe: ProbeSpec, output_dir: Path, simulator_config: Mapping[str, Any]) -> ProbeResult:
    try:
        result = simulator.run(task, candidate, probe, output_dir, simulator_config=simulator_config)
    except TypeError as exc:
        # Keep compatibility with tiny test doubles that implement the four-argument API.
        if "simulator_config" not in str(exc):
            raise
        result = simulator.run(task, candidate, probe, output_dir)
    if not isinstance(result, ProbeResult):
        raise TypeError("simulator.run must return ProbeResult")
    return result


def _write_skipped_heldout(path: Path, probe: ProbeSpec, reason: str, *, fixed_hash: str, candidate_hash: str) -> ProbeResult:
    payload = probe_record(probe)
    report = {
        "schema_name": "softsim.probe_report",
        "schema_version": 2,
        "backend": "agent",
        "probe": payload,
        "probe_id": probe.probe_id,
        "probe_hash": payload["probe_hash"],
        "stimulus_hash": payload["stimulus_hash"],
        "action_hash": payload["action_hash"],
        "config_hash": payload["config_hash"],
        "candidate_hash": candidate_hash,
        "simulator_config_hash": fixed_hash,
        "execution_status": "not_executed",
        "behavior_pass": False,
        "passed": False,
        "explicit_behavior_evidence": False,
        "metrics": {"heldout_executed": False},
        "diagnosis": {"kind": reason, "suggested_updates": {}},
        "outputs": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, report)
    return ProbeResult(False, report["metrics"], report["diagnosis"], str(path), file_sha256(path), "not_executed", False, {}, report)


def _archive_stale(task_dir: Path, old_hash: str) -> None:
    """Preserve a previous config's evidence before recomputing a task."""

    if not old_hash or not task_dir.exists():
        return
    target = task_dir / "history" / old_hash.replace("sha256:", "")[:24]
    target.mkdir(parents=True, exist_ok=True)
    for relative in ("SUCCESS.json", "prior.json", "physics.json", "agent_trace.jsonl", "task_contract.json", "heldout/report.json"):
        source = task_dir / relative
        if source.is_file():
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    for source in sorted([*task_dir.glob("candidate_*"), task_dir / "heldout"]):
        if not source.is_dir() or source == target or "history" in source.parts:
            continue
        destination = target / source.relative_to(task_dir)
        shutil.copytree(source, destination, dirs_exist_ok=True)


def _valid_resume(task_dir: Path, task: Mapping[str, Any]) -> dict[str, Any] | None:
    marker_path = task_dir / "SUCCESS.json"
    if not marker_path.is_file():
        return None
    try:
        marker = read_json(marker_path)
    except (OSError, ValueError):
        return None
    if marker.get("task_fingerprint") != task.get("task_fingerprint") or marker.get("config_hash") != task.get("config_hash"):
        return None
    required = (task_dir / "prior.json", task_dir / "physics.json", task_dir / "agent_trace.jsonl", task_dir / "heldout" / "report.json")
    if not all(path.is_file() for path in required):
        return None
    expected_physics_hash = marker.get("physics_sha256")
    if expected_physics_hash and file_sha256(task_dir / "physics.json") != expected_physics_hash:
        return None
    expected_trace_hash = marker.get("trace_sha256")
    if expected_trace_hash and file_sha256(task_dir / "agent_trace.jsonl") != expected_trace_hash:
        return None
    expected_heldout_hash = marker.get("heldout_sha256")
    if expected_heldout_hash and file_sha256(task_dir / "heldout" / "report.json") != expected_heldout_hash:
        return None
    return marker


def _candidate_config(task: Mapping[str, Any], candidate: Mapping[str, float], probe: ProbeSpec, iteration: int, *, fixed_hash: str) -> dict[str, Any]:
    config = {
        "schema_name": "softsim.candidate_config",
        "schema_version": 2,
        "sample_id": task["sample_id"],
        "asset_revision_id": task["asset_revision_id"],
        "candidate_index": iteration,
        "route": task["route"],
        "physical_parameters": dict(candidate),
        "mutable_parameter_names": sorted(candidate),
        "fixed_solver_numerics": task["fixed_solver_numerics"],
        "fixed_solver_numerics_hash": fixed_hash,
        "proxy_mode": task["fixed_solver_numerics"].get("proxy_mode"),
        "probe": probe_record(probe),
    }
    config["config_hash"] = content_hash(config)
    return config


def _ensure_report_reference(result: ProbeResult, report_path: Path, probe: ProbeSpec, candidate: Mapping[str, float], simulator_config_hash: str) -> ProbeResult:
    """Normalize custom simulator output so every candidate has a report."""

    if result.report_path and Path(result.report_path).is_file():
        existing = Path(result.report_path)
        try:
            existing.resolve().relative_to(report_path.parent.parent.resolve())
        except ValueError:
            # Backends may write to a scheduler scratch directory.  Copy the
            # evidence into the task so release references remain portable.
            report_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(existing, report_path)
            result.report_path = str(report_path)
            result.report_hash = file_sha256(report_path)
        return result
    report = {
        "schema_name": "softsim.probe_report",
        "schema_version": 2,
        "backend": "custom",
        "probe": probe_record(probe),
        "probe_id": probe.probe_id,
        "probe_hash": probe_record(probe)["probe_hash"],
        "candidate_hash": content_hash(dict(candidate)),
        "simulator_config_hash": simulator_config_hash,
        "execution_status": result.execution_status,
        "behavior_pass": bool(result.passed and result.explicit_behavior_evidence),
        "passed": bool(result.passed and result.explicit_behavior_evidence),
        "explicit_behavior_evidence": bool(result.explicit_behavior_evidence),
        "metrics": dict(result.metrics),
        "diagnosis": dict(result.diagnosis),
        "outputs": dict(result.outputs),
    }
    atomic_write_json(report_path, report)
    result.report_path = str(report_path)
    result.report_hash = file_sha256(report_path)
    result.raw_report = report
    return result


def _enforce_probe_evidence(result: ProbeResult, probe: ProbeSpec) -> ProbeResult:
    """Require registered behavior metrics in addition to a successful process."""

    missing = [name for name in probe.required_metrics if name not in (result.metrics or {})]
    if not missing and result.explicit_behavior_evidence:
        return result
    result.passed = False
    result.explicit_behavior_evidence = False
    if result.execution_status == "completed":
        result.execution_status = SIM_EXECUTED_UNVERIFIED
    diagnosis = dict(result.diagnosis or {})
    if result.execution_status in {"failed", "timeout"}:
        # A crashed or timed-out executor stays a numerical failure; missing
        # metrics are a symptom, not a reason to reclassify it as unverified.
        diagnosis.setdefault("kind", "numerical_failure")
        diagnosis["missing_metrics"] = missing
    else:
        diagnosis.update({"kind": "behavior_unverified", "suggested_updates": {}, "missing_metrics": missing})
    result.diagnosis = diagnosis
    if result.report_path:
        report_path = Path(result.report_path)
        try:
            report = read_json(report_path)
            report.update({
                "behavior_pass": False,
                "passed": False,
                "explicit_behavior_evidence": False,
                "execution_status": result.execution_status,
                "diagnosis": diagnosis,
                "missing_required_metrics": missing,
            })
            atomic_write_json(report_path, report)
            result.report_hash = file_sha256(report_path)
            result.raw_report = report
        except (OSError, ValueError):
            pass
    return result


def _normalise_output_refs(result: ProbeResult, base_dir: Path, task_dir: Path) -> ProbeResult:
    """Convert backend-local trajectory/render paths to task-relative refs."""

    outputs = dict(result.outputs or {})
    for key, value in list(outputs.items()):
        if not isinstance(value, str):
            continue
        path = Path(value)
        if path.is_absolute():
            try:
                outputs[key] = path.resolve().relative_to(task_dir.resolve()).as_posix()
            except ValueError:
                # External output roots cannot be made release-relative; retain
                # the marker so aggregate marks the release non-publishable.
                continue
        elif (base_dir / path).is_file():
            outputs[key] = (base_dir / path).relative_to(task_dir).as_posix()
    result.outputs = outputs
    if result.report_path:
        try:
            report_path = Path(result.report_path)
            report = read_json(report_path)
            report["outputs"] = outputs
            atomic_write_json(report_path, report)
            result.report_hash = file_sha256(report_path)
            result.raw_report = report
        except (OSError, ValueError):
            pass
    return result


def _executor_failure(path: Path, probe: ProbeSpec, candidate: Mapping[str, float], error: Exception, simulator_config_hash: str) -> ProbeResult:
    report = {
        "schema_name": "softsim.probe_report",
        "schema_version": 2,
        "backend": "agent",
        "probe": probe_record(probe),
        "probe_id": probe.probe_id,
        "probe_hash": probe_record(probe)["probe_hash"],
        "stimulus_hash": probe_record(probe)["stimulus_hash"],
        "action_hash": probe_record(probe)["action_hash"],
        "config_hash": probe_record(probe)["config_hash"],
        "candidate_hash": content_hash(dict(candidate)),
        "simulator_config_hash": simulator_config_hash,
        "execution_status": "failed",
        "behavior_pass": False,
        "passed": False,
        "explicit_behavior_evidence": False,
        "metrics": {"finite_fraction": 0.0},
        "diagnosis": {"kind": "numerical_failure", "suggested_updates": {}, "error": f"{type(error).__name__}: {error}"},
        "outputs": {},
    }
    atomic_write_json(path, report)
    return ProbeResult(False, report["metrics"], report["diagnosis"], str(path), file_sha256(path), "failed", False, {}, report)


def _budget_progress(trace: list[dict[str, Any]], min_improvement: float) -> tuple[bool, dict[str, Any]]:
    """Elastic-budget gate: extend only while calibration loss keeps improving."""

    losses = [
        (event.get("metrics") or {}).get("loss")
        for event in trace
        if event.get("event") == "calibration_probe"
    ]
    losses = [float(value) for value in losses if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if len(losses) < 2:
        return False, {"reason": "insufficient_loss_history", "losses": losses}
    previous, latest = losses[-2], losses[-1]
    improved = latest < previous * (1.0 - min_improvement)
    return improved, {"loss_previous": previous, "loss_latest": latest, "min_relative_improvement": min_improvement}


def _status_from_calibration(calibration: ProbeResult, *, heldout: ProbeResult | None, candidate_index: int, proposal_status: str | None = None) -> str:
    if calibration.execution_status in {"failed", "timeout"}:
        return REJECTED_NUMERICAL_FAILURE
    if not calibration.explicit_behavior_evidence or calibration.execution_status == SIM_EXECUTED_UNVERIFIED:
        return UNCERTAIN_HUMAN_ESCALATION
    kind = str((calibration.diagnosis or {}).get("kind") or "")
    if kind in {"model_limit", "unsupported_model", "behavior_unverified"}:
        return REJECTED_MODEL_LIMIT
    if proposal_status == "proposer_unavailable":
        return UNCERTAIN_HUMAN_ESCALATION
    if heldout is not None:
        if heldout.execution_status in {"failed", "timeout"}:
            return REJECTED_NUMERICAL_FAILURE
        if not heldout.explicit_behavior_evidence:
            return UNCERTAIN_HUMAN_ESCALATION
        if not heldout.passed:
            return REJECTED_MODEL_LIMIT
        return ACCEPTED_FIRST_PASS if candidate_index == 0 else CORRECTED_THEN_ACCEPTED
    return REJECTED_MODEL_LIMIT


def run_task(task: dict[str, Any], output_root: Path, config: Mapping[str, Any], simulator: Any | None = None, proposer: Any | None = None, *, force: bool = False) -> dict[str, Any]:
    """Execute one auditable task and retain evidence for every terminal state."""

    # Accept the prototype's historical positional order:
    # run_task(task, output, simulator, config).
    if not isinstance(config, Mapping) and isinstance(simulator, Mapping):
        config, simulator = simulator, config
    config = dict(config)
    task = copy.deepcopy(task)
    raw_prior = task.get("prior")
    if not isinstance(raw_prior, Mapping):
        raise ValueError(f"{task.get('sample_id', '<unknown>')}: task has no prior")
    route = dict(task.get("route") or build_route(task, raw_prior))
    task["route"] = route
    task["prior"] = canonical_prior(task, raw_prior, route)
    supplied_fixed = task.get("fixed_solver_numerics") if isinstance(task.get("fixed_solver_numerics"), Mapping) else {}
    computed_fixed = fixed_numerics(task["prior"], route)
    computed_fixed.update(dict(supplied_fixed))
    task["fixed_solver_numerics"] = computed_fixed
    effective_config_hash = content_hash(config)
    task["config_hash"] = effective_config_hash
    core = {key: task.get(key) for key in ("sample_id", "asset_revision_id", "dataset_release_id", "route", "artifact_refs", "prior", "fixed_solver_numerics")}
    task["task_fingerprint"] = content_hash({**core, "config_hash": effective_config_hash})
    output_root = Path(output_root)
    task_dir = output_root / "tasks" / _safe_task_name(str(task["sample_id"]))
    task_dir.mkdir(parents=True, exist_ok=True)
    resumed = _valid_resume(task_dir, task)
    if resumed is not None and not force:
        return resumed
    old_marker = task_dir / "SUCCESS.json"
    if old_marker.is_file():
        try:
            old = read_json(old_marker)
            _archive_stale(task_dir, str(old.get("config_hash") or "stale"))
        except (OSError, ValueError):
            pass
    if simulator is None:
        from .simulators import build_simulator
        simulator = build_simulator(config)
    if proposer is None:
        proposer = build_proposer(config)
    context = task_context(task)
    calibration_probe, heldout_probe = build_probe_pair(config)
    fixed_hash = content_hash(context.fixed_solver_numerics)
    # Configuration can narrow, but never expand, the physical mutation set.
    allowed = frozenset(set(config.get("allowed_physical_parameters") or ALLOWED_PHYSICAL_PARAMETERS) & set(ALLOWED_PHYSICAL_PARAMETERS))
    prior = copy.deepcopy(context.prior)
    atomic_write_json(task_dir / "prior.json", prior)
    task_contract = {
        "schema_name": "softsim.task_contract",
        "schema_version": 2,
        "sample_id": context.sample_id,
        "asset_revision_id": context.asset_revision_id,
        "dataset_release_id": context.dataset_release_id,
        "route": context.route,
        "simulator_backend": getattr(simulator, "backend_name", type(simulator).__name__),
        "artifact_refs": context.artifact_refs,
        "fixed_solver_numerics": context.fixed_solver_numerics,
        "fixed_solver_numerics_hash": fixed_hash,
        "allowed_physical_parameters": sorted(allowed),
        "calibration_probe": probe_record(calibration_probe),
        "heldout_probe": probe_record(heldout_probe),
        "config_hash": context.config_hash,
        "task_fingerprint": context.task_fingerprint,
    }
    validate_task_contract(task_contract)
    atomic_write_json(task_dir / "task_contract.json", task_contract)
    # Configuration may narrow the mutable set, but can never add solver
    # numerics or arbitrary JSON fields to the physical mutation contract.
    candidate = {name: value for name, value in physical_values(prior).items() if name in allowed}
    if not candidate:
        # A malformed prior is a model-limit task, but still receives complete evidence.
        candidate = {"density_kg_m3": 600.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.35}
        prior.setdefault("limitations", []).append("missing_mutable_prior_parameters")
        atomic_write_json(task_dir / "prior.json", prior)
    bounds = _normalise_parameter_bounds(prior, candidate)
    for name in allowed:
        bounds.setdefault(name, DEFAULT_PARAMETER_BOUNDS[name])
    candidate = _clamp_candidate(candidate, bounds)
    max_iterations = max(1, int(config.get("max_iterations", config.get("budget", 3))))
    # Elastic budget: hard tasks may earn extra iterations, but only while the
    # calibration loss keeps improving; a stalled loop stops at the base budget.
    budget_cap = max(max_iterations, int(config.get("max_iterations_cap", max_iterations)))
    budget_step = max(1, int(config.get("budget_step", 2)))
    min_improvement = float(config.get("budget_min_relative_improvement", 0.05))
    trace: list[dict[str, Any]] = []
    calibration: ProbeResult | None = None
    heldout: ProbeResult | None = None
    final_candidate_index = 0
    terminal_override: str | None = None
    proposal_status: str | None = None
    candidate_refs: list[dict[str, Any]] = []
    iteration = 0
    while iteration < max_iterations:
        final_candidate_index = iteration
        candidate_config = _candidate_config(task, candidate, calibration_probe, iteration, fixed_hash=fixed_hash)
        candidate_dir = task_dir / f"candidate_{iteration:02d}"
        atomic_write_json(candidate_dir / "config.json", candidate_config)
        simulator_config = {
            "policy_version": POLICY_VERSION,
            "route": context.route,
            "fixed_solver_numerics": context.fixed_solver_numerics,
            "fixed_solver_numerics_hash": fixed_hash,
            "candidate_config_hash": candidate_config["config_hash"],
            "probe_hash": probe_record(calibration_probe)["probe_hash"],
            "seed": context.fixed_solver_numerics.get("seed"),
        }
        simulator_config_hash = content_hash(simulator_config)
        try:
            calibration = _call_simulator(simulator, task, candidate, calibration_probe, candidate_dir, simulator_config)
        except Exception as exc:  # retain a terminal numerical-failure record
            calibration = _executor_failure(candidate_dir / "report.json", calibration_probe, candidate, exc, simulator_config_hash)
        calibration = _ensure_report_reference(calibration, candidate_dir / "report.json", calibration_probe, candidate, simulator_config_hash)
        calibration = _enforce_probe_evidence(calibration, calibration_probe)
        calibration = _normalise_output_refs(calibration, candidate_dir, task_dir)
        report_ref = str(Path(calibration.report_path).relative_to(task_dir).as_posix()) if calibration.report_path else None
        trace.append({
            "event": "calibration_probe",
            "candidate_index": iteration,
            "candidate_hash": content_hash(candidate),
            "candidate_parameters": dict(candidate),
            "candidate_config_hash": candidate_config["config_hash"],
            "simulator_config_hash": simulator_config_hash,
            "fixed_solver_numerics_hash": fixed_hash,
            "fixed_solver_numerics": context.fixed_solver_numerics,
            "probe": probe_record(calibration_probe),
            "metrics": calibration.metrics,
            "diagnosis": calibration.diagnosis,
            "behavior_pass": calibration.passed,
            "explicit_behavior_evidence": calibration.explicit_behavior_evidence,
            "report_ref": report_ref,
            "budget": {"max_iterations": max_iterations, "iteration": iteration, "remaining": max_iterations - iteration - 1},
        })
        candidate_refs.append({"candidate_index": iteration, "config_ref": str((candidate_dir / "config.json").relative_to(task_dir).as_posix()), "report_ref": report_ref, "outputs": calibration.outputs, "config_sha256": file_sha256(candidate_dir / "config.json"), "report_sha256": calibration.report_hash})
        if calibration.passed and calibration.explicit_behavior_evidence:
            break
        if iteration >= max_iterations - 1:
            exhausted = True
            if max_iterations < budget_cap:
                progress, detail = _budget_progress(trace, min_improvement)
                if progress:
                    # Escalate and fall through to the proposer: the extra
                    # iterations are spent on new candidates, never on
                    # re-running the current one.
                    max_iterations = min(budget_cap, max_iterations + budget_step)
                    trace.append({
                        "event": "budget_escalation",
                        "candidate_index": iteration,
                        "new_budget": max_iterations,
                        "budget_cap": budget_cap,
                        "reason": "calibration_loss_improving",
                        "detail": detail,
                    })
                    exhausted = False
            if exhausted:
                terminal_override = _status_from_calibration(calibration, heldout=None, candidate_index=iteration)
                break
        calibration_kind = str((calibration.diagnosis or {}).get("kind") or "")
        if calibration_kind in {"model_limit", "unsupported_model"}:
            # Family capability gate: an unsupported route terminates as a
            # model limit; the proposer is never invoked (no fallback, no
            # wasted paid calls).
            terminal_override = REJECTED_MODEL_LIMIT
            break
        try:
            proposal = proposer.propose(prior=prior, candidate=candidate, calibration=calibration, bounds=bounds, iteration=iteration)
        except Exception as exc:
            proposal = type("ProposalFailure", (), {"status": "proposer_unavailable", "patch": {}, "rule_id": None, "rationale": "proposer raised an exception", "details": {"error": f"{type(exc).__name__}: {exc}"}})()
        proposal_status = proposal.status
        trace.append({
            "event": "candidate_proposal",
            "candidate_index": iteration,
            "proposal_status": proposal.status,
            "rule_id": proposal.rule_id,
            "patch": proposal.patch,
            "rationale": proposal.rationale,
            "details": proposal.details,
            "feedback_source": "calibration_only",
            "heldout_feedback_used": False,
        })
        if proposal.status != "proposed":
            terminal_override = _status_from_calibration(calibration, heldout=None, candidate_index=iteration, proposal_status=proposal.status)
            if proposal.status == "model_limit":
                terminal_override = REJECTED_MODEL_LIMIT
            elif proposal.status == "numerical_failure":
                terminal_override = REJECTED_NUMERICAL_FAILURE
            elif proposal.status in {"proposer_unavailable", "evidence_unavailable"}:
                # An unavailable proposer escalates only when the diagnosis is
                # not already a definitive model/numerical limit.
                if terminal_override not in {REJECTED_MODEL_LIMIT, REJECTED_NUMERICAL_FAILURE}:
                    terminal_override = UNCERTAIN_HUMAN_ESCALATION
            break
        try:
            patch = validate_patch(proposal.patch, allowed)
        except (TypeError, ValueError) as exc:
            terminal_override = UNCERTAIN_HUMAN_ESCALATION
            trace.append({
                "event": "candidate_proposal_rejected",
                "candidate_index": iteration,
                "proposal_status": "invalid_patch",
                "patch": proposal.patch,
                "error": f"{type(exc).__name__}: {exc}",
                "feedback_source": "calibration_only",
                "heldout_feedback_used": False,
            })
            break
        next_candidate = dict(candidate)
        next_candidate.update(patch)
        next_candidate = _clamp_candidate(next_candidate, bounds)
        if next_candidate == candidate:
            terminal_override = REJECTED_MODEL_LIMIT
            break
        candidate = next_candidate
        iteration += 1
    if calibration is None:
        raise RuntimeError(f"{context.sample_id}: calibration probe did not execute")
    if calibration.passed and calibration.explicit_behavior_evidence:
        trace.append({
            "event": "candidate_frozen",
            "candidate_index": final_candidate_index,
            "candidate_hash": content_hash(candidate),
            "candidate_config_ref": f"candidate_{final_candidate_index:02d}/config.json",
            "candidate_report_ref": str((task_dir / f"candidate_{final_candidate_index:02d}" / "report.json").relative_to(task_dir).as_posix()),
            "fixed_solver_numerics_hash": fixed_hash,
            "eligible_for_heldout": True,
        })
        heldout_dir = task_dir / "heldout"
        heldout_config = {
            "policy_version": POLICY_VERSION,
            "route": context.route,
            "fixed_solver_numerics": context.fixed_solver_numerics,
            "fixed_solver_numerics_hash": fixed_hash,
            "candidate_hash": content_hash(candidate),
            "probe": probe_record(heldout_probe),
            "feedback_allowed": False,
        }
        atomic_write_json(heldout_dir / "config.json", heldout_config)
        try:
            heldout = _call_simulator(simulator, task, candidate, heldout_probe, heldout_dir, heldout_config)
        except Exception as exc:  # retain held-out execution failure as evidence
            heldout = _executor_failure(heldout_dir / "report.json", heldout_probe, candidate, exc, content_hash(heldout_config))
        heldout = _ensure_report_reference(heldout, heldout_dir / "report.json", heldout_probe, candidate, content_hash(heldout_config))
        heldout = _enforce_probe_evidence(heldout, heldout_probe)
        heldout = _normalise_output_refs(heldout, heldout_dir, task_dir)
        trace.append({
            "event": "heldout_probe",
            "candidate_index": final_candidate_index,
            "candidate_hash": content_hash(candidate),
            "candidate_parameters": dict(candidate),
            "simulator_config_hash": content_hash(heldout_config),
            "fixed_solver_numerics_hash": fixed_hash,
            "fixed_solver_numerics": context.fixed_solver_numerics,
            "probe": probe_record(heldout_probe),
            "metrics": heldout.metrics,
            "diagnosis": heldout.diagnosis,
            "behavior_pass": heldout.passed,
            "explicit_behavior_evidence": heldout.explicit_behavior_evidence,
            "feedback_allowed": False,
            "report_ref": str(Path(heldout.report_path).relative_to(task_dir).as_posix()) if heldout.report_path else None,
        })
    else:
        trace.append({
            "event": "candidate_frozen",
            "candidate_index": final_candidate_index,
            "candidate_hash": content_hash(candidate),
            "candidate_config_ref": f"candidate_{final_candidate_index:02d}/config.json",
            "candidate_report_ref": f"candidate_{final_candidate_index:02d}/report.json",
            "fixed_solver_numerics_hash": fixed_hash,
            "eligible_for_heldout": False,
        })
        heldout = _write_skipped_heldout(task_dir / "heldout" / "report.json", heldout_probe, "calibration_not_accepted", fixed_hash=fixed_hash, candidate_hash=content_hash(candidate))
    status = terminal_override or _status_from_calibration(calibration, heldout=heldout, candidate_index=final_candidate_index, proposal_status=proposal_status)
    if calibration.passed and calibration.explicit_behavior_evidence and heldout is not None and not heldout.passed:
        status = _status_from_calibration(calibration, heldout=heldout, candidate_index=final_candidate_index)
    accepted = status in {ACCEPTED_FIRST_PASS, CORRECTED_THEN_ACCEPTED}
    trace.append({
        "event": "decision",
        "status": status,
        "accepted": accepted,
        "candidate_index": final_candidate_index,
        "calibration_pass": calibration.passed,
        "heldout_executed": bool(heldout and heldout.execution_status != "not_executed"),
        "heldout_pass": bool(heldout and heldout.passed),
        "heldout_feedback_used": False,
    })
    trace_path = task_dir / "agent_trace.jsonl"
    atomic_write_jsonl(trace_path, trace)
    calibration_ref = str(Path(calibration.report_path).relative_to(task_dir).as_posix()) if calibration.report_path else None
    heldout_ref = str(Path(heldout.report_path).relative_to(task_dir).as_posix()) if heldout and heldout.report_path else "heldout/report.json"
    physics = {
        "schema_name": "softsim.physics_bundle",
        "schema_version": 2,
        "sample_id": context.sample_id,
        "asset_revision_id": context.asset_revision_id,
        "dataset_release_id": context.dataset_release_id,
        "status": status,
        "success": accepted,
        "terminal": True,
        "simulation_calibrated": accepted,
        "prior_compatible": status == ACCEPTED_FIRST_PASS,
        "correction_applied": final_candidate_index > 0,
        "verification_state": "verified_behavior_contract" if accepted else status,
        "route": context.route,
        "simulator_backend": getattr(simulator, "backend_name", type(simulator).__name__),
        "agent_source_sha256": package_source_hash(),
        "artifact_refs": context.artifact_refs,
        "prior_ref": "prior.json",
        "trace_ref": "agent_trace.jsonl",
        "calibration_report_ref": calibration_ref,
        "heldout_report_ref": heldout_ref,
        "prior_status": prior.get("status", "predicted_prior"),
        "candidate": {"index": final_candidate_index, "parameters": dict(candidate), "config_ref": f"candidate_{final_candidate_index:02d}/config.json", "report_ref": calibration_ref},
        "parameters": [{"name": name, "value": value, "source": "behavior_calibration" if final_candidate_index > 0 else "prior_compatible"} for name, value in sorted(candidate.items())],
        "solver_adapter": {"solver": context.route.get("solver_adapter"), "proxy_mode": context.fixed_solver_numerics.get("proxy_mode"), "parameters": dict(candidate)},
        "candidate_refs": candidate_refs,
        "solver_numerics": context.fixed_solver_numerics,
        "solver_numerics_hash": fixed_hash,
        "mutable_parameter_names": sorted(allowed),
        "verification": {
            "calibration": {"probe_id": calibration_probe.probe_id, "probe_hash": probe_record(calibration_probe)["probe_hash"], "stimulus_hash": content_hash(calibration_probe.stimulus), "action_hash": content_hash(calibration_probe.action), "config_hash": content_hash(calibration_probe.config), "report_ref": calibration_ref, "report_sha256": calibration.report_hash, "behavior_pass": calibration.passed, "metrics": calibration.metrics, "outputs": calibration.outputs},
            "heldout": {"probe_id": heldout_probe.probe_id, "probe_hash": probe_record(heldout_probe)["probe_hash"], "stimulus_hash": content_hash(heldout_probe.stimulus), "action_hash": content_hash(heldout_probe.action), "config_hash": content_hash(heldout_probe.config), "report_ref": heldout_ref, "report_sha256": heldout.report_hash if heldout else None, "executed": bool(heldout and heldout.execution_status != "not_executed"), "behavior_pass": bool(heldout and heldout.passed), "metrics": heldout.metrics if heldout else {}, "outputs": heldout.outputs if heldout else {}},
            "heldout_feedback_used": False,
        },
        "agent": {"policy_version": POLICY_VERSION, "proposer_backend": type(proposer).__name__, "last_proposal_status": proposal_status, "trace_ref": "agent_trace.jsonl", "trace_ref_root": f"tasks/{_safe_task_name(context.sample_id)}/agent_trace.jsonl", "trace_sha256": file_sha256(trace_path), "iteration_count": len([item for item in trace if item.get("event") == "calibration_probe"]), "budget_final": max_iterations, "budget_cap": budget_cap, "termination_reason": status, "decision": "accept" if accepted else "reject"},
        "provenance": {"simulation_calibrated": accepted, "is_ground_truth": False, "synthetic_fixture": str(task.get("source_kind") or "").startswith("fixture") or getattr(simulator, "backend_name", "") == "mock", "prior_only": False},
        "limitations": ([] if accepted else ["behavior_contract_not_accepted"]),
    }
    physics["physical_bundle_id"] = content_hash({key: value for key, value in physics.items() if key not in {"physical_bundle_id", "record_sha256"}})
    physics["record_sha256"] = content_hash(physics)
    validate_physics_bundle(physics)
    physics_path = task_dir / "physics.json"
    atomic_write_json(physics_path, physics)
    marker = {
        "schema_name": "softsim.task_success_marker",
        "schema_version": 2,
        "sample_id": context.sample_id,
        "asset_revision_id": context.asset_revision_id,
        "status": status,
        "config_hash": context.config_hash,
        "task_fingerprint": context.task_fingerprint,
        "physics_ref": "physics.json",
        "physics_path": "physics.json",
        "physics_sha256": file_sha256(physics_path),
        "physics_record_sha256": physics["record_sha256"],
        "agent_source_sha256": physics["agent_source_sha256"],
        "trace_ref": "agent_trace.jsonl",
        "trace_path": "agent_trace.jsonl",
        "trace_sha256": file_sha256(trace_path),
        "heldout_ref": heldout_ref,
        "heldout_sha256": file_sha256(task_dir / "heldout" / "report.json"),
        "resumable": True,
    }
    atomic_write_json(task_dir / "SUCCESS.json", marker)
    return marker
