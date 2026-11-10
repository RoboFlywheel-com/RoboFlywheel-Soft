# SPDX-License-Identifier: Apache-2.0
"""Genesis-backed formal episode executor adapter (production draft).

``softsim-exp run --episode-executor softsim_genesis_runner:run_episode
--process-isolation`` loads :func:`run_episode` inside the coordinator worker
interpreter.  The adapter never imports Genesis itself; it serializes a job
spec into the task attempt directory and spawns ``genesis_worker.py`` under
``GENESIS_PYTHON`` (the pinned Genesis 1.3.3 venv), then folds the worker
result into a canonical ``softsim.episode_record`` that satisfies
``configs/formal_executor_contract.json``.

Draft boundaries (until asset v2 / physical v12 / benchmark manifest freeze):

- every record produced here carries ``synthetic=True`` markers when the run
  manifest is synthetic, and the adapter never fabricates evidence: a failed
  or missing Genesis runtime produces ``failed``/``blocked`` records, never a
  formula-generated pass;
- E2 hidden targets are requested exclusively through ``target_provider`` and
  are used only inside the evaluator-side worker job; neither the returned
  record nor the agent steps/calls may contain target parameter values;
- E3-A damage is the frozen ``TofuDamage.continuum_damage_stiffness_loss_surrogate``
  proxy, not explicit fracture; rigid-MPM contact force is currently
  unavailable in Genesis 1.3.3 LegacyCoupler, so force fields are reported as
  ``not_available_rigid_mpm_coupler`` instead of being guessed.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from softsim_formal.canonical import fingerprint, read_json
from softsim_formal.io import atomic_write_json
from softsim_formal.records import make_call_record, make_episode_record

__all__ = ["run_episode", "GENESIS_PYTHON_ENV"]

GENESIS_PYTHON_ENV = "GENESIS_PYTHON"
_DEFAULT_TIMEOUT_S = 1800.0
_WORKER = Path(__file__).resolve().parent / "genesis_worker.py"
_REPO_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if not _REPO_SCRIPTS.is_dir():
    import softsim_tools

    _REPO_SCRIPTS = Path(softsim_tools.__file__).resolve().parent

DAMAGE_MODEL_ID = "TofuDamage.continuum_damage_stiffness_loss_surrogate"
CONTACT_FORCE_STATUS = "not_available_rigid_mpm_coupler"

# Frozen v12.6 probe semantics (configs/agent_genesis_qwen_command_v126exp.json
# mirrored 2026-09-23).  The flags replay exactly what the rebuild_v4 full run
# used: mesh_adaptation=v5, behavior_contract_version=v12.3, calibrated gates.
_V126_FLAGS: dict[str, Any] = {
    "mesh_adaptation": "v5",
    "behavior_contract_version": "v12.3",
    "gate_overrides": {
        "settle_gate": "off",
        "drop_deformation_band": [0.005, 0.98],
        "recovery_band_mpm": [0.45, 1.16],
        "recovery_band_cloth": [0.0, 1.16],
    },
}
E1_RUNTIME_PROBE: dict[str, Any] = {
    "probe_id": "e1.runtime.stability_drop.v126",
    "stimulus": {"kind": "drop", "height_m": 0.18, "orientation": "corner_first", "floor": "rigid_plane"},
    "action": {"kind": "release", "gripper": "none", "contact_schedule": "single_impact"},
    "config": {"steps": 420, "dt_s": 0.0007, "record_every": 4, "seed_offset": 11, **_V126_FLAGS},
    "tolerance": 0.05,
    "required_metrics": ["contact_fraction", "settled", "deformation_ratio", "finite_fraction"],
}
E2_CALIBRATION_PROBE: dict[str, Any] = {
    "probe_id": "calibration.drop.corner_first.v126",
    "stimulus": {"kind": "drop", "height_m": 0.18, "orientation": "corner_first", "floor": "rigid_plane"},
    "action": {"kind": "release", "gripper": "none", "contact_schedule": "single_impact"},
    "config": {"steps": 420, "dt_s": 0.0007, "record_every": 4, "seed_offset": 11, **_V126_FLAGS},
    "tolerance": 0.05,
    "required_metrics": ["contact_fraction", "settled", "deformation_ratio", "finite_fraction"],
}
E2_HELDOUT_PROBE: dict[str, Any] = {
    "probe_id": "heldout.compression.lateral.v126",
    "stimulus": {"kind": "compression", "compression_ratio": 0.22, "axis": "x", "rate": "quasi_static"},
    "action": {"kind": "compress_release", "hold_steps": 28, "release_steps": 28},
    "config": {"steps": 560, "dt_s": 0.0006, "record_every": 8, "seed_offset": 97, **_V126_FLAGS},
    "tolerance": 0.08,
    "required_metrics": ["contact_fraction", "recovery_ratio", "settled", "finite_fraction"],
}
_E2_LOSS_METRICS = ("finite_fraction", "contact_fraction", "settled", "deformation_ratio", "recovery_ratio")


class ExecutorError(RuntimeError):
    """Internal adapter error; converted into a terminal failure record."""


def _genesis_python() -> str:
    return os.environ.get(GENESIS_PYTHON_ENV) or "python3"


def _resolve_root(alias: Any, run_manifest: Mapping[str, Any]) -> Path | None:
    if not alias:
        return None
    env_name = "SOFTSIM_ROOT_" + re.sub(r"[^A-Za-z0-9]", "_", str(alias)).upper()
    root = os.environ.get(env_name) or (run_manifest.get("logical_roots") or {}).get(alias)
    if not root or str(root).startswith("<"):
        return None
    return Path(str(root))


def _verify_file_sha256(path: Path, expected: Any) -> bool:
    from softsim_formal.canonical import sha256_file

    if not expected or not str(expected).startswith("sha256:"):
        return False
    return sha256_file(path) == str(expected)


_BUNDLE_INDEX_CACHE: dict[str, dict[str, str]] = {}


def _bundle_path_by_id(root: Path, physical_bundle_id: Any) -> str | None:
    """Resolve a bundle path from the release index for rows that reference
    bundles only by identity (E3-A benchmark rows carry no bundle path)."""
    if not physical_bundle_id:
        return None
    key = str(root)
    if key not in _BUNDLE_INDEX_CACHE:
        mapping: dict[str, str] = {}
        index_path = root / "physical_bundle_index.jsonl"
        if index_path.is_file():
            with index_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    bundle = entry.get("bundle") if isinstance(entry.get("bundle"), Mapping) else {}
                    rel = bundle.get("path") or bundle.get("relative_path")
                    if entry.get("physical_bundle_id") and rel:
                        mapping[str(entry["physical_bundle_id"])] = str(rel)
        _BUNDLE_INDEX_CACHE[key] = mapping
    return _BUNDLE_INDEX_CACHE[key].get(str(physical_bundle_id))


def _load_bundle_physics(task: Mapping[str, Any], run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Load the frozen bundle physics.json referenced by the physical index row."""
    row = task.get("source_row") if isinstance(task.get("source_row"), Mapping) else {}
    bundle = row.get("bundle") if isinstance(row.get("bundle"), Mapping) else {}
    rel = bundle.get("path") or bundle.get("relative_path")
    root = _resolve_root(bundle.get("root_alias") or "release", run_manifest)
    if not rel and root is not None:
        rel = _bundle_path_by_id(root, row.get("physical_bundle_id") or task.get("physical_bundle_id"))
    if not rel:
        return {}
    if root is None:
        return {}
    path = root / str(rel)
    if not path.is_file():
        return {}
    value = read_json(path)
    return dict(value) if isinstance(value, dict) else {}


def _resolve_inputs(task: Mapping[str, Any], run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve mesh/parameters/route/numerics/scale from frozen release rows.

    Raises ExecutorError with a failure code when a required input cannot be
    resolved or its content hash mismatches.
    """
    row = task.get("source_row") if isinstance(task.get("source_row"), Mapping) else {}
    physics = _load_bundle_physics(task, run_manifest)

    mesh_path: Path | None = None
    mesh_sha256: Any = None
    for container in (row, task):
        for key in ("solver_mesh_path", "mesh_path"):
            value = container.get(key)
            if value:
                mesh_path = Path(str(value))
                break
        if mesh_path is not None:
            break
    if mesh_path is None:
        ref = row.get("asset_geometry_ref") or row.get("geometry_ref")
        ref = ref if isinstance(ref, Mapping) else {}
        relative = ref.get("relative_path")
        root = _resolve_root(ref.get("root_alias"), run_manifest)
        if relative and root is not None:
            mesh_path = root / str(relative)
            mesh_sha256 = ref.get("sha256")
    if mesh_path is None:
        # E3-A benchmark rows carry no geometry_ref; the frozen physical bundle
        # is the mesh authority via artifact_refs (e.g. collision_mesh under
        # root_alias p0_root).
        artifact_refs = physics.get("artifact_refs") if isinstance(physics.get("artifact_refs"), Mapping) else {}
        for role in ("collision_mesh", "visual_mesh"):
            aref = artifact_refs.get(role)
            if not isinstance(aref, Mapping):
                continue
            relative = aref.get("path") or aref.get("relative_path")
            root = _resolve_root(aref.get("root_alias"), run_manifest)
            if relative and root is not None:
                mesh_path = root / str(relative)
                mesh_sha256 = aref.get("sha256")
                break
    if mesh_path is None or not mesh_path.is_file():
        raise ExecutorError("mesh_unresolved")
    if mesh_sha256 and not _verify_file_sha256(mesh_path, mesh_sha256):
        raise ExecutorError("mesh_hash_mismatch")

    adapter = physics.get("solver_adapter") if isinstance(physics.get("solver_adapter"), Mapping) else {}
    if not adapter:
        adapter = row.get("solver_adapter") if isinstance(row.get("solver_adapter"), Mapping) else {}
    raw_params = adapter.get("parameters") or task.get("prior_parameters") or {}
    parameters = {str(key): float(value) for key, value in raw_params.items() if isinstance(value, (int, float))}

    route = physics.get("route") if isinstance(physics.get("route"), Mapping) else {}
    if not route:
        route = row.get("route") if isinstance(row.get("route"), Mapping) else {}
    numerics = row.get("solver_numerics") if isinstance(row.get("solver_numerics"), Mapping) else {}
    if not numerics:
        numerics = physics.get("solver_numerics") if isinstance(physics.get("solver_numerics"), Mapping) else {}

    scale_row = row.get("asset_metric_scale") or row.get("metric_scale")
    scale_row = scale_row if isinstance(scale_row, Mapping) else {}
    meters_per_unit = scale_row.get("meters_per_unit") or scale_row.get("scale_to_meter")
    scale_m = numerics.get("scale_m")
    return {
        "mesh_path": mesh_path.as_posix(),
        "mesh_sha256_verified": bool(mesh_sha256),
        "parameters": parameters,
        "route": {str(k): v for k, v in route.items() if isinstance(v, (str, int, float, bool))},
        "numerics": dict(numerics),
        "scale": {"meters_per_unit": meters_per_unit, "scale_m": scale_m},
    }


def _task_parameters(task: Mapping[str, Any]) -> dict[str, float]:
    row = task.get("source_row") if isinstance(task.get("source_row"), Mapping) else {}
    adapter = row.get("solver_adapter") if isinstance(row.get("solver_adapter"), Mapping) else {}
    raw = adapter.get("parameters") or task.get("prior_parameters") or {}
    return {str(key): float(value) for key, value in raw.items() if isinstance(value, (int, float))}


def _run_worker(job: Mapping[str, Any], attempt_dir: Path, *, timeout_s: float | None = None) -> dict[str, Any]:
    """Spawn the Genesis worker process and return its parsed result."""
    job_path = attempt_dir / "genesis_job.json"
    result_path = attempt_dir / "genesis_result.json"
    atomic_write_json(job_path, dict(job))
    python = _genesis_python()
    if shutil.which(python) is None and not Path(python).is_file():
        return {"status": "error", "error_type": "GenesisRuntimeMissing", "detail": f"{GENESIS_PYTHON_ENV} not resolvable: {python}"}
    env = _bridge_env()
    timeout = float(timeout_s or os.environ.get("SOFTSIM_GENESIS_EPISODE_TIMEOUT_S", _DEFAULT_TIMEOUT_S))
    started = time.time()
    try:
        completed = subprocess.run(
            [python, str(_WORKER), "--job", str(job_path), "--result", str(result_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(attempt_dir),
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "error_type": "GenesisTimeout", "detail": f"worker exceeded {timeout}s"}
    except OSError as exc:
        return {"status": "error", "error_type": "GenesisSpawnFailure", "detail": type(exc).__name__}
    wall_s = time.time() - started
    (attempt_dir / "genesis_stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (attempt_dir / "genesis_stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if not result_path.is_file():
        return {
            "status": "error",
            "error_type": "GenesisWorkerCrash",
            "detail": f"returncode={completed.returncode}; see genesis_stderr.log",
            "wall_s": round(wall_s, 6),
        }
    result = read_json(result_path)
    if not isinstance(result, dict):
        return {"status": "error", "error_type": "GenesisResultMalformed", "detail": "result JSON is not an object"}
    result.setdefault("wall_s", round(wall_s, 6))
    return result


def _bridge_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    preload = env.get("SOFTSIM_GENESIS_LD_PRELOAD")
    default_candidate = Path("/opt/nvidia/nsight-systems/2026.1.3/host-linux-x64/libstdc++.so.6")
    if preload is None and default_candidate.is_file():
        preload = str(default_candidate)
    if preload:
        env["LD_PRELOAD"] = preload + ((":" + env["LD_PRELOAD"]) if env.get("LD_PRELOAD") else "")
    return env


def _run_bridge_probe(
    attempt_dir: Path,
    *,
    label: str,
    mesh_path: str,
    parameters: Mapping[str, Any],
    route: Mapping[str, Any],
    numerics: Mapping[str, Any],
    probe: Mapping[str, Any],
    sample_id: str,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Run one production behavior probe through the frozen bridge.

    ``code/scripts/genesis_probe.py`` (stdlib-only, CLI interpreter) wraps
    ``code/scripts/genesis_behavior_runner.py`` under ``GENESIS_PYTHON`` — the
    same runner that produced the v12.6 full-run evidence.  Returns the probe
    report dict, or an error sentinel.
    """
    work = attempt_dir / label
    work.mkdir(parents=True, exist_ok=True)
    probe_script = Path(os.environ.get("SOFTSIM_GENESIS_PROBE_SCRIPT") or (_REPO_SCRIPTS / "genesis_probe.py"))
    behavior_runner = os.environ.get("SOFTSIM_GENESIS_BEHAVIOR_RUNNER") or str(_REPO_SCRIPTS / "genesis_behavior_runner.py")
    if not probe_script.is_file():
        return {"status": "error", "error_type": "ProbeBridgeMissing", "detail": str(probe_script)}
    asset = {
        "sample_id": sample_id,
        "resolved_artifacts": {"collision_mesh": mesh_path},
        "route": dict(route),
        "fixed_solver_numerics": dict(numerics),
    }
    atomic_write_json(work / "asset.json", asset)
    atomic_write_json(work / "candidate.json", {str(k): float(v) for k, v in parameters.items() if isinstance(v, (int, float))})
    atomic_write_json(work / "probe.json", dict(probe))
    report_path = work / "report.json"
    env = _bridge_env()
    env["GENESIS_PYTHON"] = _genesis_python()
    env["SOFTSIM_GENESIS_BEHAVIOR_RUNNER"] = behavior_runner
    timeout = float(timeout_s or os.environ.get("SOFTSIM_GENESIS_EPISODE_TIMEOUT_S", _DEFAULT_TIMEOUT_S))
    started = time.time()
    try:
        completed = subprocess.run(
            [sys.executable, str(probe_script), "--asset-json", str(work / "asset.json"), "--candidate-json", str(work / "candidate.json"), "--probe-json", str(work / "probe.json"), "--output-dir", str(work / "out"), "--report-json", str(report_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(work),
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "error_type": "GenesisTimeout", "detail": f"bridge probe exceeded {timeout}s"}
    except OSError as exc:
        return {"status": "error", "error_type": "GenesisSpawnFailure", "detail": type(exc).__name__}
    wall_s = time.time() - started
    (work / "bridge_stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (work / "bridge_stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if not report_path.is_file():
        return {"status": "error", "error_type": "ProbeReportMissing", "detail": f"returncode={completed.returncode}"}
    report = read_json(report_path)
    if not isinstance(report, dict):
        return {"status": "error", "error_type": "ProbeReportMalformed", "detail": label}
    report["wall_s"] = round(wall_s, 6)
    report["bridge_returncode"] = completed.returncode
    return {"status": "ok", "report": report}


def _report_behavior_loss(candidate_report: Mapping[str, Any], reference_report: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    cand = candidate_report.get("metrics") if isinstance(candidate_report.get("metrics"), Mapping) else {}
    ref = reference_report.get("metrics") if isinstance(reference_report.get("metrics"), Mapping) else {}
    total = 0.0
    residual: dict[str, Any] = {}
    used = 0
    for key in _E2_LOSS_METRICS:
        if key not in cand or key not in ref:
            continue
        c, r = cand[key], ref[key]
        if isinstance(c, bool) or isinstance(r, bool):
            diff = 0.0 if bool(c) == bool(r) else 1.0
        else:
            try:
                diff = min(1.0, abs(float(c) - float(r)) / max(abs(float(r)), 1e-6))
            except (TypeError, ValueError):
                continue
        residual[key] = {"candidate": c, "reference": r}
        total += diff
        used += 1
    return (total / used if used else 1.0), residual


def _failure_record(
    task: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    *,
    failure_code: str,
    failure_domain: str = "infrastructure",
    execution_status: str = "failed",
    outcome_status: str = "failure",
    retriable: bool = True,
    detail: str,
    experiment_id: str,
) -> dict[str, Any]:
    outcomes: dict[str, Any] = {}
    measurements: dict[str, Any] = {"error_detail": detail, "synthetic_formula_used": False}
    if experiment_id == "e3a":
        outcomes = {
            "task_success": False,
            "grasp_success": False,
            "lift_success": False,
            "hold_success": False,
            "intact_success": False,
            "damage_failure": False,
        }
        measurements = {
            "damage_fraction": 0.0,
            "integrity": 1.0,
            "safe_force_window": {"low_n": None, "high_n": None, "width_n": 0.0, "non_empty": False},
            "slip": None,
            "contact_count": 0,
            "convergence": False,
            "finite_fraction": 0.0,
            "error_detail": detail,
            "damage_model_id": DAMAGE_MODEL_ID,
            "contact_force_status": CONTACT_FORCE_STATUS,
            "synthetic_formula_used": False,
        }
    return make_episode_record(
        run_manifest=run_manifest,
        task=task,
        execution_status=execution_status,
        outcome_status=outcome_status,
        failure_domain=failure_domain,
        failure_code=failure_code,
        retriable=retriable,
        metrics={"real_executor_used": True, "genesis_completed": False},
        outcomes=outcomes,
        measurements=measurements,
        extra={"executor_id": "softsim_genesis_runner:run_episode", "synthetic_formula_used": False},
    )


def _e1_record(task: Mapping[str, Any], run_manifest: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    """Map a production bridge probe report onto the E1 runtime/contract gate."""
    report = result.get("report") if isinstance(result.get("report"), Mapping) else {}
    metrics = dict(report.get("metrics") or {})
    diagnosis = report.get("diagnosis") if isinstance(report.get("diagnosis"), Mapping) else {}
    finite = float(metrics.get("finite_fraction") or 0.0)
    executed = str(report.get("execution_status")) == "completed" and bool(report.get("explicit_behavior_evidence"))
    runtime_pass = bool(executed and finite >= 1.0)
    contract_pass = bool(report.get("behavior_pass")) if executed else False
    outcome = "success" if (runtime_pass and contract_pass) else "failure"
    if outcome == "success":
        failure_domain = "none"
        failure_code = None
    elif finite < 1.0:
        failure_domain = "numerical"
        failure_code = "nan_or_instability"
    elif not executed:
        failure_domain = "infrastructure"
        failure_code = str(diagnosis.get("kind") or "probe_not_executed")
    else:
        failure_domain = "task"
        failure_code = str(diagnosis.get("kind") or "behavior_contract_fail")
    return make_episode_record(
        run_manifest=run_manifest,
        task=task,
        execution_status="completed" if executed else "failed",
        outcome_status=outcome,
        failure_domain=failure_domain,
        failure_code=failure_code,
        retriable=False,
        metrics={
            "finite_fraction": finite,
            "runtime_s": float(result.get("wall_s", report.get("wall_s", 0.0))),
            "gpu_seconds": float(result.get("wall_s", report.get("wall_s", 0.0))),
            "penetration_m": float(metrics.get("penetration_m") or 0.0),
            "behavior_pass": bool(report.get("behavior_pass", False)),
        },
        outcomes={"contract_pass": contract_pass, "runtime_pass": runtime_pass},
        measurements={
            "finite_fraction": finite,
            "runtime_s": float(result.get("wall_s", report.get("wall_s", 0.0))),
            "contact_fraction": metrics.get("contact_fraction"),
            "settled": metrics.get("settled"),
            "deformation_ratio": metrics.get("deformation_ratio"),
            "recovery_ratio": metrics.get("recovery_ratio"),
            "probe_id": str(report.get("probe", {}).get("probe_id") if isinstance(report.get("probe"), Mapping) else E1_RUNTIME_PROBE["probe_id"]),
            "diagnosis_kind": diagnosis.get("kind"),
            "solver_route": metrics.get("mode") or ((report.get("provenance") or {}).get("executed_solver") if isinstance(report.get("provenance"), Mapping) else None),
            "coupler": "LegacyCoupler",
            "ipc_used": False,
            "uipc_used": False,
            "limitations": report.get("limitations") or [],
        },
        extra={"executor_id": "softsim_genesis_runner:run_episode", "synthetic_formula_used": False, "claim_boundary": "frozen Genesis v12.6 proxy contract; not measured material ground truth"},
    )


def _bounded_rule_update(candidate: Mapping[str, float], residual_report: Mapping[str, Any]) -> dict[str, float]:
    """Deterministic RuleTune-style update from calibration residuals only."""
    updated = dict(candidate)
    deform = residual_report.get("deformation_ratio") if isinstance(residual_report, Mapping) else None
    if isinstance(deform, Mapping):
        cand = float(deform.get("candidate", 0.0))
        ref = float(deform.get("reference", 0.0))
        if ref > 0 and "youngs_modulus_pa" in updated:
            ratio = min(4.0, max(0.25, cand / ref))
            updated["youngs_modulus_pa"] = float(updated["youngs_modulus_pa"]) * ratio
    settle = residual_report.get("settled") if isinstance(residual_report, Mapping) else None
    if isinstance(settle, Mapping) and "density_kg_m3" in updated:
        if not bool(settle.get("candidate", True)) and bool(settle.get("reference", True)):
            updated["density_kg_m3"] = float(updated["density_kg_m3"]) * 0.8
    return {key: round(float(value), 8) for key, value in updated.items()}


class _Abstain(RuntimeError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


_PRIMES = (2, 3, 5, 7)


def _halton(index: int, base: int) -> float:
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def _param_bounds(name: str, base: float) -> tuple[float, float]:
    """Draft identifiability-box bounds; the formal E2.2 box freezes at the
    parameter-response gate."""
    value = float(base)
    if 0.0 < value < 0.5:
        return (0.05, 0.49)
    return (max(value / 4.0, 1e-9), value * 4.0)


def _map_unit(base: float, lo: float, hi: float, u: float) -> float:
    if 0.0 < float(base) < 0.5:
        return round(lo + u * (hi - lo), 8)
    return round(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))), 8)


def _sobol_candidates(prior: Mapping[str, float], mutable: Sequence[str], budget: int, seed: int) -> list[dict[str, float]]:
    """Open-loop seeded low-discrepancy schedule; first candidate is the prior
    (protocol: every adaptive method starts from the same one-shot prior)."""
    candidates = [dict(prior)]
    if budget <= 1 or not mutable:
        return candidates
    unit: list[list[float]] = []
    try:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=len(mutable), scramble=True, seed=int(seed))
        count = budget - 1
        unit = [list(map(float, row)) for row in sampler.random_base2(m=max(1, math.ceil(math.log2(count))))[:count]]
    except Exception:
        unit = [[_halton(i + 1, _PRIMES[j % len(_PRIMES)]) for j in range(len(mutable))] for i in range(budget - 1)]
    for row in unit:
        candidate = dict(prior)
        for j, name in enumerate(mutable):
            lo, hi = _param_bounds(name, prior[name])
            candidate[name] = _map_unit(prior[name], lo, hi, row[j])
        candidates.append(candidate)
    return candidates


_VLM_METHODS = {"vid2sid_style_reimplementation", "swim2real_style_reimplementation"}


def _vlm_propose(prior: Mapping[str, float], candidate: Mapping[str, float], history: Sequence[Mapping[str, Any]], mutable: Sequence[str], attempt_dir: Path, index: int) -> dict[str, float]:
    """Vid2Sid/Swim2Real-style VLM update via the repo's Qwen transport.

    Structured JSON only (no images), full calibration history in, bounded
    patch out.  Unavailable key/transport -> _Abstain (uncertain), never a
    silent heuristic fallback.
    """
    if not (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY") or os.environ.get("QWEN_VL_API_KEY")):
        raise _Abstain("vlm_proposer_unavailable", "no Qwen/DashScope key in environment")
    transport = Path(os.environ.get("SOFTSIM_QWEN_TRANSPORT") or (_REPO_SCRIPTS / "qwen_proposer_transport.py"))
    if not transport.is_file():
        raise _Abstain("vlm_proposer_unavailable", f"transport script missing: {transport}")
    work = attempt_dir / "vlm"
    work.mkdir(parents=True, exist_ok=True)
    request_path = work / f"request-{index:02d}.json"
    response_path = work / f"response-{index:02d}.json"
    atomic_write_json(request_path, {
        "schema_name": "softsim.e2_vlm_propose_request",
        "schema_version": 1,
        "prior": dict(prior),
        "candidate": dict(candidate),
        "calibration_history": [dict(item) for item in history],
        "mutable_parameters": list(mutable),
    })
    completed = subprocess.run(
        [sys.executable, str(transport), "--request", str(request_path), "--response", str(response_path), "--model", os.environ.get("SOFTSIM_QWEN_MODEL", "qwen3.8-max")],
        check=False, capture_output=True, text=True, timeout=300, env=_bridge_env(),
    )
    if completed.returncode != 0 or not response_path.is_file():
        raise _Abstain("vlm_proposer_unavailable", f"transport exit {completed.returncode}")
    response = read_json(response_path)
    patch = response.get("patch") if isinstance(response, Mapping) else None
    if not isinstance(patch, Mapping) or not patch:
        raise _Abstain("vlm_proposer_unavailable", "empty/invalid patch")
    updated = dict(candidate)
    for name, value in patch.items():
        if name not in mutable or not isinstance(value, (int, float)):
            continue
        lo, hi = _param_bounds(name, float(updated.get(name, value)))
        updated[name] = round(min(hi, max(lo, float(value))), 8)
    return updated


def _e2_record(
    task: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    attempt_dir: Path,
    mesh: Path,
    target_provider: Any,
    *,
    route: Mapping[str, Any] | None = None,
    numerics: Mapping[str, Any] | None = None,
    bundle_parameters: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    experiment_id = str(task.get("experiment_id") or run_manifest.get("experiment_id"))
    route = dict(route or {})
    numerics = dict(numerics or {})
    if target_provider is None:
        record = _failure_record(task, run_manifest, failure_code="hidden_target_provider_missing", execution_status="invalid", outcome_status="blocked", retriable=False, detail="formal E2 requires an evaluator-side target provider", experiment_id=experiment_id)
        return record, [], []
    try:
        provided = target_provider(dict(task))
        if not isinstance(provided, Mapping) or not provided:
            raise TypeError("empty target")
        evaluator_parameters = {str(key): float(value) for key, value in provided.items()}
    except Exception as exc:
        record = _failure_record(task, run_manifest, failure_code="hidden_target_provider_error", execution_status="invalid", outcome_status="blocked", retriable=False, detail=type(exc).__name__, experiment_id=experiment_id)
        return record, [], []
    method = str(task.get("method_id", "fixed_profile"))
    prior = dict(task.get("prior_parameters") or bundle_parameters or _task_parameters(task))
    budget = max(1, int(task.get("candidate_budget", 8)))
    mutable = [name for name in (task.get("mutable_parameters") or list(prior)) if name in prior][:4]
    tolerance = float(task.get("calibration_tolerance", 0.05))
    steps: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    candidate = dict(prior)
    candidate_count = 0
    best: tuple[float, dict[str, float]] | None = None
    calibration_probe = dict(E2_CALIBRATION_PROBE)
    calibration_probe["probe_id"] = str(task.get("calibration_probe_id") or calibration_probe["probe_id"])
    heldout_probe = dict(E2_HELDOUT_PROBE)
    heldout_probe["probe_id"] = str(task.get("heldout_probe_id") or heldout_probe["probe_id"])
    probe_kwargs = {
        "mesh_path": mesh.as_posix(),
        "route": dict(route),
        "numerics": dict(numerics),
        "sample_id": str(task.get("sample_id") or task["task_id"]),
    }

    def _evaluate(params: Mapping[str, float], probe: Mapping[str, Any], label: str) -> tuple[float, dict[str, Any], dict[str, Any] | None]:
        """One candidate rollout + cached evaluator-side reference; loss only."""
        result = _run_bridge_probe(attempt_dir, label=label, parameters=params, probe=probe, **probe_kwargs)
        if result.get("status") != "ok" or str((result.get("report") or {}).get("execution_status")) != "completed":
            return 1.0, {}, result
        reference = _run_bridge_probe(attempt_dir / "evaluator_sidecar", label=label, parameters=evaluator_parameters, probe=probe, **probe_kwargs)
        if reference.get("status") != "ok" or str((reference.get("report") or {}).get("execution_status")) != "completed":
            return 1.0, {}, reference
        loss, residual = _report_behavior_loss(result["report"], reference["report"])
        return loss, residual, None

    # Method dispatch (E2.2 frozen matrix, EXPERIMENT_PLAN §4.4):
    #   A fixed_profile / B one_shot_prior: prior only, no feedback;
    #   C ruletune: structured residual -> frozen mechanical rules;
    #   D seeded_sobol: open-loop low-discrepancy schedule over the box;
    #   F/G VLM-style: full history -> bounded patch (G adds beta=0.5 backtrack);
    #   H full_numeric_agent: rule update + backtrack + abstain on flat response.
    sobol_schedule = _sobol_candidates(prior, mutable, budget, seed=int(task.get("seed", 0))) if method == "seeded_sobol" else None
    history: list[dict[str, Any]] = []
    backtracks = 0
    stalls = 0
    abstain: _Abstain | None = None
    for index in range(budget):
        loss, residual, error = _evaluate(candidate, calibration_probe, f"calibration-{index:02d}")
        candidate_count += 1
        if error is not None:
            steps.append({
                "schema_name": "softsim.agent_step",
                "schema_version": 1,
                "task_id": task["task_id"],
                "step_index": index,
                "phase": "calibration",
                "candidate_hash": fingerprint(candidate),
                "candidate_parameters": dict(candidate),
                "probe_id": calibration_probe["probe_id"],
                "feedback_source": "calibration_only",
                "heldout_feedback_used": False,
                "simulator_evaluation": True,
                "loss": 1.0,
                "passed": False,
                "executor_error": str(error.get("error_type")),
            })
            calls.append(make_call_record(run_manifest=run_manifest, task={**task, "call_id": f"{task['task_id']}-cal-{index:02d}", "episode_id": task["task_id"]}, execution_status="failed", outcome_status="failure", failure_domain="infrastructure", failure_code=str(error.get("error_type", "genesis_error")), metrics={"loss": 1.0}, measurements={"phase": "calibration", "feedback_allowed": True}, extra={"call_index": candidate_count - 1, "heldout_feedback_used": False}))
            continue
        passed = bool(loss <= tolerance)
        steps.append({
            "schema_name": "softsim.agent_step",
            "schema_version": 1,
            "task_id": task["task_id"],
            "step_index": index,
            "phase": "calibration",
            "candidate_hash": fingerprint(candidate),
            "candidate_parameters": dict(candidate),
            "probe_id": calibration_probe["probe_id"],
            "feedback_source": "calibration_only",
            "heldout_feedback_used": False,
            "simulator_evaluation": True,
            "loss": round(loss, 8),
            "passed": passed,
        })
        calls.append(make_call_record(run_manifest=run_manifest, task={**task, "call_id": f"{task['task_id']}-cal-{index:02d}", "episode_id": task["task_id"]}, execution_status="completed", outcome_status="success" if passed else "failure", failure_domain="none" if passed else "task", failure_code=None if passed else "calibration_behavior_mismatch", metrics={"loss": round(loss, 8)}, measurements={"phase": "calibration", "feedback_allowed": True}, extra={"call_index": candidate_count - 1, "heldout_feedback_used": False}))
        history.append({"candidate": dict(candidate), "loss": round(loss, 8), "residual_metrics": residual})
        improved = best is None or loss < best[0]
        if improved:
            best = (loss, dict(candidate))
            stalls = 0
        else:
            stalls += 1
        if passed or method in {"fixed_profile", "one_shot_prior"}:
            break
        if method == "seeded_sobol":
            assert sobol_schedule is not None
            if index + 1 >= len(sobol_schedule):
                break
            candidate = dict(sobol_schedule[index + 1])
            continue
        try:
            if method == "ruletune":
                candidate = _bounded_rule_update(candidate, residual)
            elif method in _VLM_METHODS:
                proposed = _vlm_propose(prior, candidate, history, mutable, attempt_dir, index)
                if method == "swim2real_style_reimplementation" and best is not None and not improved and backtracks < 3:
                    # beta=0.5 backtrack toward the best-so-far candidate.
                    backtracks += 1
                    proposed = {name: round(float(best[1].get(name, proposed.get(name))) + 0.5 * (float(proposed.get(name, best[1].get(name, 0.0))) - float(best[1].get(name, 0.0))), 8) for name in proposed}
                candidate = proposed
            elif method == "full_numeric_agent":
                if stalls >= 2:
                    raise _Abstain("abstain_non_identifiable", "two consecutive updates improved calibration loss by nothing; parameter response is flat")
                if not improved and best is not None:
                    backtracks += 1
                    proposed = _bounded_rule_update(best[1], residual)
                    candidate = {name: round(float(best[1].get(name, proposed.get(name))) + 0.5 * (float(proposed.get(name, best[1].get(name, 0.0))) - float(best[1].get(name, 0.0))), 8) for name in proposed}
                else:
                    candidate = _bounded_rule_update(candidate, residual)
            else:
                break
        except _Abstain as exc:
            abstain = exc
            break
    if abstain is not None:
        steps.append({
            "schema_name": "softsim.agent_step",
            "schema_version": 1,
            "task_id": task["task_id"],
            "step_index": len(steps),
            "phase": "abstain",
            "candidate_hash": fingerprint(best[1] if best else candidate),
            "candidate_parameters": dict(best[1] if best else candidate),
            "feedback_source": "none",
            "heldout_feedback_used": False,
            "feedback_allowed": False,
            "reason": abstain.code,
        })
        frozen_for_record = dict(best[1]) if best else dict(prior)
        record = make_episode_record(
            run_manifest=run_manifest,
            task=task,
            execution_status="completed",
            outcome_status="failure",
            failure_domain="policy",
            failure_code=abstain.code,
            retriable=False,
            metrics={"candidate_count": candidate_count},
            measurements={
                "candidate_count": candidate_count,
                "prior_parameters": prior,
                "final_parameters": frozen_for_record,
                "accepted_first_pass": False,
                "correction_success": False,
                "target_handle": task.get("hidden_target_handle"),
                "target_access_policy": task.get("target_access_policy", "evaluator_sidecar_required"),
                "abstain_detail": abstain.detail,
            },
            extra={
                "executor_id": "softsim_genesis_runner:run_episode",
                "synthetic_formula_used": False,
                "heldout_feedback_used": False,
                "prior_parameters": prior,
                "final_parameters": frozen_for_record,
                "candidate_count": candidate_count,
                "agent_status": "uncertain_human_escalation",
            },
        )
        return record, steps, calls
        record = _failure_record(task, run_manifest, failure_code="genesis_calibration_failed", failure_domain="infrastructure", detail="no successful calibration evaluation", experiment_id=experiment_id)
        return record, steps, calls
    calibration_loss, frozen_candidate = best
    steps.append({
        "schema_name": "softsim.agent_step",
        "schema_version": 1,
        "task_id": task["task_id"],
        "step_index": len(steps),
        "phase": "candidate_frozen",
        "candidate_hash": fingerprint(frozen_candidate),
        "candidate_parameters": dict(frozen_candidate),
        "feedback_source": "none",
        "heldout_feedback_used": False,
        "feedback_allowed": False,
    })
    heldout_loss, _residual, heldout_error = _evaluate(frozen_candidate, heldout_probe, "heldout")
    if heldout_error is not None:
        record = _failure_record(task, run_manifest, failure_code="genesis_heldout_failed", failure_domain="infrastructure", detail=str(heldout_error.get("error_type")), experiment_id=experiment_id)
        steps.append({
            "schema_name": "softsim.agent_step",
            "schema_version": 1,
            "task_id": task["task_id"],
            "step_index": len(steps),
            "phase": "heldout_verification",
            "candidate_hash": fingerprint(frozen_candidate),
            "candidate_parameters": dict(frozen_candidate),
            "probe_id": heldout_probe["probe_id"],
            "feedback_source": "heldout_no_feedback",
            "heldout_feedback_used": False,
            "feedback_allowed": False,
            "simulator_evaluation": True,
            "passed": False,
            "executor_error": str(heldout_error.get("error_type")),
        })
        return record, steps, calls
    heldout_pass = bool(heldout_loss <= 0.08)
    steps.append({
        "schema_name": "softsim.agent_step",
        "schema_version": 1,
        "task_id": task["task_id"],
        "step_index": len(steps),
        "phase": "heldout_verification",
        "candidate_hash": fingerprint(frozen_candidate),
        "candidate_parameters": dict(frozen_candidate),
        "probe_id": heldout_probe["probe_id"],
        "feedback_source": "heldout_no_feedback",
        "heldout_feedback_used": False,
        "feedback_allowed": False,
        "simulator_evaluation": True,
        "loss": round(heldout_loss, 8),
        "passed": heldout_pass,
    })
    calls.append(make_call_record(run_manifest=run_manifest, task={**task, "call_id": f"{task['task_id']}-heldout", "episode_id": task["task_id"]}, execution_status="completed", outcome_status="success" if heldout_pass else "failure", failure_domain="none" if heldout_pass else "task", failure_code=None if heldout_pass else "heldout_behavior_mismatch", metrics={"loss": round(heldout_loss, 8)}, measurements={"phase": "heldout_verification", "feedback_allowed": False}, extra={"call_index": candidate_count, "heldout_feedback_used": False}))
    first_passed = bool(steps and steps[0].get("passed"))
    accepted_first_pass = bool(first_passed and candidate_count == 1 and heldout_pass)
    correction_success = bool(candidate_count > 1 and heldout_pass)
    measurements = {
        "heldout_behavior_error": round(heldout_loss, 8),
        "calibration_behavior_error": round(calibration_loss, 8),
        "candidate_count": candidate_count,
        "prior_parameters": prior,
        "final_parameters": frozen_candidate,
        "accepted_first_pass": accepted_first_pass,
        "correction_success": correction_success,
        "target_handle": task.get("hidden_target_handle"),
        "target_access_policy": task.get("target_access_policy", "evaluator_sidecar_required"),
    }
    record = make_episode_record(
        run_manifest=run_manifest,
        task=task,
        execution_status="completed",
        outcome_status="success" if heldout_pass else "failure",
        failure_domain="none" if heldout_pass else "task",
        failure_code=None if heldout_pass else "heldout_behavior_mismatch",
        retriable=False,
        metrics={"heldout_behavior_error": round(heldout_loss, 8), "calibration_behavior_error": round(calibration_loss, 8), "candidate_count": candidate_count},
        measurements=measurements,
        extra={
            "executor_id": "softsim_genesis_runner:run_episode",
            "synthetic_formula_used": False,
            "heldout_feedback_used": False,
            "calibration_probe_id": task.get("calibration_probe_id"),
            "heldout_probe_id": task.get("heldout_probe_id"),
            "prior_parameters": prior,
            "final_parameters": frozen_candidate,
            "candidate_count": candidate_count,
            "agent_status": "accepted_first_pass" if accepted_first_pass else ("corrected_then_accepted" if correction_success else "rejected_model_limit"),
        },
    )
    return record, steps, calls


def _e3a_record(task: Mapping[str, Any], run_manifest: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(result.get("metrics") or {})
    damage = float(metrics.get("damage_fraction", 0.0))
    integrity = float(metrics.get("integrity", 1.0 - damage))
    slip = metrics.get("slip")
    task_success = bool(metrics.get("task_success", False))
    intact = bool(task_success and integrity >= 0.75)
    window = task.get("safe_force_window") if isinstance(task.get("safe_force_window"), Mapping) else metrics.get("safe_force_window")
    if not isinstance(window, Mapping):
        window = {"low_n": None, "high_n": None, "width_n": 0.0, "non_empty": False}
    failure_code = None
    failure_domain = "none"
    outcome = "success" if task_success else "failure"
    if not task_success:
        damage_failure = bool(metrics.get("damage_failure", damage >= 0.25))
        failure_domain = "damage" if damage_failure else "task"
        failure_code = "damage_failure" if damage_failure else str(metrics.get("failure_reason") or "slip_before_hold")
    outcomes = {
        "task_success": task_success,
        "grasp_success": bool(metrics.get("grasp_success", False)),
        "lift_success": bool(metrics.get("lift_success", False)),
        "hold_success": bool(metrics.get("hold_success", False)),
        "intact_success": intact,
        "damage_failure": bool(metrics.get("damage_failure", damage >= 0.25)),
    }
    measurements = {
        "damage_fraction": round(damage, 8),
        "integrity": round(integrity, 8),
        "safe_force_window": dict(window),
        "slip": slip,
        "contact_count": int(metrics.get("contact_count", 0)),
        "convergence": bool(metrics.get("convergence", False)),
        "finite_fraction": float(metrics.get("finite_fraction", 0.0)),
        "peak_contact_force_n": metrics.get("peak_contact_force_n"),
        "retained_lifted_mass_fraction": metrics.get("retained_lifted_mass_fraction"),
        "residual_set": metrics.get("residual_set"),
        "drop_or_slip": bool(metrics.get("drop_or_slip", False)),
        "damage_model_id": DAMAGE_MODEL_ID,
        "contact_force_status": CONTACT_FORCE_STATUS,
    }
    return make_episode_record(
        run_manifest=run_manifest,
        task=task,
        execution_status="completed",
        outcome_status=outcome,
        failure_domain=failure_domain,
        failure_code=failure_code,
        retriable=False,
        metrics={"task_success": task_success, "damage_fraction": round(damage, 8), "integrity": round(integrity, 8), "slip": slip, "convergence": bool(metrics.get("convergence", False))},
        outcomes=outcomes,
        measurements=measurements,
        extra={
            "executor_id": "softsim_genesis_runner:run_episode",
            "synthetic_formula_used": False,
            "damage_model_id": DAMAGE_MODEL_ID,
            "commanded_force_n": task.get("force_n"),
            "measured_contact_force_n": None,
            "contact_force_status": CONTACT_FORCE_STATUS,
        },
    )


def run_episode(
    task: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    attempt_dir: str | Path,
    target_provider: Any | None = None,
) -> Any:
    """Formal episode executor entry point (``module:function`` contract).

    Returns a canonical episode record, or ``(record, agent_steps, calls)``
    for E2.  Never falls back to a deterministic formula backend: any runtime
    problem becomes a terminal failure/blocked record.
    """
    attempt_path = Path(attempt_dir)
    attempt_path.mkdir(parents=True, exist_ok=True)
    experiment_id = str(task.get("experiment_id") or run_manifest.get("experiment_id"))
    if experiment_id.startswith("e2") and target_provider is None:
        # The evaluator-side target boundary is a contract violation, so it is
        # reported before any infrastructure (mesh/runtime) failure.
        record = _failure_record(task, run_manifest, failure_code="hidden_target_provider_missing", execution_status="invalid", outcome_status="blocked", retriable=False, detail="formal E2 requires an evaluator-side target provider", experiment_id=experiment_id)
        return record, [], []
    try:
        inputs = _resolve_inputs(task, run_manifest)
    except ExecutorError as exc:
        record = _failure_record(task, run_manifest, failure_code=str(exc), failure_domain="infrastructure" if str(exc) == "mesh_unresolved" else "contract", detail="input resolution failed; see genesis_job context", experiment_id=experiment_id)
        return (record, [], []) if experiment_id.startswith("e2") else record
    mesh = Path(inputs["mesh_path"])
    if experiment_id.startswith("e2"):
        return _e2_record(task, run_manifest, attempt_path, mesh, target_provider, route=inputs["route"], numerics=inputs["numerics"], bundle_parameters=inputs["parameters"])
    if experiment_id == "e3a":
        numerics_e3a = dict(inputs["numerics"])
        # Grasp starts from rest contact, not the drop-probe height.
        numerics_e3a["drop_height_m"] = float(numerics_e3a.get("grasp_spawn_height_m", 0.005))
        job = {
            "schema_name": "softsim.genesis_runner_job",
            "schema_version": 1,
            "experiment_id": experiment_id,
            "mesh_path": mesh.as_posix(),
            "seed": int(task.get("seed", 0)),
            "numerics": numerics_e3a,
            "parameters": inputs["parameters"],
            "route": inputs["route"],
            "scale": inputs["scale"],
            "kind": "e3a_grasp",
            "method_id": str(task.get("method_id") or "fixed"),
            "sample_id": str(task.get("sample_id") or ""),
            "force_n": float(task.get("force_n", 15.0)),
            "fragility_regime": str(task.get("fragility_regime", "medium")),
            "condition_id": str(task.get("condition_id", "medium")),
        }
        if task.get("demo_log"):
            # Track L data interface: per-step canonical observations/actions
            # (1024-pt cloud, centroid, EE pose, gripper width, 7D delta action).
            job["demo_log"] = (attempt_path / "demo.jsonl").as_posix()
        result = _run_worker(job, attempt_path)
        if result.get("status") != "ok":
            return _failure_record(task, run_manifest, failure_code=str(result.get("error_type") or "genesis_error"), failure_domain="infrastructure", detail=str(result.get("detail", "genesis worker failed"))[:400], experiment_id=experiment_id)
        return _e3a_record(task, run_manifest, result)
    if experiment_id == "e1-task":
        # E1-C stratified task audit: drop/compress replay through the
        # production bridge; drape/push/pinch_lift/place run the worker probes.
        probe_kind = str(task.get("task_probe") or "drop")
        if probe_kind in {"drop", "compress"}:
            probe = dict(E1_RUNTIME_PROBE if probe_kind == "drop" else E2_HELDOUT_PROBE)
            probe["probe_id"] = f"e1.task.{probe_kind}.v126"
            result = _run_bridge_probe(
                attempt_path,
                label=f"e1_task_{probe_kind}",
                mesh_path=mesh.as_posix(),
                parameters=inputs["parameters"],
                route=inputs["route"],
                numerics=inputs["numerics"],
                probe=probe,
                sample_id=str(task.get("sample_id") or task["task_id"]),
            )
            if result.get("status") != "ok":
                return _failure_record(task, run_manifest, failure_code=str(result.get("error_type") or "genesis_error"), failure_domain="infrastructure", detail=str(result.get("detail", "bridge probe failed"))[:400], experiment_id=experiment_id)
            record = _e1_record(task, run_manifest, result)
            record["task_probe"] = probe_kind
            from softsim_formal.records import record_hash as _rh

            record["record_sha256"] = _rh(record)
            return record
        worker_kind = {"drape": "e1_task_drape", "push": "e1_task_push", "pinch_lift": "e1_task_pinch_lift", "place": "e1_task_place"}.get(probe_kind)
        if worker_kind is None:
            return _failure_record(task, run_manifest, failure_code="unsupported_task_probe", failure_domain="contract", retriable=False, detail=f"unknown task probe {probe_kind!r}", experiment_id=experiment_id)
        numerics_task = dict(inputs["numerics"])
        # Worker task probes start from rest contact; the drop height in the
        # frozen row numerics targets the drop probe only.
        numerics_task["drop_height_m"] = float(numerics_task.get("task_spawn_height_m", 0.005))
        job = {
            "schema_name": "softsim.genesis_runner_job",
            "schema_version": 1,
            "experiment_id": experiment_id,
            "mesh_path": mesh.as_posix(),
            "seed": int(task.get("seed", 0)),
            "numerics": numerics_task,
            "settle_steps": 240,
            "parameters": inputs["parameters"],
            "route": inputs["route"],
            "scale": inputs["scale"],
            "kind": worker_kind,
            "probe_id": f"e1.task.{probe_kind}.v126",
        }
        result = _run_worker(job, attempt_path)
        if result.get("status") != "ok":
            return _failure_record(task, run_manifest, failure_code=str(result.get("error_type") or "genesis_error"), failure_domain="infrastructure", detail=str(result.get("detail", "genesis worker failed"))[:400], experiment_id=experiment_id)
        metrics = dict(result.get("metrics") or {})
        finite = float(metrics.get("finite_fraction") or 0.0)
        task_pass = bool(metrics.get("task_pass", False))
        outcome = "success" if (finite >= 1.0 and task_pass) else "failure"
        record = make_episode_record(
            run_manifest=run_manifest,
            task=task,
            execution_status="completed",
            outcome_status=outcome,
            failure_domain="none" if outcome == "success" else ("numerical" if finite < 1.0 else "task"),
            failure_code=None if outcome == "success" else ("nan_or_instability" if finite < 1.0 else "task_probe_fail"),
            retriable=False,
            metrics={"finite_fraction": finite, "runtime_s": float(result.get("wall_s", 0.0)), "task_pass": task_pass},
            outcomes={"contract_pass": task_pass, "runtime_pass": finite >= 1.0},
            measurements=metrics,
            extra={"executor_id": "softsim_genesis_runner:run_episode", "synthetic_formula_used": False, "task_probe": probe_kind, "claim_boundary": "frozen Genesis v12.6 proxy contract; not measured material ground truth"},
        )
        return record
    # E1: replay the frozen v12.6 runtime probe through the production bridge.
    result = _run_bridge_probe(
        attempt_path,
        label="e1_runtime",
        mesh_path=mesh.as_posix(),
        parameters=inputs["parameters"],
        route=inputs["route"],
        numerics=inputs["numerics"],
        probe=E1_RUNTIME_PROBE,
        sample_id=str(task.get("sample_id") or task["task_id"]),
    )
    if result.get("status") != "ok":
        return _failure_record(task, run_manifest, failure_code=str(result.get("error_type") or "genesis_error"), failure_domain="infrastructure", detail=str(result.get("detail", "bridge probe failed"))[:400], experiment_id=experiment_id)
    return _e1_record(task, run_manifest, result)
