#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Explicit Genesis/MPM bridge contract for the server phase.

The legacy stage3 runner proves execution and mesh preparation, but it does not
emit a behavior contract.  This bridge therefore refuses to report a behavior
pass unless the selected runner supplies a boolean ``behavior_pass``, non-empty
metrics, and a diagnosis object.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


def _read(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _write_unverified(report_path: Path, probe: dict[str, Any], detail: str, *, execution_status: str = "not_executed") -> int:
    report = {
        "schema_name": "softsim.probe_report",
        "schema_version": 2,
        "backend": "genesis",
        "probe": probe,
        "execution_status": execution_status,
        "behavior_pass": False,
        "passed": False,
        "explicit_behavior_evidence": False,
        "metrics": {},
        "diagnosis": {"kind": "behavior_unverified", "suggested_updates": {}, "detail": detail},
        "outputs": {},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="SoftSim Genesis behavior-evidence bridge")
    parser.add_argument("--asset-json", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--probe-json", required=True)
    parser.add_argument("--simulator-config-json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()
    asset = _read(args.asset_json)
    candidate = _read(args.candidate_json)
    probe = _read(args.probe_json)
    simulator_config = _read(args.simulator_config_json) if args.simulator_config_json else {}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_kind = str((probe.get("stimulus") or {}).get("kind") or "")
    generic_runner = os.environ.get("SOFTSIM_GENESIS_BEHAVIOR_RUNNER")
    drop_runner = os.environ.get("GENESIS_MPM_DROP_RUNNER")
    raw_report_path = output_dir / "genesis" / "report.json"
    if generic_runner:
        command = [
            os.environ.get("GENESIS_PYTHON", "python3"), generic_runner,
            "--asset-json", args.asset_json,
            "--candidate-json", args.candidate_json,
            "--probe-json", args.probe_json,
            "--output-dir", str(output_dir / "genesis"),
            "--report-json", str(raw_report_path),
        ]
        if args.simulator_config_json:
            command.extend(["--simulator-config-json", args.simulator_config_json])
    elif probe_kind == "drop" and drop_runner:
        command = None
    else:
        return _write_unverified(Path(args.report_json), probe, f"no runner registered for probe stimulus {probe_kind!r}; set SOFTSIM_GENESIS_BEHAVIOR_RUNNER for non-drop probes")
    mesh_ref = ((asset.get("artifact_refs") or {}).get("collision_mesh") or {})
    mesh = mesh_ref.get("path")
    if not mesh:
        mesh = (asset.get("resolved_artifacts") or {}).get("collision_mesh")
    elif not Path(str(mesh)).is_absolute():
        alias = str(mesh_ref.get("root_alias") or "source")
        env_name = "SOFTSIM_ROOT_" + re.sub(r"[^A-Za-z0-9]", "_", alias).upper()
        root = os.environ.get(env_name) or os.environ.get("SOFTSIM_ASSET_ROOT")
        if root:
            mesh = str(Path(root) / str(mesh))
    fixed = asset.get("fixed_solver_numerics") or (asset.get("prior") or {}).get("fixed_solver_numerics") or {}
    route = asset.get("route") if isinstance(asset.get("route"), dict) else {}
    def value(name: str, default: Any) -> Any:
        raw = fixed.get(name, default)
        return raw.get("value", default) if isinstance(raw, dict) else raw
    if command is None:
        route_solver = str(route.get("solver_adapter") or route.get("model_family") or "mpm").lower()
        default_solver = "pbd-cloth" if "cloth" in route_solver else "mpm"
        command = [
            os.environ.get("GENESIS_PYTHON", "python3"), drop_runner,
            "--mesh", str(mesh or ""),
            "--output-dir", str(output_dir / "genesis"),
            "--sample-id", str(asset.get("sample_id", "agent")),
            "--backend", str(os.environ.get("GENESIS_BACKEND", "gpu")),
            "--solver", str(os.environ.get("GENESIS_SOLVER", default_solver)),
            "--proxy-mode", str(value("proxy_mode", "alpha-wrap")),
            "--material-preset", "none",
            "--youngs-modulus", str(candidate.get("youngs_modulus_pa", 50000.0)),
            "--poisson-ratio", str(candidate.get("poisson_ratio", 0.35)),
            "--density", str(candidate.get("density_kg_m3", 600.0)),
            "--particle-size", str(value("particle_size_m", 0.01)),
            "--grid-density", str(value("grid_density", 96.0)),
            "--dt", str(value("dt_s", 0.001)),
            "--substeps", str(int(value("substeps", 16))),
            "--drop-height", str(value("drop_height_m", 0.2)),
            "--asset-size", str(value("scale_m", 0.32)),
            "--orientation-preset", str(os.environ.get("GENESIS_ORIENTATION_PRESET", "largest-face-down")),
            "--visual-source", "proxy",
            "--renderer", "rasterizer",
            "--overwrite",
        ]
    try:
        process = subprocess.run(command, check=False, text=True, capture_output=True, timeout=float(os.environ.get("GENESIS_PROBE_TIMEOUT_SEC", "1800")))
    except subprocess.TimeoutExpired as exc:
        return _write_unverified(Path(args.report_json), probe, f"runner timeout: {exc}", execution_status="failed")
    except OSError as exc:
        return _write_unverified(Path(args.report_json), probe, f"runner start failure: {exc}", execution_status="failed")
    raw = _read(str(raw_report_path)) if raw_report_path.is_file() else {}
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
    behavior = raw.get("behavior_pass")
    if behavior is None:
        behavior = metrics.get("behavior_pass")
    diagnosis = raw.get("diagnosis") if isinstance(raw.get("diagnosis"), dict) else {}
    explicit = isinstance(behavior, bool) and bool(metrics) and bool(diagnosis)
    raw_status = str(raw.get("execution_status") or "")
    if explicit and raw_status == "not_executed":
        # The runner executed cleanly but refused the probe (e.g. unsupported
        # stimulus); propagate its status instead of conflating it with a crash.
        execution_status = "not_executed"
    elif explicit and process.returncode == 0:
        execution_status = "completed"
    else:
        execution_status = "sim_executed_unverified" if process.returncode == 0 else "failed"
    payload = {
        "schema_name": "softsim.probe_report",
        "schema_version": 2,
        "backend": str(raw.get("backend") or "genesis_mpm_drop"),
        "probe": probe,
        "execution_status": execution_status,
        "return_code": process.returncode,
        "behavior_pass": bool(behavior) if explicit and process.returncode == 0 else False,
        "passed": bool(behavior) if explicit and process.returncode == 0 else False,
        "explicit_behavior_evidence": explicit,
        "metrics": metrics,
        "diagnosis": diagnosis if explicit else {"kind": "behavior_unverified", "suggested_updates": {}, "detail": "Genesis report lacks explicit behavior evidence"},
        "outputs": {"raw_report": "genesis/report.json" if raw_report_path.is_file() else None},
        "stdout_tail": process.stdout[-4000:],
        "stderr_tail": process.stderr[-4000:],
        "simulator_config": simulator_config,
    }
    payload["outputs"] = {key: value for key, value in payload["outputs"].items() if value}
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # A completed probe whose behavior contract failed is still a completed
    # execution; non-zero exits are reserved for genuine executor failures so
    # the caller does not misclassify behavior failure as numerical failure.
    return 0 if payload["execution_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
