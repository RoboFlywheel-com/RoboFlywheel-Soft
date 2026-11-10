# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .io import atomic_write_json, content_hash, read_json, file_sha256
from .models import ProbeResult, ProbeSpec
from .probes import probe_record
from .routing import physical_values


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value and abs(float(value)) != float("inf")


def _probe_payload(probe: ProbeSpec | Mapping[str, Any]) -> dict[str, Any]:
    return probe_record(probe) if isinstance(probe, ProbeSpec) else dict(probe)


class MockSimulator:
    """Deterministic local simulator with explicit behavior evidence.

    It is a qualification backend only.  ``mock_target`` is hidden from the
    report's metrics but is used internally to produce a realistic diagnosis.
    """

    backend_name = "mock"

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config = dict(config or {})

    def _target(self, asset: Mapping[str, Any], candidate: Mapping[str, float]) -> dict[str, float]:
        explicit = asset.get("mock_target")
        if isinstance(explicit, Mapping):
            result = {str(k): float(v) for k, v in explicit.items() if str(k) in candidate and _finite_number(v)}
            if result:
                return result
        # A stable offset makes a prior-only fixture deterministic without storing labels in reports.
        digest = content_hash(str(asset.get("sample_id", "fixture"))).split(":", 1)[1]
        factor = 1.04 + (int(digest[:4], 16) % 9) / 100.0
        return {str(k): float(v) * factor for k, v in candidate.items()}

    def run(self, asset: Mapping[str, Any], candidate: Mapping[str, float], probe: ProbeSpec, output_dir: Path, *, simulator_config: Mapping[str, Any] | None = None) -> ProbeResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = _probe_payload(probe)
        candidate_hash = content_hash(dict(candidate))
        sim_config = dict(simulator_config or {})
        sim_config_hash = content_hash(sim_config)
        behavior_mode = str(asset.get("mock_behavior") or "normal")
        if behavior_mode in {"numerical_failure", "executor_failure"}:
            report = self._report(payload, candidate_hash, sim_config_hash, passed=False, metrics={"finite_fraction": 0.0}, diagnosis={"kind": "numerical_failure", "suggested_updates": {}}, execution_status="failed", simulator_config=sim_config)
            path = output_dir / "report.json"
            atomic_write_json(path, report)
            return ProbeResult(False, report["metrics"], report["diagnosis"], str(path), file_sha256(path), "failed", True, {}, report)
        if behavior_mode == "model_limit":
            report = self._report(payload, candidate_hash, sim_config_hash, passed=False, metrics={"finite_fraction": 1.0, "loss": 1.0, "contact_fraction": 1.0, "settled": True, "deformation_ratio": 1.0, "recovery_ratio": 0.0, "behavior_pass": False}, diagnosis={"kind": "model_limit", "suggested_updates": {}}, execution_status="completed", simulator_config=sim_config)
            path = output_dir / "report.json"
            atomic_write_json(path, report)
            return ProbeResult(False, report["metrics"], report["diagnosis"], str(path), file_sha256(path), "completed", True, {}, report)
        target = self._target(asset, candidate)
        relative_errors = {name: (target[name] - float(value)) / max(abs(target[name]), 1.0e-12) for name, value in candidate.items() if name in target}
        loss = sum(error * error for error in relative_errors.values()) / max(len(relative_errors), 1)
        tolerance = float(payload.get("tolerance", self.config.get("tolerance", 0.05)))
        passed = loss <= tolerance * tolerance
        phase = str(payload.get("stimulus", {}).get("kind", "drop"))
        # These metrics are explicit behavior observations, not merely process status.
        metrics = {
            "loss": loss,
            "relative_errors": relative_errors,
            "contact_fraction": 1.0 if phase in {"drop", "compression", "incline"} else 0.5,
            "settled": bool(loss <= max(tolerance * tolerance * 4.0, 0.01)),
            "deformation_ratio": min(1.0, max(0.0, 0.2 + loss)),
            "recovery_ratio": min(1.0, max(0.0, 1.0 - loss)),
            "finite_fraction": 1.0,
            "behavior_pass": passed,
        }
        updates = {name: target[name] - float(value) for name, value in candidate.items() if name in target}
        diagnosis = {"kind": "pass" if passed else "parameter_correctable", "suggested_updates": updates, "relative_errors": relative_errors, "mechanism": "physical_parameter_error" if not passed else "none"}
        outputs: dict[str, Any] = {}
        if self.config.get("emit_trajectory"):
            trajectory_path = output_dir / "trajectory.json"
            atomic_write_json(trajectory_path, {"sample_id": asset.get("sample_id"), "probe_id": payload.get("probe_id"), "samples": [{"step": 0, "candidate_hash": candidate_hash, "loss": loss}, {"step": 1, "candidate_hash": candidate_hash, "loss": loss}]})
            outputs["trajectory"] = "trajectory.json"
        report = self._report(payload, candidate_hash, sim_config_hash, passed=passed, metrics=metrics, diagnosis=diagnosis, execution_status="completed", simulator_config=sim_config, outputs=outputs)
        path = output_dir / "report.json"
        atomic_write_json(path, report)
        return ProbeResult(passed, metrics, diagnosis, str(path), file_sha256(path), "completed", True, outputs, report)

    @staticmethod
    def _report(probe: Mapping[str, Any], candidate_hash: str, sim_config_hash: str, *, passed: bool, metrics: dict[str, Any], diagnosis: dict[str, Any], execution_status: str, simulator_config: Mapping[str, Any] | None = None, outputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
        simulator_config = simulator_config or {}
        return {
            "schema_name": "softsim.probe_report",
            "schema_version": 2,
            "backend": "mock",
            "probe": dict(probe),
            "probe_id": probe.get("probe_id"),
            "probe_hash": str(probe.get("probe_hash") or content_hash(probe)),
            "stimulus_hash": str(probe.get("stimulus_hash") or content_hash(probe.get("stimulus", {}))),
            "action_hash": str(probe.get("action_hash") or content_hash(probe.get("action", {}))),
            "config_hash": str(probe.get("config_hash") or content_hash(probe.get("config", {}))),
            "candidate_hash": candidate_hash,
            "simulator_config_hash": sim_config_hash,
            "candidate_config_hash": simulator_config.get("candidate_config_hash"),
            "execution_status": execution_status,
            "status": "pass" if passed else "failed",
            "behavior_pass": bool(passed),
            "passed": bool(passed),
            "metrics": metrics,
            "diagnosis": diagnosis,
            "outputs": dict(outputs or {}),
        }


class CommandSimulator:
    """Run a server-supplied Genesis/command probe with strict evidence checks."""

    backend_name = "command"

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        command = self.config.get("command") or self.config.get("command_template")
        if isinstance(command, str):
            command = shlex.split(command)
        if not isinstance(command, list) or not command:
            raise ValueError("command backend requires a non-empty command list")
        self.command = [str(item) for item in command]

    def run(self, asset: Mapping[str, Any], candidate: Mapping[str, float], probe: ProbeSpec, output_dir: Path, *, simulator_config: Mapping[str, Any] | None = None) -> ProbeResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = _probe_payload(probe)
        candidate_hash = content_hash(dict(candidate))
        sim_config = dict(simulator_config or {})
        sim_config_hash = content_hash(sim_config)
        asset_path = output_dir / "asset_context.json"
        candidate_path = output_dir / "candidate.json"
        probe_path = output_dir / "probe.json"
        simulator_config_path = output_dir / "simulator_config.json"
        report_path = output_dir / "report.json"
        atomic_write_json(asset_path, dict(asset))
        atomic_write_json(candidate_path, dict(candidate))
        atomic_write_json(probe_path, payload)
        atomic_write_json(simulator_config_path, sim_config)
        replacements = {
            "{asset_json}": str(asset_path.resolve()),
            "{candidate_json}": str(candidate_path.resolve()),
            "{probe_json}": str(probe_path.resolve()),
            "{simulator_config_json}": str(simulator_config_path.resolve()),
            "{output_dir}": str(output_dir.resolve()),
            "{report_json}": str(report_path.resolve()),
            "{sample_id}": str(asset.get("sample_id", "")),
        }
        replacements["{python}"] = os.environ.get("GENESIS_PYTHON", "python3")
        replacements["{genesis_python}"] = os.environ.get("GENESIS_PYTHON", "python3")
        command = [self._replace(item, replacements) for item in self.command]
        command = [os.environ.get("GENESIS_PYTHON", "python3") if item == "${GENESIS_PYTHON}" else item for item in command]
        package_root = Path(__file__).resolve().parents[2]
        command = [
            str(package_root / item)
            if item.endswith(".py") and not Path(item).is_absolute() and (package_root / item).is_file()
            else item
            for item in command
        ]
        # Wheels ship the shared tools as a Python package beside the runners.
        tools_root = package_root / "scripts"
        if not tools_root.is_dir():
            import softsim_tools

            tools_root = Path(softsim_tools.__file__).resolve().parent
        command = [
            str(tools_root / Path(item).name)
            if item.startswith(("scripts/", "code/scripts/")) and (tools_root / Path(item).name).is_file()
            else item
            for item in command
        ]
        timeout = self.config.get("timeout_sec")
        try:
            process = subprocess.run(command, check=False, text=True, capture_output=True, timeout=float(timeout) if timeout is not None else None)
        except subprocess.TimeoutExpired as exc:
            report = self._validate_report({}, payload, candidate_hash, sim_config_hash, _failed_process(124, str(exc), "command timed out"), sim_config)
            report["execution_status"] = "timeout"
            report["diagnosis"] = {"kind": "numerical_failure", "suggested_updates": {}, "detail": "command timeout"}
            atomic_write_json(report_path, report)
            return ProbeResult(False, report.get("metrics") or {}, report["diagnosis"], str(report_path), file_sha256(report_path), "timeout", False, {}, report)
        except OSError as exc:
            report = self._validate_report({}, payload, candidate_hash, sim_config_hash, _failed_process(127, str(exc), "command could not start"), sim_config)
            report["execution_status"] = "failed"
            report["diagnosis"] = {"kind": "numerical_failure", "suggested_updates": {}, "detail": str(exc)}
            atomic_write_json(report_path, report)
            return ProbeResult(False, report.get("metrics") or {}, report["diagnosis"], str(report_path), file_sha256(report_path), "failed", False, {}, report)
        raw: dict[str, Any]
        if report_path.is_file():
            try:
                raw = read_json(report_path)
            except (OSError, ValueError) as exc:
                raw = {"parse_error": str(exc)}
        else:
            raw = {}
        result = self._validate_report(raw, payload, candidate_hash, sim_config_hash, process, sim_config)
        atomic_write_json(report_path, result)
        status = str(result.get("execution_status", "failed"))
        return ProbeResult(bool(result.get("behavior_pass")), result.get("metrics") or {}, result.get("diagnosis") or {}, str(report_path), file_sha256(report_path), status, bool(result.get("explicit_behavior_evidence")), result.get("outputs") or {}, result)

    @classmethod
    def _validate_report(cls, raw: Mapping[str, Any], probe: Mapping[str, Any], candidate_hash: str, sim_config_hash: str, process: subprocess.CompletedProcess[str], simulator_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
        simulator_config = simulator_config or {}
        metrics = raw.get("metrics")
        if not isinstance(metrics, Mapping) and isinstance(raw.get("behavior_metrics"), Mapping):
            metrics = raw.get("behavior_metrics")
        diagnosis = raw.get("diagnosis")
        behavior = raw.get("behavior_pass")
        if behavior is None and isinstance(metrics, Mapping):
            behavior = metrics.get("behavior_pass")
        explicit = isinstance(behavior, bool) and isinstance(metrics, Mapping) and bool(metrics) and isinstance(diagnosis, Mapping) and bool(diagnosis)
        required = probe.get("required_metrics") if isinstance(probe.get("required_metrics"), (list, tuple)) else []
        metric_map = metrics if isinstance(metrics, Mapping) else {}
        missing_required = [str(name) for name in required if str(name) not in metric_map]
        if missing_required:
            explicit = False
        execution_status = "completed" if process.returncode == 0 else "failed"
        if not explicit:
            # A status=pass, return code, GIF, or empty report is not behavior evidence.
            raw_status = str(raw.get("execution_status") or "")
            raw_kind = str((raw.get("diagnosis") or {}).get("kind") or "") if isinstance(raw.get("diagnosis"), Mapping) else ""
            if raw_status in {"not_executed", "sim_executed_unverified"} or raw_kind == "behavior_unverified":
                normalized_status = raw_status or "sim_executed_unverified"
            else:
                normalized_status = "sim_executed_unverified" if process.returncode == 0 else "failed"
            return {
                "schema_name": "softsim.probe_report",
                "schema_version": 2,
                "backend": "command",
                "probe": dict(probe),
                "probe_id": probe.get("probe_id"),
                "probe_hash": str(probe.get("probe_hash") or content_hash(probe)),
                "stimulus_hash": str(probe.get("stimulus_hash") or content_hash(probe.get("stimulus", {}))),
                "action_hash": str(probe.get("action_hash") or content_hash(probe.get("action", {}))),
                "config_hash": str(probe.get("config_hash") or content_hash(probe.get("config", {}))),
                "candidate_hash": candidate_hash,
                "simulator_config_hash": sim_config_hash,
                "candidate_config_hash": simulator_config.get("candidate_config_hash"),
                "execution_status": normalized_status,
                "behavior_pass": False,
                "passed": False,
                "explicit_behavior_evidence": False,
                "metrics": dict(metrics) if isinstance(metrics, Mapping) else {"return_code": process.returncode},
                "diagnosis": {"kind": "behavior_unverified", "suggested_updates": {}, "detail": "report requires boolean behavior_pass, non-empty registered behavior metrics, and diagnosis", "missing_metrics": missing_required},
                "outputs": dict(raw.get("outputs") or {}) if isinstance(raw.get("outputs"), Mapping) else {},
                "stdout_tail": process.stdout[-4000:],
                "stderr_tail": process.stderr[-4000:],
            }
        if not isinstance(diagnosis, Mapping):
            diagnosis = {"kind": "behavior_unverified", "suggested_updates": {}, "detail": "diagnosis object missing"}
            explicit = False
            behavior = False
        if missing_required:
            explicit = False
            behavior = False
            diagnosis = {"kind": "behavior_unverified", "suggested_updates": {}, "detail": "registered behavior metrics missing", "missing_metrics": missing_required}
        result = dict(raw)
        result.update({
            "schema_name": "softsim.probe_report",
            "schema_version": 2,
            "backend": "command",
            "probe": dict(probe),
            "probe_id": probe.get("probe_id"),
            "probe_hash": str(probe.get("probe_hash") or content_hash(probe)),
            "stimulus_hash": str(probe.get("stimulus_hash") or content_hash(probe.get("stimulus", {}))),
            "action_hash": str(probe.get("action_hash") or content_hash(probe.get("action", {}))),
            "config_hash": str(probe.get("config_hash") or content_hash(probe.get("config", {}))),
            "candidate_hash": candidate_hash,
            "simulator_config_hash": sim_config_hash,
            "candidate_config_hash": simulator_config.get("candidate_config_hash"),
            "behavior_pass": bool(behavior) if process.returncode == 0 else False,
            "passed": bool(behavior) if process.returncode == 0 else False,
            "explicit_behavior_evidence": bool(explicit),
            "execution_status": execution_status,
            "metrics": dict(metrics),
            "diagnosis": dict(diagnosis),
            "stdout_tail": process.stdout[-4000:],
            "stderr_tail": process.stderr[-4000:],
        })
        return result

    @staticmethod
    def _replace(value: str, replacements: Mapping[str, str]) -> str:
        for key, replacement in replacements.items():
            value = value.replace(key, replacement)
        return value


# Explicit name for callers that want to distinguish the Genesis adapter from
# a generic command backend; both share the strict report contract.
GenesisSimulator = CommandSimulator


def _failed_process(returncode: int, stderr: str, stdout: str) -> subprocess.CompletedProcess[str]:
    """Create a CompletedProcess-shaped value for executor errors."""

    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def build_simulator(config: Mapping[str, Any] | None = None):
    config = config or {"backend": "mock"}
    backend = str(config.get("backend", "mock")).lower()
    if backend == "mock":
        return MockSimulator(config)
    if backend in {"command", "genesis", "mpm"}:
        return CommandSimulator(config)
    raise ValueError(f"unknown simulator backend: {backend}")
