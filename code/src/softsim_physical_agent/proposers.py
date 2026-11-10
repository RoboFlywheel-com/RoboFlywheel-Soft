# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import json
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .io import content_hash
from .models import Proposal, ProbeResult
from .routing import ALLOWED_PHYSICAL_PARAMETERS, validate_patch


@dataclass
class RuleTuneProposer:
    """Deterministic, one-mechanism-at-a-time candidate proposer.

    The simulator diagnosis is evidence, not an instruction to mutate arbitrary
    JSON.  Only a registered physical parameter can cross this boundary, and
    the resulting proposal carries a stable rule id and input hash.
    """

    gain: float = 0.6
    allowed_parameters: frozenset[str] = ALLOWED_PHYSICAL_PARAMETERS
    max_relative_step: float = 0.5

    def propose(
        self,
        *,
        prior: Mapping[str, Any],
        candidate: Mapping[str, float],
        calibration: ProbeResult,
        bounds: Mapping[str, tuple[float, float]],
        iteration: int,
    ) -> Proposal:
        diagnosis = calibration.diagnosis if isinstance(calibration.diagnosis, Mapping) else {}
        kind = str(diagnosis.get("kind") or "")
        if kind == "behavior_unverified":
            return Proposal("evidence_unavailable", rationale="behavior evidence is incomplete; human/evaluator review is required", details={"diagnosis_kind": kind})
        if kind in {"model_limit", "unsupported_model"}:
            return Proposal("model_limit", rationale="calibration diagnosis declares an unsupported model", details={"diagnosis_kind": kind})
        if kind in {"executor_failure", "numerical_failure", "timeout"} or calibration.execution_status in {"failed", "timeout"}:
            return Proposal("numerical_failure", rationale="calibration executor failed", details={"diagnosis_kind": kind})
        raw_updates = diagnosis.get("suggested_updates")
        if not isinstance(raw_updates, Mapping) or not raw_updates:
            return Proposal("no_update", rationale="diagnosis supplied no bounded physical update", details={"diagnosis_kind": kind})
        # Keep the first deterministic key only: each candidate changes one mechanism.
        patch: dict[str, float] = {}
        relative_errors = diagnosis.get("relative_errors")
        names = sorted(raw_updates, key=str)
        if isinstance(relative_errors, Mapping):
            names.sort(key=lambda name: (-abs(float(relative_errors.get(name, 0.0))) if isinstance(relative_errors.get(name, 0.0), (int, float)) else 0.0, str(name)))
        for raw_name in names:
            name = str(raw_name)
            if name not in candidate or name not in self.allowed_parameters:
                continue
            try:
                delta = float(raw_updates[raw_name])
            except (TypeError, ValueError):
                continue
            if not (delta == delta and abs(delta) != float("inf")):
                continue
            lower, upper = bounds.get(name, (candidate[name] * 0.05, candidate[name] * 20.0))
            current = float(candidate[name])
            step_limit = max(abs(current) * float(self.max_relative_step), 1.0e-12)
            bounded_delta = min(step_limit, max(-step_limit, delta * float(self.gain)))
            value = min(float(upper), max(float(lower), current + bounded_delta))
            if value == float(candidate[name]):
                continue
            patch[name] = value
            break
        if not patch:
            return Proposal("no_update", rationale="all suggested changes were outside the mutable physical contract", details={"diagnosis_kind": kind})
        patch = validate_patch(patch, self.allowed_parameters)
        evidence = {
            "prior_id": prior.get("physical_bundle_id") or prior.get("prior_id"),
            "candidate": dict(candidate),
            "diagnosis": dict(diagnosis),
            "iteration": iteration,
        }
        return Proposal(
            "proposed",
            patch=patch,
            rule_id="RuleTune.v1.single_parameter_diagnosis",
            rationale="apply the first bounded physical update suggested by the calibration diagnosis",
            details={"input_hash": content_hash(evidence), "gain": self.gain},
        )


class QwenProposer:
    """Optional transport boundary; never pretends a missing key was used."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config = dict(config or {})

    def propose(self, **_: Any) -> Proposal:
        key_env = str(self.config.get("api_key_env") or "DASHSCOPE_API_KEY")
        if not any(os.environ.get(name) for name in (key_env, "QWEN_API_KEY", "QWEN_VL_API_KEY")):
            return Proposal(
                "proposer_unavailable",
                rationale="Qwen backend is optional and no API key is present",
                details={"reason": "missing_api_key", "api_key_env": key_env, "transport_called": False},
            )
        command = self.config.get("proposer_command") or self.config.get("command") or self.config.get("command_template")
        if command:
            if isinstance(command, str):
                command = shlex.split(command)
            if not isinstance(command, list) or not command:
                return Proposal("proposer_unavailable", rationale="configured Qwen command is invalid", details={"reason": "invalid_transport_command", "transport_called": False})
            # Resolve relative script paths against the package root (and its
            # parent for repo-relative spellings like code/scripts/...) so the
            # transport works regardless of the caller's working directory.
            package_root = Path(__file__).resolve().parents[2]
            tools_root = package_root / "scripts"
            if not tools_root.is_dir():
                import softsim_tools

                tools_root = Path(softsim_tools.__file__).resolve().parent
            resolved_command = []
            for item in command:
                item = str(item)
                if item.endswith(".py") and not Path(item).is_absolute():
                    for base in (package_root, package_root.parent):
                        if (base / item).is_file():
                            item = str(base / item)
                            break
                    if item.startswith(("scripts/", "code/scripts/")) and (tools_root / Path(item).name).is_file():
                        item = str(tools_root / Path(item).name)
                resolved_command.append(item)
            command = resolved_command
            # A configured server-side transport receives a JSON request and
            # must return {"patch": {...}, "rationale": "..."}.  The local
            # phase never configures this command, so no paid call is made here.
            kwargs = _
            raw_metrics = dict(kwargs.get("calibration").metrics) if kwargs.get("calibration") else {}
            # Keep the transport payload compact: bulk trajectory samples are
            # evidence for the trace, not prompt material (large payloads time
            # out the VLM call).
            slim_metrics = {
                key: value
                for key, value in raw_metrics.items()
                if key != "trajectory" and not (isinstance(value, list) and len(value) > 8)
            }
            slim_prior = None
            raw_prior = kwargs.get("prior")
            if isinstance(raw_prior, Mapping):
                slim_prior = {
                    key: raw_prior.get(key)
                    for key in ("sample_id", "model_family", "simulation_domain", "representation", "status")
                    if raw_prior.get(key) is not None
                }
                slim_prior["parameters"] = [
                    {"name": item.get("name"), "value": item.get("value"), "valid_range": item.get("valid_range")}
                    for item in raw_prior.get("parameters") or []
                    if isinstance(item, Mapping)
                ]
            request = {
                "prior": slim_prior,
                "candidate": kwargs.get("candidate"),
                "calibration": {
                    "metrics": slim_metrics,
                    "diagnosis": (kwargs.get("calibration").diagnosis if kwargs.get("calibration") else {}),
                },
            }
            with tempfile.TemporaryDirectory(prefix="softsim-qwen-") as temp_dir:
                request_path = Path(temp_dir) / "request.json"
                response_path = Path(temp_dir) / "response.json"
                request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
                replacements = {"{request_json}": str(request_path), "{response_json}": str(response_path)}
                command = [str(item).replace("{request_json}", replacements["{request_json}"]).replace("{response_json}", replacements["{response_json}"]) for item in command]
                try:
                    process = subprocess.run(command, check=False, text=True, capture_output=True, timeout=float(self.config.get("proposer_timeout_sec", self.config.get("timeout_sec", 120))))
                except (OSError, subprocess.TimeoutExpired) as exc:
                    return Proposal("proposer_unavailable", rationale="Qwen transport command failed", details={"reason": type(exc).__name__, "transport_called": True})
                if process.returncode != 0:
                    return Proposal("proposer_unavailable", rationale="Qwen transport command returned non-zero", details={"reason": "nonzero_return_code", "return_code": process.returncode, "transport_called": True})
                try:
                    raw = json.loads(response_path.read_text(encoding="utf-8")) if response_path.is_file() else json.loads(process.stdout)
                except (OSError, ValueError):
                    return Proposal("proposer_unavailable", rationale="Qwen transport returned invalid JSON", details={"reason": "invalid_response", "return_code": process.returncode, "transport_called": True})
            patch = raw.get("patch") if isinstance(raw, dict) else None
            if not isinstance(patch, dict):
                return Proposal("proposer_unavailable", rationale="Qwen response has no structured patch", details={"reason": "patch_missing", "transport_called": True})
            try:
                patch = validate_patch(patch, ALLOWED_PHYSICAL_PARAMETERS)
            except ValueError as exc:
                return Proposal("proposer_unavailable", rationale="Qwen patch violates physical parameter contract", details={"reason": str(exc), "transport_called": True})
            return Proposal("proposed", patch=patch, rule_id="Qwen.command.v1", rationale=str(raw.get("rationale") or "Qwen structured proposal"), details={"transport_called": True, "return_code": process.returncode})
        # Paid/network transport is intentionally not part of the local package.
        return Proposal(
            "proposer_unavailable",
            rationale="Qwen transport is not enabled in the local phase",
            details={"reason": "transport_not_enabled_local_phase", "api_key_env": key_env, "transport_called": False},
        )


def build_proposer(config: Mapping[str, Any] | None = None):
    config = config or {}
    backend = str(config.get("proposer_backend") or config.get("proposer") or "ruletune").lower()
    if backend in {"ruletune", "rule_tune", "deterministic"}:
        allowed = frozenset(set(config.get("allowed_physical_parameters") or ALLOWED_PHYSICAL_PARAMETERS) & set(ALLOWED_PHYSICAL_PARAMETERS))
        return RuleTuneProposer(gain=float(config.get("update_gain", 0.6)), allowed_parameters=allowed, max_relative_step=float(config.get("max_relative_step", 0.5)))
    if backend in {"qwen", "qwen_vl", "qwen-proposer"}:
        return QwenProposer(config)
    raise ValueError(f"unknown proposer backend: {backend}")
