# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .io import atomic_write_json, atomic_write_jsonl, content_hash, file_sha256, iter_jsonl, read_json


def _copy_task_evidence(task_dir: Path, bundle_dir: Path) -> None:
    """Copy only task-local evidence; source asset paths are never copied."""

    for source in sorted(task_dir.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(task_dir)
        if relative.parts and relative.parts[0] in {"history", "SUCCESS.json"}:
            continue
        if relative.as_posix() in {"physics.json", "SUCCESS.json"}:
            continue
        target = bundle_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _rewrite_trace(source: Path, target: Path, sample_id: str) -> None:
    """Rewrite task-local refs inside JSONL trace while preserving each event."""

    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as input_handle, target.open("w", encoding="utf-8") as output_handle:
        for line in input_handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError:
                output_handle.write(line)
                continue
            if isinstance(event, dict):
                _rewrite_trace_refs(event, sample_id)
            output_handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _rewrite_trace_refs(value: Any, sample_id: str) -> None:
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if isinstance(item, str) and key.endswith("_ref") and (item.startswith("candidate_") or item.startswith("heldout/") or item == "agent_trace.jsonl"):
                value[key] = _bundle_ref(sample_id, item)
            else:
                _rewrite_trace_refs(item, sample_id)
    elif isinstance(value, list):
        for item in value:
            _rewrite_trace_refs(item, sample_id)


def _bundle_ref(sample_id: str, relative: str | None) -> str | None:
    if not relative:
        return None
    return f"bundles/{sample_id}/{Path(str(relative)).as_posix()}"


def _task_local_ref(sample_id: str, relative: str | None) -> str | None:
    if not relative:
        return None
    raw = Path(str(relative)).as_posix()
    prefix = f"tasks/{sample_id}/"
    if raw.startswith(prefix):
        raw = raw[len(prefix):]
    return _bundle_ref(sample_id, raw)


def _rewrite_physics(physics: Mapping[str, Any], sample_id: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(physics))
    result["prior_ref"] = _bundle_ref(sample_id, str(result.get("prior_ref") or "prior.json"))
    for key in ("trace_ref", "trace_ref_root", "calibration_report_ref", "heldout_report_ref"):
        if result.get(key):
            result[key] = _task_local_ref(sample_id, str(result[key]))
    agent = result.get("agent")
    if isinstance(agent, dict):
        agent["trace_ref"] = _bundle_ref(sample_id, str(agent.get("trace_ref") or "agent_trace.jsonl"))
    candidate = result.get("candidate")
    if isinstance(candidate, dict):
        for key in ("config_ref", "report_ref"):
            if candidate.get(key):
                candidate[key] = _bundle_ref(sample_id, str(candidate[key]))
    for item in result.get("candidate_refs") or []:
        if isinstance(item, dict):
            for key in ("config_ref", "report_ref"):
                if item.get(key):
                    item[key] = _bundle_ref(sample_id, str(item[key]))
            outputs = item.get("outputs")
            if isinstance(outputs, dict):
                for key, value in list(outputs.items()):
                    if isinstance(value, str) and (key.endswith("_ref") or key in {"trajectory", "preview", "report"}):
                        outputs[key] = _bundle_ref(sample_id, value)
    verification = result.get("verification")
    if isinstance(verification, dict):
        for phase in ("calibration", "heldout"):
            record = verification.get(phase)
            if isinstance(record, dict):
                for key in ("report_ref", "trajectory_ref", "preview_ref", "output_ref"):
                    if record.get(key):
                        record[key] = _bundle_ref(sample_id, str(record[key]))
                outputs = record.get("outputs")
                if isinstance(outputs, dict):
                    for key, value in list(outputs.items()):
                        if isinstance(value, str) and (key.endswith("_ref") or key in {"trajectory", "preview", "report"}):
                            outputs[key] = _bundle_ref(sample_id, value)
    for key, value in list(result.items()):
        if key.endswith("_ref") and isinstance(value, str) and key not in {"prior_ref"}:
            # Top-level task-local references, if supplied by a custom backend.
            if value.startswith("candidate_") or value.startswith("heldout/") or value == "agent_trace.jsonl":
                result[key] = _bundle_ref(sample_id, value)
    return result


def _read_marker(path: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    try:
        marker = read_json(path)
        physics = read_json(path.parent / str(marker.get("physics_ref") or "physics.json"))
    except (OSError, ValueError, KeyError):
        return None
    return marker, physics


def _expected_ids(output_root: Path, expected_count: int | None, explicit: Iterable[str] | None) -> list[str]:
    if explicit:
        return sorted(set(str(item) for item in explicit))
    for path in (output_root / "run_plan.json", output_root / "tasks.jsonl"):
        if path.name == "run_plan.json" and path.is_file():
            value = read_json(path)
            ids = value.get("task_ids")
            if isinstance(ids, list):
                return sorted(set(str(item) for item in ids))
        if path.name == "tasks.jsonl" and path.is_file():
            return sorted(set(str(item.get("sample_id")) for item in iter_jsonl(path) if item.get("sample_id")))
    return [] if expected_count is None else []


def aggregate_run(
    output_root: Path,
    *,
    release_id: str = "softsim-physical-agent-rebuild-v0.1",
    expected_count: int | None = None,
    expected_task_ids: Iterable[str] | None = None,
    strict_publish: bool = False,
) -> dict[str, Any]:
    """Create a deterministic release index while retaining rejects/failures."""

    output_root = Path(output_root)
    bundle_root = output_root / "bundles"
    bundle_root.mkdir(parents=True, exist_ok=True)
    # Emit the exact proxy contract alongside the index so a downstream formal
    # reader can verify method/hash identity without relying on a global config.
    method_contract_path = output_root / "method_contract.json"
    if not method_contract_path.is_file():
        atomic_write_json(method_contract_path, {
            "schema_name": "softsim.method_contract",
            "schema_version": 1,
            "method_id": "softsim.genesis_proxy.v12.2",
            "genesis_version": "1.3.3",
            "solver": {"cloth": "PBD.Cloth", "volume": "MPM.Elastic/MPM.ElastoPlastic", "coupler": "LegacyCoupler"},
            "ipc_used": False,
            "uipc_used": False,
            "proxy": True,
            "synthetic": True,
            "simulation_calibrated_definition": "behavior contract pass only; not real-material ground truth",
        })
    method_contract_hash = file_sha256(method_contract_path)
    # A pre-placed contract file (e.g. the v12.6 contract for supplemental
    # releases) also supplies the manifest method_id.
    try:
        contract_method_id = str(read_json(method_contract_path).get("method_id") or "softsim.genesis_proxy.v12.2")
    except (OSError, ValueError):
        contract_method_id = "softsim.genesis_proxy.v12.2"
    markers = sorted((path for path in (output_root / "tasks").glob("*/SUCCESS.json") if path.is_file()), key=lambda path: path.parent.name)
    rows: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    family_status: defaultdict[str, Counter[str]] = defaultdict(Counter)
    failures: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    synthetic = False
    missing_heldout: list[str] = []
    missing_trace: list[str] = []
    missing_outputs: list[str] = []
    source_hashes: set[str] = set()
    for marker_path in markers:
        loaded = _read_marker(marker_path)
        if loaded is None:
            continue
        marker, physics = loaded
        sample_id = str(physics.get("sample_id") or marker.get("sample_id") or marker_path.parent.name)
        if sample_id in seen:
            duplicate_ids.append(sample_id)
            continue
        seen.add(sample_id)
        bundle_dir = bundle_root / sample_id
        _copy_task_evidence(marker_path.parent, bundle_dir)
        source_trace = marker_path.parent / "agent_trace.jsonl"
        if source_trace.is_file():
            _rewrite_trace(source_trace, bundle_dir / "agent_trace.jsonl", sample_id)
        rewritten = _rewrite_physics(physics, sample_id)
        rewritten.setdefault("agent", {})["trace_ref"] = f"bundles/{sample_id}/agent_trace.jsonl"
        rewritten["agent"]["trace_sha256"] = file_sha256(bundle_dir / "agent_trace.jsonl") if (bundle_dir / "agent_trace.jsonl").is_file() else None
        # The release copy's own hash is recomputed after references are rewritten.
        rewritten.pop("record_sha256", None)
        rewritten["physical_bundle_id"] = content_hash({key: value for key, value in rewritten.items() if key not in {"physical_bundle_id", "record_sha256"}})
        rewritten["record_sha256"] = content_hash(rewritten)
        bundle_path = bundle_dir / "physics.json"
        atomic_write_json(bundle_path, rewritten)
        status = str(rewritten.get("status") or marker.get("status") or "uncertain_human_escalation")
        statuses[status] += 1
        family = str((rewritten.get("route") or {}).get("geometry_family") or "unknown")
        family_status[family][status] += 1
        provenance = rewritten.get("provenance") or {}
        if rewritten.get("agent_source_sha256"):
            source_hashes.add(str(rewritten["agent_source_sha256"]))
        synthetic = synthetic or bool(provenance.get("synthetic_fixture")) or bool(rewritten.get("synthetic_fixture")) or str((rewritten.get("agent") or {}).get("proposer_backend")) == "MockProposer" or str(rewritten.get("simulator_backend", "")).lower() == "mock"
        verification = rewritten.get("verification") or {}
        heldout = verification.get("heldout") if isinstance(verification, Mapping) else {}
        if not isinstance(heldout, Mapping) or not heldout.get("executed") or not heldout.get("report_ref"):
            missing_heldout.append(sample_id)
        trace_ref = (rewritten.get("agent") or {}).get("trace_ref")
        if not trace_ref or not (output_root / str(trace_ref)).is_file():
            # trace_ref is release-relative after rewrite, so resolve directly under output root.
            if not trace_ref or not (output_root / str(trace_ref)).is_file():
                missing_trace.append(sample_id)
        trace_path = output_root / str(trace_ref) if trace_ref else None
        trace_sha = file_sha256(trace_path) if trace_path and trace_path.is_file() else None
        heldout_ref_value = heldout.get("report_ref") if isinstance(heldout, Mapping) else None
        heldout_path = output_root / str(heldout_ref_value) if heldout_ref_value else None
        heldout_sha = file_sha256(heldout_path) if heldout_path and heldout_path.is_file() else None
        for phase_name in ("calibration", "heldout"):
            phase_record = verification.get(phase_name) if isinstance(verification, Mapping) else {}
            outputs = phase_record.get("outputs") if isinstance(phase_record, Mapping) else {}
            if isinstance(outputs, Mapping):
                for key, value in outputs.items():
                    if not isinstance(value, str) or not (key.endswith("_ref") or key in {"trajectory", "preview", "report"}):
                        continue
                    if Path(value).is_absolute() or not (output_root / value).is_file():
                        missing_outputs.append(sample_id)
        trajectory_refs: list[str] = []
        if isinstance(verification, Mapping):
            for phase_name in ("calibration", "heldout"):
                phase_record = verification.get(phase_name)
                if not isinstance(phase_record, Mapping):
                    continue
                outputs = phase_record.get("outputs")
                if not isinstance(outputs, Mapping):
                    continue
                for key, value in outputs.items():
                    if isinstance(value, str) and ("trajectory" in str(key) or str(key).endswith("trajectory_ref")):
                        trajectory_refs.append(value)
        # Formal status is deliberately separate from the agent's richer
        # decision vocabulary.  Keep the latter in ``agent_status`` for audit.
        agent_status = status
        formal_status = "verified" if status in {"accepted_first_pass", "corrected_then_accepted"} else ("task_failure" if status == "rejected_numerical_failure" else "rejected")
        def _ref(relative: str | None) -> dict[str, Any] | None:
            if not relative:
                return None
            candidate = output_root / relative
            if not candidate.is_file():
                return None
            return {"root_alias": "release", "relative_path": Path(relative).as_posix(), "sha256": file_sha256(candidate)}
        trace_artifact = _ref(str((rewritten.get("agent") or {}).get("trace_ref") or ""))
        calibration_artifact = _ref(str((verification.get("calibration") or {}).get("report_ref") or "")) if isinstance(verification, Mapping) else None
        heldout_artifact = _ref(str((verification.get("heldout") or {}).get("report_ref") or "")) if isinstance(verification, Mapping) else None
        row = {
            "schema_name": "softsim.physical_bundle_index_record",
            "schema_version": 12,
            "sample_id": sample_id,
            "asset_revision_id": rewritten.get("asset_revision_id"),
            "dataset_release_id": rewritten.get("dataset_release_id"),
            "physical_bundle_id": rewritten.get("physical_bundle_id"),
            "status": formal_status,
            "agent_status": agent_status,
            "geometry_family": family,
            "bundle": {"path": f"bundles/{sample_id}/physics.json", "sha256": file_sha256(bundle_path)},
            "trace_ref": (rewritten.get("agent") or {}).get("trace_ref"),
            "agent_trace_ref": trace_artifact,
            "calibration_evidence_ref": calibration_artifact,
            "heldout_evidence_ref": heldout_artifact,
            "method_contract_hash": method_contract_hash,
            "heldout_report_ref": heldout_ref_value,
            "heldout_report_sha256": heldout_sha,
            "trace_sha256": trace_sha,
            "trajectory_refs": trajectory_refs,
            "task_fingerprint": marker.get("task_fingerprint"),
            "config_hash": marker.get("config_hash"),
            "synthetic_fixture": bool(provenance.get("synthetic_fixture")),
        }
        if formal_status != "verified":
            row["failure_code"] = str((rewritten.get("limitations") or ["agent_rejected"])[0])
        row["record_sha256"] = content_hash(row)
        rows.append(row)
        if formal_status != "verified":
            failures.append({"sample_id": sample_id, "status": formal_status, "agent_status": agent_status, "route": rewritten.get("route"), "limitations": rewritten.get("limitations", []), "candidate": rewritten.get("candidate")})
    rows.sort(key=lambda row: row["sample_id"])
    index_path = output_root / "physical_bundle_index.jsonl"
    atomic_write_jsonl(index_path, rows)
    index_hash = file_sha256(index_path)
    index_digest = index_hash.split(":", 1)[1] if index_hash.startswith("sha256:") else index_hash
    (output_root / "physical_bundle_index.sha256").write_text(index_digest + "  physical_bundle_index.jsonl\n", encoding="utf-8")
    expected_ids = _expected_ids(output_root, expected_count, expected_task_ids)
    plan_population_count: int | None = None
    plan_path = output_root / "run_plan.json"
    if plan_path.is_file():
        try:
            raw_plan = read_json(plan_path)
            if isinstance(raw_plan.get("population_count"), int):
                plan_population_count = int(raw_plan["population_count"])
        except (OSError, ValueError):
            pass
    if expected_count is None and plan_population_count is not None:
        expected_count = plan_population_count
    expected = expected_count if expected_count is not None else (len(expected_ids) if expected_ids else len(rows))
    missing = sorted(set(expected_ids) - seen) if expected_ids else []
    unexpected = sorted(seen - set(expected_ids)) if expected_ids else []
    reasons: list[str] = []
    if expected <= 0 or not rows:
        reasons.append("empty_population")
    if len(rows) != expected:
        reasons.append("population_count_mismatch")
    if missing:
        reasons.append("missing_expected_tasks")
    if unexpected:
        reasons.append("unexpected_tasks")
    if duplicate_ids:
        reasons.append("duplicate_sample_id")
    if synthetic:
        reasons.append("mock_or_synthetic_backend")
    if missing_trace:
        reasons.append("missing_trace_evidence")
    if missing_outputs:
        reasons.append("missing_output_evidence")
    if missing_heldout:
        reasons.append("missing_heldout_evidence")
    # ``status`` in the release index is the formal vocabulary above
    # (verified/rejected/task_failure).  The richer agent decision is retained
    # separately as ``agent_status``; comparing it here would make every
    # successful aggregate permanently non-publishable.
    if any(row["status"] != "verified" for row in rows):
        reasons.append("rejected_or_uncertain_tasks_present")
    if expected_count is None and not expected_ids:
        reasons.append("expected_population_not_declared")
    if not source_hashes:
        reasons.append("source_hash_missing")
    if len(source_hashes) > 1:
        reasons.append("source_hash_mismatch")
    if any(
        bool((read_json(output_root / str(row["bundle"]["path"])).get("provenance") or {}).get("prior_only"))
        or bool(read_json(output_root / str(row["bundle"]["path"])).get("prior_only"))
        or read_json(output_root / str(row["bundle"]["path"])).get("status") == "predicted_prior"
        for row in rows
    ):
        reasons.append("prior_only_bundle")
    manifest = {
        "schema_name": "softsim.physical_release_manifest",
        "schema_version": 12,
        "release_id": release_id,
        "physical_release_id": release_id,
        "method_id": contract_method_id,
        "release_gate_version": "physical_release_v12",
        "genesis_version": "1.3.3",
        "solver_contract": {"cloth": "PBD.Cloth", "volume": "MPM.Elastic/MPM.ElastoPlastic", "coupler": "LegacyCoupler", "ipc_used": False, "uipc_used": False},
        "source_sha256": (next(iter(source_hashes)) if len(source_hashes) == 1 else sorted(source_hashes)),
        "source_sha256s": sorted(source_hashes),
        "config_hashes": sorted(set(str(row.get("config_hash")) for row in rows if row.get("config_hash"))),
        "code_revision": "local-package-v0.1",
        "expected_population": expected,
        "population_count": expected,
        "bundle_count": len(rows),
        "status_counts": dict(sorted({status: sum(row.get("status") == status for row in rows) for status in {"verified", "rejected", "task_failure"}}.items())),
        "agent_status_counts": dict(sorted(statuses.items())),
        "index_path": "physical_bundle_index.jsonl",
        "physical_bundle_index": "physical_bundle_index.jsonl",
        "index_sha256": index_hash,
        "method_contract_hash": method_contract_hash,
        "method_contract_ref": "method_contract.json",
        "coverage_complete": not missing and not unexpected and len(rows) == expected,
        "publishable": not reasons,
        "non_publishable_reasons": sorted(set(reasons)),
        "synthetic_fixture": synthetic,
        "partial_shard": bool(missing or (expected_count is not None and len(rows) < expected_count)),
        "mock_backend": synthetic,
        "prior_only": "prior_only_bundle" in reasons,
        "heldout_evidence_complete": not missing_heldout,
        "trace_evidence_complete": not missing_trace,
        "output_evidence_complete": not missing_outputs,
    }
    atomic_write_json(output_root / "physical_release_manifest.json", manifest)
    formal_status_counts = {status: sum(row.get("status") == status for row in rows) for status in ("verified", "rejected", "task_failure")}
    atomic_write_json(output_root / "summaries" / "coverage.json", {"expected_population": expected, "observed_population": len(rows), "missing_sample_ids": missing, "unexpected_sample_ids": unexpected, "duplicate_sample_ids": sorted(set(duplicate_ids)), "status_counts": dict(sorted(formal_status_counts.items())), "agent_status_counts": dict(sorted(statuses.items()))})
    atomic_write_json(output_root / "summaries" / "status_by_family.json", {family: dict(sorted(counter.items())) for family, counter in sorted(family_status.items())})
    atomic_write_jsonl(output_root / "summaries" / "failure_taxonomy.jsonl", failures)
    if strict_publish and not manifest["publishable"]:
        raise RuntimeError("release is non-publishable: " + ", ".join(manifest["non_publishable_reasons"]))
    return manifest
