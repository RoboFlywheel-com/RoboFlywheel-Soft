# SPDX-License-Identifier: Apache-2.0
"""Canonical run, case, episode, and simulator-call records."""

from __future__ import annotations

import copy
from typing import Any, Mapping, MutableMapping, Optional

from .canonical import fingerprint
from .schema import SchemaError, validate_episode_record, validate_run_manifest, require_hash, require_id


def record_hash(record: Mapping[str, Any]) -> str:
    """Hash a record without its self-referential hash field."""
    return fingerprint({key: value for key, value in record.items() if key not in {"record_sha256", "record_hash"}})


def task_id(spec: Mapping[str, Any]) -> str:
    return fingerprint(dict(spec))


def case_id(spec: Mapping[str, Any]) -> str:
    return fingerprint(dict(spec))


def episode_id(spec: Mapping[str, Any]) -> str:
    return fingerprint(dict(spec))


def call_id(spec: Mapping[str, Any]) -> str:
    return fingerprint(dict(spec))


def make_task_spec(
    *,
    experiment_id: str,
    protocol_hash: str,
    population: str,
    sample_id: str | None = None,
    asset_revision_id: str | None = None,
    physical_bundle_id: str | None = None,
    benchmark_instance_id: str | None = None,
    method_id: str = "scripted.mock.v1",
    condition_id: str = "default",
    paired_scenario_key: str = "default",
    seed: int = 0,
    trial_index: int = 0,
    canonical_strata: Mapping[str, Any] | None = None,
    data_role: str = "test",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "experiment_id": experiment_id,
        "protocol_hash": protocol_hash,
        "population": population,
        "data_role": data_role,
        "sample_id": sample_id,
        "asset_revision_id": asset_revision_id,
        "physical_bundle_id": physical_bundle_id,
        "benchmark_instance_id": benchmark_instance_id,
        "method_id": method_id,
        "condition_id": condition_id,
        "paired_scenario_key": paired_scenario_key,
        "seed": int(seed),
        "trial_index": int(trial_index),
        "canonical_strata": dict(canonical_strata or {}),
    }
    if extra:
        spec.update(copy.deepcopy(dict(extra)))
    spec["task_id"] = task_id({key: value for key, value in spec.items() if key != "task_id"})
    return spec


def make_run_manifest(
    *,
    run_id: str,
    experiment_id: str,
    protocol_hash: str,
    expected_task_count: int,
    task_manifest_sha256: str,
    config_hash: str,
    code_hash: str,
    population: str,
    data_role: str = "test",
    mode: str = "local",
    scheduler: str = "local",
    source_hashes: Mapping[str, str] | None = None,
    method_id: str = "scripted.mock.v1",
    denominator_scope: str = "planned_tasks",
    logical_roots: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_name": "softsim.run_manifest",
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "protocol_hash": protocol_hash,
        "expected_task_count": int(expected_task_count),
        "task_manifest_sha256": task_manifest_sha256,
        "config_hash": config_hash,
        "code_hash": code_hash,
        "population": population,
        "data_role": data_role,
        "mode": mode,
        "scheduler": scheduler,
        "method_id": method_id,
        "denominator_scope": denominator_scope,
        "denominator": {"population": population, "scope": denominator_scope, "expected": int(expected_task_count), "includes_failures": True},
        "strata_keys": ["family", "canonical_id", "split"],
        "source_hashes": dict(source_hashes or {}),
        "logical_roots": dict(logical_roots or {}),
        "status": "planned",
    }
    validate_run_manifest(manifest)
    return manifest


def _base_record(
    *,
    schema_name: str,
    run_manifest: Mapping[str, Any],
    task: Mapping[str, Any],
    execution_status: str,
    outcome_status: str,
    failure_domain: str = "none",
    failure_code: str | None = None,
    retriable: bool = False,
    metrics: Mapping[str, Any] | None = None,
    outcomes: Mapping[str, Any] | None = None,
    measurements: Mapping[str, Any] | None = None,
    artifacts: list[Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_name": schema_name,
        "schema_version": 1,
        "run_id": str(run_manifest["run_id"]),
        "task_id": str(task["task_id"]),
        "case_id": str(task.get("case_id") or task["task_id"]),
        "episode_id": str(task.get("episode_id") or task["task_id"]),
        "call_id": task.get("call_id"),
        "experiment_id": str(run_manifest["experiment_id"]),
        "protocol_hash": str(run_manifest["protocol_hash"]),
        "data_role": str(task.get("data_role") or run_manifest.get("data_role") or "test"),
        "population": str(task.get("population") or run_manifest.get("population") or "unknown"),
        "denominator_scope": str(run_manifest.get("denominator_scope") or "planned_tasks"),
        "sample_id": task.get("sample_id"),
        "asset_revision_id": task.get("asset_revision_id"),
        "physical_bundle_id": task.get("physical_bundle_id"),
        "benchmark_instance_id": task.get("benchmark_instance_id"),
        "method_id": str(task.get("method_id") or run_manifest.get("method_id") or "unknown"),
        "condition_id": str(task.get("condition_id") or "default"),
        "paired_scenario_key": str(task.get("paired_scenario_key") or "default"),
        "seed": int(task.get("seed", 0)),
        "trial_index": int(task.get("trial_index", 0)),
        "canonical_strata": dict(task.get("canonical_strata") or {}),
        "strata": dict(task.get("canonical_strata") or {}),
        "execution_status": execution_status,
        "outcome_status": outcome_status,
        "status": outcome_status,
        "failure_domain": failure_domain,
        "retriable": bool(retriable),
        "metrics": dict(metrics or {}),
        "outcomes": dict(outcomes or {}),
        "measurements": dict(measurements or {}),
        "artifact_refs": [dict(item) for item in (artifacts or [])],
        "source_hashes": dict(run_manifest.get("source_hashes") or {}),
    }
    if failure_code is not None:
        record["failure_code"] = str(failure_code)
    if extra:
        record.update(copy.deepcopy(dict(extra)))
    record["record_sha256"] = record_hash(record)
    return record


def make_case_record(**kwargs: Any) -> dict[str, Any]:
    kwargs["schema_name"] = "softsim.case_record"
    return _base_record(**kwargs)


def make_episode_record(**kwargs: Any) -> dict[str, Any]:
    kwargs["schema_name"] = "softsim.episode_record"
    return _base_record(**kwargs)


def make_call_record(**kwargs: Any) -> dict[str, Any]:
    kwargs["schema_name"] = "softsim.call_record"
    return _base_record(**kwargs)


def validate_record(record: Mapping[str, Any]) -> None:
    validate_episode_record(record)
    expected = record_hash(record)
    if record.get("record_sha256") != expected:
        raise SchemaError("record_sha256 does not match canonical record")


def validate_task_spec(task: Mapping[str, Any]) -> None:
    for key in ("task_id", "experiment_id", "protocol_hash", "population"):
        require_id(task.get(key), key)
    require_hash(task.get("protocol_hash"), "protocol_hash")
    expected = task_id({key: value for key, value in task.items() if key != "task_id"})
    if task.get("task_id") != expected:
        raise SchemaError("task_id does not match frozen task specification")


def validate_hashes(record: Mapping[str, Any]) -> None:
    for key in ("protocol_hash", "record_sha256"):
        require_hash(record.get(key), key)
