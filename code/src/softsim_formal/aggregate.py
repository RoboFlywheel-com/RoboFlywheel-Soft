# SPDX-License-Identifier: Apache-2.0
"""Deterministic merge and denominator-preserving summaries."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonical_dumps, fingerprint, read_json, sha256_file
from .io import atomic_write_json, write_jsonl, read_jsonl
from .records import validate_record
from .schema import SchemaError


class AggregateError(ValueError):
    pass


def _all_shard_records(root: Path, shard_count: int) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for index in range(shard_count):
        path = root / "shard_records" / f"shard-{index:04d}.jsonl"
        if not path.is_file():
            missing.append(path.name)
            continue
        summary_path = root / "shard_records" / f"shard-{index:04d}.json"
        if summary_path.is_file():
            try:
                shard_summary = read_json(summary_path)
                if shard_summary.get("record_file_sha256") and shard_summary["record_file_sha256"] != sha256_file(path):
                    missing.append(f"{path.name}:hash_mismatch")
                    continue
            except (OSError, ValueError):
                missing.append(f"{summary_path.name}:invalid")
                continue
        rows.extend(read_jsonl(path))
    return rows, missing


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return sha256_file(path)


def _write_parquet_fallback(path: Path, rows: list[dict[str, Any]]) -> tuple[str, str]:
    """Write an explicit JSONL fallback when optional pyarrow is absent.

    The file keeps the requested stable `.parquet` location while metadata
    states the physical format.  A server with pyarrow can replace it through
    the documented conversion command; no false Parquet claim is made.
    """
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        columns: dict[str, list[Any]] = defaultdict(list)
        keys = sorted({key for row in rows for key in row})
        for row in rows:
            for key in keys:
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
                columns[key].append(value)
        table = pa.table(columns)
        pq.write_table(table, path, compression="zstd", use_dictionary=False)
        return sha256_file(path), "parquet"
    except Exception:
        write_jsonl(path, rows)
        return sha256_file(path), "jsonl_fallback_not_parquet"


def _summary(rows: list[dict[str, Any]], manifest: Mapping[str, Any], expected: int, missing: list[str], duplicate: list[str], invalid: list[str]) -> dict[str, Any]:
    statuses = Counter(str(row.get("outcome_status")) for row in rows)
    execution = Counter(str(row.get("execution_status")) for row in rows)
    failures = Counter(str(row.get("failure_code")) for row in rows if row.get("failure_code"))
    domains = Counter(str(row.get("failure_domain")) for row in rows)
    strata: dict[str, dict[str, int]] = defaultdict(lambda: {"planned": 0, "observed": 0})
    for task in _read_tasks(Path(manifest.get("_run_dir", "."))):
        key = _strata_key(task.get("canonical_strata"))
        strata[key]["planned"] += 1
    for row in rows:
        key = _strata_key(row.get("canonical_strata"))
        strata[key]["observed"] += 1
    physical_population = manifest.get("physical_execution_population")
    verified_population = manifest.get("verified_physical_count")
    complete = not missing and not duplicate and not invalid and len(rows) == expected and len({row.get("task_id") for row in rows}) == expected
    return {
        "schema_name": "softsim.aggregate_summary",
        "schema_version": 1,
        "run_id": manifest.get("run_id"),
        "experiment_id": manifest.get("experiment_id"),
        "population": manifest.get("population"),
        "denominator_scope": manifest.get("denominator_scope", "planned_tasks_including_failures"),
        "planned_denominator": expected,
        "observed_records": len(rows),
        "complete": complete,
        "missing_task_ids": sorted(missing),
        "duplicate_task_ids": sorted(duplicate),
        "invalid_records": sorted(invalid),
        "execution_status_counts": dict(sorted(execution.items())),
        "outcome_status_counts": dict(sorted(statuses.items())),
        "failure_domain_counts": dict(sorted(domains.items())),
        "failure_code_counts": dict(sorted(failures.items())),
        "canonical_strata": {key: strata[key] for key in sorted(strata)},
        "physical_execution_population": physical_population,
        "verified_candidate_population": verified_population,
        "task_failure_count": sum(1 for row in rows if row.get("failure_domain") == "infrastructure" or row.get("failure_code") == "source_task_failure"),
        "source_release_ids": manifest.get("source_release_ids", {}),
        "source_hashes": manifest.get("source_hashes", {}),
    }


def _strata_key(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "unknown"
    family = str(value.get("family", "unknown"))
    canonical = str(value.get("canonical_id", "unknown"))
    split = str(value.get("split", "unknown"))
    return f"{family}|{canonical}|{split}"


def _read_tasks(root: Path) -> list[dict[str, Any]]:
    path = root / "task_manifest.jsonl"
    return list(read_jsonl(path)) if path.is_file() else []


def aggregate_run(run_dir: str | Path, *, strict: bool = False) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("status") == "blocked":
        completion = {
            "schema_name": "softsim.run_completion",
            "schema_version": 1,
            "run_id": manifest.get("run_id"),
            "experiment_id": manifest.get("experiment_id"),
            "status": "blocked",
            "publishable": False,
            "synthetic": False,
            "blocked_code": manifest.get("blocked_code"),
            "blocked_message": manifest.get("blocked_message"),
            "expected_task_count": 0,
            "observed_task_count": 0,
            "denominator": {"population": manifest.get("population"), "planned": 0, "observed": 0, "scope": "blocked_no_tasks"},
            "source_hashes": manifest.get("source_hashes", {}),
        }
        atomic_write_json(root / "run_completion.json", completion)
        return {"summary": {"complete": False, "blocked_code": manifest.get("blocked_code")}, "completion": completion}
    manifest_with_root = dict(manifest)
    manifest_with_root["_run_dir"] = root.as_posix()
    tasks = _read_tasks(root)
    expected = int(manifest.get("expected_task_count", len(tasks)))
    task_by_id = {str(task.get("task_id")): task for task in tasks}
    shard_count = max(1, int(manifest.get("shard_count", 1)))
    shard_rows, missing_shards = _all_shard_records(root, shard_count)
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid: list[str] = []
    for row in shard_rows:
        task_key = str(row.get("task_id", "<missing>"))
        try:
            validate_record(row)
            task = task_by_id.get(task_key)
            if task is None:
                raise SchemaError("record task_id is absent from frozen task manifest")
            for field, expected_value in (
                ("run_id", manifest.get("run_id")),
                ("experiment_id", manifest.get("experiment_id")),
                ("protocol_hash", manifest.get("protocol_hash")),
            ):
                if expected_value is not None and row.get(field) != expected_value:
                    raise SchemaError(f"record {field} does not match run manifest")
            if row.get("attempt_id") is None:
                raise SchemaError("record attempt_id is missing")
        except (SchemaError, ValueError) as exc:
            invalid.append(f"{task_key}:{type(exc).__name__}")
            continue
        by_id[task_key].append(row)
    duplicate = [key for key, values in by_id.items() if len(values) > 1]
    expected_ids = {str(task.get("task_id")) for task in tasks}
    observed_ids = set(by_id)
    missing = sorted(expected_ids - observed_ids)
    if len(tasks) != expected:
        missing.append(f"manifest_count:{len(tasks)}!={expected}")
    rows = [values[0] for key, values in by_id.items() if len(values) == 1]
    rows.sort(key=lambda row: (str(row.get("task_id")), str(row.get("episode_id")), str(row.get("record_sha256"))))
    canonical_dir = root / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = canonical_dir / "episodes.jsonl"
    canonical_hash = write_jsonl(canonical_path, rows)
    # Keep a conventional root-level alias for downstream scripts.
    write_jsonl(root / "canonical_records.jsonl", rows)
    parquet_hash, parquet_format = _write_parquet_fallback(canonical_dir / "episodes.parquet", rows)
    # Agent steps and simulator calls are separate canonical streams.  They
    # are sorted and hashed independently so E2 budgets/isolation remain
    # auditable without changing the episode denominator.
    step_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    for task in tasks:
        task_key = str(task.get("task_id"))
        unique = by_id.get(task_key, [])
        if not unique:
            continue
        attempt_dirs = sorted((root / "tasks" / task_key).glob("attempt-*/"))
        if not attempt_dirs:
            continue
        attempt = attempt_dirs[-1]
        if (attempt / "agent_steps.jsonl").is_file():
            step_rows.extend(read_jsonl(attempt / "agent_steps.jsonl"))
        if (attempt / "calls.jsonl").is_file():
            call_rows.extend(read_jsonl(attempt / "calls.jsonl"))
    step_rows.sort(key=lambda row: (str(row.get("task_id")), int(row.get("step_index", 0))))
    call_rows.sort(key=lambda row: (str(row.get("task_id")), int(row.get("call_index", 0))))
    steps_hash = write_jsonl(canonical_dir / "agent_steps.jsonl", step_rows)
    calls_hash = write_jsonl(canonical_dir / "calls.jsonl", call_rows)
    flat_rows = []
    for row in rows:
        flat_rows.append({
            "task_id": row.get("task_id"),
            "episode_id": row.get("episode_id"),
            "experiment_id": row.get("experiment_id"),
            "population": row.get("population"),
            "family": (row.get("canonical_strata") or {}).get("family"),
            "canonical_id": (row.get("canonical_strata") or {}).get("canonical_id"),
            "method_id": row.get("method_id"),
            "condition_id": row.get("condition_id"),
            "execution_status": row.get("execution_status"),
            "outcome_status": row.get("outcome_status"),
            "failure_domain": row.get("failure_domain"),
            "failure_code": row.get("failure_code"),
            "record_sha256": row.get("record_sha256"),
        })
    csv_hash = _write_csv(root / "canonical_records.csv", flat_rows, list(flat_rows[0].keys()) if flat_rows else ["task_id", "episode_id", "outcome_status", "record_sha256"])
    summary = _summary(rows, manifest_with_root, expected, missing + missing_shards, duplicate, invalid)
    summary["canonical_jsonl_sha256"] = canonical_hash
    summary["agent_steps_sha256"] = steps_hash
    summary["calls_sha256"] = calls_hash
    summary["parquet_sha256"] = parquet_hash
    summary["parquet_format"] = parquet_format
    summary["csv_sha256"] = csv_hash
    summary["blocked_record_count"] = sum(1 for row in rows if row.get("outcome_status") == "blocked")
    summary["invalid_record_count"] = sum(1 for row in rows if row.get("execution_status") == "invalid")
    disabled_methods = set((manifest.get("disabled_methods") or {}).keys())
    summary["disabled_method_record_count"] = sum(1 for row in rows if str(row.get("method_id")) in disabled_methods)
    summary["publication_gate"] = {
        "coverage_complete": bool(summary["complete"]),
        "blocked_records_absent": summary["blocked_record_count"] == 0,
        "invalid_records_absent": summary["invalid_record_count"] == 0,
        "disabled_method_records_absent": summary["disabled_method_record_count"] == 0,
        "synthetic": bool(manifest.get("synthetic", False)),
    }
    selection_path = root / "selection_manifest.jsonl"
    if selection_path.is_file():
        summary["selection_manifest_sha256"] = sha256_file(selection_path)
        summary["selection_manifest_ref"] = "selection_manifest.jsonl"
    summary["deterministic_aggregate_hash"] = fingerprint({key: value for key, value in summary.items() if key not in {"deterministic_aggregate_hash"}})
    atomic_write_json(root / "aggregate_summary.json", summary)
    completion = {
        "schema_name": "softsim.run_completion",
        "schema_version": 1,
        "run_id": manifest.get("run_id"),
        "experiment_id": manifest.get("experiment_id"),
        "status": "completed" if summary["complete"] else "incomplete",
        "expected_task_count": expected,
        "observed_task_count": len(rows),
        "canonical_records_ref": "canonical/episodes.jsonl",
        "canonical_records_sha256": canonical_hash,
        "agent_steps_ref": "canonical/agent_steps.jsonl",
        "agent_steps_sha256": steps_hash,
        "calls_ref": "canonical/calls.jsonl",
        "calls_sha256": calls_hash,
        "aggregate_summary_ref": "aggregate_summary.json",
        "aggregate_summary_sha256": sha256_file(root / "aggregate_summary.json"),
        "publishable": bool(
            summary["complete"]
            and manifest.get("status") != "blocked"
            and not manifest.get("synthetic", False)
            and summary["blocked_record_count"] == 0
            and summary["invalid_record_count"] == 0
            and summary["disabled_method_record_count"] == 0
        ),
        "synthetic": bool(manifest.get("synthetic", False) or str(manifest.get("mode")) == "fixture"),
        "denominator": {"population": manifest.get("population"), "planned": expected, "observed": len(rows), "scope": manifest.get("denominator_scope")},
        "source_hashes": manifest.get("source_hashes", {}),
    }
    atomic_write_json(root / "run_completion.json", completion)
    if strict and (not summary["complete"] or not completion["publishable"]):
        raise AggregateError("run failed publication gate: " + canonical_dumps(summary))
    return {"summary": summary, "completion": completion}
