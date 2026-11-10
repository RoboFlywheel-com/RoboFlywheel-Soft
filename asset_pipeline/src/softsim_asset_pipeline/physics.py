# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    physical_bundle_id,
    validate_asset_ledger_record,
    validate_physical_bundle,
    with_record_hash,
)
from .io import (
    atomic_text_writer,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_bytes,
    content_id,
    iter_jsonl,
    read_json,
    safe_join,
    sha256_file,
)
from .transport import (
    FatalTransportError,
    MockTransport,
    QwenTransport,
    SlidingWindowLimiter,
    TransportError,
    complete_with_retries,
)


def load_catalog(
    config: dict[str, Any],
    legacy_src: Path | None,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    catalog_json = config.get("catalog_json")
    if catalog_json:
        path = Path(str(catalog_json)).expanduser()
        if not path.is_absolute() and config_dir is not None:
            path = config_dir / path
        path = path.resolve()
        catalog = read_json(path)
        catalog_kind = str(catalog.get("catalog_kind") or "synthetic_fixture")
        if catalog_kind == "production_solver_catalog":
            catalog.setdefault("catalog_source", "versioned_production_catalog_json")
        else:
            catalog["catalog_source"] = "synthetic_fixture_json"
        catalog["catalog_source_sha256"] = sha256_file(path)
        catalog.setdefault("core_parameters", [str(value) for value in config.get("core_parameters") or []])
        catalog["catalog_fingerprint"] = content_id(
            {key: value for key, value in catalog.items() if not key.startswith("catalog_")}
        )
        return catalog
    if legacy_src is None:
        raise ValueError("Production planning requires --legacy-src for the existing wjy catalog")
    source = legacy_src.resolve()
    catalog_path = source / "physical_property_prediction" / "catalog.py"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Legacy catalog not found: {catalog_path}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("physical_property_prediction.catalog")
    parameters = {
        str(name): {
            "lower": float(spec.lower),
            "upper": float(spec.upper),
            "unit": str(spec.unit),
        }
        for name, spec in module.PARAMETERS.items()
    }
    numeric_names = set(str(value) for value in config.get("solver_numeric_parameters") or [])
    models: dict[str, Any] = {}
    core_parameters = [
        str(name)
        for name in config.get("core_parameters", ["longest_dimension_m", "mass_kg"])
        if str(name) in parameters
    ]
    for model_family, adapter in module.GENESIS_ADAPTERS.items():
        required = list(
            dict.fromkeys(
                [*core_parameters, *[str(name) for name in adapter.get("parameters") or [] if str(name) in parameters]]
            )
        )
        representations = [str(value) for value in adapter.get("supported_representations") or []]
        domain = (
            "surface_dynamics"
            if any("surface" in value or "shell" in value for value in representations)
            else "volumetric_soft_body"
        )
        models[str(model_family)] = {
            "simulation_domain": domain,
            "representations": representations or ["unspecified"],
            "solver": str(adapter.get("solver") or "unsupported"),
            "required_parameters": required,
            "fixed": adapter.get("fixed") or {},
        }
    allowlist = set(str(value) for value in config.get("allowed_model_families") or [])
    if allowlist:
        models = {key: value for key, value in models.items() if key in allowlist}
        unknown = sorted(allowlist - models.keys())
        if unknown:
            raise ValueError("Unknown allowed model families: " + ",".join(unknown))
    if not models:
        raise ValueError("Physical catalog exposes no allowed models")
    catalog = {
        "parameters": parameters,
        "models": models,
        "solver_numeric_parameters": sorted(numeric_names),
        "core_parameters": core_parameters,
        "catalog_source": "wjy.physical_property_prediction.catalog",
        "catalog_source_sha256": sha256_file(catalog_path),
    }
    catalog["catalog_fingerprint"] = content_id(
        {key: value for key, value in catalog.items() if not key.startswith("catalog_")}
    )
    return catalog


def _stable_shard(sample_id: str, num_shards: int) -> int:
    return int(content_id(sample_id).split(":", 1)[1][:16], 16) % num_shards


def _task_from_ledger(
    row: dict[str, Any],
    *,
    ledger_sha: str,
    config_sha: str,
    catalog_fingerprint: str,
    num_shards: int,
    fixture_ordinal: int,
) -> dict[str, Any]:
    source_image = next((item for item in row["artifacts"] if item["role"] == "source_image"), None)
    visual_mesh = next((item for item in row["artifacts"] if item["role"] == "visual_mesh"), None)
    payload = {
        "sample_id": row["sample_id"],
        "asset_revision_id": row["asset_revision_id"],
        "asset_ledger_sha256": ledger_sha,
        "physical_config_sha256": config_sha,
        "catalog_fingerprint": catalog_fingerprint,
        "prompt_version": None,
        "source_image_sha256": source_image.get("sha256") if source_image else None,
        "visual_mesh_sha256": visual_mesh.get("sha256") if visual_mesh else None,
    }
    return {
        "schema_name": "softsim.physical_task",
        "schema_version": 1,
        "sample_id": row["sample_id"],
        "asset_revision_id": row["asset_revision_id"],
        "dataset_release_id": row["dataset_release_id"],
        "object_type": row.get("object_type"),
        "geometry_family": row.get("geometry_family"),
        "material_family": row.get("material_family"),
        "geometry": row.get("geometry") or {},
        "metric_scale": row.get("metric_scale") or {},
        "source_image": source_image,
        "visual_mesh": visual_mesh,
        "shard_index": _stable_shard(str(row["sample_id"]), num_shards),
        "task_fingerprint": content_id(payload),
        "fixture_ordinal": fixture_ordinal,
    }


def plan_physics(
    *,
    asset_ledger: Path,
    output_root: Path,
    config_path: Path,
    legacy_src: Path | None,
    num_shards: int,
    expected_ready_count: int | None,
    selection_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    config = read_json(config_path)
    catalog = load_catalog(config, legacy_src, config_path.resolve().parent)
    ledger_sha = sha256_file(asset_ledger)
    config_sha = sha256_file(config_path)
    tasks_by_shard: dict[int, list[dict[str, Any]]] = {index: [] for index in range(num_shards)}
    selection = (
        {str(row["sample_id"]) for row in iter_jsonl(selection_path)}
        if selection_path is not None
        else None
    )
    selected_seen: set[str] = set()
    source_count = 0
    rejected_count = 0
    for ordinal, row in enumerate(iter_jsonl(asset_ledger)):
        source_count += 1
        validate_asset_ledger_record(row)
        if selection is not None and str(row["sample_id"]) not in selection:
            continue
        if row["asset_status"] not in {"ready", "ready_with_warnings"}:
            rejected_count += 1
            continue
        if limit is not None and sum(len(value) for value in tasks_by_shard.values()) >= limit:
            continue
        selected_seen.add(str(row["sample_id"]))
        task = _task_from_ledger(
            row,
            ledger_sha=ledger_sha,
            config_sha=config_sha,
            catalog_fingerprint=catalog["catalog_fingerprint"],
            num_shards=num_shards,
            fixture_ordinal=ordinal,
        )
        task["task_fingerprint"] = content_id(
            {
                **{key: value for key, value in task.items() if key != "task_fingerprint"},
                "prompt_version": config["prompt_version"],
            }
        )
        tasks_by_shard[task["shard_index"]].append(task)
    ready_count = sum(len(value) for value in tasks_by_shard.values())
    if selection is not None and selected_seen != selection:
        missing = sorted(selection - selected_seen)
        raise ValueError("selection_missing_or_not_geometry_ready:" + ",".join(missing[:100]))
    if expected_ready_count is not None and ready_count != expected_ready_count:
        raise ValueError(f"geometry_ready_count_mismatch:{expected_ready_count}:{ready_count}")
    plans_dir = output_root / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    shard_counts: dict[str, int] = {}
    for shard, tasks in tasks_by_shard.items():
        tasks.sort(key=lambda row: row["sample_id"])
        path = plans_dir / f"shard_{shard:05d}.jsonl"
        atomic_write_jsonl(path, tasks)
        shard_counts[str(shard)] = len(tasks)
    batch_size = int(config.get("batch_size", 1))
    semantic_attempts = int(config.get("max_semantic_attempts", 2))
    base_requests = 2 * sum((count + batch_size - 1) // batch_size for count in shard_counts.values())
    maximum_requests = base_requests * semantic_attempts
    estimated_input_tokens = ready_count * int(config.get("estimated_input_tokens_per_asset", 1800))
    estimated_output_tokens = ready_count * int(config.get("estimated_output_tokens_per_asset", 900))
    estimated_run_metadata_bytes = ready_count * int(config.get("estimated_run_bytes_per_asset", 65536))
    price_input = config.get("price_per_million_input_tokens")
    price_output = config.get("price_per_million_output_tokens")
    estimated_cost = None
    if price_input is not None and price_output is not None:
        estimated_cost = estimated_input_tokens * float(price_input) / 1_000_000 + estimated_output_tokens * float(price_output) / 1_000_000
    summary = {
        "schema_name": "softsim.physical_run_plan",
        "schema_version": 1,
        "asset_ledger_sha256": ledger_sha,
        "physical_config_sha256": config_sha,
        "catalog_source": catalog["catalog_source"],
        "catalog_source_sha256": catalog["catalog_source_sha256"],
        "catalog_fingerprint": catalog["catalog_fingerprint"],
        "source_asset_count": source_count,
        "geometry_ready_count": ready_count,
        "rejected_asset_count": rejected_count,
        "num_shards": num_shards,
        "shard_counts": shard_counts,
        "batch_size": batch_size,
        "base_request_count": base_requests,
        "maximum_request_count": maximum_requests,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_paid_cost": estimated_cost,
        "estimated_bundle_disk_bytes": ready_count * int(config.get("estimated_bundle_bytes_per_asset", 16_384)),
        "estimated_run_metadata_bytes": estimated_run_metadata_bytes,
        "estimated_total_output_bytes": estimated_run_metadata_bytes
        + ready_count * int(config.get("estimated_bundle_bytes_per_asset", 16_384)),
        "selection_sha256": sha256_file(selection_path) if selection_path else None,
        "deterministic_limit": limit,
        "paid_run_started": False,
        "config_snapshot": config,
    }
    atomic_write_json(output_root / "physical_run_plan.json", summary)
    atomic_write_json(output_root / "catalog_snapshot.json", catalog)
    return summary


def build_physics_selection(
    *,
    asset_ledger: Path,
    count: int,
    output: Path,
    strata_fields: list[str],
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("selection count must be positive")
    groups: dict[tuple[str, ...], list[str]] = {}
    for row in iter_jsonl(asset_ledger):
        validate_asset_ledger_record(row)
        if row["asset_status"] not in {"ready", "ready_with_warnings"}:
            continue
        key = tuple(str(row.get(field) or "unknown") for field in strata_fields)
        groups.setdefault(key, []).append(str(row["sample_id"]))
    queues = {key: deque(sorted(values)) for key, values in groups.items()}
    selected: list[dict[str, Any]] = []
    ordered_keys = sorted(queues)
    while len(selected) < count:
        made_progress = False
        for key in ordered_keys:
            if queues[key] and len(selected) < count:
                selected.append(
                    {
                        "sample_id": queues[key].popleft(),
                        "stratum": {field: value for field, value in zip(strata_fields, key)},
                    }
                )
                made_progress = True
        if not made_progress:
            break
    if len(selected) != count:
        raise ValueError(f"selection_count_unavailable:{count}:{len(selected)}")
    selected.sort(key=lambda row: row["sample_id"])
    atomic_write_jsonl(output, selected)
    summary = {
        "schema_name": "softsim.physical_selection",
        "schema_version": 1,
        "asset_ledger_sha256": sha256_file(asset_ledger),
        "count": len(selected),
        "strata_fields": strata_fields,
        "strata_available": len(groups),
        "selection_sha256": sha256_file(output),
    }
    atomic_write_json(output.with_suffix(".summary.json"), summary)
    return summary


def _prepared_valid(path: Path, task_fingerprint: str) -> bool:
    if not path.is_file():
        return False
    prepared = read_json(path)
    if prepared.get("task_fingerprint") != task_fingerprint:
        return False
    for item in prepared.get("verified_inputs") or []:
        absolute = Path(str(item["absolute_path"]))
        if not absolute.is_file() or sha256_file(absolute) != item["sha256"]:
            return False
    return True


def _prepare_one(
    task: dict[str, Any],
    run_root: Path,
    roots: dict[str, Path],
    allow_metadata_only: bool,
) -> dict[str, Any]:
    shard = int(task["shard_index"])
    task_dir = run_root / "shards" / f"{shard:05d}" / "tasks" / str(task["sample_id"])
    path = task_dir / "PREPARED.json"
    if _prepared_valid(path, str(task["task_fingerprint"])):
        return {"sample_id": task["sample_id"], "status": "skipped"}
    verified: list[dict[str, Any]] = []
    for role in ("source_image", "visual_mesh"):
        artifact = task.get(role)
        if not artifact:
            if role == "source_image" and allow_metadata_only:
                continue
            raise ValueError(f"{task['sample_id']}:{role}_artifact_missing")
        alias = str(artifact["root_alias"])
        if alias not in roots:
            if allow_metadata_only:
                continue
            raise ValueError(f"{task['sample_id']}:root_alias_not_mounted:{alias}")
        absolute = safe_join(roots[alias], str(artifact["path"]), require_exists=True)
        digest = sha256_file(absolute)
        if digest != artifact["sha256"]:
            raise ValueError(f"{task['sample_id']}:{role}_hash_mismatch")
        verified.append(
            {
                "role": role,
                "root_alias": alias,
                "path": artifact["path"],
                "sha256": digest,
                "absolute_path": str(absolute),
            }
        )
    prepared = {
        "schema_name": "softsim.physical_prepared_input",
        "schema_version": 1,
        "sample_id": task["sample_id"],
        "task_fingerprint": task["task_fingerprint"],
        "input_mode": "metadata_only" if not any(item["role"] == "source_image" for item in verified) else "source_image",
        "verified_inputs": verified,
    }
    atomic_write_json(path, prepared)
    return {"sample_id": task["sample_id"], "status": "prepared"}


def prepare_physics(
    *,
    run_root: Path,
    shard_index: int,
    roots: dict[str, Path],
    workers: int,
    allow_metadata_only: bool,
) -> dict[str, Any]:
    plan_path = run_root / "plans" / f"shard_{shard_index:05d}.jsonl"
    tasks = list(iter_jsonl(plan_path))
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_prepare_one, task, run_root, roots, allow_metadata_only): task["sample_id"]
            for task in tasks
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except BaseException as exc:
                results.append(
                    {
                        "sample_id": futures[future],
                        "status": "failed",
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
    summary = {
        "schema_version": 1,
        "shard_index": shard_index,
        "total": len(tasks),
        "prepared": sum(row["status"] == "prepared" for row in results),
        "skipped": sum(row["status"] == "skipped" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
    }
    atomic_write_json(run_root / "shards" / f"{shard_index:05d}" / "prepare_summary.json", summary)
    return summary


def _task_directory(run_root: Path, task: dict[str, Any]) -> Path:
    return run_root / "shards" / f"{int(task['shard_index']):05d}" / "tasks" / str(task["sample_id"])


def _success_valid(task_dir: Path, task_fingerprint: str) -> bool:
    path = task_dir / "SUCCESS.json"
    if not path.is_file():
        return False
    marker = read_json(path)
    if marker.get("task_fingerprint") != task_fingerprint:
        return False
    bundle = task_dir / str(marker.get("physics_path") or "")
    return bundle.is_file() and sha256_file(bundle) == marker.get("physics_sha256")


def _routes(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "model_family": name,
            "simulation_domain": spec["simulation_domain"],
            "representations": spec["representations"],
        }
        for name, spec in sorted(catalog["models"].items())
    ]


def _validate_routing(
    task: dict[str, Any], value: Any, catalog: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(value, dict):
        return None, "routing_not_object"
    if value.get("sample_id") != task["sample_id"]:
        return None, "routing_sample_id_mismatch"
    model_family = str(value.get("model_family") or "")
    model = catalog["models"].get(model_family)
    if not model:
        return None, "routing_model_unknown"
    representation = str(value.get("representation") or "")
    if representation not in model["representations"]:
        return None, "routing_representation_incompatible"
    if value.get("simulation_domain") != model["simulation_domain"]:
        return None, "routing_domain_incompatible"
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        return None, "routing_confidence_invalid"
    return {
        "simulation_domain": model["simulation_domain"],
        "representation": representation,
        "model_family": model_family,
        "confidence": float(confidence),
        "limitations": [str(item) for item in value.get("limitations") or []],
    }, None


def _parameter_specs(route: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    model = catalog["models"][route["model_family"]]
    return [
        {"name": name, **catalog["parameters"][name]}
        for name in model["required_parameters"]
    ]


def _validate_parameters(
    task: dict[str, Any],
    route: dict[str, Any],
    value: Any,
    catalog: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not isinstance(value, dict) or value.get("sample_id") != task["sample_id"]:
        return None, "parameter_sample_id_mismatch"
    values = value.get("parameters")
    if not isinstance(values, list):
        return None, "parameters_not_list"
    by_name: dict[str, dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict) or not item.get("name"):
            return None, "parameter_item_invalid"
        name = str(item["name"])
        if name in by_name:
            return None, f"parameter_duplicate:{name}"
        by_name[name] = item
    output: list[dict[str, Any]] = []
    for spec in _parameter_specs(route, catalog):
        name = spec["name"]
        item = by_name.get(name)
        if item is None:
            return None, f"parameter_missing:{name}"
        value_number = item.get("value")
        confidence = item.get("confidence")
        if isinstance(value_number, bool) or not isinstance(value_number, (int, float)):
            return None, f"parameter_non_numeric:{name}"
        if not float(spec["lower"]) <= float(value_number) <= float(spec["upper"]):
            return None, f"parameter_out_of_range:{name}"
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            return None, f"parameter_confidence_invalid:{name}"
        output.append(
            {
                "name": name,
                "value": float(value_number),
                "unit": spec["unit"],
                "valid_range": [float(spec["lower"]), float(spec["upper"])],
                "confidence": float(confidence),
                "source_type": "vlm_prior",
                "evidence_refs": [],
                "rationale": str(item.get("rationale") or ""),
            }
        )
    unexpected = sorted(set(by_name) - {item["name"] for item in output})
    if unexpected:
        return None, "unexpected_parameters:" + ",".join(unexpected)
    return output, None


def _compile_bundle(
    *,
    task: dict[str, Any],
    route: dict[str, Any],
    parameters: list[dict[str, Any]],
    catalog: dict[str, Any],
    estimator: dict[str, Any],
    assignment_method: str,
    limitations: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    numeric_names = set(catalog.get("solver_numeric_parameters") or [])
    physical_parameters = [item for item in parameters if item["name"] not in numeric_names]
    solver_numerics = {
        item["name"]: {"value": item["value"], "unit": item["unit"], "source_type": item["source_type"]}
        for item in parameters
        if item["name"] in numeric_names
    }
    model = catalog["models"][route["model_family"]]
    record: dict[str, Any] = {
        "schema_name": "softsim.physical_bundle",
        "schema_version": 1,
        "sample_id": task["sample_id"],
        "asset_revision_id": task["asset_revision_id"],
        "physical_bundle_id": None,
        "status": "predicted",
        "simulation_domain": route["simulation_domain"],
        "representation": route["representation"],
        "model_family": route["model_family"],
        "parameters": physical_parameters,
        "solver_adapter": {
            "solver": model["solver"],
            "parameters": {item["name"]: item["value"] for item in physical_parameters},
            "fixed": model.get("fixed") or {},
        },
        "solver_numerics": solver_numerics,
        "proxy_artifacts": [],
        "estimator": estimator,
        "agent": {
            "policy_version": None,
            "trace_ref": None,
            "iteration_count": 0,
            "termination_reason": "prior_only",
        },
        "verification": {
            "calibration_probe_suite_hash": None,
            "heldout_probe_suite_hash": None,
            "calibration_result": None,
            "heldout_result": None,
            "acceptance_contract_hash": None,
        },
        "limitations": sorted(set(["not_empirically_calibrated", "not_sim_verified", *limitations])),
        "warnings": sorted(set(warnings)),
        "provenance": {
            "dataset_release_id": task["dataset_release_id"],
            "assignment_method": assignment_method,
            "catalog_fingerprint": catalog["catalog_fingerprint"],
            "task_fingerprint": task["task_fingerprint"],
            "is_ground_truth": False,
            "simulation_calibrated": False,
        },
    }
    record["physical_bundle_id"] = physical_bundle_id(record)
    record = with_record_hash(record)
    validate_physical_bundle(record)
    return record


def _unsupported_bundle(
    task: dict[str, Any], catalog: dict[str, Any], estimator: dict[str, Any], reason: str
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_name": "softsim.physical_bundle",
        "schema_version": 1,
        "sample_id": task["sample_id"],
        "asset_revision_id": task["asset_revision_id"],
        "physical_bundle_id": None,
        "status": "unsupported",
        "simulation_domain": "unsupported",
        "representation": "unsupported",
        "model_family": "unsupported",
        "parameters": [],
        "solver_adapter": {},
        "solver_numerics": {},
        "proxy_artifacts": [],
        "estimator": estimator,
        "agent": {"policy_version": None, "trace_ref": None, "iteration_count": 0, "termination_reason": "unsupported"},
        "verification": {
            "calibration_probe_suite_hash": None,
            "heldout_probe_suite_hash": None,
            "calibration_result": None,
            "heldout_result": None,
            "acceptance_contract_hash": None,
        },
        "limitations": [reason, "not_empirically_calibrated", "not_sim_verified"],
        "warnings": [],
        "provenance": {
            "dataset_release_id": task["dataset_release_id"],
            "assignment_method": "unsupported",
            "catalog_fingerprint": catalog["catalog_fingerprint"],
            "task_fingerprint": task["task_fingerprint"],
            "is_ground_truth": False,
            "simulation_calibrated": False,
        },
    }
    record["physical_bundle_id"] = physical_bundle_id(record)
    record = with_record_hash(record)
    validate_physical_bundle(record)
    return record


def _fallback_bundle(
    task: dict[str, Any],
    catalog: dict[str, Any],
    fallback_policy: dict[str, Any],
    estimator: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    for profile in fallback_policy.get("profiles") or []:
        match = profile.get("match") or {}
        if any(task.get(key) != value for key, value in match.items()):
            continue
        if not profile.get("review_id"):
            raise ValueError("fallback_profile_missing_review_id")
        route = {
            "model_family": profile["model_family"],
            "representation": profile["representation"],
            "simulation_domain": profile["simulation_domain"],
            "confidence": float(profile.get("confidence", 0.0)),
            "limitations": [str(item) for item in profile.get("limitations") or []],
        }
        model = catalog["models"].get(route["model_family"])
        if not model or route["representation"] not in model["representations"]:
            raise ValueError("fallback_profile_incompatible_with_catalog")
        raw_parameters = profile.get("parameters") or {}
        values: list[dict[str, Any]] = []
        for spec in _parameter_specs(route, catalog):
            if spec["name"] not in raw_parameters:
                raise ValueError(f"fallback_profile_parameter_missing:{spec['name']}")
            value = float(raw_parameters[spec["name"]])
            if not float(spec["lower"]) <= value <= float(spec["upper"]):
                raise ValueError(f"fallback_profile_parameter_out_of_range:{spec['name']}")
            values.append(
                {
                    "name": spec["name"],
                    "value": value,
                    "unit": spec["unit"],
                    "valid_range": [float(spec["lower"]), float(spec["upper"])],
                    "confidence": float(profile.get("confidence", 0.0)),
                    "source_type": "reviewed_family_prior",
                    "evidence_refs": [str(profile["review_id"])],
                    "rationale": "explicit reviewed fallback profile",
                }
            )
        return _compile_bundle(
            task=task,
            route=route,
            parameters=values,
            catalog=catalog,
            estimator=estimator,
            assignment_method="reviewed_family_prior",
            limitations=[reason, *route["limitations"]],
            warnings=["vlm_prediction_unavailable_or_invalid"],
        )
    return _unsupported_bundle(task, catalog, estimator, reason)


def _commit_bundle(run_root: Path, task: dict[str, Any], bundle: dict[str, Any]) -> None:
    task_dir = _task_directory(run_root, task)
    revision = str(task["task_fingerprint"]).split(":", 1)[1]
    relative = f"attempts/{revision}/physics.json"
    path = task_dir / relative
    atomic_write_json(path, bundle)
    marker = {
        "schema_name": "softsim.physical_task_success",
        "schema_version": 1,
        "sample_id": task["sample_id"],
        "task_fingerprint": task["task_fingerprint"],
        "physical_bundle_id": bundle["physical_bundle_id"],
        "status": bundle["status"],
        "physics_path": relative,
        "physics_sha256": sha256_file(path),
    }
    atomic_write_json(task_dir / "SUCCESS.json", marker)


class _ShardRunner:
    def __init__(
        self,
        *,
        run_root: Path,
        config: dict[str, Any],
        catalog: dict[str, Any],
        fallback_policy: dict[str, Any],
        transport: QwenTransport | MockTransport,
        workers: int,
        shard_index: int,
    ):
        self.run_root = run_root
        self.config = config
        self.catalog = catalog
        self.fallback_policy = fallback_policy
        self.transport = transport
        self.shard_index = shard_index
        self.api_root = run_root / "shards" / f"{shard_index:05d}" / "api"
        self.semaphore = asyncio.Semaphore(max(1, workers))
        self.limiter = SlidingWindowLimiter(
            int(config.get("requests_per_minute", 60)),
            int(config.get("tokens_per_minute", 100_000)),
        )
        self.request_count = 0
        self.cache_hits = 0
        self.transport_errors: Counter[str] = Counter()
        self.in_flight = 0
        self.observed_peak_in_flight = 0
        # Prepared inputs are content-pinned and their roots are mounted
        # read-only for a run. Avoid re-hashing the same files for routing and
        # parameter stages while retaining the first-use integrity check.
        self._prepared_cache: dict[str, dict[str, Any]] = {}

    def _prepared(self, task: dict[str, Any]) -> dict[str, Any]:
        fingerprint = str(task["task_fingerprint"])
        cached = self._prepared_cache.get(fingerprint)
        if cached is not None:
            return cached
        path = _task_directory(self.run_root, task) / "PREPARED.json"
        if not _prepared_valid(path, fingerprint):
            raise ValueError(f"{task['sample_id']}:prepared_input_missing_or_stale")
        prepared = read_json(path)
        self._prepared_cache[fingerprint] = prepared
        return prepared

    async def _request(
        self,
        *,
        stage: str,
        semantic_attempt: int,
        items: list[dict[str, Any]],
        tasks: list[dict[str, Any]],
        prior_issues: dict[str, str],
    ) -> tuple[dict[str, Any], str]:
        if stage == "routing":
            instruction = (
                "Return exactly one assets entry per requested sample_id using ONLY the routing fields "
                "listed below. Each entry MUST contain sample_id, model_family, simulation_domain, "
                "representation, confidence (number 0..1), and limitations (array of strings). "
                "Use model_family and representation values exactly from allowed_routes. "
                "Do not put these fields inside simulation_priors, recommended_route, or any nested object; "
                "do not return an alternate schema. Values are uncertain priors only; do not claim "
                "simulation calibration or verification."
            )
            output_contract = {
                "assets_entry": {
                    "sample_id": "string (exact requested id)",
                    "model_family": "string (exact allowed route)",
                    "simulation_domain": "string (from the selected route)",
                    "representation": "string (from the selected route)",
                    "confidence": "number in [0,1]",
                    "limitations": ["string"],
                }
            }
        elif stage == "parameters":
            instruction = (
                "Return exactly one assets entry per requested sample_id. Each entry MUST contain only "
                "sample_id and parameters. parameters MUST be an array containing every required parameter "
                "exactly once. Copy each required parameter name verbatim; each item MUST contain name, "
                "numeric value, confidence (number 0..1), and rationale. Keep values within the supplied "
                "valid ranges. Do not use aliases such as density_kg_per_m3 or yield_strength_pa, and do "
                "not put values inside simulation_priors. Values are uncertain priors only; do not claim "
                "simulation calibration or verification."
            )
            output_contract = {
                "assets_entry": {
                    "sample_id": "string (exact requested id)",
                    "parameters": [
                        {
                            "name": "exact required_parameters.name",
                            "value": "number within the supplied lower/upper range",
                            "confidence": "number in [0,1]",
                            "rationale": "string",
                        }
                    ],
                    "parameter_rule": "include every required parameter exactly once",
                }
            }
        else:
            raise ValueError(f"unknown_physical_stage:{stage}")
        request_object = {
            "instruction": instruction,
            "output_contract": output_contract,
            "stage": stage,
            "prompt_version": self.config["prompt_version"],
            "semantic_attempt": semantic_attempt,
            "prior_validation_issues": prior_issues,
            "assets": items,
        }
        prompt = canonical_json_bytes(request_object).decode("utf-8")
        image_paths: list[Path] = []
        image_hashes: list[str] = []
        for task in tasks:
            prepared = self._prepared(task)
            for input_item in prepared["verified_inputs"]:
                if input_item["role"] == "source_image":
                    image_paths.append(Path(input_item["absolute_path"]))
                    image_hashes.append(input_item["sha256"])
        request_identity = {
            "model_revision": self.transport.model_revision,
            "prompt_sha256": content_id(prompt),
            "image_sha256": image_hashes,
            "temperature": self.config.get("temperature", 0.1),
            "max_tokens": self.config.get("max_tokens", 4096),
        }
        request_fingerprint = content_id(request_identity)
        fingerprint_name = request_fingerprint.split(":", 1)[1]
        cache_path = self.api_root / "cache" / fingerprint_name / "response.json"
        if cache_path.is_file():
            cached = read_json(cache_path)
            self.cache_hits += 1
            return cached["response"], request_fingerprint
        estimated_tokens = max(
            1,
            len(prompt) // 4 + len(items) * int(self.config.get("estimated_output_tokens_per_asset", 900)),
        )
        await self.limiter.acquire(estimated_tokens)
        async with self.semaphore:
            self.in_flight += 1
            self.observed_peak_in_flight = max(self.observed_peak_in_flight, self.in_flight)
            try:
                result, attempts = await complete_with_retries(
                    self.transport,
                    prompt,
                    image_paths,
                    max_retries=int(self.config.get("max_transport_retries", 5)),
                    backoff_seconds=float(self.config.get("retry_backoff_seconds", 2.0)),
                    jitter_ratio=float(self.config.get("retry_jitter_ratio", 0.25)),
                )
            except TransportError as exc:
                self.transport_errors[exc.category] += 1
                raise
            finally:
                self.in_flight -= 1
        self.request_count += 1
        record = {
            "schema_name": "softsim.physical_api_request",
            "schema_version": 1,
            "request_fingerprint": request_fingerprint,
            "stage": stage,
            "semantic_attempt": semantic_attempt,
            "sample_ids": [task["sample_id"] for task in tasks],
            "model": self.transport.model,
            "model_revision": self.transport.model_revision,
            "image_sha256": image_hashes,
            "transport_attempts": attempts,
            "metadata": result.metadata,
            "response": result.response,
        }
        atomic_write_json(cache_path, record)
        atomic_write_json(self.api_root / "request_records" / f"{fingerprint_name}.json", record)
        return result.response, request_fingerprint

    async def _routing_stage(
        self, tasks: list[dict[str, Any]]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[str]]:
        resolved: dict[str, dict[str, Any]] = {}
        issues: dict[str, str] = {}
        fingerprints: list[str] = []
        pending = {str(task["sample_id"]): task for task in tasks}
        for semantic_attempt in range(1, int(self.config.get("max_semantic_attempts", 2)) + 1):
            if not pending:
                break
            active = list(pending.values())
            items = [
                {
                    "sample_id": task["sample_id"],
                    "object_type": task.get("object_type"),
                    "geometry_family": task.get("geometry_family"),
                    "material_family": task.get("material_family"),
                    "geometry": task.get("geometry"),
                    "metric_scale": task.get("metric_scale"),
                    "allowed_routes": _routes(self.catalog),
                    "fixture_ordinal": task.get("fixture_ordinal", 0),
                }
                for task in active
            ]
            response, fingerprint = await self._request(
                stage="routing",
                semantic_attempt=semantic_attempt,
                items=items,
                tasks=active,
                prior_issues=issues,
            )
            fingerprints.append(fingerprint)
            values = response.get("assets") if isinstance(response, dict) else None
            by_sample = {
                str(value.get("sample_id")): value
                for value in values or []
                if isinstance(value, dict) and value.get("sample_id")
            }
            for sample_id, task in list(pending.items()):
                route, issue = _validate_routing(task, by_sample.get(sample_id), self.catalog)
                if route is not None:
                    resolved[sample_id] = route
                    pending.pop(sample_id)
                    issues.pop(sample_id, None)
                else:
                    issues[sample_id] = issue or "routing_unknown_error"
        return resolved, {sample_id: issues.get(sample_id, "routing_unresolved") for sample_id in pending}, fingerprints

    async def _parameter_stage(
        self,
        tasks: list[dict[str, Any]],
        routes: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], list[str]]:
        resolved: dict[str, list[dict[str, Any]]] = {}
        issues: dict[str, str] = {}
        fingerprints: list[str] = []
        pending = {str(task["sample_id"]): task for task in tasks if str(task["sample_id"]) in routes}
        for semantic_attempt in range(1, int(self.config.get("max_semantic_attempts", 2)) + 1):
            if not pending:
                break
            active = list(pending.values())
            items = [
                {
                    "sample_id": task["sample_id"],
                    "route": routes[str(task["sample_id"])],
                    "required_parameters": _parameter_specs(routes[str(task["sample_id"])], self.catalog),
                    "fixture_ordinal": task.get("fixture_ordinal", 0),
                }
                for task in active
            ]
            response, fingerprint = await self._request(
                stage="parameters",
                semantic_attempt=semantic_attempt,
                items=items,
                tasks=active,
                prior_issues=issues,
            )
            fingerprints.append(fingerprint)
            values = response.get("assets") if isinstance(response, dict) else None
            by_sample = {
                str(value.get("sample_id")): value
                for value in values or []
                if isinstance(value, dict) and value.get("sample_id")
            }
            for sample_id, task in list(pending.items()):
                parameters, issue = _validate_parameters(task, routes[sample_id], by_sample.get(sample_id), self.catalog)
                if parameters is not None:
                    resolved[sample_id] = parameters
                    pending.pop(sample_id)
                    issues.pop(sample_id, None)
                else:
                    issues[sample_id] = issue or "parameters_unknown_error"
        return resolved, {sample_id: issues.get(sample_id, "parameters_unresolved") for sample_id in pending}, fingerprints

    async def process_batch(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            routes, route_issues, routing_fingerprints = await self._routing_stage(tasks)
            parameters, parameter_issues, parameter_fingerprints = await self._parameter_stage(tasks, routes)
        except FatalTransportError:
            raise
        except (TransportError, ValueError) as exc:
            routes = {}
            parameters = {}
            route_issues = {str(task["sample_id"]): f"transport_failure:{type(exc).__name__}" for task in tasks}
            parameter_issues = {}
            routing_fingerprints = []
            parameter_fingerprints = []
        outcomes: list[dict[str, Any]] = []
        is_mock = isinstance(self.transport, MockTransport)
        for task in tasks:
            sample_id = str(task["sample_id"])
            request_fingerprints = [*routing_fingerprints, *parameter_fingerprints]
            estimator = {
                "model": self.transport.model,
                "model_revision": self.transport.model_revision,
                "prompt_version": self.config["prompt_version"],
                "request_fingerprint": content_id(request_fingerprints),
                "request_fingerprints": request_fingerprints,
                "transport": "mock" if is_mock else "qwen",
            }
            if sample_id in routes and sample_id in parameters:
                values = parameters[sample_id]
                if is_mock:
                    values = [{**item, "source_type": "synthetic_fixture"} for item in values]
                bundle = _compile_bundle(
                    task=task,
                    route=routes[sample_id],
                    parameters=values,
                    catalog=self.catalog,
                    estimator=estimator,
                    assignment_method="synthetic_fixture" if is_mock else "qwen_vlm_prior",
                    limitations=routes[sample_id]["limitations"],
                    warnings=[],
                )
            else:
                reason = route_issues.get(sample_id) or parameter_issues.get(sample_id) or "semantic_validation_failed"
                bundle = _fallback_bundle(task, self.catalog, self.fallback_policy, estimator, reason)
            _commit_bundle(self.run_root, task, bundle)
            outcomes.append({"sample_id": sample_id, "status": bundle["status"]})
        return outcomes


async def _run_shard_async(
    *,
    run_root: Path,
    shard_index: int,
    config: dict[str, Any],
    catalog: dict[str, Any],
    fallback_policy: dict[str, Any],
    transport: QwenTransport | MockTransport,
    workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    tasks = list(iter_jsonl(run_root / "plans" / f"shard_{shard_index:05d}.jsonl"))
    pending: list[dict[str, Any]] = []
    resumed = 0
    for task in tasks:
        task_dir = _task_directory(run_root, task)
        if _success_valid(task_dir, str(task["task_fingerprint"])):
            resumed += 1
        else:
            pending.append(task)
    batch_size = max(1, int(config.get("batch_size", 1)))
    batches = [pending[index : index + batch_size] for index in range(0, len(pending), batch_size)]
    runner = _ShardRunner(
        run_root=run_root,
        config=config,
        catalog=catalog,
        fallback_policy=fallback_policy,
        transport=transport,
        workers=workers,
        shard_index=shard_index,
    )
    results: list[dict[str, Any]] = []
    futures = [asyncio.create_task(runner.process_batch(batch)) for batch in batches]
    try:
        for future in asyncio.as_completed(futures):
            results.extend(await future)
    except FatalTransportError:
        for future in futures:
            future.cancel()
        raise
    by_status = Counter(row["status"] for row in results)
    summary = {
        "schema_version": 1,
        "shard_index": shard_index,
        "total": len(tasks),
        "resumed": resumed,
        "newly_completed": len(results),
        "by_status": dict(sorted(by_status.items())),
        "api_requests": runner.request_count,
        "request_cache_hits": runner.cache_hits,
        "transport_errors": dict(sorted(runner.transport_errors.items())),
        "transport": "mock" if isinstance(transport, MockTransport) else "qwen",
        "publishable": not isinstance(transport, MockTransport),
        "requested_workers": workers,
        "wall_time_seconds": time.monotonic() - started,
        "observed_peak_in_flight": runner.observed_peak_in_flight,
    }
    summary_path = run_root / "shards" / f"{shard_index:05d}" / "run_summary.json"
    if summary_path.is_file():
        previous = read_json(summary_path)
        summary["observed_peak_in_flight"] = max(
            int(previous.get("observed_peak_in_flight", 0)), summary["observed_peak_in_flight"]
        )
        summary["cumulative_request_records"] = len(
            list((run_root / "shards" / f"{shard_index:05d}" / "api" / "request_records").glob("*.json"))
        )
    atomic_write_json(summary_path, summary)
    return summary


def run_physics_shard(
    *,
    run_root: Path,
    shard_index: int,
    config_path: Path,
    fallback_policy_path: Path,
    transport_name: str,
    workers: int,
    confirm_paid_run: bool,
) -> dict[str, Any]:
    config = read_json(config_path)
    planned = read_json(run_root / "physical_run_plan.json")
    if sha256_file(config_path) != planned["physical_config_sha256"]:
        raise ValueError("physical_config_changed_since_plan")
    catalog = read_json(run_root / "catalog_snapshot.json")
    fallback_policy = read_json(fallback_policy_path)
    if transport_name == "qwen":
        if not confirm_paid_run:
            raise ValueError("Paid Qwen run requires --confirm-paid-run")
        transport: QwenTransport | MockTransport = QwenTransport(config)
    elif transport_name == "mock":
        transport = MockTransport(config.get("mock_transport") or {})
    else:
        raise ValueError(f"Unknown transport: {transport_name}")
    return asyncio.run(
        _run_shard_async(
            run_root=run_root,
            shard_index=shard_index,
            config=config,
            catalog=catalog,
            fallback_policy=fallback_policy,
            transport=transport,
            workers=workers,
        )
    )


def _all_tasks(run_root: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for path in sorted((run_root / "plans").glob("shard_*.jsonl")):
        tasks.extend(iter_jsonl(path))
    tasks.sort(key=lambda row: row["sample_id"])
    return tasks


def aggregate_physics_release(
    *,
    run_root: Path,
    asset_release_dir: Path,
    physical_release_id: str,
    strict: bool,
    allow_nonpublishable: bool,
    output_root: Path | None = None,
) -> dict[str, Any]:
    destination = (output_root or asset_release_dir) / "physical" / physical_release_id
    plan = read_json(run_root / "physical_run_plan.json")
    asset_release = read_json(asset_release_dir / "release_manifest.json")
    if plan["asset_ledger_sha256"] != asset_release["asset_ledger_sha256"]:
        raise ValueError("physical_plan_asset_ledger_mismatch")
    if destination.exists():
        release = read_json(destination / "physical_release_manifest.json")
        expected = (destination / "physical_bundle_index.sha256").read_text(encoding="utf-8").strip()
        if sha256_file(destination / "physical_bundle_index.jsonl") != expected:
            raise ValueError("existing_physical_release_hash_mismatch")
        return release
    tasks = _all_tasks(run_root)
    seen: set[str] = set()
    records: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    failures: list[dict[str, Any]] = []
    publishable = True
    for task in tasks:
        sample_id = str(task["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate_physical_task:{sample_id}")
        seen.add(sample_id)
        task_dir = _task_directory(run_root, task)
        if not _success_valid(task_dir, str(task["task_fingerprint"])):
            failures.append({"sample_id": sample_id, "failure": "missing_or_stale_success"})
            continue
        marker = read_json(task_dir / "SUCCESS.json")
        bundle_path = task_dir / marker["physics_path"]
        bundle = read_json(bundle_path)
        try:
            validate_physical_bundle(bundle)
            if bundle["sample_id"] != sample_id or bundle["asset_revision_id"] != task["asset_revision_id"]:
                raise ValueError("bundle_identity_mismatch")
        except ValueError as exc:
            failures.append({"sample_id": sample_id, "failure": str(exc)})
            continue
        if bundle["estimator"].get("transport") != "qwen":
            publishable = False
        records.append((task, bundle, bundle_path))
    if strict and failures:
        raise ValueError(f"physical_aggregate_incomplete:{len(failures)}")
    if not publishable and not allow_nonpublishable:
        raise ValueError("mock_or_synthetic_results_cannot_be_published_without_explicit_fixture_override")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{physical_release_id}.", dir=destination.parent))
    try:
        index_rows: list[dict[str, Any]] = []
        by_status: Counter[str] = Counter()
        by_method: Counter[str] = Counter()
        by_family: dict[str, Counter[str]] = {}
        for task, bundle, source_path in records:
            sample_id = str(task["sample_id"])
            target = temporary / "bundles" / sample_id / "physics.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
            target_sha = sha256_file(target)
            method = str(bundle["provenance"]["assignment_method"])
            index = with_record_hash(
                {
                    "schema_name": "softsim.physical_bundle_index",
                    "schema_version": 1,
                    "physical_release_id": physical_release_id,
                    "sample_id": sample_id,
                    "asset_revision_id": bundle["asset_revision_id"],
                    "physical_bundle_id": bundle["physical_bundle_id"],
                    "status": bundle["status"],
                    "assignment_method": method,
                    "confidence": (
                        sum(float(item["confidence"]) for item in bundle["parameters"]) / len(bundle["parameters"])
                        if bundle["parameters"]
                        else 0.0
                    ),
                    "bundle": {
                        "root_alias": "physical_release",
                        "path": f"bundles/{sample_id}/physics.json",
                        "sha256": target_sha,
                        "size_bytes": target.stat().st_size,
                    },
                }
            )
            index_rows.append(index)
            by_status[bundle["status"]] += 1
            by_method[method] += 1
            by_family.setdefault(str(task.get("geometry_family") or "unknown"), Counter())[bundle["status"]] += 1
        index_rows.sort(key=lambda row: row["sample_id"])
        atomic_write_jsonl(temporary / "physical_bundle_index.jsonl", index_rows)
        index_sha = sha256_file(temporary / "physical_bundle_index.jsonl")
        with atomic_text_writer(temporary / "physical_bundle_index.sha256") as handle:
            handle.write(index_sha + "\n")
        summaries = temporary / "summaries"
        atomic_write_json(
            summaries / "coverage.json",
            {
                "expected_geometry_ready": len(tasks),
                "bundle_presence": len(index_rows),
                "missing_or_invalid": len(failures),
                "by_status": dict(sorted(by_status.items())),
                "by_assignment_method": dict(sorted(by_method.items())),
                "vlm_success": by_method.get("qwen_vlm_prior", 0),
                "fallback": by_method.get("reviewed_family_prior", 0),
                "unsupported": by_status.get("unsupported", 0),
                "calibrated": by_status.get("calibrated_semantic", 0) + by_status.get("calibrated_behavior", 0),
                "verified": by_status.get("verified", 0),
            },
        )
        atomic_write_json(
            summaries / "status_by_family.json",
            {family: dict(sorted(counts.items())) for family, counts in sorted(by_family.items())},
        )
        atomic_write_jsonl(summaries / "failures.jsonl", failures)
        request_by_fingerprint: dict[str, dict[str, Any]] = {}
        for path in sorted((run_root / "shards").glob("*/api/request_records/*.json")):
            request = read_json(path)
            fingerprint = str(request["request_fingerprint"])
            previous = request_by_fingerprint.get(fingerprint)
            if previous is not None and previous != request:
                raise ValueError(f"request_fingerprint_collision:{fingerprint}")
            request_by_fingerprint[fingerprint] = request
        request_records = [request_by_fingerprint[key] for key in sorted(request_by_fingerprint)]
        api_dir = temporary / "api"
        atomic_write_jsonl(api_dir / "request_records.jsonl", request_records)
        latencies = sorted(
            float(record.get("metadata", {}).get("latency_seconds", 0.0)) for record in request_records
        )

        def percentile(values: list[float], fraction: float) -> float | None:
            if not values:
                return None
            return values[min(len(values) - 1, int(round((len(values) - 1) * fraction)))]

        retry_categories: Counter[str] = Counter()
        total_transport_attempts = 0
        for record in request_records:
            for attempt in record.get("transport_attempts") or []:
                total_transport_attempts += 1
                if attempt.get("category"):
                    retry_categories[str(attempt["category"])] += 1
        shard_summaries = [
            read_json(path) for path in sorted((run_root / "shards").glob("*/run_summary.json"))
        ]
        concurrency_report = {
            "schema_version": 1,
            "request_records": len(request_records),
            "total_transport_attempts": total_transport_attempts,
            "retry_categories": dict(sorted(retry_categories.items())),
            "rate_limit_count": retry_categories.get("rate_limited", 0),
            "timeout_count": retry_categories.get("timeout", 0),
            "server_error_count": retry_categories.get("server_error", 0),
            "p50_latency_seconds": percentile(latencies, 0.50),
            "p95_latency_seconds": percentile(latencies, 0.95),
            "requested_workers_by_shard": {
                str(summary["shard_index"]): summary.get("requested_workers") for summary in shard_summaries
            },
            "observed_peak_in_flight_by_shard": {
                str(summary["shard_index"]): summary.get("observed_peak_in_flight") for summary in shard_summaries
            },
            "transport": "qwen" if publishable else "mock_or_synthetic",
        }
        atomic_write_json(api_dir / "concurrency_report.json", concurrency_report)
        request_records_sha = sha256_file(api_dir / "request_records.jsonl")
        release = {
            "schema_name": "softsim.physical_release",
            "schema_version": 1,
            "physical_release_id": physical_release_id,
            "dataset_release_id": asset_release["dataset_release_id"],
            "asset_ledger_sha256": plan["asset_ledger_sha256"],
            "physical_bundle_index_sha256": index_sha,
            "request_records_sha256": request_records_sha,
            "expected_asset_count": len(tasks),
            "bundle_count": len(index_rows),
            "publishable": publishable,
            "transport": "qwen" if publishable else "mock_or_synthetic",
            "status_counts": dict(sorted(by_status.items())),
            "immutable": True,
        }
        atomic_write_json(temporary / "physical_release_manifest.json", release)
        os.replace(temporary, destination)
        return release
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
