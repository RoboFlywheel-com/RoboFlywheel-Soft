# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .agent import build_tasks, run_task
from .aggregate import aggregate_run
from .audit import build_golden_reference_audit
from .input_adapter import load_asset_release, load_legacy_assets, load_previous_staging, load_stage3_manifest, merge_rows
from .io import atomic_write_json, atomic_write_jsonl, content_hash, iter_jsonl, read_json, resolve_roots
from .proposers import build_proposer
from .simulators import build_simulator
from . import __version__


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {
            "schema_name": "softsim.agent_config",
            "schema_version": 2,
            "backend": "mock",
            "proposer_backend": "ruletune",
            "max_iterations": 3,
            "update_gain": 0.7,
        }
    return read_json(Path(path))


def _rows_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    roots = resolve_roots(args.root or [])
    if args.asset_release_dir:
        groups.append(load_asset_release(Path(args.asset_release_dir).resolve(), roots))
    if args.manifest:
        groups.append(load_stage3_manifest(Path(args.manifest).resolve()))
    if args.legacy_root:
        groups.append(load_legacy_assets(Path(args.legacy_root).resolve()))
    if args.staging_root:
        groups.append(load_previous_staging(Path(args.staging_root).resolve()))
    if args.prior_root:
        groups.append(load_previous_staging(Path(args.prior_root).resolve()))
    if args.prior_index:
        groups.append(load_previous_staging(Path(args.prior_index).resolve().parent))
    if not groups:
        raise ValueError("provide --manifest, --legacy-root, or --staging-root")
    rows = merge_rows(*groups)
    if not rows:
        raise ValueError("input sources yielded no tasks")
    return rows


def _tasks_from_plan(path: str | None, config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]] | None:
    if not path:
        return None
    tasks = list(iter_jsonl(Path(path)))
    config_hash = content_hash(config)
    # Plans are immutable inputs; changing config intentionally changes every fingerprint.
    result = []
    for task in tasks:
        copied = dict(task)
        core = {key: copied.get(key) for key in ("sample_id", "asset_revision_id", "dataset_release_id", "route", "artifact_refs", "prior", "fixed_solver_numerics")}
        copied["config_hash"] = config_hash
        copied["task_fingerprint"] = content_hash({**core, "config_hash": config_hash})
        result.append(copied)
    return result


def _select_tasks(tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = set(str(item) for item in (args.sample_id or []))
    if selected:
        tasks = [task for task in tasks if str(task.get("sample_id")) in selected]
    if args.limit is not None:
        tasks = tasks[: int(args.limit)]
    num_shards = max(1, int(args.num_shards or 1))
    shard_index = int(args.shard_index or 0)
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if num_shards > 1:
        tasks = [task for task in tasks if int(content_hash(str(task["sample_id"])).split(":", 1)[1][:8], 16) % num_shards == shard_index]
    return tasks


def cmd_plan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    population_tasks = build_tasks(_rows_from_args(args), config)
    tasks = _select_tasks(population_tasks, args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output / "tasks.jsonl", tasks)
    atomic_write_json(output / "run_plan.json", {"schema_name": "softsim.run_plan", "schema_version": 2, "population_count": len(population_tasks), "task_count": len(tasks), "task_ids": [task["sample_id"] for task in population_tasks], "selected_task_ids": [task["sample_id"] for task in tasks], "config_hash": content_hash(config), "backend": config.get("backend", "mock"), "proposer_backend": config.get("proposer_backend", "ruletune"), "shard_index": int(args.shard_index or 0), "num_shards": int(args.num_shards or 1)})
    print(json.dumps({"population_count": len(population_tasks), "task_count": len(tasks), "output": str(output), "config_hash": content_hash(config)}, ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    planned = _tasks_from_plan(args.plan, config, args)
    population_tasks = planned if planned is not None else build_tasks(_rows_from_args(args), config)
    tasks = _select_tasks(population_tasks, args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Every shard records the complete population; aggregate can therefore
    # detect missing IDs instead of relying on a count-only check.
    atomic_write_json(output / "run_plan.json", {
        "schema_name": "softsim.run_plan",
        "schema_version": 2,
        "population_count": len(population_tasks),
        "task_count": len(tasks),
        "task_ids": [task["sample_id"] for task in population_tasks],
        "selected_task_ids": [task["sample_id"] for task in tasks],
        "config_hash": content_hash(config),
        "backend": config.get("backend", "mock"),
        "proposer_backend": config.get("proposer_backend", "ruletune"),
        "shard_index": int(args.shard_index or 0),
        "num_shards": int(args.num_shards or 1),
    })
    simulator = build_simulator(config)
    proposer = build_proposer(config)
    records: list[dict[str, Any]] = []
    workers = max(1, int(args.workers or 1))
    if workers == 1:
        for task in tasks:
            records.append(run_task(task, output, config, simulator, proposer, force=bool(args.force)))
    else:
        # Each worker gets its own backend/proposer to avoid mutable command state.
        def worker(task: dict[str, Any]) -> dict[str, Any]:
            return run_task(task, output, config, build_simulator(config), build_proposer(config), force=bool(args.force))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, task) for task in tasks]
            for future in as_completed(futures):
                records.append(future.result())
    records.sort(key=lambda item: str(item.get("sample_id")))
    summary = {"schema_name": "softsim.run_summary", "schema_version": 2, "task_count": len(tasks), "population_count": len(population_tasks), "completed": len(records), "status_counts": {status: sum(item.get("status") == status for item in records) for status in sorted({str(item.get("status")) for item in records})}, "backend": config.get("backend", "mock"), "proposer_backend": config.get("proposer_backend", "ruletune"), "config_hash": content_hash(config), "shard_index": int(args.shard_index or 0), "num_shards": int(args.num_shards or 1), "task_ids": [task["sample_id"] for task in tasks]}
    atomic_write_json(output / f"run_summary.shard_{int(args.shard_index or 0):04d}_of_{int(args.num_shards or 1):04d}.json", summary)
    if int(args.num_shards or 1) == 1:
        atomic_write_json(output / "run_summary.json", summary)
    print(json.dumps({"task_count": len(tasks), "completed": len(records), "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    manifest = aggregate_run(Path(args.output).resolve(), release_id=args.release_id, expected_count=args.expected_count, expected_task_ids=args.expected_task_id or None, strict_publish=bool(args.strict_publish))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest.get("publishable") else 2


def _fixture_task(sample_id: str, target: dict[str, float], *, behavior: str = "normal", source_kind: str = "fixture") -> dict[str, Any]:
    prior = {
        "schema_name": "softsim.physical_prior",
        "schema_version": 2,
        "sample_id": sample_id,
        "asset_revision_id": f"{sample_id}-revision",
        "status": "predicted_prior",
        "route": {"route_id": "fixture.solid", "geometry_family": "solid", "material_family": "fixture", "representation": "watertight_volume", "model_family": "MPM.Elastic", "solver_adapter": "MPM.Elastic", "proxy_mode": "mock"},
        "parameters": [
            {"name": "density_kg_m3", "value": 600.0, "unit": "kg/m^3", "valid_range": [10.0, 30000.0], "source_type": "synthetic_fixture"},
            {"name": "youngs_modulus_pa", "value": 50000.0, "unit": "Pa", "valid_range": [10.0, 1.0e10], "source_type": "synthetic_fixture"},
            {"name": "poisson_ratio", "value": 0.30, "unit": "1", "valid_range": [-0.49, 0.499], "source_type": "synthetic_fixture"},
        ],
        "solver_adapter": {"solver": "mock", "parameters": {"density_kg_m3": 600.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30}},
        "solver_numerics": {"particle_size_m": 0.01, "dt_s": 0.001, "substeps": 8, "grid_density": 64, "scale_m": 0.12, "seed": 7, "proxy_mode": "mock"},
        "provenance": {"prior_only": True, "synthetic_fixture": True, "is_ground_truth": False},
    }
    route = prior["route"]
    return {"sample_id": sample_id, "asset_revision_id": prior["asset_revision_id"], "dataset_release_id": "fixture", "geometry_family": "solid", "material_family": "fixture", "artifact_refs": {}, "prior": prior, "route": route, "fixed_solver_numerics": prior["solver_numerics"], "mock_target": target, "mock_behavior": behavior, "source_kind": source_kind}


def cmd_smoke(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    config = {"schema_name": "softsim.agent_config", "schema_version": 2, "backend": "mock", "proposer_backend": "ruletune", "max_iterations": 4, "update_gain": 0.8, "calibration_probe": {"probe_id": "calibration.fixture.drop", "stimulus": {"kind": "drop", "height_m": 0.18}, "action": {"kind": "release", "direction": "z"}, "config": {"steps": 20, "dt_s": 0.001, "seed_offset": 11}, "tolerance": 0.04, "required_metrics": ["loss", "contact_fraction", "settled"]}, "heldout_probe": {"probe_id": "heldout.fixture.compression", "stimulus": {"kind": "compression", "ratio": 0.2}, "action": {"kind": "compress_release", "axis": "x"}, "config": {"steps": 24, "dt_s": 0.0008, "seed_offset": 97}, "tolerance": 0.08, "required_metrics": ["loss", "contact_fraction", "settled"]}}
    tasks = [
        _fixture_task("fixture_first_pass", {"density_kg_m3": 600.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30}),
        _fixture_task("fixture_corrected", {"density_kg_m3": 840.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30}),
        _fixture_task("fixture_rejected", {"density_kg_m3": 840.0, "youngs_modulus_pa": 72000.0, "poisson_ratio": 0.36}, behavior="model_limit"),
    ]
    built = []
    for task in tasks:
        # Preserve fixture-specific fields through build_tasks.
        built.extend(build_tasks([task], config))
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output / "tasks.jsonl", built)
    atomic_write_json(output / "run_plan.json", {"schema_name": "softsim.run_plan", "schema_version": 2, "task_count": len(built), "task_ids": [task["sample_id"] for task in built], "config_hash": content_hash(config), "backend": "mock"})
    simulator = build_simulator(config)
    proposer = build_proposer(config)
    records = [run_task(task, output, config, simulator, proposer, force=True) for task in built]
    summary = {"records": records, "status_counts": {status: sum(item.get("status") == status for item in records) for status in sorted({str(item.get("status")) for item in records})}, "synthetic": True}
    atomic_write_json(output / "smoke_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status_counts"].get("accepted_first_pass") == 1 and summary["status_counts"].get("corrected_then_accepted") == 1 and summary["status_counts"].get("rejected_model_limit") == 1 else 2


def cmd_audit(args: argparse.Namespace) -> int:
    audit = build_golden_reference_audit(Path(args.stage3_root).resolve(), Path(args.output).resolve(), sample_per_family=int(args.sample_per_family))
    print(json.dumps({"output": str(Path(args.output).resolve()), "asset_count": audit["asset_count"], "selected_count": audit["selected_count"], "families": audit["geometry_family_counts"]}, ensure_ascii=False, indent=2))
    return 0


def _common_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--asset-release-dir")
    parser.add_argument("--manifest")
    parser.add_argument("--legacy-root")
    parser.add_argument("--staging-root")
    parser.add_argument("--prior-root")
    parser.add_argument("--prior-index")
    parser.add_argument("--root", action="append", default=[], metavar="ALIAS=PATH")
    parser.add_argument("--plan")
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SoftSim auditable physical parameter estimation agent")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="normalize stage3/legacy/staging input into immutable tasks")
    _common_source_arguments(plan)
    plan.set_defaults(handler=cmd_plan)
    run = sub.add_parser("run", help="execute calibration, bounded updates, and held-out probes")
    _common_source_arguments(run)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--force", action="store_true")
    run.set_defaults(handler=cmd_run)
    aggregate = sub.add_parser("aggregate", help="copy evidence and build a release-relative index")
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--release-id", default="softsim-physical-agent-rebuild-v0.1")
    aggregate.add_argument("--expected-count", type=int)
    aggregate.add_argument("--expected-task-id", action="append", default=[])
    aggregate.add_argument("--strict-publish", action="store_true")
    aggregate.set_defaults(handler=cmd_aggregate)
    smoke = sub.add_parser("smoke", help="run the three local fixture scenarios")
    smoke.add_argument("--output", required=True)
    smoke.set_defaults(handler=cmd_smoke)
    audit = sub.add_parser("audit-golden", help="hash a five-family stage3 golden reference")
    audit.add_argument("--stage3-root", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--sample-per-family", type=int, default=1)
    audit.set_defaults(handler=cmd_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
