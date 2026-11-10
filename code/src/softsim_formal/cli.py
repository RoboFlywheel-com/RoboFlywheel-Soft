# SPDX-License-Identifier: Apache-2.0
"""Unified ``softsim-exp`` command line interface."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .aggregate import aggregate_run
from .canonical import fingerprint, read_json, sha256_file
from .e1 import run_e1_smoke
from .e2 import run_e2_smoke
from .e3a import run_e3a_smoke
from .e2 import load_target_provider
from .io import atomic_write_json, write_jsonl, read_jsonl
from .planner import EXPERIMENTS, EXPERIMENT_ALIASES, plan_run
from .release import BlockedRelease, ReleaseError, ReleaseReader
from .render import check_render, render_run
from .asset_release import convert_v1_to_v2
from .runner import load_executor, run_all_shards, run_shard, status_run
from .schema import SchemaError, validate_run_manifest
from .records import validate_task_spec


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2))


def _config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    return value


def _plan(args: argparse.Namespace) -> int:
    bindings = {}
    for value in args.root_binding:
        alias, separator, path = value.partition("=")
        if not separator or not alias or not path:
            raise ValueError(f"--root must be ALIAS=PATH, got {value!r}")
        bindings[alias] = path
    result = plan_run(
        args.experiment,
        output_dir=args.output,
        config=_config(args.config),
        release_root=args.release_root,
        fixture=bool(args.fixture),
        count=args.count,
        shard_count=args.shards,
        mode=args.mode,
        scheduler=args.scheduler,
        allow_synthetic=not args.reject_synthetic,
        root_bindings=bindings,
        benchmark_join_map=args.benchmark_join_map,
        hash_workers=getattr(args, "hash_workers", 1),
        verify_artifact_content=not getattr(args, "skip_artifact_content_hash", False),
        selection_file=getattr(args, "selection_file", None),
    )
    _json(result.as_dict())
    return 0 if result.status == "planned" else 2


def _run(args: argparse.Namespace, resume: bool = False) -> int:
    run_dir = args.run or args.manifest
    if not run_dir:
        raise ValueError("--run is required")
    isolated = bool(getattr(args, "process_isolation", False))
    provider = load_target_provider(args.target_provider) if getattr(args, "target_provider", None) and not isolated else None
    executor = load_executor(args.episode_executor) if getattr(args, "episode_executor", None) and not isolated else None
    result = run_shard(
        run_dir,
        shard_index=args.shard_index,
        workers=args.workers,
        force=bool(args.force and not resume),
        target_provider=provider,
        episode_executor=executor,
        process_isolation=isolated,
        episode_executor_reference=args.episode_executor if isolated else None,
        target_provider_reference=args.target_provider if isolated else None,
    )
    _json(result)
    return 0 if result.get("status") != "blocked" else 2


def _run_all(args: argparse.Namespace) -> int:
    if not args.run:
        raise ValueError("--run is required")
    isolated = bool(getattr(args, "process_isolation", False))
    provider = load_target_provider(args.target_provider) if getattr(args, "target_provider", None) and not isolated else None
    executor = load_executor(args.episode_executor) if getattr(args, "episode_executor", None) and not isolated else None
    result = run_all_shards(
        args.run,
        workers=args.workers,
        force=args.force,
        target_provider=provider,
        episode_executor=executor,
        process_isolation=isolated,
        episode_executor_reference=args.episode_executor if isolated else None,
        target_provider_reference=args.target_provider if isolated else None,
    )
    _json(result)
    return 0 if result.get("status") in {"completed", "running"} else 2


def _shard(args: argparse.Namespace) -> int:
    """(Re)materialize deterministic shard files from a frozen task manifest."""
    if not args.run and not args.manifest_path:
        raise ValueError("--run or --manifest is required")
    if args.run:
        root = Path(args.run).resolve()
    else:
        manifest_file = Path(args.manifest_path).resolve()
        root = Path(args.shard_output or manifest_file.parent).resolve()
        root.mkdir(parents=True, exist_ok=True)
        # Read the explicit manifest below instead of requiring a run manifest.
        manifest = {"run_id": "unbound-shard-plan", "shard_count": args.shards or 1}
        tasks_path = manifest_file
        tasks = list(read_jsonl(tasks_path))
        shard_count = int(args.shards or 1)
        partitions: dict[int, list[dict[str, Any]]] = {index: [] for index in range(shard_count)}
        for task in tasks:
            task_copy = dict(task)
            task_copy["shard_index"] = int(fingerprint(str(task_copy.get("benchmark_instance_id") or task_copy.get("task_id") or ""))[7:15], 16) % shard_count
            partitions[task_copy["shard_index"]].append(task_copy)
        shard_dir = root / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for index, rows in partitions.items():
            rows.sort(key=lambda row: str(row.get("benchmark_instance_id") or row.get("task_id")))
            write_jsonl(shard_dir / f"shard-{index:04d}.jsonl", rows)
        _json({"status": "planned", "task_count": len(tasks), "shard_count": shard_count, "output": shard_dir.as_posix()})
        return 0
    manifest = read_json(root / "run_manifest.json")
    tasks_path = root / "task_manifest.jsonl"
    tasks = list(read_jsonl(tasks_path))
    shard_count = int(args.shards or manifest.get("shard_count", 1))
    if shard_count <= 0:
        raise ValueError("--shards must be positive")
    partitions: dict[int, list[dict[str, Any]]] = {index: [] for index in range(shard_count)}
    for task in tasks:
        task_copy = dict(task)
        task_copy["shard_index"] = int(fingerprint(str(task_copy.get("task_id", "")))[7:15], 16) % shard_count
        partitions[task_copy["shard_index"]].append(task_copy)
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for index, rows in partitions.items():
        rows.sort(key=lambda row: str(row.get("task_id")))
        write_jsonl(shard_dir / f"shard-{index:04d}.jsonl", rows)
    manifest["shard_count"] = shard_count
    # The frozen task list is unchanged; only its derived shard materialization
    # is refreshed.  A changed shard count is explicitly recorded in status.
    atomic_write_json(root / "run_status.json", {"status": "planned", "run_id": manifest.get("run_id"), "expected_task_count": len(tasks), "completed_task_count": 0, "shard_count": shard_count, "resharded": True})
    _json({"status": "planned", "run_id": manifest.get("run_id"), "task_count": len(tasks), "shard_count": shard_count})
    return 0


def _aggregate(args: argparse.Namespace) -> int:
    if not args.run:
        raise ValueError("--run is required")
    result = aggregate_run(args.run, strict=args.strict)
    _json(result)
    return 0 if result["completion"].get("status") == "completed" else 2


def _render(args: argparse.Namespace) -> int:
    if args.check:
        result = check_render(args.output or args.run)
        _json(result)
        return 0 if result.get("valid") else 2
    if not args.run:
        raise ValueError("--run is required")
    _json(render_run(args.run, spec_path=args.analysis_spec, output_dir=args.output))
    return 0


def _videos(args: argparse.Namespace) -> int:
    """Emit an explicit video manifest; Phase-I mock never fabricates video."""
    if args.check:
        root = Path(args.output or args.run)
        path = root / "video_manifest.jsonl"
        valid = path.is_file()
        _json({"valid": valid, "generated": False, "manifest": path.as_posix()})
        return 0 if valid else 2
    if not args.run:
        raise ValueError("--run is required")
    root = Path(args.run).resolve()
    completion = read_json(root / "run_completion.json")
    records_path = root / str(completion.get("canonical_records_ref", "canonical/episodes.jsonl"))
    rows = list(read_jsonl(records_path)) if records_path.is_file() else []
    manifests = [{"schema_name": "softsim.video_manifest_record", "schema_version": 1, "episode_id": row.get("episode_id"), "video_status": "not_generated_local_mock", "is_evidence": False, "source_record_sha256": row.get("record_sha256")} for row in rows]
    path = root / "video_manifest.jsonl"
    write_jsonl(path, manifests)
    _json({"manifest": path.as_posix(), "count": len(manifests), "generated": False, "reason": "Phase-I mock; no GIF/video enters formal tables"})
    return 0


def _status(args: argparse.Namespace) -> int:
    _json(status_run(args.run))
    return 0


def _validate_run(args: argparse.Namespace) -> int:
    root = Path(args.run).resolve()
    errors: list[str] = []
    try:
        manifest = read_json(root / "run_manifest.json")
        validate_run_manifest(manifest)
    except Exception as exc:
        _json({"valid": False, "errors": [f"manifest:{exc}"]})
        return 2
    if manifest.get("status") == "blocked":
        _json({"valid": False, "status": "blocked", "blocked_code": manifest.get("blocked_code"), "errors": []})
        return 2
    task_path = root / "task_manifest.jsonl"
    tasks = list(read_jsonl(task_path)) if task_path.is_file() else []
    expected = int(manifest.get("expected_task_count", len(tasks)))
    if len(tasks) != expected:
        errors.append(f"task_count:{len(tasks)}!={expected}")
    if task_path.is_file() and manifest.get("task_manifest_sha256") != sha256_file(task_path):
        errors.append("task_manifest_hash")
    for task in tasks:
        try:
            validate_task_spec(task)
        except (SchemaError, ValueError) as exc:
            errors.append(f"task:{task.get('task_id', '<missing>')}:{exc}")
    completion = root / "run_completion.json"
    if not completion.is_file():
        errors.append("missing:run_completion.json")
    else:
        value = read_json(completion)
        if value.get("canonical_records_sha256"):
            record_path = root / str(value.get("canonical_records_ref", "canonical/episodes.jsonl"))
            if not record_path.is_file():
                errors.append("missing:canonical_records")
            elif sha256_file(record_path) != value["canonical_records_sha256"]:
                errors.append("canonical_records_hash")
    _json({"valid": not errors, "errors": errors, "run_id": manifest.get("run_id"), "expected_task_count": expected})
    return 0 if not errors else 2


def _inventory(args: argparse.Namespace) -> int:
    bindings = {}
    for value in args.root_binding:
        alias, separator, path = value.partition("=")
        if not separator or not alias or not path:
            raise ValueError(f"--root must be ALIAS=PATH, got {value!r}")
        bindings[alias] = path
    try:
        bundle = ReleaseReader(
            args.release_root,
            asset_manifest=args.asset_manifest,
            asset_ledger=args.asset_ledger,
            physical_manifest=args.physical_manifest,
            physical_index=args.physical_index,
            benchmark_manifest=args.benchmark_manifest,
            benchmark_join_map=args.benchmark_join_map,
            method_contract=args.method_contract,
            allow_synthetic=not args.reject_synthetic,
            require_benchmark=not args.no_benchmark,
            root_bindings=bindings,
            hash_workers=getattr(args, "hash_workers", 1),
        ).read()
    except BlockedRelease as exc:
        _json({"status": "blocked", "code": exc.code, "message": str(exc)})
        return 2
    except ReleaseError as exc:
        _json({"status": "rejected", "code": exc.code, "message": str(exc)})
        return 2
    _json({"status": "valid", **bundle.as_dict()})
    return 0


def _fixture(args: argparse.Namespace) -> int:
    if args.experiment == "e1":
        result = run_e1_smoke(args.output, stage=args.stage, count=args.count, shard_count=args.shards)
    elif args.experiment == "e2":
        result = run_e2_smoke(args.output, count=args.count)
    elif args.experiment == "e3a":
        result = run_e3a_smoke(args.output, count=args.count)
    else:
        raise ValueError("fixture experiment must be e1, e2, or e3a")
    _json(result)
    return 0


def _archive(args: argparse.Namespace) -> int:
    # Keep archive creation in Python so it is also available after install.
    from .archive import create_archive, verify_archive
    result = create_archive(args.root, args.output)
    result["verified"] = verify_archive(args.output)
    _json(result)
    return 0


def _convert_asset_release(args: argparse.Namespace) -> int:
    roots = {}
    for value in args.root:
        alias, separator, path = value.partition("=")
        if not separator or not alias or not path:
            raise ValueError(f"--root must be ALIAS=PATH, got {value!r}")
        roots[alias] = path
    result = convert_v1_to_v2(
        args.ledger,
        output_dir=args.output,
        artifact_root=args.artifact_root,
        artifact_roots=roots,
        release_id=args.release_id,
        source_manifest_path=args.source_manifest,
        expected_count=args.expected_count,
        allow_ready_with_warnings=args.allow_ready_with_warnings,
    )
    _json(result)
    return 0 if result["manifest"].get("publishable") else 2


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="softsim-exp", description="SoftSim-100K formal experiment runner")
    parser.add_argument("--version", action="version", version="0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="freeze tasks and deterministic shards")
    plan.add_argument("experiment", choices=sorted(EXPERIMENTS | set(EXPERIMENT_ALIASES)))
    plan.add_argument("--output", required=True)
    plan.add_argument("--config")
    plan.add_argument("--release-root")
    plan.add_argument("--benchmark-join-map", help="explicit benchmark-to-physical join map")
    plan.add_argument("--fixture", action="store_true")
    plan.add_argument("--reject-synthetic", action="store_true")
    plan.add_argument("--count", type=int)
    plan.add_argument("--shards", type=int, default=1)
    plan.add_argument("--mode", choices=("local", "scheduler"), default="local")
    plan.add_argument("--scheduler", default=None)
    plan.add_argument("--hash-workers", type=int, default=1, help="parallel content-hash verification (I/O-bound roots)")
    plan.add_argument("--skip-artifact-content-hash", action="store_true", help="plan-time structural checks only; full-population content hashing stays with inventory validate, per-episode mesh sha256 with the executor")
    plan.add_argument("--selection-file", help="frozen selection JSONL (sample_id/asset_revision_id/physical_bundle_id) restricting the plan population")
    plan.add_argument("--root", dest="root_binding", action="append", default=[], metavar="ALIAS=PATH")
    plan.set_defaults(func=_plan)

    for name, callback in (("run", lambda args: _run(args, False)), ("resume", lambda args: _run(args, True))):
        command = sub.add_parser(name, help="execute or resume one shard")
        command.add_argument("--run")
        command.add_argument("--manifest")
        command.add_argument("--shard-index", type=int, default=0)
        command.add_argument("--workers", type=int, default=1)
        command.add_argument("--force", action="store_true")
        command.add_argument("--target-provider", help="E2 evaluator-side module:function; never put hidden targets in task JSON")
        command.add_argument("--episode-executor", help="production module:function returning a canonical episode record")
        command.add_argument("--process-isolation", action="store_true", help="run each task in a fresh worker process")
        command.set_defaults(func=callback)
    all_run = sub.add_parser("run-all", help="execute all planned shards locally")
    all_run.add_argument("--run", required=True)
    all_run.add_argument("--workers", type=int, default=1)
    all_run.add_argument("--force", action="store_true")
    all_run.add_argument("--target-provider", help="E2 evaluator-side module:function")
    all_run.add_argument("--episode-executor", help="production module:function returning canonical records")
    all_run.add_argument("--process-isolation", action="store_true", help="run each task in a fresh worker process")
    all_run.set_defaults(func=_run_all)

    status = sub.add_parser("status", help="show run status")
    status.add_argument("--run", required=True)
    status.set_defaults(func=_status)
    aggregate = sub.add_parser("aggregate", help="merge shard records deterministically")
    aggregate.add_argument("experiment", nargs="?", choices=sorted(EXPERIMENTS | {"e1", "e2"}), help="optional experiment label (recorded in the run manifest)")
    aggregate.add_argument("--run", required=True)
    aggregate.add_argument("--strict", action="store_true")
    aggregate.set_defaults(func=_aggregate)
    render = sub.add_parser("render", help="build CSV/LaTeX/PNG/PDF/review outputs")
    render.add_argument("--run")
    render.add_argument("--analysis-spec")
    render.add_argument("--output")
    render.add_argument("--check", action="store_true")
    render.set_defaults(func=_render)
    figures = sub.add_parser("figures", help="alias for declarative figure rendering")
    figure_sub = figures.add_subparsers(dest="figure_command", required=True)
    figure_build = figure_sub.add_parser("build")
    figure_build.add_argument("--run", required=True)
    figure_build.add_argument("--analysis-spec")
    figure_build.add_argument("--output")
    figure_build.set_defaults(func=_render, check=False)
    figure_check = figure_sub.add_parser("check")
    figure_check.add_argument("--output", required=True)
    figure_check.add_argument("--run")
    figure_check.add_argument("--check", action="store_true", default=True)
    figure_check.set_defaults(func=_render)
    videos = sub.add_parser("videos", help="build/check explicit video manifest")
    videos.add_argument("action", choices=("build", "check"), default="build")
    videos.add_argument("--run")
    videos.add_argument("--output")
    videos.add_argument("--check", action="store_true")
    videos.set_defaults(func=_videos)
    validate = sub.add_parser("validate-run", help="validate completion and canonical hashes")
    validate.add_argument("--run", required=True)
    validate.set_defaults(func=_validate_run)

    inventory = sub.add_parser("inventory", help="read hashed releases")
    inventory.add_argument("action", nargs="?", choices=("build", "validate"), default="validate")
    inventory.add_argument("release_root", nargs="?", default=".")
    inventory.add_argument("--release-root", dest="release_root_opt")
    inventory.add_argument("--asset-manifest")
    inventory.add_argument("--asset-ledger")
    inventory.add_argument("--physical-manifest")
    inventory.add_argument("--physical-index")
    inventory.add_argument("--benchmark-manifest")
    inventory.add_argument("--benchmark-join-map")
    inventory.add_argument("--method-contract")
    inventory.add_argument("--reject-synthetic", action="store_true")
    inventory.add_argument("--no-benchmark", action="store_true")
    inventory.add_argument("--hash-workers", type=int, default=1, help="parallel content-hash verification (I/O-bound roots)")
    inventory.add_argument("--root", dest="root_binding", action="append", default=[], metavar="ALIAS=PATH")
    inventory.set_defaults(func=lambda args: _inventory(argparse.Namespace(
        release_root=args.release_root_opt or args.release_root,
        asset_manifest=args.asset_manifest,
        asset_ledger=args.asset_ledger,
        physical_manifest=args.physical_manifest,
        physical_index=args.physical_index,
        benchmark_manifest=args.benchmark_manifest,
        benchmark_join_map=args.benchmark_join_map,
        method_contract=args.method_contract,
        reject_synthetic=args.reject_synthetic,
        no_benchmark=args.no_benchmark,
        root_binding=args.root_binding,
    )))

    shard = sub.add_parser("shard", help="materialize deterministic shard manifests")
    shard.add_argument("--run")
    shard.add_argument("--manifest", dest="manifest_path")
    shard.add_argument("--output", dest="shard_output")
    shard.add_argument("--shards", type=int)
    shard.set_defaults(func=_shard)

    fixture = sub.add_parser("fixture", help="run local smoke fixture end to end")
    fixture.add_argument("experiment", choices=("e1", "e2", "e3a"))
    fixture.add_argument("--output", required=True)
    fixture.add_argument("--count", type=int, default=10)
    fixture.add_argument("--stage", default="fixture")
    fixture.add_argument("--shards", type=int, default=1)
    fixture.set_defaults(func=_fixture)

    archive = sub.add_parser("archive", help="build a portable deployment archive")
    archive.add_argument("--root", default=".")
    archive.add_argument("--output", required=True)
    archive.set_defaults(func=_archive)

    release = sub.add_parser("release", help="release conversion and validation helpers")
    release_sub = release.add_subparsers(dest="release_command", required=True)
    convert = release_sub.add_parser("convert-v1-to-v2", help="convert a v1 asset ledger into an independent v2 release gate")
    convert.add_argument("--ledger", required=True)
    convert.add_argument("--artifact-root", required=True)
    convert.add_argument("--root", action="append", default=[], metavar="ALIAS=PATH", help="additional v1 artifact root aliases")
    convert.add_argument("--output", required=True)
    convert.add_argument("--release-id", default="softsim-100k-assets-v2")
    convert.add_argument("--source-manifest")
    convert.add_argument("--expected-count", type=int)
    convert.add_argument("--allow-ready-with-warnings", action="store_true")
    convert.set_defaults(func=_convert_asset_release)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (BlockedRelease, ReleaseError, SchemaError, ValueError, OSError, KeyError) as exc:
        _json({"status": "error", "error_type": type(exc).__name__, "message": str(exc)})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
