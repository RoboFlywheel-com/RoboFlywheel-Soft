# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .asset import audit_assets, inventory_assets, plan_assets, publish_asset_release
from .benchmark import run_mock_concurrency_sweep
from .cloth import run_cloth_backfill
from .fixtures import create_asset_fixture, create_ledger_fixture
from .io import parse_root_aliases
from .physics import aggregate_physics_release, plan_physics, prepare_physics, run_physics_shard
from .physics import build_physics_selection


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _print(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _asset_plan(args: argparse.Namespace) -> int:
    summary = plan_assets(
        manifest=_path(args.manifest),
        output=_path(args.output),
        mesh_roots=parse_root_aliases(args.mesh_root or []),
        overlay_root=_path(args.overlay_root),
        expected_count=args.expected_count,
        shard_size=args.shard_size,
        estimated_cloth_bytes_per_asset=args.estimated_cloth_bytes_per_asset,
    )
    _print(summary)
    return 1 if summary["promotion_gate_failures"] else 0


def _asset_inventory(args: argparse.Namespace) -> int:
    mesh_indexes: dict[str, set[str]] = {}
    for value in args.mesh_index or []:
        alias, separator, raw_path = value.partition("=")
        if not separator or not alias or not raw_path:
            raise ValueError(f"Expected ROOT_ALIAS=INDEX_FILE, got {value!r}")
        if alias in mesh_indexes:
            raise ValueError(f"Duplicate mesh index alias: {alias}")
        index_path = _path(raw_path)
        with index_path.open("r", encoding="utf-8") as handle:
            mesh_indexes[alias] = {line.strip() for line in handle if line.strip()}
    summary = inventory_assets(
        manifest=_path(args.manifest),
        mesh_roots=parse_root_aliases(args.mesh_root),
        image_root=_path(args.image_root) if args.image_root else None,
        output_dir=_path(args.output_dir),
        scan_orphans=args.scan_orphans,
        workers=args.workers,
        mesh_indexes=mesh_indexes or None,
        verify_glb_headers=not args.skip_glb_headers,
        verify_image_paths=not args.skip_image_check,
    )
    _print(summary)
    return 0


def _asset_audit(args: argparse.Namespace) -> int:
    summary = audit_assets(
        inventory_path=_path(args.inventory),
        qa_config_path=_path(args.qa_config),
        output_dir=_path(args.output_dir),
        workers=args.workers,
    )
    _print(summary)
    return 1 if args.strict and summary["failure_count"] else 0


def _cloth_backfill(args: argparse.Namespace) -> int:
    summary = run_cloth_backfill(
        manifest=_path(args.manifest),
        image_root=_path(args.image_root),
        overlay_root=_path(args.overlay_root),
        config_path=_path(args.config),
        result_path=_path(args.result),
        workers=args.workers,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    _print(summary)
    return 1 if summary["failed"] else 0


def _asset_publish(args: argparse.Namespace) -> int:
    summary = publish_asset_release(
        manifest=_path(args.manifest),
        inventory_path=_path(args.inventory),
        audit_path=_path(args.audit),
        backfill_paths=[_path(value) for value in args.backfill_result or []],
        releases_root=_path(args.releases_root),
        dataset_release_id=args.dataset_release_id,
        expected_count=args.expected_count,
        strict=args.strict,
    )
    _print(summary)
    return 0


def _physics_plan(args: argparse.Namespace) -> int:
    summary = plan_physics(
        asset_ledger=_path(args.asset_ledger),
        output_root=_path(args.output_root),
        config_path=_path(args.config),
        legacy_src=_path(args.legacy_src) if args.legacy_src else None,
        num_shards=args.num_shards,
        expected_ready_count=args.expected_ready_count,
        selection_path=_path(args.selection) if args.selection else None,
        limit=args.limit,
    )
    _print(summary)
    return 0


def _physics_selection(args: argparse.Namespace) -> int:
    summary = build_physics_selection(
        asset_ledger=_path(args.asset_ledger),
        count=args.count,
        output=_path(args.output),
        strata_fields=[value for value in args.strata.split(",") if value],
    )
    _print(summary)
    return 0


def _physics_prepare(args: argparse.Namespace) -> int:
    summary = prepare_physics(
        run_root=_path(args.run_root),
        shard_index=args.shard_index,
        roots=parse_root_aliases(args.root or []),
        workers=args.workers,
        allow_metadata_only=args.allow_metadata_only,
    )
    _print(summary)
    return 1 if summary["failed"] else 0


def _physics_run(args: argparse.Namespace) -> int:
    summary = run_physics_shard(
        run_root=_path(args.run_root),
        shard_index=args.shard_index,
        config_path=_path(args.config),
        fallback_policy_path=_path(args.fallback_policy),
        transport_name=args.transport,
        workers=args.workers,
        confirm_paid_run=args.confirm_paid_run,
    )
    _print(summary)
    return 0


def _physics_aggregate(args: argparse.Namespace) -> int:
    summary = aggregate_physics_release(
        run_root=_path(args.run_root),
        asset_release_dir=_path(args.asset_release_dir),
        physical_release_id=args.physical_release_id,
        strict=args.strict,
        allow_nonpublishable=args.allow_nonpublishable,
        output_root=_path(args.output_root) if args.output_root else None,
    )
    _print(summary)
    return 0


def _fixture_assets(args: argparse.Namespace) -> int:
    _print(create_asset_fixture(_path(args.output_root), args.count))
    return 0


def _fixture_ledger(args: argparse.Namespace) -> int:
    _print(create_ledger_fixture(_path(args.output_root), args.count))
    return 0


def _integer_list(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return result


def _mock_sweep(args: argparse.Namespace) -> int:
    report = run_mock_concurrency_sweep(
        samples=args.samples,
        batch_sizes=args.batch_sizes,
        worker_levels=args.worker_levels,
        latency_seconds=args.latency_seconds,
        output=_path(args.output),
    )
    _print(report)
    return 0


def _add_common_workers(parser: argparse.ArgumentParser, default: int = 4) -> None:
    parser.add_argument("--workers", type=int, default=default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="softsim-assets",
        description="SoftSim read-only asset maintenance and physical bundle production",
    )
    parser.add_argument("--version", action="version", version="softsim-assets 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("asset-plan", help="Stream the frozen primary manifest and write a dry-run plan")
    command.add_argument("--manifest", required=True)
    command.add_argument("--mesh-root", action="append", default=[])
    command.add_argument("--overlay-root", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--expected-count", type=int)
    command.add_argument("--shard-size", type=int, default=250)
    command.add_argument("--estimated-cloth-bytes-per-asset", type=int, default=1572864)
    command.set_defaults(handler=_asset_plan)

    command = subparsers.add_parser("asset-inventory", help="Run read-only lightweight mesh inventory")
    command.add_argument("--manifest", required=True)
    command.add_argument("--mesh-root", action="append", required=True)
    command.add_argument("--image-root")
    command.add_argument("--output-dir", required=True)
    command.add_argument("--scan-orphans", action="store_true")
    command.add_argument("--mesh-index", action="append", default=[], help="ROOT_ALIAS=file listing relative mesh paths")
    command.add_argument("--skip-glb-headers", action="store_true", help="Use a trusted mesh index; full audit still validates GLB content")
    command.add_argument("--skip-image-check", action="store_true", help="Trust a prior image-root preflight and check only that the root is mounted")
    _add_common_workers(command, 1)
    command.set_defaults(handler=_asset_inventory)

    command = subparsers.add_parser("asset-audit", help="Run isolated full static GLB audit")
    command.add_argument("--inventory", required=True)
    command.add_argument("--qa-config", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--strict", action="store_true")
    _add_common_workers(command)
    command.set_defaults(handler=_asset_audit)

    command = subparsers.add_parser("cloth-backfill", help="Generate content-addressed procedural cloth revisions")
    command.add_argument("--manifest", required=True)
    command.add_argument("--image-root", required=True)
    command.add_argument("--overlay-root", required=True)
    command.add_argument("--config", required=True)
    command.add_argument("--result", required=True)
    command.add_argument("--shard-index", type=int, default=0)
    command.add_argument("--num-shards", type=int, default=1)
    _add_common_workers(command)
    command.set_defaults(handler=_cloth_backfill)

    command = subparsers.add_parser("asset-publish", help="Publish immutable versioned asset ledger")
    command.add_argument("--manifest", required=True)
    command.add_argument("--inventory", required=True)
    command.add_argument("--audit", required=True)
    command.add_argument("--backfill-result", action="append", default=[])
    command.add_argument("--releases-root", required=True)
    command.add_argument("--dataset-release-id", required=True)
    command.add_argument("--expected-count", type=int)
    command.add_argument("--strict", action="store_true")
    command.set_defaults(handler=_asset_publish)

    command = subparsers.add_parser("physics-plan", help="Create static physical task shards and paid-run estimates")
    command.add_argument("--asset-ledger", required=True)
    command.add_argument("--output-root", required=True)
    command.add_argument("--config", required=True)
    command.add_argument("--legacy-src")
    command.add_argument("--num-shards", type=int, default=1)
    command.add_argument("--expected-ready-count", type=int)
    command.add_argument("--selection", help="JSONL containing sample_id values for a frozen gate subset")
    command.add_argument("--limit", type=int, help="Deterministic smoke limit; use --selection for stratified gates")
    command.set_defaults(handler=_physics_plan)

    command = subparsers.add_parser("physics-selection", help="Build a deterministic stratified gate selection")
    command.add_argument("--asset-ledger", required=True)
    command.add_argument("--count", type=int, required=True)
    command.add_argument("--strata", default="geometry_family,material_family")
    command.add_argument("--output", required=True)
    command.set_defaults(handler=_physics_selection)

    command = subparsers.add_parser("physics-prepare", help="Verify pinned physical task inputs")
    command.add_argument("--run-root", required=True)
    command.add_argument("--shard-index", type=int, required=True)
    command.add_argument("--root", action="append", default=[])
    command.add_argument("--allow-metadata-only", action="store_true", help="Fixture/mock only")
    _add_common_workers(command, 8)
    command.set_defaults(handler=_physics_prepare)

    command = subparsers.add_parser("physics-run", help="Run one physical prediction shard")
    command.add_argument("--run-root", required=True)
    command.add_argument("--shard-index", type=int, required=True)
    command.add_argument("--config", required=True)
    command.add_argument("--fallback-policy", required=True)
    command.add_argument("--transport", choices=("mock", "qwen"), required=True)
    command.add_argument("--confirm-paid-run", action="store_true")
    _add_common_workers(command, 8)
    command.set_defaults(handler=_physics_run)

    command = subparsers.add_parser("physics-aggregate", help="Aggregate immutable physical bundle release")
    command.add_argument("--run-root", required=True)
    command.add_argument("--asset-release-dir", required=True)
    command.add_argument("--physical-release-id", required=True)
    command.add_argument("--strict", action="store_true")
    command.add_argument("--allow-nonpublishable", action="store_true", help="Fixture/mock release only")
    command.add_argument("--output-root", help="Use an isolated gate root instead of writing under asset release")
    command.set_defaults(handler=_physics_aggregate)

    command = subparsers.add_parser("fixture-assets", help="Create a synthetic asset-maintenance fixture")
    command.add_argument("--output-root", required=True)
    command.add_argument("--count", type=int, default=10)
    command.set_defaults(handler=_fixture_assets)

    command = subparsers.add_parser("fixture-ledger", help="Create a synthetic ledger for mock scale gates")
    command.add_argument("--output-root", required=True)
    command.add_argument("--count", type=int, required=True)
    command.set_defaults(handler=_fixture_ledger)

    command = subparsers.add_parser("mock-concurrency-sweep", help="Benchmark bounded mock transport concurrency")
    command.add_argument("--samples", type=int, default=256)
    command.add_argument("--batch-sizes", type=_integer_list, default=[1, 4, 8])
    command.add_argument("--worker-levels", type=_integer_list, default=[1, 4, 8, 16])
    command.add_argument("--latency-seconds", type=float, default=0.01)
    command.add_argument("--output", required=True)
    command.set_defaults(handler=_mock_sweep)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
