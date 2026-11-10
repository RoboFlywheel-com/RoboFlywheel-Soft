# SPDX-License-Identifier: Apache-2.0
"""Command line entry points for local smoke and server construction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .adapter import P0Adapter
from .archive import create_deployment_archive, verify_deployment_archive
from .builder import build_instances, write_jsonl
from .qualification import make_default_qualification_instances, run_qualification
from .validation import validate_output_dir
from .sharding import partition_instances
from .release import promote_e3_a_manifest
from .runner import load_episode_runner
from .physical_join import make_join_template


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
# ``PACKAGE_ROOT`` is the paper workspace root (the package lives in
# ``<root>/code``), so fixture/config paths remain portable after installation.
REPO_ROOT = PACKAGE_ROOT
DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "p0_legacy"


def _adapter(args: argparse.Namespace) -> P0Adapter:
    if args.p0_root:
        return P0Adapter.from_directory(args.p0_root)
    manifest = Path(args.manifest) if args.manifest else DEFAULT_FIXTURE / "manifest.json"
    profiles = Path(args.profiles) if args.profiles else DEFAULT_FIXTURE / "material_profiles.json"
    return P0Adapter.from_files(
        manifest,
        profiles,
        summary_json_path=args.summary_json or DEFAULT_FIXTURE / "summary.json",
        summary_csv_path=args.summary_csv or DEFAULT_FIXTURE / "summary.csv",
        video_manifest_path=args.video_manifest or DEFAULT_FIXTURE / "videos.json",
        state_manifest_path=args.state_manifest or DEFAULT_FIXTURE / "states.json",
        source_root=manifest.parent,
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--p0-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--video-manifest", type=Path)
    parser.add_argument("--state-manifest", type=Path)
    parser.add_argument("--physical-index", type=Path)
    parser.add_argument("--physical-join-map", type=Path)
    parser.add_argument("--physical-release-id")


def command_adapt(args: argparse.Namespace) -> int:
    adapter = _adapter(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(adapter.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"asset_count": len(adapter.assets), "source_sha256": adapter.source_sha256, "output": output.as_posix()}, ensure_ascii=False))
    return 0


def command_build(args: argparse.Namespace) -> int:
    if not args.server and (args.limit is None or args.limit > 6):
        raise SystemExit("local build requires --limit <= 6; use --server for production matrix")
    if args.require_physical_join and not args.physical_join_map:
        raise SystemExit("--require-physical-join requires --physical-join-map")
    if args.require_physical_join and not args.physical_release_id:
        raise SystemExit("--require-physical-join requires --physical-release-id")
    adapter = _adapter(args)
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    join_report: dict[str, Any] = {}
    rows = build_instances(
        adapter,
        config,
        limit=args.limit,
        physical_index=args.physical_index,
        physical_join_map=args.physical_join_map,
        physical_release_id=args.physical_release_id,
        require_physical_join=bool(args.require_physical_join),
        join_report_out=join_report,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "instance_manifest.jsonl", rows)
    write_jsonl(output / "material_records.jsonl", [{"benchmark_instance_id": row["benchmark_instance_id"], "material": row["material"], "physical_bundle_id": row["physical_bundle_id"]} for row in rows])
    write_jsonl(output / "material_config_records.jsonl", [{"benchmark_instance_id": row["benchmark_instance_id"], "material": row["material"], "material_model_id": row["material_model_id"], "damage_model_id": row.get("damage_model_id"), "difficulty": row["difficulty"], "scenario": row["scenario"]} for row in rows])
    write_jsonl(output / "config_records.jsonl", [{"benchmark_instance_id": row["benchmark_instance_id"], "difficulty": row["difficulty"], "scenario": row["scenario"]} for row in rows])
    # A build is a plan-only stage: create explicit empty/blocked record files
    # so downstream readers never infer that missing files mean success.
    write_jsonl(output / "episode_records.jsonl", [])
    write_jsonl(output / "metric_records.jsonl", [])
    write_jsonl(output / "video_keyframe_manifest.jsonl", [{"benchmark_instance_id": row["benchmark_instance_id"], "video_status": "not_run", "keyframe_status": "not_run", "is_evidence": False} for row in rows])
    write_jsonl(output / "video_manifest.jsonl", [{"benchmark_instance_id": row["benchmark_instance_id"], "status": "not_run", "is_evidence": False} for row in rows])
    write_jsonl(output / "keyframe_manifest.jsonl", [{"benchmark_instance_id": row["benchmark_instance_id"], "status": "not_run", "is_evidence": False} for row in rows])
    write_jsonl(output / "failed_repair_replace.jsonl", [{"benchmark_instance_id": row["benchmark_instance_id"], "disposition": "pending_qualification", "failed_gates": ["qualification_not_run"], "reason": "plan_only_build"} for row in rows])
    (output / "physical_join_report.json").write_text(json.dumps(join_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "qualification_report.json").write_text(json.dumps({"schema_version": "fragile-soft.qualification.v1", "qualification_id": "not-run", "instance_count": len(rows), "qualified_count": 0, "rejected_count": 0, "status": "rejected", "e3_a_eligible": False, "formal_e3_a_manifest": False, "release_scope": "plan_only", "instance_reports": [], "qualification_report_hash": "sha256:" + "0" * 64}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"instance_count": len(rows), "population": "benchmark instances", "output": output.as_posix()}, ensure_ascii=False))
    return 0


def command_qualify(args: argparse.Namespace) -> int:
    if not args.server and (args.limit is None or args.limit > 6):
        raise SystemExit("local qualification requires --limit <= 6; use --server for 36 instances")
    if args.server and not args.episode_runner:
        raise SystemExit("--server qualification requires --episode-runner module:function and --formal-scope")
    if args.formal_scope and not args.episode_runner:
        raise SystemExit("--formal-scope requires --episode-runner module:function")
    if args.episode_runner and not args.formal_scope:
        raise SystemExit("--episode-runner requires --formal-scope")
    if args.formal_scope and not args.physical_index:
        raise SystemExit("--formal-scope requires --physical-index and an explicit physical join")
    if args.formal_scope and not args.physical_join_map:
        raise SystemExit("--formal-scope requires --physical-join-map")
    if args.formal_scope and not args.physical_release_id:
        raise SystemExit("--formal-scope requires --physical-release-id")
    adapter = _adapter(args)
    config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    instances = make_default_qualification_instances(
        adapter,
        config,
        physical_index=args.physical_index,
        physical_join_map=args.physical_join_map,
        physical_release_id=args.physical_release_id,
        require_physical_join=args.formal_scope,
    )
    if args.limit is not None:
        # Keep a calibration and hidden group in local smoke subsets; a prefix
        # would contain only one group and would correctly fail split gating.
        by_group: dict[str, list[dict[str, Any]]] = {}
        for instance in instances:
            by_group.setdefault(str(instance.get("canonical_group_id")), []).append(instance)
        selected: list[dict[str, Any]] = []
        while len(selected) < args.limit and by_group:
            for group in sorted(list(by_group)):
                rows = by_group.get(group, [])
                if rows:
                    selected.append(rows.pop(0))
                    if len(selected) >= args.limit:
                        break
                if not rows:
                    by_group.pop(group, None)
        instances = selected
    runner = load_episode_runner(args.episode_runner) if args.episode_runner else None
    result = run_qualification(
        instances,
        output_dir=args.output,
        seeds=tuple(config.get("seeds", (0, 1))),
        thresholds=config.get("thresholds"),
        backend_id=args.backend_id,
        formal_scope=args.formal_scope,
        episode_runner=runner,
    )
    print(json.dumps(result["report"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["report"]["status"] == "qualified" else 2


def command_archive(args: argparse.Namespace) -> int:
    result = create_deployment_archive(args.root, args.output)
    verified = verify_deployment_archive(args.output)
    print(json.dumps({"archive": result["archive"], "sha256": verified["sha256"], "file_count": result["file_count"], "verified": verified["verified"]}, ensure_ascii=False))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    result = validate_output_dir(args.output, expected_count=args.expected_count)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


def command_shard(args: argparse.Namespace) -> int:
    rows = []
    with Path(args.manifest).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    partitions = partition_instances(rows, args.shard_count)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for index, shard_rows in partitions.items():
        write_jsonl(output / f"shard-{index:04d}.jsonl", shard_rows)
    print(json.dumps({"shard_count": args.shard_count, "instance_count": len(rows), "output": output.as_posix()}, ensure_ascii=False))
    return 0


def command_promote(args: argparse.Namespace) -> int:
    instances = []
    with Path(args.instance_manifest).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                instances.append(json.loads(line))
    qualification = json.loads(Path(args.qualification_report).read_text(encoding="utf-8"))
    result = promote_e3_a_manifest(instances, qualification, output_path=args.output, allow_mock=args.allow_mock)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_join_template(args: argparse.Namespace) -> int:
    instances = []
    with Path(args.instance_manifest).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                instances.append(json.loads(line))
    rows = make_join_template(instances, physical_release_id=args.physical_release_id)
    write_jsonl(args.output, rows)
    print(json.dumps({"instance_count": len(rows), "output": str(args.output), "status": "needs_verified_physical_mapping"}, ensure_ascii=False))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fragile-soft-benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    adapt = sub.add_parser("adapt", help="adapt legacy P0 records")
    _common(adapt); adapt.add_argument("--output", type=Path, required=True); adapt.set_defaults(func=command_adapt)
    build = sub.add_parser("build", help="build benchmark instance manifests")
    _common(build); build.add_argument("--config", type=Path); build.add_argument("--output", type=Path, required=True); build.add_argument("--limit", type=int); build.add_argument("--server", action="store_true"); build.add_argument("--require-physical-join", action="store_true"); build.set_defaults(func=command_build)
    qualify = sub.add_parser("qualify", help="run qualification")
    _common(qualify); qualify.add_argument("--config", type=Path); qualify.add_argument("--output", type=Path, required=True); qualify.add_argument("--limit", type=int); qualify.add_argument("--server", action="store_true"); qualify.add_argument("--episode-runner"); qualify.add_argument("--backend-id", default="mock.deterministic.v1"); qualify.add_argument("--formal-scope", action="store_true"); qualify.set_defaults(func=command_qualify)
    archive = sub.add_parser("archive", help="create portable deployment archive")
    archive.add_argument("--root", type=Path, default=REPO_ROOT); archive.add_argument("--output", type=Path, required=True); archive.set_defaults(func=command_archive)
    validate = sub.add_parser("validate", help="validate emitted output contracts")
    validate.add_argument("--output", type=Path, required=True); validate.add_argument("--expected-count", type=int); validate.set_defaults(func=command_validate)
    shard = sub.add_parser("shard", help="partition an instance manifest")
    shard.add_argument("--manifest", type=Path, required=True); shard.add_argument("--shard-count", type=int, default=20); shard.add_argument("--output", type=Path, required=True); shard.set_defaults(func=command_shard)
    promote = sub.add_parser("promote", help="guard formal E3-A promotion")
    promote.add_argument("--instance-manifest", type=Path, required=True); promote.add_argument("--qualification-report", type=Path, required=True); promote.add_argument("--output", type=Path, required=True); promote.add_argument("--allow-mock", action="store_true"); promote.set_defaults(func=command_promote)
    join = sub.add_parser("join-template", help="create an explicit benchmark/physical mapping template")
    join.add_argument("--instance-manifest", type=Path, required=True); join.add_argument("--physical-release-id"); join.add_argument("--output", type=Path, required=True); join.set_defaults(func=command_join_template)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
