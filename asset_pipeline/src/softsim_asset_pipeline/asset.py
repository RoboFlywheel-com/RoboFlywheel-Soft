# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import shutil
import struct
import tempfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any, Iterable

from .cloth import is_cloth
from .contracts import validate_asset_ledger_record, with_record_hash
from .glb import audit_glb
from .io import (
    atomic_text_writer,
    atomic_write_json,
    atomic_write_jsonl,
    content_id,
    iter_jsonl,
    read_json,
    safe_join,
    sha256_file,
    validate_relative_path,
)


def mesh_relative(row: dict[str, Any]) -> str:
    explicit = row.get("mesh_relpath") or row.get("mesh")
    if explicit:
        return validate_relative_path(str(explicit))
    sample_id = str(row["sample_id"])
    return f"samples/{sample_id}/mesh/{sample_id}.glb"


def image_relative(row: dict[str, Any]) -> str | None:
    explicit = row.get("input_image_relpath") or row.get("input_image")
    return validate_relative_path(str(explicit)) if explicit else None


def plan_assets(
    *,
    manifest: Path,
    output: Path,
    mesh_roots: dict[str, Path],
    overlay_root: Path,
    expected_count: int | None,
    shard_size: int,
    estimated_cloth_bytes_per_asset: int,
) -> dict[str, Any]:
    sample_ids: set[str] = set()
    duplicate_ids: list[str] = []
    family_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    row_count = 0
    invalid_rows: list[dict[str, Any]] = []
    for row_count, row in enumerate(iter_jsonl(manifest), 1):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            invalid_rows.append({"row": row_count, "failure": "sample_id_missing"})
            continue
        if sample_id in sample_ids:
            duplicate_ids.append(sample_id)
        sample_ids.add(sample_id)
        family_counts[str(row.get("geometry_family") or "unknown")] += 1
        route_counts[str(row.get("generation_route") or "unknown")] += 1
        object_counts[str(row.get("object_type") or "unknown")] += 1
    cloth_count = family_counts["cloth_sheet"]
    for route, count in route_counts.items():
        if route == "procedural_cloth_grid" and cloth_count == 0:
            cloth_count = count
    gates: list[str] = []
    if expected_count is not None and row_count != expected_count:
        gates.append(f"expected_count_mismatch:{expected_count}:{row_count}")
    if duplicate_ids:
        gates.append("duplicate_sample_id")
    if invalid_rows:
        gates.append("invalid_manifest_row")
    summary = {
        "schema_name": "softsim.asset_inventory_plan",
        "schema_version": 1,
        "dry_run": True,
        "source_manifest_sha256": sha256_file(manifest),
        "manifest_rows": row_count,
        "unique_sample_ids": len(sample_ids),
        "dynamic_cloth_count": sum(
            1 for row in iter_jsonl(manifest) if is_cloth(row)
        ),
        "family_counts": dict(sorted(family_counts.items())),
        "generation_route_counts": dict(sorted(route_counts.items())),
        "unique_object_types": len(object_counts),
        "duplicate_sample_ids": sorted(set(duplicate_ids))[:100],
        "invalid_rows": invalid_rows[:100],
        "mesh_root_aliases": list(mesh_roots),
        "mesh_roots_mounted": {alias: root.is_dir() for alias, root in mesh_roots.items()},
        "overlay_root": str(overlay_root),
        "overlay_conflicts_with_source": any(
            root.resolve() == overlay_root.resolve() for root in mesh_roots.values()
        ),
        "shard_size": shard_size,
        "estimated_shards": (row_count + shard_size - 1) // shard_size,
        "estimated_cloth_bytes_per_asset": estimated_cloth_bytes_per_asset,
        "estimated_cloth_overlay_bytes": sum(
            1 for row in iter_jsonl(manifest) if is_cloth(row)
        )
        * estimated_cloth_bytes_per_asset,
        "promotion_gate_failures": gates,
    }
    atomic_write_json(output, summary)
    return summary


def _glb_metadata_status(path: Path) -> tuple[str, list[str]]:
    failures: list[str] = []
    try:
        stat = path.stat()
        if not path.is_file():
            failures.append("not_regular_file")
        if stat.st_size <= 0:
            failures.append("zero_byte_file")
        with path.open("rb") as handle:
            header = handle.read(12)
        if len(header) != 12:
            failures.append("glb_header_truncated")
        else:
            magic, version, declared_length = struct.unpack("<III", header)
            if magic != 0x46546C67:
                failures.append("glb_bad_magic")
            if version != 2:
                failures.append("glb_unsupported_version")
            if declared_length != stat.st_size:
                failures.append("glb_declared_length_mismatch")
    except OSError as exc:
        failures.append(f"file_error:{exc.errno}")
    return ("pass" if not failures else "corrupt", sorted(set(failures)))


def _inventory_record(
    row: dict[str, Any],
    mesh_roots: dict[str, Path],
    image_root: Path | None,
    mesh_indexes: dict[str, set[str]] | None = None,
    verify_glb_headers: bool = True,
    verify_image_paths: bool = True,
    image_root_mounted: bool | None = None,
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    sample_id = str(row["sample_id"])
    failures: list[str] = []
    try:
        relative = mesh_relative(row)
    except ValueError as exc:
        relative = f"samples/{sample_id}/mesh/{sample_id}.glb"
        failures.append(f"unsafe_mesh_path:{exc}")
    placements: list[dict[str, Any]] = []
    expected_paths: dict[str, set[str]] = {alias: set() for alias in mesh_roots}
    for alias, root in mesh_roots.items():
        indexed = mesh_indexes and alias in mesh_indexes
        if indexed and not verify_glb_headers:
            # mesh_relative() already validates the POSIX path. Avoid a FUSE
            # resolve/stat for every row when a trusted flat index is used.
            candidate = root / Path(relative)
        else:
            try:
                candidate = safe_join(root, relative)
            except ValueError as exc:
                failures.append(f"path_escape:{alias}:{exc}")
                continue
        expected_paths[alias].add(str(candidate))
        if (relative in mesh_indexes.get(alias, set()) if indexed else candidate.exists()):
            placements.append({"root_alias": alias, "path": relative, "absolute_path": str(candidate)})
    if failures:
        status = "corrupt"
    elif len(placements) > 1:
        status = "duplicate"
        failures.append("duplicate_placement")
    elif not placements:
        status = "cloth_pending" if is_cloth(row) else "missing"
        failures.append("cloth_mesh_pending" if is_cloth(row) else "mesh_missing")
    elif verify_glb_headers:
        status, metadata_failures = _glb_metadata_status(Path(placements[0]["absolute_path"]))
        failures.extend(metadata_failures)
    else:
        status = "pass"
    image_ref: dict[str, Any] | None = None
    relative_image = image_relative(row)
    if relative_image:
        if verify_image_paths:
            image_path = safe_join(image_root, relative_image) if image_root else None
            mounted = bool(image_path and image_path.is_file())
            image_check = "path"
        else:
            image_path = None
            mounted = bool(image_root_mounted if image_root_mounted is not None else image_root)
            image_check = "root_only"
        image_ref = {
            "root_alias": "image_bank",
            "path": relative_image,
            "sha256": row.get("input_image_sha256"),
            "mounted": mounted,
            "check": image_check,
        }
    record = {
        "schema_name": "softsim.asset_inventory_record",
        "schema_version": 1,
        "sample_id": sample_id,
        "canonical_id": row.get("canonical_id"),
        "object_type": row.get("object_type"),
        "geometry_family": row.get("geometry_family"),
        "material_family": row.get("material_family"),
        "generation_route": row.get("generation_route"),
        "status": status,
        "failure_codes": sorted(set(failures)),
        "placements": placements,
        "image": image_ref,
    }
    return record, expected_paths


def inventory_assets(
    *,
    manifest: Path,
    mesh_roots: dict[str, Path],
    image_root: Path | None,
    output_dir: Path,
    scan_orphans: bool = False,
    workers: int = 1,
    mesh_indexes: dict[str, set[str]] | None = None,
    verify_glb_headers: bool = True,
    verify_image_paths: bool = True,
) -> dict[str, Any]:
    if not mesh_roots:
        raise ValueError("At least one explicit --mesh-root is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_paths: dict[str, set[str]] = {alias: set() for alias in mesh_roots}
    records: list[dict[str, Any]] = []
    by_status: Counter[str] = Counter()
    image_root_mounted = (
        bool(image_root and image_root.is_dir()) if not verify_image_paths else None
    )
    if workers > 1 and not scan_orphans:
        # Keep only a small bounded queue: a 100k manifest must not create 100k
        # futures (which can consume multiple GB before the first result lands).
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = set()
            rows = iter_jsonl(manifest)
            for _ in range(max(1, workers * 4)):
                try:
                    pending.add(
                        executor.submit(
                            _inventory_record,
                            next(rows),
                            mesh_roots,
                            image_root,
                            mesh_indexes,
                            verify_glb_headers,
                            verify_image_paths,
                            image_root_mounted,
                        )
                    )
                except StopIteration:
                    break
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    record, paths = future.result()
                    records.append(record)
                    by_status[record["status"]] += 1
                    for alias, values in paths.items():
                        expected_paths[alias].update(values)
                    try:
                        pending.add(
                            executor.submit(
                                _inventory_record,
                                next(rows),
                                mesh_roots,
                                image_root,
                                mesh_indexes,
                                verify_glb_headers,
                                verify_image_paths,
                                image_root_mounted,
                            )
                        )
                    except StopIteration:
                        pass
    else:
        for row in iter_jsonl(manifest):
            record, paths = _inventory_record(
                row,
                mesh_roots,
                image_root,
                mesh_indexes,
                verify_glb_headers,
                verify_image_paths,
                image_root_mounted,
            )
            records.append(record)
            by_status[record["status"]] += 1
            for alias, values in paths.items():
                expected_paths[alias].update(values)
    records.sort(key=lambda row: row["sample_id"])
    atomic_write_jsonl(output_dir / "inventory_records.jsonl", records)

    orphan_records: list[dict[str, Any]] = []
    if scan_orphans:
        for alias, root in mesh_roots.items():
            for candidate in root.rglob("*.glb"):
                if str(candidate.resolve()) not in expected_paths[alias]:
                    orphan_records.append(
                        {
                            "root_alias": alias,
                            "path": candidate.resolve().relative_to(root.resolve()).as_posix(),
                            "status": "orphan",
                        }
                    )
    orphan_records.sort(key=lambda row: (row["root_alias"], row["path"]))
    atomic_write_jsonl(output_dir / "orphans.jsonl", orphan_records)
    summary = {
        "schema_version": 1,
        "source_manifest_sha256": sha256_file(manifest),
        "total": len(records),
        "by_status": dict(sorted(by_status.items())),
        "orphans_scanned": scan_orphans,
        "orphan_count": len(orphan_records),
        "source_roots_open_mode": "read_only",
        "mesh_index_aliases": sorted(mesh_indexes or {}),
        "glb_header_check": "path_header" if verify_glb_headers else "deferred",
        "image_path_check": "path" if verify_image_paths else "root_only",
    }
    atomic_write_json(output_dir / "inventory_summary.json", summary)
    return summary


def _audit_one(record: dict[str, Any], qa_config: dict[str, Any]) -> dict[str, Any]:
    sample_id = record["sample_id"]
    if record["status"] != "pass":
        return {
            "sample_id": sample_id,
            "status": record["status"],
            "failure_codes": record["failure_codes"],
            "repair_action": "cloth_backfill" if record["status"] == "cloth_pending" else "block",
        }
    placement = record["placements"][0]
    path = Path(placement["absolute_path"])
    try:
        result = audit_glb(path)
        family_profile = (qa_config.get("families") or {}).get(record.get("geometry_family"), {})
        max_faces = int(family_profile.get("max_faces", qa_config.get("max_faces", 2_000_000)))
        max_components = int(family_profile.get("max_components", qa_config.get("max_components", 128)))
        if result["geometry"]["faces"] > max_faces:
            result["failure_codes"].append("face_budget_exceeded")
        if result["geometry"]["components"] > max_components:
            result["failure_codes"].append("component_budget_exceeded")
        result["failure_codes"] = sorted(set(result["failure_codes"]))
        result["status"] = "pass" if not result["failure_codes"] else "qa_failed"
        return {
            "sample_id": sample_id,
            "status": result["status"],
            "failure_codes": result["failure_codes"],
            "warnings": result["warnings"],
            "repair_action": "none" if result["status"] == "pass" else "manual_review",
            "artifact": {
                "role": "visual_mesh",
                "root_alias": placement["root_alias"],
                "path": placement["path"],
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "media_type": "model/gltf-binary",
            },
            "geometry": result["geometry"],
            "materials": result["materials"],
        }
    except BaseException as exc:
        return {
            "sample_id": sample_id,
            "status": "corrupt",
            "failure_codes": [f"glb_parse_error:{type(exc).__name__}:{exc}"],
            "repair_action": "trellis_regenerate" if not is_cloth(record) else "cloth_backfill",
        }


def audit_assets(
    *,
    inventory_path: Path,
    qa_config_path: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    qa_config = read_json(qa_config_path)
    records = list(iter_jsonl(inventory_path))
    output: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_audit_one, record, qa_config): record["sample_id"] for record in records}
        for future in as_completed(futures):
            try:
                output.append(future.result())
            except BaseException as exc:
                output.append(
                    {
                        "sample_id": futures[future],
                        "status": "corrupt",
                        "failure_codes": [f"audit_worker_crash:{type(exc).__name__}:{exc}"],
                        "repair_action": "manual_review",
                    }
                )
    output.sort(key=lambda row: row["sample_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output_dir / "asset_audit_records.jsonl", output)
    by_status = Counter(row["status"] for row in output)
    failures = [row for row in output if row["status"] != "pass"]
    atomic_write_jsonl(output_dir / "failures.jsonl", failures)
    summary = {
        "schema_version": 1,
        "inventory_sha256": sha256_file(inventory_path),
        "qa_config_sha256": sha256_file(qa_config_path),
        "total": len(output),
        "by_status": dict(sorted(by_status.items())),
        "failure_count": len(failures),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _load_by_sample(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in iter_jsonl(path):
            sample_id = str(row["sample_id"])
            if sample_id in result:
                raise ValueError(f"Duplicate result for {sample_id} across {path}")
            result[sample_id] = row
    return result


def _backfill_artifact(backfill: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    revision = str(backfill["revision"])
    artifact = next(item for item in backfill["artifacts"] if item["role"] == "visual_mesh")
    sample_id = str(backfill["sample_id"])
    path = f"samples/{sample_id}/revisions/{revision}/{artifact['relative_to_revision']}"
    return (
        {
            "role": "visual_mesh",
            "root_alias": "cloth_overlay",
            "path": path,
            "sha256": artifact["sha256"],
            "size_bytes": artifact["size_bytes"],
            "media_type": artifact["media_type"],
        },
        backfill["geometry"],
    )


def _metric_scale(row: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    scale = row.get("metric_scale")
    if isinstance(scale, dict) and scale.get("scale_to_meter") is not None:
        return dict(scale), []
    return (
        {
            "unit": "normalized_asset_unit",
            "scale_to_meter": None,
            "provenance": "unassigned_requires_physical_prior_or_measurement",
            "confidence": 0.0,
        },
        ["metric_scale_unassigned"],
    )


def publish_asset_release(
    *,
    manifest: Path,
    inventory_path: Path,
    audit_path: Path,
    backfill_paths: list[Path],
    releases_root: Path,
    dataset_release_id: str,
    expected_count: int | None,
    strict: bool,
    producer_version: str = "1.0.0",
) -> dict[str, Any]:
    # The manifest is a large immutable input; hash it once and reuse the
    # digest for the release check, every ledger row, and the release record.
    source_manifest_sha256 = sha256_file(manifest)
    destination = releases_root / dataset_release_id
    if destination.exists():
        release_manifest = read_json(destination / "release_manifest.json")
        if release_manifest.get("source_manifest_sha256") != source_manifest_sha256:
            raise ValueError("immutable_release_id_already_exists_with_different_input")
        if sha256_file(destination / "asset_ledger.jsonl") != (destination / "asset_ledger.sha256").read_text().strip():
            raise ValueError("existing_release_hash_mismatch")
        return release_manifest
    inventory = _load_by_sample([inventory_path])
    audits = _load_by_sample([audit_path])
    backfills = _load_by_sample(backfill_paths) if backfill_paths else {}
    rows: list[dict[str, Any]] = []
    by_status: Counter[str] = Counter()
    missing_results: list[str] = []
    for source in iter_jsonl(manifest):
        sample_id = str(source["sample_id"])
        inv = inventory.get(sample_id)
        audit = audits.get(sample_id)
        artifacts: list[dict[str, Any]] = []
        geometry: dict[str, Any] = {}
        failure_codes: list[str] = []
        backfill_status = "not_applicable"
        static_status = audit.get("status") if audit else "missing"
        if audit and audit.get("status") == "pass" and audit.get("artifact"):
            artifacts.append(audit["artifact"])
            geometry = audit.get("geometry") or {}
        elif is_cloth(source) and sample_id in backfills and backfills[sample_id].get("status") in {"generated", "skipped"}:
            artifact, geometry = _backfill_artifact(backfills[sample_id])
            artifacts.append(artifact)
            backfill_status = "pass"
            static_status = "pass"
        else:
            failure_codes.extend((audit or inv or {}).get("failure_codes") or ["maintenance_result_missing"])
            missing_results.append(sample_id)
        relative_image = image_relative(source)
        if relative_image and source.get("input_image_sha256"):
            artifacts.append(
                {
                    "role": "source_image",
                    "root_alias": "image_bank",
                    "path": relative_image,
                    "sha256": source["input_image_sha256"],
                    "size_bytes": None,
                    "media_type": "image/png",
                }
            )
        scale, scale_warnings = _metric_scale(source)
        if not artifacts or not any(item["role"] == "visual_mesh" for item in artifacts):
            asset_status = "deferred_cloth_generation" if is_cloth(source) else "missing_artifact"
        elif failure_codes:
            asset_status = "invalid"
        elif scale_warnings:
            asset_status = "ready_with_warnings"
        else:
            asset_status = "ready"
        failure_codes.extend(scale_warnings)
        revision_payload = {
            "sample_id": sample_id,
            "artifacts": artifacts,
            "metric_scale": scale,
            "producer_version": producer_version,
        }
        ledger_record = with_record_hash(
            {
                "schema_name": "softsim.asset_ledger",
                "schema_version": 1,
                "sample_id": sample_id,
                "asset_revision_id": content_id(revision_payload),
                "dataset_release_id": dataset_release_id,
                "canonical_id": source.get("canonical_id"),
                "object_type": source.get("object_type"),
                "geometry_family": source.get("geometry_family"),
                "material_family": source.get("material_family"),
                "generation_route": source.get("generation_route"),
                "asset_status": asset_status,
                "failure_codes": sorted(set(failure_codes)),
                "repairable": asset_status in {"missing_artifact", "deferred_cloth_generation"},
                "metric_scale": scale,
                "artifacts": artifacts,
                "geometry": geometry,
                "generation_provenance": {
                    "source_manifest_hash": source_manifest_sha256,
                    "producer_version": producer_version,
                },
                "maintenance": {
                    "cloth_backfill_status": backfill_status,
                    "static_check_status": static_status,
                },
                "physical_bundle_ref": None,
            }
        )
        validate_asset_ledger_record(ledger_record)
        rows.append(ledger_record)
        by_status[asset_status] += 1
    rows.sort(key=lambda row: row["sample_id"])
    failures: list[str] = []
    if expected_count is not None and len(rows) != expected_count:
        failures.append(f"expected_count_mismatch:{expected_count}:{len(rows)}")
    if strict and any(row["asset_status"] not in {"ready", "ready_with_warnings"} for row in rows):
        failures.append("geometry_not_ready")
    if failures:
        raise ValueError("asset_release_gate_failed:" + ",".join(failures))
    releases_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{dataset_release_id}.", dir=releases_root))
    try:
        audit_dir = temporary / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_jsonl(temporary / "asset_ledger.jsonl", rows)
        ledger_sha = sha256_file(temporary / "asset_ledger.jsonl")
        with atomic_text_writer(temporary / "asset_ledger.sha256") as handle:
            handle.write(ledger_sha + "\n")
        shutil.copy2(audit_path, audit_dir / "asset_audit_records.jsonl")
        release_manifest = {
            "schema_name": "softsim.asset_release",
            "schema_version": 1,
            "dataset_release_id": dataset_release_id,
            "source_manifest_sha256": source_manifest_sha256,
            "asset_ledger_sha256": ledger_sha,
            "asset_count": len(rows),
            "asset_status_counts": dict(sorted(by_status.items())),
            "root_aliases": sorted(
                {
                    artifact["root_alias"]
                    for row in rows
                    for artifact in row["artifacts"]
                }
            ),
            "producer_version": producer_version,
            "immutable": True,
        }
        atomic_write_json(temporary / "release_manifest.json", release_manifest)
        os.replace(temporary, destination)
        return release_manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
