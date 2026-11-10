# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .io import atomic_write_json, content_hash, file_sha256, iter_jsonl, read_json


def classify_legacy_report(report: dict[str, Any]) -> str:
    """Classify an old Genesis report without upgrading execution to behavior verification."""

    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    top_level = report.get("behavior_pass")
    nested = metrics.get("behavior_pass") if isinstance(metrics.get("behavior_pass"), bool) else None
    behavior_value = top_level if isinstance(top_level, bool) else nested
    has_behavior = isinstance(behavior_value, bool) and bool(metrics) and any(
        key in metrics for key in ("contact_fraction", "settled", "deformation_ratio", "recovery_ratio", "behavior_pass")
    )
    if not has_behavior:
        return "sim_executed_unverified"
    return "verified" if behavior_value is True else "behavior_evidence_rejected"


def _hash_ref(root: Path, relative: str | None) -> dict[str, Any]:
    if not relative:
        return {"path": None, "exists": False, "sha256": content_hash({"missing": None})}
    path = root / relative
    return {"path": str(relative).replace("\\", "/"), "exists": path.is_file(), "sha256": file_sha256(path) if path.is_file() else content_hash({"missing": str(relative)})}


def build_golden_reference_audit(stage3_root: Path, output: Path, *, sample_per_family: int = 1) -> dict[str, Any]:
    """Record stage3's old facts and hashes without copying its assets."""

    stage3_root = Path(stage3_root).resolve()
    manifest_path = stage3_root / "manifests" / "delivery_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows = list(iter_jsonl(manifest_path))
    family_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        family_rows.setdefault(str(row.get("geometry_family") or "unknown"), []).append(row)
    selected: list[dict[str, Any]] = []
    for family in sorted(family_rows):
        selected.extend(sorted(family_rows[family], key=lambda item: str(item.get("sample_id")))[:sample_per_family])
    assets: list[dict[str, Any]] = []
    for row in selected:
        asset_dir = str(row.get("asset_dir") or "")
        asset_root = stage3_root / asset_dir
        metadata_path = asset_root / "metadata.json"
        physical_path = asset_root / "physical_properties.json"
        report_path = asset_root / "validation" / "genesis_mpm_report.json"
        metadata = read_json(metadata_path) if metadata_path.is_file() else {}
        physical = read_json(physical_path) if physical_path.is_file() else {}
        report = read_json(report_path) if report_path.is_file() else {}
        paths = row.get("paths") if isinstance(row.get("paths"), dict) else {}
        artifact_hashes: dict[str, Any] = {}
        for role, raw in sorted(paths.items()):
            if not isinstance(raw, str):
                continue
            relative = (Path(asset_dir) / raw).as_posix()
            artifact_hashes[str(role)] = _hash_ref(stage3_root, relative)
        legacy_evidence = classify_legacy_report(report)
        assets.append({
            "sample_id": row.get("sample_id"),
            "asset_revision_id": row.get("asset_revision_id") or metadata.get("asset_revision_id") or content_hash({"metadata": metadata, "physical": physical}),
            "geometry_family": row.get("geometry_family"),
            "material_family": row.get("material_family") or row.get("physical_profile"),
            "relative_asset_dir": asset_dir,
            "artifact_hashes": artifact_hashes,
            "old_metadata_sha256": file_sha256(metadata_path) if metadata_path.is_file() else None,
            "old_physical_properties_sha256": file_sha256(physical_path) if physical_path.is_file() else None,
            "old_genesis_report_sha256": file_sha256(report_path) if report_path.is_file() else None,
            "old_config_hash": content_hash(physical.get("mpm", {})),
            "old_report_hash": file_sha256(report_path) if report_path.is_file() else None,
            "old_config": physical.get("mpm", {}),
            "old_solver": physical.get("solver"),
            "old_proxy": (report.get("metrics", {}).get("mesh", {}) if isinstance(report.get("metrics"), dict) else {}).get("proxy_mode") or (report.get("metrics", {}).get("mesh", {}) if isinstance(report.get("metrics"), dict) else {}).get("proxy_used"),
            "old_material_profile": physical.get("profile"),
            "old_report_status": report.get("status"),
            "old_report_metrics_keys": sorted((report.get("metrics") or {}).keys()) if isinstance(report.get("metrics"), dict) else [],
            "old_behavior_evidence": legacy_evidence,
            "old_report_relative_ref": (Path(asset_dir) / "validation" / "genesis_mpm_report.json").as_posix(),
        })
    summary_paths = {}
    for relative in ("reports/physical_parameter_assignment.json", "reports/showcase_mpm_status.json", "reports/delivery_summary.json", "manifests/delivery_manifest.jsonl"):
        summary_paths[relative] = _hash_ref(stage3_root, relative)
    family_counts = Counter(str(row.get("geometry_family") or "unknown") for row in rows)
    audit = {
        "schema_name": "softsim.golden_reference_audit",
        "schema_version": 1,
        "audit_kind": "reference_only",
        "reference_project": "wjy/stage3_showcase_150_genesis_mpm_20260724",
        "source_root_alias": "stage3_showcase_150_genesis_mpm_20260724",
        "source_root_policy": "resolved at audit invocation; authoritative records use source_root_alias and relative paths only",
        "asset_count": len(rows),
        "geometry_family_counts": dict(sorted(family_counts.items())),
        "selected_count": len(assets),
        "selected_per_family": sample_per_family,
        "old_delivery_status": "pass",
        "old_results_are_not_new_experiment": True,
        "source_summary_hashes": summary_paths,
        "selected_assets": assets,
        "limitations": ["stage3_report_lacks_behavior_metrics_where_marked_sim_executed_unverified", "reference_only_not_agent_result", "source_root_is_an_alias_resolved_by_the_invoker"],
    }
    audit["audit_sha256"] = content_hash(audit)
    atomic_write_json(output, audit)
    return audit
