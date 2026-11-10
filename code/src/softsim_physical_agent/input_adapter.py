# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

from .io import content_hash, file_sha256, is_hash, iter_jsonl, read_json
from .routing import canonical_prior, build_route


ARTIFACT_ROLE_ALIASES = {
    "collision_mesh": "collision_mesh",
    "collision": "collision_mesh",
    "visual_mesh": "visual_mesh",
    "visual": "visual_mesh",
    "input_image": "source_image",
    "source_image": "source_image",
    "image": "source_image",
    "physical_properties": "physical_properties",
    "metadata": "metadata",
    "urdf": "urdf",
    "mpm_preview_gif": "preview_gif",
    "preview_gif": "preview_gif",
    "genesis_report": "genesis_report",
    "report": "genesis_report",
}


def _artifact_ref(path: Path, logical_root: Path, *, root_alias: str = "source") -> dict[str, Any]:
    try:
        logical = path.resolve().relative_to(logical_root.resolve()).as_posix()
    except ValueError:
        logical = path.name
    exists = path.is_file()
    return {
        "root_alias": root_alias,
        "path": logical,
        "sha256": file_sha256(path) if exists else content_hash({"missing": logical}),
        "exists": exists,
    }


def _metadata_physical(asset_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = asset_dir / "metadata.json"
    physical_path = asset_dir / "physical_properties.json"
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    physical = read_json(physical_path) if physical_path.is_file() else {}
    return metadata, physical


def _legacy_prior(row: Mapping[str, Any], physical: Mapping[str, Any]) -> dict[str, Any]:
    mpm = physical.get("mpm") if isinstance(physical.get("mpm"), Mapping) else physical
    mpm = mpm or {}
    def number(names: Iterable[str], default: float) -> float:
        for name in names:
            raw = mpm.get(name)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return float(raw)
        return default
    values = {
        "density_kg_m3": number(("density", "density_kg_m3"), 600.0),
        "youngs_modulus_pa": number(("youngs_modulus", "youngs_modulus_pa"), 50000.0),
        "poisson_ratio": number(("poisson_ratio",), 0.35),
    }
    params = [
        {"name": name, "value": value, "unit": {"density_kg_m3": "kg/m^3", "youngs_modulus_pa": "Pa", "poisson_ratio": "1"}[name], "valid_range": bounds}
        for name, value, bounds in (
            ("density_kg_m3", values["density_kg_m3"], [1.0, 30000.0]),
            ("youngs_modulus_pa", values["youngs_modulus_pa"], [1.0, 1.0e10]),
            ("poisson_ratio", values["poisson_ratio"], [-0.49, 0.499]),
        )
    ]
    fixed = {}
    for old, new in (("particle_size", "particle_size_m"), ("dt", "dt_s"), ("substeps", "substeps"), ("grid_density", "grid_density"), ("drop_height", "drop_height_m")):
        if old in mpm:
            fixed[new] = mpm[old]
    return {
        "schema_name": "softsim.physical_prior",
        "schema_version": 2,
        "sample_id": row["sample_id"],
        "asset_revision_id": row["asset_revision_id"],
        "status": "predicted_prior",
        "simulation_domain": "volumetric_soft_body" if row.get("geometry_family") == "solid" else "surface_proxy",
        "representation": "watertight_volume" if row.get("geometry_family") == "solid" else "alpha_wrap_proxy",
        "model_family": "MPM.Elastic",
        "parameters": params,
        "solver_adapter": {"solver": physical.get("solver", "genesis_mpm_elastic"), "parameters": values},
        "solver_numerics": fixed,
        "estimator": {"transport": "legacy_prior", "source": "physical_properties.json"},
        "provenance": {"prior_only": True, "is_ground_truth": False},
        "limitations": ["legacy_prior_only", "not_empirical_measurement"],
    }


def _row_from_asset_dir(asset_dir: Path, root: Path, *, source_kind: str = "legacy") -> dict[str, Any]:
    metadata, physical = _metadata_physical(asset_dir)
    sample_id = str(metadata.get("sample_id") or asset_dir.name)
    geometry_family = metadata.get("geometry_family") or metadata.get("geometry_family_id") or "unknown"
    material_family = metadata.get("material_family") or physical.get("profile") or "generic"
    artifact_paths = {
        "metadata": asset_dir / "metadata.json",
        "physical_properties": asset_dir / "physical_properties.json",
        "collision_mesh": asset_dir / "collision" / "collision.obj",
        "visual_mesh": asset_dir / "visual" / "visual_mesh.obj",
        "source_image": asset_dir / "input_image.png",
        "urdf": asset_dir / "asset.urdf",
        "preview_gif": asset_dir / "preview.gif",
        "genesis_report": asset_dir / "validation" / "genesis_mpm_report.json",
    }
    artifact_refs = {
        role: _artifact_ref(path, root)
        for role, path in artifact_paths.items()
    }
    raw_revision = metadata.get("asset_revision_id")
    revision_payload = {"metadata": metadata, "physical": physical, "artifacts": artifact_refs}
    revision = str(raw_revision or content_hash(revision_payload))
    row = {
        "sample_id": sample_id,
        "asset_revision_id": revision,
        "dataset_release_id": root.name,
        "geometry_family": geometry_family,
        "material_family": material_family,
        "object_type": metadata.get("object_type"),
        "canonical_id": metadata.get("canonical_id"),
        "metadata": metadata,
        "physical_properties": physical,
        "artifact_refs": artifact_refs,
        "source_kind": source_kind,
        "source_root": root.name,
    }
    prior = _legacy_prior(row, physical)
    row["prior"] = canonical_prior(row, prior, build_route(row, prior))
    return row


def load_legacy_assets(legacy_root: Path) -> list[dict[str, Any]]:
    rows = [_row_from_asset_dir(path.parent, legacy_root, source_kind="legacy") for path in sorted(legacy_root.glob("assets/*/metadata.json"))]
    if not rows:
        rows = [_row_from_asset_dir(path.parent, legacy_root, source_kind="legacy") for path in sorted(legacy_root.glob("*/metadata.json"))]
    return rows


def _resolve_manifest_path(raw: Any, manifest_root: Path, asset_dir: str | None = None) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        if asset_dir and not raw.startswith(asset_dir):
            candidate = manifest_root / asset_dir / raw
            if candidate.is_file():
                return candidate
        candidate = manifest_root / raw
        return candidate
    return path


def load_stage3_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    manifest_parent = manifest_path.resolve().parent
    root = manifest_parent.parent if manifest_parent.name == "manifests" else manifest_parent
    rows: list[dict[str, Any]] = []
    for item in iter_jsonl(manifest_path):
        sample_id = str(item.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"stage3 manifest row has no sample_id: {manifest_path}")
        asset_dir_raw = item.get("asset_dir")
        asset_dir = str(asset_dir_raw) if asset_dir_raw else None
        asset_base = (root / asset_dir).resolve() if asset_dir else root
        metadata_path = asset_base / "metadata.json"
        physical_path = asset_base / "physical_properties.json"
        metadata = read_json(metadata_path) if metadata_path.is_file() else {}
        physical = read_json(physical_path) if physical_path.is_file() else {}
        paths = item.get("paths") if isinstance(item.get("paths"), Mapping) else {}
        artifact_refs: dict[str, Any] = {}
        for raw_role, raw_path in paths.items():
            role = ARTIFACT_ROLE_ALIASES.get(str(raw_role), str(raw_role))
            resolved = _resolve_manifest_path(raw_path, root, asset_dir)
            if resolved is not None:
                artifact_refs[role] = _artifact_ref(resolved, root)
        for role, fallback in (("metadata", metadata_path), ("physical_properties", physical_path)):
            artifact_refs.setdefault(role, _artifact_ref(fallback, root))
        revision = item.get("asset_revision_id") or metadata.get("asset_revision_id") or content_hash({"manifest": item, "metadata": metadata, "physical": physical, "artifacts": artifact_refs})
        row = {
            "sample_id": sample_id,
            "asset_revision_id": str(revision),
            "dataset_release_id": str(item.get("dataset_release_id") or root.name),
            "geometry_family": item.get("geometry_family") or metadata.get("geometry_family") or "unknown",
            "material_family": item.get("material_family") or item.get("physical_profile") or metadata.get("material_family") or physical.get("profile") or "generic",
            "object_type": item.get("object_type") or metadata.get("object_type"),
            "canonical_id": item.get("canonical_id") or metadata.get("canonical_id"),
            "metadata": metadata,
            "physical_properties": physical,
            "artifact_refs": artifact_refs,
            "source_kind": "stage3_manifest",
            "source_root": root.name,
            "manifest_row": copy.deepcopy(item),
        }
        prior = _legacy_prior(row, physical)
        row["prior"] = canonical_prior(row, prior, build_route(row, prior))
        rows.append(row)
    return rows


def load_asset_release(asset_release_dir: Path, roots: Mapping[str, Path] | None = None) -> list[dict[str, Any]]:
    """Load the server asset-ledger contract used by the previous release.

    Artifact paths remain logical ``root_alias/path`` references.  When a root
    is mounted locally we additionally compute the actual file hash; missing
    mounts are retained as explicit ``exists=false`` records rather than being
    silently dropped.
    """

    roots = dict(roots or {})
    ledger_path = asset_release_dir / "asset_ledger.jsonl"
    if not ledger_path.is_file():
        manifest = asset_release_dir / "manifests" / "delivery_manifest.jsonl"
        if manifest.is_file():
            return load_stage3_manifest(manifest)
        return load_legacy_assets(asset_release_dir)
    release_manifest = read_json(asset_release_dir / "release_manifest.json") if (asset_release_dir / "release_manifest.json").is_file() else {}
    dataset_release_id = str(release_manifest.get("dataset_release_id") or asset_release_dir.name)
    rows: list[dict[str, Any]] = []
    for item in iter_jsonl(ledger_path):
        sample_id = str(item.get("sample_id") or "")
        if not sample_id:
            continue
        artifact_refs: dict[str, Any] = {}
        raw_artifacts = item.get("artifacts") or item.get("artifact_refs") or {}
        if isinstance(raw_artifacts, Mapping):
            iterable = raw_artifacts.items()
        elif isinstance(raw_artifacts, list):
            iterable = ((str(value.get("role", "artifact")), value) for value in raw_artifacts if isinstance(value, Mapping))
        else:
            iterable = ()
        for raw_role, raw_value in iterable:
            role = str(raw_role)
            if isinstance(raw_value, Mapping):
                path_value = raw_value.get("path") or raw_value.get("relative_path")
                alias = str(raw_value.get("root_alias") or "")
                supplied_hash = raw_value.get("sha256")
            else:
                path_value, alias, supplied_hash = raw_value, "", None
            if not path_value:
                continue
            logical = str(path_value).replace("\\", "/").lstrip("/")
            root = roots.get(alias) if alias else asset_release_dir
            resolved = root / logical if root else None
            exists = bool(resolved and resolved.is_file())
            digest = file_sha256(resolved) if exists else str(supplied_hash or content_hash({"missing": logical}))
            if not digest.startswith("sha256:"):
                digest = "sha256:" + digest
            artifact_refs[role] = {"root_alias": alias or "asset_release", "path": logical, "sha256": digest, "exists": exists}
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        physical = item.get("physical_properties") if isinstance(item.get("physical_properties"), Mapping) else {}
        prior = item.get("prior") if isinstance(item.get("prior"), Mapping) else item.get("physical_bundle") if isinstance(item.get("physical_bundle"), Mapping) else None
        row = {
            "sample_id": sample_id,
            "asset_revision_id": str(item.get("asset_revision_id") or content_hash({"sample_id": sample_id, "artifacts": artifact_refs})),
            "dataset_release_id": dataset_release_id,
            "geometry_family": item.get("geometry_family") or item.get("geometry_family_id") or "unknown",
            "material_family": item.get("material_family") or item.get("physical_profile") or "generic",
            "object_type": item.get("object_type"),
            "canonical_id": item.get("canonical_id"),
            "metadata": dict(metadata),
            "physical_properties": dict(physical),
            "artifact_refs": artifact_refs,
            "source_kind": "asset_release_ledger",
            "source_root": asset_release_dir.name,
            "asset_status": item.get("asset_status"),
        }
        if not prior:
            prior = _legacy_prior(row, physical)
        row["prior"] = canonical_prior(row, prior, build_route(row, prior))
        rows.append(row)
    return rows


def load_previous_staging(staging_root: Path) -> list[dict[str, Any]]:
    """Read a prior physical staging/index without treating it as verified input."""

    release_meta = read_json(staging_root / "physical_release_manifest.json") if (staging_root / "physical_release_manifest.json").is_file() else {}
    staging_release_id = str(release_meta.get("physical_release_id") or release_meta.get("dataset_release_id") or staging_root.name)
    index_candidates = [staging_root / "physical_bundle_index.jsonl", staging_root / "bundle_index.jsonl", staging_root / "index.jsonl"]
    index = next((path for path in index_candidates if path.is_file()), None)
    rows: list[dict[str, Any]] = []
    if index:
        for item in iter_jsonl(index):
            bundle_info = item.get("bundle") if isinstance(item.get("bundle"), Mapping) else {}
            raw_path = bundle_info.get("path") or item.get("physics_path") or item.get("path")
            bundle_path = staging_root / str(raw_path) if raw_path and not Path(str(raw_path)).is_absolute() else Path(str(raw_path)) if raw_path else None
            bundle = read_json(bundle_path) if bundle_path and bundle_path.is_file() else {}
            sample_id = str(item.get("sample_id") or bundle.get("sample_id") or "")
            if not sample_id:
                continue
            prior = copy.deepcopy(bundle) if bundle else {"sample_id": sample_id, "asset_revision_id": item.get("asset_revision_id"), "status": "predicted_prior", "parameters": []}
            row = {
                "sample_id": sample_id,
                "asset_revision_id": str(item.get("asset_revision_id") or bundle.get("asset_revision_id") or content_hash(bundle)),
                "dataset_release_id": str(item.get("dataset_release_id") or bundle.get("dataset_release_id") or staging_release_id),
                "geometry_family": item.get("geometry_family") or bundle.get("geometry_family") or (bundle.get("route") or {}).get("geometry_family") or "unknown",
                "material_family": item.get("material_family") or bundle.get("material_family") or (bundle.get("route") or {}).get("material_family") or "generic",
                "object_type": item.get("object_type"),
                "artifact_refs": {"prior_bundle": {"root_alias": "staging", "path": str(bundle_path.relative_to(staging_root).as_posix()) if bundle_path and bundle_path.is_relative_to(staging_root) else str(raw_path), "sha256": file_sha256(bundle_path) if bundle_path and bundle_path.is_file() else content_hash(bundle), "exists": bool(bundle_path and bundle_path.is_file())}},
                "source_kind": "physical_staging",
                "source_root": staging_root.name,
            }
            row["prior"] = canonical_prior(row, prior, build_route(row, prior))
            rows.append(row)
    if not rows:
        for physics_path in sorted(staging_root.glob("bundles/*/physics.json")):
            bundle = read_json(physics_path)
            sample_id = str(bundle.get("sample_id") or physics_path.parent.name)
            row = {
                "sample_id": sample_id,
                "asset_revision_id": str(bundle.get("asset_revision_id") or content_hash(bundle)),
                "dataset_release_id": staging_root.name,
                "geometry_family": bundle.get("geometry_family") or "unknown",
                "material_family": bundle.get("material_family") or "generic",
                "artifact_refs": {"prior_bundle": {"root_alias": "staging", "path": physics_path.relative_to(staging_root).as_posix(), "sha256": file_sha256(physics_path), "exists": True}},
                "source_kind": "physical_staging",
                "source_root": staging_root.name,
            }
            row["prior"] = canonical_prior(row, bundle, build_route(row, bundle))
            rows.append(row)
    if not rows:
        # A resumable staging run may have task-local priors before aggregation.
        for prior_path in sorted(staging_root.glob("tasks/*/prior.json")):
            prior = read_json(prior_path)
            sample_id = str(prior.get("sample_id") or prior_path.parent.name)
            row = {
                "sample_id": sample_id,
                "asset_revision_id": str(prior.get("asset_revision_id") or content_hash(prior)),
                "dataset_release_id": staging_root.name,
                "geometry_family": prior.get("geometry_family") or (prior.get("route") or {}).get("geometry_family") or "unknown",
                "material_family": prior.get("material_family") or (prior.get("route") or {}).get("material_family") or "generic",
                "artifact_refs": {"prior_task": {"root_alias": "staging", "path": prior_path.relative_to(staging_root).as_posix(), "sha256": file_sha256(prior_path), "exists": True}},
                "source_kind": "physical_staging",
                "source_root": staging_root.name,
            }
            row["prior"] = canonical_prior(row, prior, build_route(row, prior))
            rows.append(row)
    return rows


def merge_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge sources by sample identity, preferring explicit stage3/asset rows."""

    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for row in group:
            sample_id = str(row.get("sample_id") or "")
            if not sample_id:
                continue
            if sample_id not in merged:
                merged[sample_id] = copy.deepcopy(row)
            else:
                current = merged[sample_id]
                for key, value in row.items():
                    if key == "artifact_refs":
                        current.setdefault(key, {}).update(value or {})
                    elif key == "prior" and row.get("source_kind") == "physical_staging":
                        current[key] = value
                    elif key == "asset_revision_id" and row.get("source_kind") == "physical_staging" and value:
                        # Keep the source asset revision when available, but let a
                        # newer staging bundle provide the prior parameters.
                        current[key] = current.get(key) or value
                    elif current.get(key) in (None, "", {}, []):
                        current[key] = value
    return [merged[key] for key in sorted(merged)]


# Name retained for callers of the earlier prototype.
load_release_assets = load_asset_release
