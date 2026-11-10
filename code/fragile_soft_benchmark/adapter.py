# SPDX-License-Identifier: Apache-2.0
"""Adapter for the heterogeneous P0 JSON/CSV/video/state records.

The adapter is a read-only compatibility layer.  It emits compact source
references and hashes; it never treats the P0 pilot rows as new benchmark
episodes.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .hashing import combined_source_hash, normalise_relative_path, reference_digest, sha256_json


ADAPTER_SCHEMA_VERSION = "fragile-soft.p0-adapter.v1"


def _read_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    if path.suffix.lower() == ".csv":
        return _read_csv(path)
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # A few P0 exports are JSONL despite a .json suffix.
        rows = []
        for line in text.splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, Mapping):
                    rows.append(dict(value))
        return rows


def _as_rows(value: Any, key: str = "assets") -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    if isinstance(value, Mapping):
        rows = value.get(key)
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, Mapping)]
        if isinstance(rows, Mapping):
            return [dict(row, id=str(name)) for name, row in rows.items() if isinstance(row, Mapping)]
        # Legacy exports sometimes use {asset_id: {...}} rather than a list.
        if rows is None and key == "assets":
            return [dict(row, id=str(name)) for name, row in value.items() if isinstance(row, Mapping)]
    return []


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _case_level(row: Mapping[str, Any]) -> str:
    return str(_first(row, "force_level", "level", "difficulty", default="unknown")).lower()


@dataclass(frozen=True)
class P0Adapter:
    """Normalized, immutable view of old P0 materials and evidence."""

    schema_version: str
    source_files: tuple[dict[str, Any], ...]
    assets: tuple[dict[str, Any], ...]
    source_sha256: str

    @classmethod
    def from_files(
        cls,
        manifest_path: str | Path,
        material_profiles_path: str | Path | None = None,
        *,
        summary_json_path: str | Path | None = None,
        summary_csv_path: str | Path | None = None,
        video_manifest_path: str | Path | None = None,
        state_manifest_path: str | Path | None = None,
        source_root: str | Path | None = None,
    ) -> "P0Adapter":
        manifest_file = Path(manifest_path)
        material_file = Path(material_profiles_path) if material_profiles_path else None
        summary_json_file = Path(summary_json_path) if summary_json_path else None
        summary_csv_file = Path(summary_csv_path) if summary_csv_path else None
        video_file = Path(video_manifest_path) if video_manifest_path else None
        state_file = Path(state_manifest_path) if state_manifest_path else None
        root = Path(source_root).resolve() if source_root else manifest_file.parent.resolve()

        manifest = _read_json(manifest_file) or {}
        profiles = _read_json(material_file) or {}
        summary = _read_json(summary_json_file) or {}
        video_manifest = _read_json(video_file) or {}
        state_manifest = _read_json(state_file) or {}

        manifest_assets = _as_rows(manifest)
        profile_rows = _as_rows(profiles) or _as_rows(profiles, key="profiles")
        profile_assets = {str(_first(row, "id", "asset_id")): row for row in profile_rows}
        summary_assets = {str(_first(row, "asset_id", "id")): row for row in _as_rows(summary)}
        video_rows = _as_rows(video_manifest, key="videos")
        videos_by_asset: dict[str, list[dict[str, Any]]] = {}
        for row in video_rows:
            asset_id = str(_first(row, "asset_id", "id", default="unknown"))
            videos_by_asset.setdefault(asset_id, []).append(dict(row))
        state_rows = _as_rows(state_manifest, key="states")
        states_by_asset: dict[str, list[dict[str, Any]]] = {}
        for row in state_rows:
            asset_id = str(_first(row, "asset_id", "id", default="unknown"))
            states_by_asset.setdefault(asset_id, []).append(dict(row))

        csv_rows = _read_csv(summary_csv_file)
        csv_cases: dict[str, list[dict[str, Any]]] = {}
        for row in csv_rows:
            asset_id = str(_first(row, "asset_id", "id", default="unknown"))
            csv_cases.setdefault(asset_id, []).append(row)

        source_paths = [manifest_file, material_file, summary_json_file, summary_csv_file, video_file, state_file]
        source_files: list[dict[str, Any]] = []
        for path in source_paths:
            if path is not None:
                source_files.append(reference_digest(path, root=root))

        normalized_assets: list[dict[str, Any]] = []
        for index, raw_manifest_asset in enumerate(manifest_assets):
            asset_id = str(_first(raw_manifest_asset, "id", "asset_id", "name", default=f"asset_{index:04d}"))
            profile = profile_assets.get(asset_id, {})
            summary_asset = summary_assets.get(asset_id, {})
            merged = _merge_asset(raw_manifest_asset, profile)
            merged = _sanitize_asset_paths(merged)
            merged["asset_id"] = asset_id
            merged["canonical_id"] = str(
                _first(merged, "canonical_id", default=f"p0_canonical__{asset_id}")
            )
            merged["profile_id"] = str(_first(merged, "profile_id", "material_profile_id", default=asset_id))
            merged["material_profile"] = _normalize_material(merged, profile)
            merged["geometry"] = _normalize_geometry(merged, root)
            merged["cases"] = _normalize_cases(
                asset_id,
                summary_asset,
                csv_cases.get(asset_id, []),
                profile,
                videos_by_asset.get(asset_id, []),
            )
            inline_states = merged.get("states") if isinstance(merged.get("states"), list) else []
            merged["states"] = [dict(row) for row in inline_states if isinstance(row, Mapping)] + [dict(row) for row in states_by_asset.get(asset_id, [])]
            merged["video_refs"] = sorted({str(case["video_ref"]) for case in merged["cases"] if case.get("video_ref")})
            merged["state_refs"] = sorted({str(row.get("state_path") or row.get("path")) for row in merged["states"] if row.get("state_path") or row.get("path")})
            merged["keyframe_refs"] = sorted({str(row.get("keyframe") or row.get("keyframe_path")) for row in merged["states"] if row.get("keyframe") or row.get("keyframe_path")})
            merged["source_refs"] = _source_refs(merged, root)
            merged["source_sha256"] = combined_source_hash(merged["source_refs"] + source_files)
            # These flags prevent downstream code from accidentally claiming
            # that a reused P0 geometry is an independent physical asset.
            merged["asset_kind"] = "p0_reference"
            merged["unique_geometry_asset"] = False
            normalized_assets.append(merged)

        adapter_payload = {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "source_files": source_files,
            "assets": normalized_assets,
        }
        return cls(
            schema_version=ADAPTER_SCHEMA_VERSION,
            source_files=tuple(source_files),
            assets=tuple(normalized_assets),
            source_sha256=sha256_json(adapter_payload),
        )

    @classmethod
    def from_directory(cls, root: str | Path) -> "P0Adapter":
        """Locate the conventional P0 files under ``root``."""

        base = Path(root)
        manifest = base / "assets" / "manifest.json"
        profiles = base / "material_profiles.json"
        summary_json = base / "reports" / "p0_force_sweep_summary.json"
        summary_csv = base / "reports" / "p0_force_sweep_summary.csv"
        return cls.from_files(
            manifest,
            profiles,
            summary_json_path=summary_json,
            summary_csv_path=summary_csv,
            source_root=base,
        )

    def get_asset(self, asset_id: str) -> dict[str, Any]:
        for asset in self.assets:
            if asset["asset_id"] == asset_id or asset["canonical_id"] == asset_id:
                return dict(asset)
        raise KeyError(f"unknown P0 asset: {asset_id}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_sha256": self.source_sha256,
            "source_files": [dict(row) for row in self.source_files],
            "assets": [dict(row) for row in self.assets],
        }


def adapt_p0(*args: Any, **kwargs: Any) -> P0Adapter:
    """Functional alias used by scripts and notebooks."""

    return P0Adapter.from_files(*args, **kwargs)


def _read_csv(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _merge_asset(manifest: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    for key, value in profile.items():
        if key not in result:
            result[key] = value
    if isinstance(profile.get("material"), Mapping):
        result["material"] = dict(profile["material"])
    return result


def _sanitize_asset_paths(asset: Mapping[str, Any]) -> dict[str, Any]:
    """Remove machine-specific path strings while retaining logical refs."""

    result = dict(asset)
    for key in ("preview_source", "collision_source", "video", "state_path", "mesh_path", "preview_path"):
        value = result.get(key)
        if value:
            result[f"{key}_ref"] = normalise_relative_path(str(value))
            result.pop(key, None)
    return result


def _normalize_material(asset: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    raw = asset.get("material") if isinstance(asset.get("material"), Mapping) else {}
    model = _first(raw, "model_id", "model", default=_first(asset, "material_model", "model", default="unknown"))
    params: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in {"model", "model_id", "profile_id"}:
            params[key] = value
    # Some newer profiles put numeric values under ``parameters``.
    nested = raw.get("parameters")
    if isinstance(nested, Mapping):
        params.update(nested)
    if not raw:
        material_keys = {
            "youngs_modulus_pa", "poisson_ratio", "density_kg_m3", "compressive_yield_strain",
            "tensile_yield_strain", "damage_start_compressive_strain", "damage_complete_compressive_strain",
            "residual_stiffness_fraction", "yield_lower", "yield_higher", "relaxation_time_s",
        }
        params.update({key: asset[key] for key in material_keys if key in asset})
        params.update({key: profile[key] for key in material_keys if key in profile})
    return {
        "profile_id": str(_first(asset, "profile_id", "material_profile_id", default=_first(profile, "id", default="unknown"))),
        "model": str(model),
        "parameters": params,
        "source_model_label": str(model),
    }


def _normalize_geometry(asset: Mapping[str, Any], root: Path) -> dict[str, Any]:
    collision = _first(asset, "collision", "collision_path", "mesh", default=None)
    preview = _first(asset, "preview", "preview_path", default=None)
    result: dict[str, Any] = {
        "collision_ref": reference_digest(collision, root=root) if collision else None,
        "preview_ref": reference_digest(preview, root=root) if preview else None,
        "extents_m": asset.get("extents_m") or asset.get("dimensions_m"),
        "watertight": asset.get("watertight"),
        "vertices": asset.get("vertices"),
        "faces": asset.get("faces"),
    }
    return result


def _normalize_cases(
    asset_id: str,
    summary_asset: Mapping[str, Any],
    csv_rows: Iterable[Mapping[str, Any]],
    profile: Mapping[str, Any],
    video_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    raw_cases = summary_asset.get("cases")
    if isinstance(raw_cases, Mapping):
        for level, row in raw_cases.items():
            if isinstance(row, Mapping):
                cases.append(_normalize_case(asset_id, str(level), row))
    elif isinstance(raw_cases, list):
        for row in raw_cases:
            if isinstance(row, Mapping):
                cases.append(_normalize_case(asset_id, _case_level(row), row))
    for row in csv_rows:
        cases.append(_normalize_case(asset_id, _case_level(row), row))
    # Profiles often contain only the force triplet.  Preserve it as planned
    # cases, clearly separated from observed summary rows.
    if not cases:
        forces = profile.get("forces_n_per_finger") or profile.get("force_schedule") or []
        if isinstance(forces, (list, tuple)):
            for level, force in zip(("low", "medium", "high"), forces):
                cases.append(
                    {
                        "asset_id": asset_id,
                        "force_level": level,
                        "force_n_per_finger": _float(force),
                        "observed": False,
                        "metrics": {},
                        "video_ref": None,
                    }
                )
    if video_rows:
        by_level = {_case_level(row): row for row in video_rows}
        for case in cases:
            row = by_level.get(case["force_level"])
            if row:
                video = _first(row, "video", "path", "video_path")
                if video:
                    case["video_ref"] = normalise_relative_path(str(video))
    # Deduplicate CSV + JSON copies, preferring observed JSON metrics.
    unique: dict[tuple[str, float | None], dict[str, Any]] = {}
    for case in cases:
        key = (str(case["force_level"]), case.get("force_n_per_finger"))
        previous = unique.get(key)
        if previous is None or (not previous.get("observed") and case.get("observed")):
            unique[key] = case
    return [unique[key] for key in sorted(unique, key=lambda item: (item[0], item[1] or 0.0))]


def _normalize_case(asset_id: str, level: str, row: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {
        "finite_fraction": ("finite_fraction", "finite", "finite_ratio"),
        "lift_height_m": ("lift_height_m", "final_lift_m"),
        "damage_fraction": ("damage_fraction", "final_damaged_fraction"),
        "failed_fraction": ("failed_fraction", "final_failed_fraction"),
        "integrity": ("integrity", "hold_integrity", "integrity_score"),
        "contact_fraction": ("contact_fraction", "contact"),
        "slip_distance_m": ("slip_distance_m", "slip_m"),
        "max_penetration_m": ("max_penetration_m", "penetration_m"),
    }
    metrics: dict[str, Any] = {}
    for target, keys in aliases.items():
        value = _first(row, *keys)
        if value is not None:
            metrics[target] = _float(value, default=value)
    for key in ("final_damaged_percent", "final_failed_percent", "final_floor_percent", "final_lift_mm", "final_rms_deformation_mm"):
        if key in row:
            metrics[key] = _float(row[key], default=row[key])
    video = _first(row, "video", "video_path", "video_ref")
    return {
        "asset_id": asset_id,
        "force_level": level,
        "force_n_per_finger": _float(_first(row, "force_n_per_finger", "force_n", "force")),
        "observed": True,
        "metrics": metrics,
        "video_ref": normalise_relative_path(str(video)) if video else None,
        "raw_state_fields": {
            str(key): value
            for key, value in row.items()
            if key in {"state", "states", "state_path", "status", "lifted", "particle_count"}
        },
    }


def _source_refs(asset: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    geometry = asset.get("geometry")
    if isinstance(geometry, Mapping):
        for key in ("collision_ref", "preview_ref"):
            value = geometry.get(key)
            if isinstance(value, Mapping):
                refs.append(dict(value))
    for key in ("preview_source", "collision_source", "video", "state_path", "preview_source_ref", "collision_source_ref", "video_ref", "state_path_ref"):
        value = asset.get(key)
        if value:
            refs.append(reference_digest(str(value), root=root))
    for key in ("video_refs", "state_refs", "keyframe_refs"):
        values = asset.get(key)
        if isinstance(values, (list, tuple)):
            refs.extend(reference_digest(str(value), root=root) for value in values if value)
    for case in asset.get("cases", ()):
        if isinstance(case, Mapping) and case.get("video_ref"):
            refs.append(reference_digest(str(case["video_ref"]), root=root))
    return refs
