# SPDX-License-Identifier: Apache-2.0
"""Instance manifest construction from the P0 adapter."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adapter import P0Adapter
from .difficulty import DIFFICULTY_STATES, construct_difficulty
from .hashing import combined_source_hash, sha256_json
from .materials import construct_material
from .physical_join import join_physical_bundles
from .scenario import immutable_contract_hashes, make_scenario


BUILDER_SCHEMA_VERSION = "fragile-soft.instance.v1"
DEFAULT_CANONICAL_GROUPS = (
    ("soft_food_form", "tofu", "calibration"),
    ("compressible_fruit", "banana", "calibration"),
    ("gel_block", "avocado", "calibration"),
    ("mashed_food", "play_dough", "calibration"),
    ("sponge_foam", "slow_rise_foam_toy", "calibration"),
    ("soft_fruit", "tomato", "hidden_test"),
    ("thin_filled_shell", "inflatable_bladder", "hidden_test"),
    ("paper_film_shell", "memory_foam_cushion", "hidden_test"),
    ("weak_rubber", "silicone_organ", "hidden_test"),
    ("fragile_composite", "bean_filled_toy", "hidden_test"),
)


def _groups(config: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    raw = (config or {}).get("canonical_groups") or (config or {}).get("qualification_groups")
    if not raw:
        return [
            {"canonical_group_id": group, "source_asset_id": asset, "split": split}
            for group, asset, split in DEFAULT_CANONICAL_GROUPS
        ]
    result: list[dict[str, Any]] = []
    for row in raw:
        if isinstance(row, Mapping):
            result.append(
                {
                    "canonical_group_id": str(row.get("canonical_group_id") or row.get("id")),
                    "source_asset_id": str(row.get("source_asset_id") or row.get("asset_id")),
                    "split": str(row.get("split", "calibration")),
                }
            )
    return result


def _revision_descriptor(asset: Mapping[str, Any], revision_index: int) -> dict[str, Any]:
    geometry = copy.deepcopy(dict(asset.get("geometry") or {}))
    # Revision parameters are metadata for a future independent mesh generator;
    # they do not claim that a copied P0 mesh is a unique object.
    descriptor = {
        "geometry_revision_id": f"{asset['asset_id']}.r{revision_index:03d}",
        "geometry_revision_index": revision_index,
        "geometry_revision_kind": "p0_reference_revision",
        "unique_geometry_asset": False,
        "base_asset_id": asset["asset_id"],
        "base_geometry_hash": sha256_json(geometry),
        "revision_parameters": {
            "scale_xyz": [1.0, 1.0, 1.0],
            "pose_offset_xyz_m": [0.0, 0.0, 0.0],
            "source": "reserved_revision_slot; no mesh duplication in local phase",
        },
    }
    descriptor["geometry_revision_hash"] = sha256_json(descriptor)
    return descriptor


def _base_force_schedule(asset: Mapping[str, Any]) -> dict[str, float]:
    schedule = {str(case.get("force_level")): float(case["force_n_per_finger"]) for case in asset.get("cases", ()) if case.get("force_n_per_finger") is not None}
    if schedule:
        return schedule
    profile = asset.get("material_profile") or {}
    return {level: float(force) for level, force in zip(("low", "medium", "high"), profile.get("forces_n_per_finger", (6.0, 15.0, 30.0)))}


def _source_hash(asset: Mapping[str, Any]) -> str:
    refs = asset.get("source_refs") or []
    if refs:
        return combined_source_hash(refs)
    return sha256_json({"asset_id": asset.get("asset_id"), "canonical_id": asset.get("canonical_id")})


def make_instance(
    asset: Mapping[str, Any],
    *,
    canonical_group_id: str,
    split: str,
    revision_index: int,
    difficulty_id: str,
    scenario: Mapping[str, Any] | None = None,
    strict_material: bool = False,
    difficulty_registry: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Build one auditable instance without running a simulator."""

    scenario_value = make_scenario() if scenario is None else copy.deepcopy(dict(scenario))
    if "contract_hash" not in scenario_value:
        scenario_value = make_scenario(scenario_value)
    contract_hashes = immutable_contract_hashes(scenario_value)
    material_profile = dict(asset.get("material_profile") or {})
    # Adapter shape has ``model`` and nested ``parameters``; constructor accepts
    # both this shape and the original P0 flat ``material`` object.
    material_input = {
        "model": material_profile.get("model") or asset.get("material_model") or "unknown",
        "profile_id": material_profile.get("profile_id") or asset.get("profile_id"),
        **dict(material_profile.get("parameters") or {}),
    }
    requested_model = str(material_input.get("model") or "unknown")
    requested_capability = None
    model_override = None
    canonical = str(asset.get("canonical_id", ""))
    profile_id = str(material_profile.get("profile_id") or asset.get("profile_id") or "")
    # P0's inflatable entry is an effective elastic bulk response, not a
    # pressure-shell solver.  Give it a distinct surrogate ID instead of
    # silently presenting ordinary elasticity as pressure physics.
    if "inflatable" in canonical or profile_id == "P-SHELL-INFLATABLE":
        requested_capability = "pressure_shell"
        model_override = "effective_elastic_bulk_surrogate"
    material = construct_material(
        material_input,
        strict=strict_material,
        model_override=model_override,
        requested_model_id=requested_capability or requested_model,
        capability_scope=requested_capability,
    )
    material_dict = material.as_dict()
    geometry = _revision_descriptor(asset, revision_index)
    difficulty = construct_difficulty(
        material_dict,
        _base_force_schedule(asset),
        difficulty_id=difficulty_id,
        geometry_revision_id=geometry["geometry_revision_id"],
        scenario_hash=contract_hashes["scenario_hash"],
        action_protocol_hash=contract_hashes["action_protocol_hash"],
        evaluator_hash=contract_hashes["evaluator_hash"],
        registry=difficulty_registry,
    )
    provisional_physical_bundle_id = "pb-" + sha256_json(
        {
            "base_asset_id": asset["asset_id"],
            "geometry_revision_hash": geometry["geometry_revision_hash"],
            "material_parameter_hash": material.parameter_hash,
            "difficulty_hash": difficulty["difficulty_hash"],
        }
    )[7:19]
    instance_id = f"fsb-{canonical_group_id}-r{revision_index:03d}-{difficulty_id}"
    status = "candidate" if material.support_status in {"supported", "supported_surrogate"} else ("rejected_invalid_material" if material.support_status == "invalid" else "rejected_unsupported_material")
    failure_codes = [] if status == "candidate" else ["invalid_material_profile" if status == "rejected_invalid_material" else "unsupported_material_model"]
    damage_model_id = "TofuDamage.continuum_damage_stiffness_loss_surrogate" if material.model_id == "tofu_damage" else "none"
    row = {
        "schema_version": BUILDER_SCHEMA_VERSION,
        "benchmark_instance_id": instance_id,
        "benchmark_population": "benchmark instances",
        "unique_geometry_asset": False,
        "canonical_group_id": canonical_group_id,
        "canonical_id": asset["canonical_id"],
        "base_asset_id": asset["asset_id"],
        "geometry_revision_id": geometry["geometry_revision_id"],
        "geometry": geometry,
        "physical_bundle_id": None,
        "provisional_physical_bundle_id": provisional_physical_bundle_id,
        "physical_bundle_status": "unjoined",
        "physical_join_status": "unjoined",
        "physical_release_id": None,
        "sample_id": None,
        "asset_revision_id": None,
        "formal_join_ready": False,
        "material_model_id": material.model_id,
        "damage_model_id": damage_model_id,
        "material": material_dict,
        "difficulty_id": difficulty_id,
        "difficulty": difficulty,
        "split": split,
        "calibration_or_hidden_split": split,
        "calibration_split": split == "calibration",
        "hidden_test_split": split in {"hidden", "hidden_test"},
        "scenario": scenario_value,
        "action_protocol_id": scenario_value["scenario_id"],
        "evaluator_id": scenario_value["evaluator_id"],
        "source": {
            "p0_asset_source_sha256": _source_hash(asset),
            "p0_adapter_schema": "fragile-soft.p0-adapter.v1",
            "source_status": asset.get("source_status", "unknown"),
            "source_refs": copy.deepcopy(asset.get("source_refs", [])),
        },
        "source_hash": _source_hash(asset),
        "status": status,
        "failure_codes": failure_codes,
        "generation_rule": "P0 reference + registered geometry revision slot + one difficulty state; physical join required before formal E3-A",
    }
    row["instance_hash"] = sha256_json(row)
    return row


def build_instances(
    adapter: P0Adapter,
    config: Mapping[str, Any] | None = None,
    *,
    limit: int | None = None,
    difficulty_ids: Sequence[str] | None = None,
    revision_count: int | None = None,
    physical_index: str | Path | Sequence[Mapping[str, Any]] | None = None,
    physical_join_map: str | Path | Sequence[Mapping[str, Any]] | None = None,
    physical_release_id: str | None = None,
    require_physical_join: bool = False,
    join_report_out: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a deterministic matrix (default 10 x 20 x 10 = 2,000 rows)."""

    cfg = config or {}
    groups = _groups(cfg)
    revisions = int(revision_count or cfg.get("geometry_revisions_per_group", 20))
    states = list(difficulty_ids or cfg.get("difficulty_ids") or DIFFICULTY_STATES.keys())
    scenario = make_scenario(cfg.get("scenario_overrides"))
    difficulty_registry = cfg.get("difficulty_states")
    if isinstance(difficulty_registry, Mapping) and isinstance(difficulty_registry.get("states"), Mapping):
        difficulty_registry = difficulty_registry["states"]
    assets = {asset["asset_id"]: asset for asset in adapter.assets}
    rows: list[dict[str, Any]] = []
    for group in groups:
        if group["source_asset_id"] not in assets:
            raise KeyError(f"canonical group references unknown asset: {group['source_asset_id']}")
        asset = assets[group["source_asset_id"]]
        for revision_index in range(revisions):
            for difficulty_id in states:
                rows.append(
                    make_instance(
                        asset,
                        canonical_group_id=group["canonical_group_id"],
                        split=group["split"],
                        revision_index=revision_index,
                        difficulty_id=str(difficulty_id),
                        scenario=scenario,
                        strict_material=False,
                        difficulty_registry=difficulty_registry,
                    )
                )
                if limit is not None and len(rows) >= limit:
                    break
            if limit is not None and len(rows) >= limit:
                break
        if limit is not None and len(rows) >= limit:
            break
    if require_physical_join and physical_index is None:
        raise ValueError("formal benchmark construction requires --physical-index")
    join_report: dict[str, Any] = {
        "schema_version": "fragile-soft.physical-join.v1",
        "physical_release_id": physical_release_id,
        "index_count": 0,
        "instance_count": len(rows),
        "counts": {"joined_verified": 0, "joined_nonverified": 0, "missing": len(rows), "ambiguous": 0},
        "errors": [],
        "all_joined_verified": False,
    }
    if physical_index is not None:
        rows, join_report = join_physical_bundles(
            rows,
            physical_index,
            join_map=physical_join_map,
            physical_release_id=physical_release_id,
            require_all=require_physical_join,
        )
        for row in rows:
            row["physical_join_report_hash"] = sha256_json(join_report)
            row["instance_hash"] = sha256_json({key: value for key, value in row.items() if key != "instance_hash"})
    if join_report_out is not None:
        join_report_out.clear()
        join_report_out.update(join_report)
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
