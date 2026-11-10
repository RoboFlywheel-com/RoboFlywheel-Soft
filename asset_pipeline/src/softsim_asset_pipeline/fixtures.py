# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import binascii
import struct
import zlib
from pathlib import Path
from typing import Any

from .cloth import build_cloth_glb
from .contracts import with_record_hash
from .io import atomic_text_writer, atomic_write_bytes, atomic_write_json, atomic_write_jsonl, content_id, sha256_file


def _png(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(color) * width for _ in range(height))
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(raw, 9)),
            chunk(b"IEND", b""),
        )
    )


def _fixture_recipe() -> dict[str, Any]:
    return {
        "width_m": 1.0,
        "height_m": 0.7,
        "cols": 8,
        "rows": 6,
        "broad_fold_amplitude_m": 0.01,
        "edge_wave_amplitude_m": 0.002,
        "edge_wave_frequency": 2,
        "corner_lift_m": 0.005,
        "uv_scale": 1.0,
    }


def create_asset_fixture(output_root: Path, count: int) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("fixture count must be positive")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"fixture output already exists and is non-empty: {output_root}")
    image_root = output_root / "images"
    mesh_root = output_root / "meshes"
    overlay_root = output_root / "overlay"
    rows: list[dict[str, Any]] = []
    cloth_count = 0
    for ordinal in range(count):
        sample_id = f"fixture_{ordinal:06d}"
        cloth = ordinal % 3 == 0
        cloth_count += int(cloth)
        image_path = image_root / "samples" / sample_id / "input_image.png"
        image = _png(8, 8, ((ordinal * 31) % 255, (ordinal * 67) % 255, (ordinal * 97) % 255))
        atomic_write_bytes(image_path, image)
        row = {
            "sample_id": sample_id,
            "canonical_id": "fixture_cloth" if cloth else "fixture_solid",
            "object_type": "Fixture Cloth Sheet" if cloth else "Fixture Soft Solid",
            "geometry_family": "cloth_sheet" if cloth else "solid",
            "material_family": "textile" if cloth else "soft_food",
            "generation_route": "procedural_cloth_grid" if cloth else "trellis_visual_mesh",
            "input_image_relpath": f"samples/{sample_id}/input_image.png",
            "input_image_sha256": sha256_file(image_path),
            "cloth_geometry": _fixture_recipe() if cloth else None,
        }
        rows.append(row)
        if not cloth:
            mesh_path = mesh_root / "samples" / sample_id / "mesh" / f"{sample_id}.glb"
            atomic_write_bytes(mesh_path, build_cloth_glb(_fixture_recipe(), image))
    manifest = output_root / "primary.jsonl"
    atomic_write_jsonl(manifest, rows)
    cloth_config = {
        "schema_version": 1,
        "default_geometry": _fixture_recipe(),
        "geometry_profiles": [],
        "qa": {"max_faces": 20000, "required_components": 1},
    }
    qa_config = {
        "schema_version": 1,
        "max_faces": 2000000,
        "max_components": 128,
        "families": {"cloth_sheet": {"max_faces": 20000, "max_components": 1}},
    }
    atomic_write_json(output_root / "cloth_config.json", cloth_config)
    atomic_write_json(output_root / "qa_config.json", qa_config)
    summary = {
        "schema_version": 1,
        "synthetic_fixture": True,
        "count": count,
        "dynamic_cloth_count": cloth_count,
        "manifest": str(manifest),
        "image_root": str(image_root),
        "mesh_root": str(mesh_root),
        "overlay_root": str(overlay_root),
        "cloth_config": str(output_root / "cloth_config.json"),
        "qa_config": str(output_root / "qa_config.json"),
    }
    atomic_write_json(output_root / "fixture_summary.json", summary)
    return summary


def create_ledger_fixture(output_root: Path, count: int) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("fixture count must be positive")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"fixture output already exists and is non-empty: {output_root}")
    asset_release = output_root / "asset_release"
    release_id = f"fixture-assets-{count}"
    rows: list[dict[str, Any]] = []
    for ordinal in range(count):
        sample_id = f"fixture_physics_{ordinal:06d}"
        artifacts = [
            {
                "role": "visual_mesh",
                "root_alias": "fixture_mesh",
                "path": f"samples/{sample_id}/mesh/{sample_id}.glb",
                "sha256": f"{ordinal:064x}"[-64:],
                "size_bytes": 1,
                "media_type": "model/gltf-binary",
            },
            {
                "role": "source_image",
                "root_alias": "fixture_images",
                "path": f"samples/{sample_id}/input_image.png",
                "sha256": f"{ordinal + 1:064x}"[-64:],
                "size_bytes": 1,
                "media_type": "image/png",
            },
        ]
        revision = content_id({"sample_id": sample_id, "artifacts": artifacts})
        rows.append(
            with_record_hash(
                {
                    "schema_name": "softsim.asset_ledger",
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "asset_revision_id": revision,
                    "dataset_release_id": release_id,
                    "canonical_id": "fixture_soft_solid",
                    "object_type": "Fixture Soft Solid",
                    "geometry_family": "solid",
                    "material_family": "fixture_material",
                    "generation_route": "synthetic_fixture",
                    "asset_status": "ready_with_warnings",
                    "failure_codes": ["metric_scale_unassigned"],
                    "repairable": False,
                    "metric_scale": {
                        "unit": "normalized_asset_unit",
                        "scale_to_meter": None,
                        "provenance": "synthetic_fixture",
                        "confidence": 0.0,
                    },
                    "artifacts": artifacts,
                    "geometry": {"vertices": 4, "faces": 2, "components": 1},
                    "generation_provenance": {"source_manifest_hash": "synthetic", "producer_version": "1.0.0"},
                    "maintenance": {"cloth_backfill_status": "not_applicable", "static_check_status": "pass"},
                    "physical_bundle_ref": None,
                }
            )
        )
    ledger = asset_release / "asset_ledger.jsonl"
    atomic_write_jsonl(ledger, rows)
    ledger_sha = sha256_file(ledger)
    with atomic_text_writer(asset_release / "asset_ledger.sha256") as handle:
        handle.write(ledger_sha + "\n")
    atomic_write_json(
        asset_release / "release_manifest.json",
        {
            "schema_name": "softsim.asset_release",
            "schema_version": 1,
            "dataset_release_id": release_id,
            "asset_ledger_sha256": ledger_sha,
            "asset_count": count,
            "immutable": True,
            "synthetic_fixture": True,
        },
    )
    catalog = {
        "schema_version": 1,
        "parameters": {
            "density_kg_m3": {"lower": 100.0, "upper": 1000.0, "unit": "kg/m^3"},
            "youngs_modulus_pa": {"lower": 1000.0, "upper": 100000.0, "unit": "Pa"},
        },
        "models": {
            "fixture_elastic_solid": {
                "simulation_domain": "volumetric_soft_body",
                "representations": ["watertight_volume"],
                "solver": "Fixture.Elastic",
                "required_parameters": ["density_kg_m3", "youngs_modulus_pa"],
                "fixed": {},
            }
        },
        "solver_numeric_parameters": [],
    }
    catalog_path = output_root / "fixture_catalog.json"
    atomic_write_json(catalog_path, catalog)
    config = {
        "schema_version": 1,
        "prompt_version": "physical-production-fixture-v1",
        "catalog_json": str(catalog_path),
        "batch_size": 8,
        "max_semantic_attempts": 2,
        "max_transport_retries": 2,
        "retry_backoff_seconds": 0.001,
        "retry_jitter_ratio": 0.0,
        "requests_per_minute": 1000000,
        "tokens_per_minute": 1000000000,
        "estimated_input_tokens_per_asset": 100,
        "estimated_output_tokens_per_asset": 50,
        "estimated_bundle_bytes_per_asset": 4096,
        "mock_transport": {
            "model": "mock-physical-prior-v1",
            "model_revision": "fixture",
            "invalid_first_modulo": 17,
            "latency_seconds": 0.0,
        },
    }
    atomic_write_json(output_root / "physical_config.json", config)
    atomic_write_json(
        output_root / "fallback_policy.json",
        {"schema_version": 1, "unknown_action": "unsupported", "profiles": []},
    )
    summary = {
        "schema_version": 1,
        "synthetic_fixture": True,
        "count": count,
        "asset_release": str(asset_release),
        "asset_ledger": str(ledger),
        "config": str(output_root / "physical_config.json"),
        "fallback_policy": str(output_root / "fallback_policy.json"),
        "publishable": False,
    }
    atomic_write_json(output_root / "fixture_summary.json", summary)
    return summary
