# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .glb import audit_glb
from .io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_bytes,
    content_id,
    iter_jsonl,
    read_json,
    safe_join,
    sha256_file,
    validate_relative_path,
)


def is_cloth(row: dict[str, Any]) -> bool:
    return row.get("geometry_family") == "cloth_sheet" or row.get("generation_route") == "procedural_cloth_grid"


def stable_shard(sample_id: str, num_shards: int) -> int:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    return int(content_id(sample_id).split(":", 1)[1][:16], 16) % num_shards


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("frozen_albedo_not_png")
    width, height = struct.unpack(">II", data[16:24])
    if width <= 0 or height <= 0:
        raise ValueError("frozen_albedo_invalid_dimensions")
    return width, height


def _align(buffer: bytearray) -> None:
    while len(buffer) % 4:
        buffer.append(0)


def _append_view(buffer: bytearray, payload: bytes) -> tuple[int, int]:
    _align(buffer)
    offset = len(buffer)
    buffer.extend(payload)
    return offset, len(payload)


def _grid(recipe: dict[str, Any]) -> tuple[list[tuple[float, float, float]], list[tuple[float, float]], list[int]]:
    width = float(recipe["width_m"])
    height = float(recipe["height_m"])
    cols = int(recipe["cols"])
    rows = int(recipe["rows"])
    amplitude = float(recipe.get("broad_fold_amplitude_m", 0.0))
    edge_amplitude = float(recipe.get("edge_wave_amplitude_m", 0.0))
    edge_frequency = int(recipe.get("edge_wave_frequency", 1))
    corner_lift = float(recipe.get("corner_lift_m", 0.0))
    uv_scale = float(recipe.get("uv_scale", 1.0))
    if not (0.01 <= width <= 20.0 and 0.01 <= height <= 20.0):
        raise ValueError("cloth_dimensions_out_of_range")
    if not (1 <= cols <= 512 and 1 <= rows <= 512):
        raise ValueError("cloth_resolution_out_of_range")
    safe_amplitude = min(width, height) * 0.25
    if max(abs(amplitude), abs(edge_amplitude), abs(corner_lift)) > safe_amplitude:
        raise ValueError("cloth_deformation_amplitude_unsafe")
    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    for row_index in range(rows + 1):
        v = row_index / rows
        y = (v - 0.5) * height
        for col_index in range(cols + 1):
            u = col_index / cols
            x = (u - 0.5) * width
            broad = amplitude * math.sin(math.pi * u) * math.sin(math.pi * v)
            edge_distance = min(u, v, 1.0 - u, 1.0 - v)
            edge_weight = max(0.0, 1.0 - edge_distance * 8.0)
            wave = edge_amplitude * edge_weight * math.sin((u + v) * math.tau * max(1, edge_frequency))
            corner_distance = min(
                math.hypot(u, v),
                math.hypot(1.0 - u, v),
                math.hypot(u, 1.0 - v),
                math.hypot(1.0 - u, 1.0 - v),
            )
            z = broad + wave + corner_lift * max(0.0, 1.0 - corner_distance * 5.0)
            vertices.append((x, y, z))
            uvs.append((u * uv_scale, v * uv_scale))
    indices: list[int] = []
    stride = cols + 1
    for row_index in range(rows):
        for col_index in range(cols):
            a = row_index * stride + col_index
            b = a + 1
            c = a + stride
            d = c + 1
            indices.extend((a, b, d, a, d, c))
    return vertices, uvs, indices


def build_cloth_glb(recipe: dict[str, Any], albedo: bytes) -> bytes:
    width, height = _png_dimensions(albedo)
    if width != height:
        raise ValueError("frozen_albedo_must_be_square")
    vertices, uvs, indices = _grid(recipe)
    position_bytes = b"".join(struct.pack("<fff", *value) for value in vertices)
    uv_bytes = b"".join(struct.pack("<ff", *value) for value in uvs)
    index_component = 5123 if len(vertices) <= 65535 else 5125
    index_format = "<H" if index_component == 5123 else "<I"
    index_bytes = b"".join(struct.pack(index_format, value) for value in indices)
    binary = bytearray()
    position_offset, position_length = _append_view(binary, position_bytes)
    uv_offset, uv_length = _append_view(binary, uv_bytes)
    index_offset, index_length = _append_view(binary, index_bytes)
    image_offset, image_length = _append_view(binary, albedo)
    minimum = [min(row[axis] for row in vertices) for axis in range(3)]
    maximum = [max(row[axis] for row in vertices) for axis in range(3)]
    document = {
        "asset": {"version": "2.0", "generator": "softsim-asset-pipeline/1.0.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "procedural_cloth_grid"}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                        "indices": 2,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "materials": [
            {
                "name": "frozen_albedo_cloth",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.9,
                },
            }
        ],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "images": [{"bufferView": 3, "mimeType": "image/png", "name": "frozen_tileable_albedo"}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": position_offset, "byteLength": position_length, "target": 34962},
            {"buffer": 0, "byteOffset": uv_offset, "byteLength": uv_length, "target": 34962},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": index_length, "target": 34963},
            {"buffer": 0, "byteOffset": image_offset, "byteLength": image_length},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": len(vertices),
                "type": "VEC3",
                "min": minimum,
                "max": maximum,
            },
            {"bufferView": 1, "componentType": 5126, "count": len(uvs), "type": "VEC2"},
            {
                "bufferView": 2,
                "componentType": index_component,
                "count": len(indices),
                "type": "SCALAR",
                "min": [0],
                "max": [max(indices)],
            },
        ],
    }
    json_payload = canonical_json_bytes(document)
    json_payload += b" " * ((-len(json_payload)) % 4)
    binary_payload = bytes(binary) + b"\x00" * ((-len(binary)) % 4)
    total_length = 12 + 8 + len(json_payload) + 8 + len(binary_payload)
    return b"".join(
        (
            struct.pack("<III", 0x46546C67, 2, total_length),
            struct.pack("<II", len(json_payload), 0x4E4F534A),
            json_payload,
            struct.pack("<II", len(binary_payload), 0x004E4942),
            binary_payload,
        )
    )


def resolve_recipe(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    recipe = dict(config["default_geometry"])
    object_type = str(row.get("object_type") or "").lower()
    for profile in config.get("geometry_profiles") or []:
        tokens = [str(token).lower() for token in profile.get("object_type_contains") or []]
        if tokens and any(token in object_type for token in tokens):
            recipe.update(profile.get("geometry") or {})
            break
    explicit = row.get("cloth_geometry")
    if explicit is not None:
        if not isinstance(explicit, dict):
            raise ValueError("cloth_geometry_must_be_object")
        recipe.update(explicit)
    required = {"width_m", "height_m", "cols", "rows"}
    missing = sorted(required - recipe.keys())
    if missing:
        raise ValueError("cloth_recipe_missing:" + ",".join(missing))
    return recipe


def _image_relative(row: dict[str, Any]) -> str:
    value = row.get("input_image_relpath") or row.get("input_image")
    if not value:
        raise ValueError("frozen_albedo_ref_missing")
    return validate_relative_path(str(value))


def _verified_success(marker_path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not marker_path.is_file():
        return None
    marker = read_json(marker_path)
    if marker.get("task_fingerprint") != fingerprint:
        return None
    for artifact in marker.get("artifacts") or []:
        path = marker_path.parent / str(artifact["relative_to_revision"])
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            return None
    return marker


def _generate_one(
    row: dict[str, Any],
    image_root: str,
    overlay_root: str,
    config: dict[str, Any],
    source_manifest_hash: str,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    image_path = safe_join(Path(image_root), _image_relative(row), require_exists=True)
    image_sha = sha256_file(image_path)
    expected_image_sha = row.get("input_image_sha256")
    if expected_image_sha and expected_image_sha != image_sha:
        raise ValueError("frozen_albedo_hash_mismatch")
    recipe = resolve_recipe(row, config)
    config_hash = content_id(config)
    fingerprint_payload = {
        "sample_id": sample_id,
        "source_manifest_hash": source_manifest_hash,
        "input_image_sha256": image_sha,
        "recipe": recipe,
        "config_hash": config_hash,
        "producer_version": "1.0.0",
    }
    fingerprint = content_id(fingerprint_payload)
    revision_name = fingerprint.split(":", 1)[1]
    revision_dir = Path(overlay_root) / "samples" / sample_id / "revisions" / revision_name
    marker_path = revision_dir / "SUCCESS.json"
    existing = _verified_success(marker_path, fingerprint)
    if existing is not None:
        return {"sample_id": sample_id, "status": "skipped", **existing}

    attempts_dir = Path(overlay_root) / "attempts" / sample_id
    attempts_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        attempts_dir / f"{revision_name}.json",
        {"status": "started", "task_fingerprint": fingerprint, "inputs": fingerprint_payload},
    )
    staging_parent = Path(overlay_root) / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{sample_id}.{revision_name[:12]}.", dir=staging_parent))
    try:
        mesh_relative = f"mesh/{sample_id}.glb"
        mesh_path = temporary / mesh_relative
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(mesh_path, build_cloth_glb(recipe, image_path.read_bytes()))
        audit = audit_glb(mesh_path)
        qa = config.get("qa") or {}
        if audit["geometry"]["faces"] > int(qa.get("max_faces", 20000)):
            audit["failure_codes"].append("face_budget_exceeded")
        if audit["geometry"]["components"] != int(qa.get("required_components", 1)):
            audit["failure_codes"].append("component_count_invalid")
        if audit["geometry"]["non_manifold_edges"] != 0:
            audit["failure_codes"].append("non_manifold_edges")
        audit["failure_codes"] = sorted(set(audit["failure_codes"]))
        audit["status"] = "pass" if not audit["failure_codes"] else "fail"
        if audit["status"] != "pass":
            raise ValueError("cloth_qa_failed:" + ",".join(audit["failure_codes"]))
        atomic_write_json(temporary / "qa.json", audit)
        artifacts = [
            {
                "role": "visual_mesh",
                "relative_to_revision": mesh_relative,
                "sha256": sha256_file(mesh_path),
                "size_bytes": mesh_path.stat().st_size,
                "media_type": "model/gltf-binary",
            },
            {
                "role": "cloth_qa",
                "relative_to_revision": "qa.json",
                "sha256": sha256_file(temporary / "qa.json"),
                "size_bytes": (temporary / "qa.json").stat().st_size,
                "media_type": "application/json",
            },
        ]
        marker = {
            "schema_name": "softsim.cloth_backfill_success",
            "schema_version": 1,
            "sample_id": sample_id,
            "task_fingerprint": fingerprint,
            "revision": revision_name,
            "recipe": recipe,
            "input_image": {"root_alias": "image_bank", "path": _image_relative(row), "sha256": image_sha},
            "artifacts": artifacts,
            "geometry": audit["geometry"],
            "producer_version": "1.0.0",
        }
        atomic_write_json(temporary / "SUCCESS.json", marker)
        revision_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(temporary, revision_dir)
        except OSError:
            existing = _verified_success(marker_path, fingerprint)
            if existing is None:
                raise
            shutil.rmtree(temporary, ignore_errors=True)
            marker = existing
        atomic_write_json(
            attempts_dir / f"{revision_name}.json",
            {"status": "committed", "task_fingerprint": fingerprint, "revision": revision_name},
        )
        return {"sample_id": sample_id, "status": "generated", **marker}
    except BaseException as exc:
        atomic_write_json(
            attempts_dir / f"{revision_name}.json",
            {"status": "failed", "task_fingerprint": fingerprint, "error": f"{type(exc).__name__}:{exc}"},
        )
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def run_cloth_backfill(
    *,
    manifest: Path,
    image_root: Path,
    overlay_root: Path,
    config_path: Path,
    result_path: Path,
    workers: int,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root is not mounted: {image_root}")
    if image_root.resolve() == overlay_root.resolve():
        raise ValueError("overlay_root_must_differ_from_image_root")
    config = read_json(config_path)
    manifest_hash = sha256_file(manifest)
    targets = [
        row
        for row in iter_jsonl(manifest)
        if is_cloth(row) and stable_shard(str(row["sample_id"]), num_shards) == shard_index
    ]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _generate_one,
                row,
                str(image_root),
                str(overlay_root),
                config,
                manifest_hash,
            ): str(row["sample_id"])
            for row in targets
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                results.append(future.result())
            except BaseException as exc:
                results.append(
                    {"sample_id": sample_id, "status": "failed", "error": f"{type(exc).__name__}:{exc}"}
                )
    results.sort(key=lambda row: row["sample_id"])
    atomic_write_jsonl(result_path, results)
    summary = {
        "schema_version": 1,
        "source_manifest_sha256": manifest_hash,
        "dynamic_target_count": sum(1 for row in iter_jsonl(manifest) if is_cloth(row)),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "shard_target_count": len(targets),
        "generated": sum(row["status"] == "generated" for row in results),
        "skipped": sum(row["status"] == "skipped" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "result_path": str(result_path),
    }
    atomic_write_json(result_path.with_suffix(".summary.json"), summary)
    return summary
