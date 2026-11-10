# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import math
import struct
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942

COMPONENT_FORMATS = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}

TYPE_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


@dataclass(frozen=True)
class ParsedGlb:
    document: dict[str, Any]
    binary: bytes
    declared_length: int


def parse_glb(path: Path) -> ParsedGlb:
    data = path.read_bytes()
    if len(data) < 12:
        raise ValueError("glb_header_truncated")
    magic, version, declared_length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise ValueError("glb_bad_magic")
    if version != 2:
        raise ValueError(f"glb_unsupported_version:{version}")
    if declared_length != len(data):
        raise ValueError(f"glb_declared_length_mismatch:{declared_length}:{len(data)}")
    offset = 12
    json_chunk: bytes | None = None
    binary_chunks: list[bytes] = []
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("glb_chunk_header_truncated")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        end = offset + chunk_length
        if end > len(data):
            raise ValueError("glb_chunk_truncated")
        payload = data[offset:end]
        offset = end
        if chunk_type == JSON_CHUNK:
            if json_chunk is not None:
                raise ValueError("glb_duplicate_json_chunk")
            json_chunk = payload
        elif chunk_type == BIN_CHUNK:
            binary_chunks.append(payload)
    if json_chunk is None:
        raise ValueError("glb_json_chunk_missing")
    try:
        document = json.loads(json_chunk.rstrip(b" \t\r\n\x00").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"glb_invalid_json:{exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("glb_json_not_object")
    if len(binary_chunks) > 1:
        raise ValueError("glb_multiple_binary_chunks")
    return ParsedGlb(document=document, binary=binary_chunks[0] if binary_chunks else b"", declared_length=declared_length)


def _external_uri_errors(document: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for buffer in document.get("buffers") or []:
        uri = buffer.get("uri")
        if uri and not str(uri).startswith("data:"):
            failures.append("external_buffer_uri")
    for image in document.get("images") or []:
        uri = image.get("uri")
        if uri and not str(uri).startswith("data:"):
            failures.append("external_image_uri")
    return failures


def _accessor_values(parsed: ParsedGlb, accessor_index: int) -> list[tuple[float | int, ...]]:
    document = parsed.document
    accessors = document.get("accessors") or []
    if not 0 <= accessor_index < len(accessors):
        raise ValueError("accessor_index_out_of_range")
    accessor = accessors[accessor_index]
    if accessor.get("sparse"):
        raise ValueError("sparse_accessor_unsupported")
    if "bufferView" not in accessor:
        raise ValueError("accessor_buffer_view_missing")
    component_type = int(accessor.get("componentType", -1))
    if component_type not in COMPONENT_FORMATS:
        raise ValueError(f"accessor_component_type_unsupported:{component_type}")
    type_name = str(accessor.get("type"))
    if type_name not in TYPE_WIDTHS:
        raise ValueError(f"accessor_type_unsupported:{type_name}")
    count = int(accessor.get("count", -1))
    if count < 0:
        raise ValueError("accessor_invalid_count")
    views = document.get("bufferViews") or []
    view_index = int(accessor["bufferView"])
    if not 0 <= view_index < len(views):
        raise ValueError("buffer_view_index_out_of_range")
    view = views[view_index]
    if int(view.get("buffer", 0)) != 0:
        raise ValueError("non_primary_buffer_unsupported")
    fmt, component_size = COMPONENT_FORMATS[component_type]
    width = TYPE_WIDTHS[type_name]
    packed_size = component_size * width
    stride = int(view.get("byteStride", packed_size))
    if stride < packed_size:
        raise ValueError("buffer_view_stride_too_small")
    start = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    view_end = int(view.get("byteOffset", 0)) + int(view.get("byteLength", 0))
    if count and start + (count - 1) * stride + packed_size > min(view_end, len(parsed.binary)):
        raise ValueError("accessor_data_out_of_range")
    unpack = struct.Struct("<" + fmt * width).unpack_from
    return [unpack(parsed.binary, start + index * stride) for index in range(count)]


def _component_face_indices(
    vertex_count: int, faces: Iterable[tuple[int, int, int]]
) -> tuple[list[list[int]], int]:
    parent = list(range(vertex_count))
    rank = [0] * vertex_count

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    used: set[int] = set()
    edge_counts: dict[tuple[int, int], int] = {}
    face_rows = list(faces)
    for face in face_rows:
        a, b, c = face
        for value in face:
            if value < 0 or value >= vertex_count:
                raise ValueError("mesh_index_out_of_range")
            used.add(value)
        union(a, b)
        union(b, c)
        union(c, a)
        for left, right in ((a, b), (b, c), (c, a)):
            edge = (min(left, right), max(left, right))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    component_faces: dict[int, list[int]] = {}
    for face_index, face in enumerate(face_rows):
        root = find(face[0])
        component_faces.setdefault(root, []).append(face_index)
    non_manifold = sum(1 for count in edge_counts.values() if count > 2)
    return sorted(component_faces.values(), key=lambda rows: (-len(rows), rows[0])), non_manifold


def _component_face_counts(
    vertex_count: int, faces: Iterable[tuple[int, int, int]]
) -> tuple[list[int], int]:
    groups, non_manifold = _component_face_indices(vertex_count, faces)
    return [len(group) for group in groups], non_manifold


def _components(vertex_count: int, faces: Iterable[tuple[int, int, int]]) -> tuple[int, int]:
    component_faces, non_manifold = _component_face_counts(vertex_count, faces)
    return len(component_faces), non_manifold


def component_face_counts(path: Path, *, weld_coincident: bool = False) -> list[int]:
    """Return descending triangle counts for each connected component."""
    parsed = parse_glb(path)
    counts: list[int] = []
    for mesh in parsed.document.get("meshes") or []:
        for primitive in mesh.get("primitives") or []:
            attributes = primitive.get("attributes") or {}
            if "POSITION" not in attributes:
                continue
            positions = _accessor_values(parsed, int(attributes["POSITION"]))
            indices_raw = (
                _accessor_values(parsed, int(primitive["indices"]))
                if "indices" in primitive
                else [(index,) for index in range(len(positions))]
            )
            indices = [int(row[0]) for row in indices_raw]
            if int(primitive.get("mode", 4)) != 4 or not indices or len(indices) % 3:
                continue
            faces = [tuple(indices[index : index + 3]) for index in range(0, len(indices), 3)]
            vertex_count = len(positions)
            if weld_coincident:
                welded_ids: dict[tuple[float | int, ...], int] = {}
                vertex_map = [welded_ids.setdefault(position, len(welded_ids)) for position in positions]
                faces = [(vertex_map[a], vertex_map[b], vertex_map[c]) for a, b, c in faces]
                vertex_count = len(welded_ids)
            primitive_counts, _ = _component_face_counts(vertex_count, faces)
            counts.extend(primitive_counts)
    return sorted(counts, reverse=True)


def _serialize_glb(document: dict[str, Any], binary: bytes) -> bytes:
    json_payload = json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    json_payload += b" " * ((-len(json_payload)) % 4)
    binary_payload = binary + b"\x00" * ((-len(binary)) % 4)
    total_length = 12 + 8 + len(json_payload) + 8 + len(binary_payload)
    return b"".join(
        (
            struct.pack("<III", GLB_MAGIC, 2, total_length),
            struct.pack("<II", len(json_payload), JSON_CHUNK),
            json_payload,
            struct.pack("<II", len(binary_payload), BIN_CHUNK),
            binary_payload,
        )
    )


def prune_glb_components(
    source: Path,
    destination: Path,
    *,
    max_components: int,
    min_face_retention: float,
) -> dict[str, Any]:
    """Drop the smallest welded face components while preserving all vertex attributes."""
    if max_components <= 0:
        raise ValueError("max_components_must_be_positive")
    if not 0.0 <= min_face_retention <= 1.0:
        raise ValueError("min_face_retention_out_of_range")
    parsed = parse_glb(source)
    document = deepcopy(parsed.document)
    buffers = document.get("buffers") or []
    if len(buffers) != 1 or buffers[0].get("uri"):
        raise ValueError("component_prune_requires_one_embedded_buffer")

    plans: list[dict[str, Any]] = []
    ranked_components: list[tuple[int, int, int]] = []
    total_faces = 0
    for mesh_index, mesh in enumerate(document.get("meshes") or []):
        for primitive_index, primitive in enumerate(mesh.get("primitives") or []):
            attributes = primitive.get("attributes") or {}
            if "POSITION" not in attributes:
                continue
            if int(primitive.get("mode", 4)) != 4:
                raise ValueError("component_prune_requires_triangles")
            positions = _accessor_values(parsed, int(attributes["POSITION"]))
            indices_raw = (
                _accessor_values(parsed, int(primitive["indices"]))
                if "indices" in primitive
                else [(index,) for index in range(len(positions))]
            )
            indices = [int(row[0]) for row in indices_raw]
            if not indices or len(indices) % 3:
                raise ValueError("component_prune_invalid_indices")
            faces = [tuple(indices[index : index + 3]) for index in range(0, len(indices), 3)]
            welded_ids: dict[tuple[float | int, ...], int] = {}
            vertex_map = [welded_ids.setdefault(position, len(welded_ids)) for position in positions]
            welded_faces = [
                (vertex_map[face[0]], vertex_map[face[1]], vertex_map[face[2]]) for face in faces
            ]
            groups, _ = _component_face_indices(len(welded_ids), welded_faces)
            plan_index = len(plans)
            for group_index, group in enumerate(groups):
                ranked_components.append((len(group), plan_index, group_index))
            plans.append(
                {
                    "mesh_index": mesh_index,
                    "primitive_index": primitive_index,
                    "faces": faces,
                    "groups": groups,
                }
            )
            total_faces += len(faces)

    original_components = len(ranked_components)
    if not ranked_components or total_faces <= 0:
        raise ValueError("component_prune_empty_mesh")
    if original_components <= max_components:
        return {
            "status": "not_needed",
            "original_components": original_components,
            "retained_components": original_components,
            "original_faces": total_faces,
            "retained_faces": total_faces,
            "face_retention": 1.0,
        }

    ranked_components.sort(key=lambda row: (-row[0], row[1], row[2]))
    selected = {(plan_index, group_index) for _, plan_index, group_index in ranked_components[:max_components]}
    retained_faces = sum(face_count for face_count, plan_index, group_index in ranked_components if (plan_index, group_index) in selected)
    retention = retained_faces / total_faces
    result = {
        "status": "retention_below_threshold" if retention < min_face_retention else "repaired",
        "original_components": original_components,
        "retained_components": max_components,
        "dropped_components": original_components - max_components,
        "original_faces": total_faces,
        "retained_faces": retained_faces,
        "dropped_faces": total_faces - retained_faces,
        "face_retention": retention,
        "min_face_retention": min_face_retention,
    }
    if retention < min_face_retention:
        return result

    binary = bytearray(parsed.binary)
    views = document.setdefault("bufferViews", [])
    accessors = document.setdefault("accessors", [])
    for plan_index, plan in enumerate(plans):
        keep_face_indices: set[int] = set()
        for group_index, group in enumerate(plan["groups"]):
            if (plan_index, group_index) in selected:
                keep_face_indices.update(group)
        kept_indices = [
            vertex
            for face_index, face in enumerate(plan["faces"])
            if face_index in keep_face_indices
            for vertex in face
        ]
        primitive = document["meshes"][plan["mesh_index"]]["primitives"][plan["primitive_index"]]
        if not kept_indices:
            primitive["_softsim_remove"] = True
            continue
        component_type = 5123 if max(kept_indices) <= 65535 else 5125
        index_format = "<H" if component_type == 5123 else "<I"
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        payload = b"".join(struct.pack(index_format, value) for value in kept_indices)
        binary.extend(payload)
        view_index = len(views)
        views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(payload),
                "target": 34963,
            }
        )
        accessor_index = len(accessors)
        accessors.append(
            {
                "bufferView": view_index,
                "componentType": component_type,
                "count": len(kept_indices),
                "type": "SCALAR",
                "min": [min(kept_indices)],
                "max": [max(kept_indices)],
            }
        )
        primitive["indices"] = accessor_index
    for mesh in document.get("meshes") or []:
        mesh["primitives"] = [
            primitive
            for primitive in mesh.get("primitives") or []
            if not primitive.pop("_softsim_remove", False)
        ]
    buffers[0]["byteLength"] = len(binary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_serialize_glb(document, bytes(binary)))
    return result


def audit_glb(path: Path) -> dict[str, Any]:
    parsed = parse_glb(path)
    document = parsed.document
    failures = _external_uri_errors(document)
    warnings: list[str] = []
    meshes = document.get("meshes") or []
    if not meshes:
        failures.append("mesh_missing")
    all_positions: list[tuple[float, float, float]] = []
    total_faces = 0
    total_vertices = 0
    total_components = 0
    non_manifold_edges = 0
    has_uv = False
    has_material = False
    for mesh in meshes:
        for primitive in mesh.get("primitives") or []:
            attributes = primitive.get("attributes") or {}
            if "POSITION" not in attributes:
                failures.append("position_accessor_missing")
                continue
            try:
                positions_raw = _accessor_values(parsed, int(attributes["POSITION"]))
                positions = [(float(row[0]), float(row[1]), float(row[2])) for row in positions_raw]
                if any(not all(math.isfinite(value) for value in row) for row in positions):
                    failures.append("non_finite_position")
                all_positions.extend(positions)
                total_vertices += len(positions)
                indices_raw = (
                    _accessor_values(parsed, int(primitive["indices"]))
                    if "indices" in primitive
                    else [(index,) for index in range(len(positions))]
                )
                indices = [int(row[0]) for row in indices_raw]
                mode = int(primitive.get("mode", 4))
                if mode != 4:
                    failures.append(f"non_triangle_primitive_mode:{mode}")
                    continue
                if not indices or len(indices) % 3:
                    failures.append("triangle_indices_invalid")
                    continue
                faces = [tuple(indices[index : index + 3]) for index in range(0, len(indices), 3)]
                # glTF commonly duplicates vertices at UV/material seams. Those
                # vertices are geometrically connected and must not inflate the
                # component count merely because their indices differ.
                welded_ids: dict[tuple[float, float, float], int] = {}
                vertex_map: list[int] = []
                for position in positions:
                    vertex_map.append(welded_ids.setdefault(position, len(welded_ids)))
                welded_faces = [
                    (vertex_map[face[0]], vertex_map[face[1]], vertex_map[face[2]]) for face in faces
                ]
                components, primitive_non_manifold = _components(len(welded_ids), welded_faces)
                total_components += components
                non_manifold_edges += primitive_non_manifold
                total_faces += len(faces)
            except (KeyError, TypeError, ValueError, struct.error) as exc:
                failures.append(str(exc))
            has_uv = has_uv or "TEXCOORD_0" in attributes
            has_material = has_material or "material" in primitive
    if total_vertices <= 0:
        failures.append("empty_vertices")
    if total_faces <= 0:
        failures.append("empty_faces")
    if not has_uv:
        failures.append("uv_missing")
    if not has_material:
        failures.append("material_assignment_missing")
    images = document.get("images") or []
    textures = document.get("textures") or []
    embedded_images = sum(
        1 for image in images if "bufferView" in image or str(image.get("uri", "")).startswith("data:")
    )
    if not images or not textures or embedded_images != len(images):
        failures.append("embedded_texture_missing")
    if all_positions:
        minimum = [min(row[axis] for row in all_positions) for axis in range(3)]
        maximum = [max(row[axis] for row in all_positions) for axis in range(3)]
        extent = [maximum[axis] - minimum[axis] for axis in range(3)]
    else:
        minimum = maximum = extent = [0.0, 0.0, 0.0]
    if max(extent) <= 0:
        failures.append("zero_extent")
    extensions = set(document.get("extensionsRequired") or [])
    if extensions:
        warnings.append("extensions_required:" + ",".join(sorted(extensions)))
    return {
        "status": "pass" if not failures else "fail",
        "failure_codes": sorted(set(failures)),
        "warnings": sorted(set(warnings)),
        "geometry": {
            "representation": "surface_mesh",
            "vertices": total_vertices,
            "faces": total_faces,
            "components": total_components,
            "non_manifold_edges": non_manifold_edges,
            "bounds": [minimum, maximum],
            "extent": extent,
        },
        "materials": {
            "has_uv": has_uv,
            "has_material": has_material,
            "embedded_images": embedded_images,
            "image_count": len(images),
        },
        "declared_length": parsed.declared_length,
    }
