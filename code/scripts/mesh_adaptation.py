#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Production-grade generated-mesh -> Genesis proxy adaptation ("v3" / v12.5-candidate).

This module is the systematic mesh-adaptation layer for the SoftSim physical
agent.  It is imported lazily by ``genesis_behavior_runner.py`` and only when
the probe/simulator config carries ``mesh_adaptation: "v3"``; the default
(v12.2) and ``v2`` (v12.4-candidate) runner paths do not import it at all.

What v3 adds over v2 (all mechanisms traceable to the fast-visual-v11
reference pipeline, ``reference/genesis_mesh_adaptation.md``, plus the
v2 rebuild failure attribution):

* Source-mesh hygiene for everybody: float32-aware vertex welding, degenerate
  / duplicate face removal, fragmentation measurement (generated GLBs arrive
  as thousands of disjoint triangle patches; exact welding does NOT fuse
  them).
* Cloth route no longer trusts raw non-watertight sources
  (v2 ``source_boundary_surface``): fragmented shells are voxel-sealed
  (voxelize -> 3x3x3 binary closing -> fill -> marching cubes) into a
  watertight shell, then opened with the bounded semantic-opening surgery
  (v11 ``fast_visual.py:2990-3078``).  Projected mid-surface remains the
  fallback for garment/shell/cloth_sheet.
* MPM route keeps the v11 filled_volume policy verbatim, including
  ``preserve_watertight_boundary`` for watertight bag sources.
* Every adaptation emits an auditable proxy quality report (watertight,
  component count, boundary edges, area/volume, triangle aspect ratio,
  thinness) and a verdict: ``ok`` / ``degraded`` / ``unfit``.  ``unfit``
  proxies are gated to ``model_limit`` by the runner instead of being
  disguised as physics failures.
* Scale-before-rotate ordering is preserved everywhere (physical size belongs
  to the asset, not to the world-space drop pose).

The module never touches solver numerics, materials, or contract thresholds.
"""

from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Any

VOXEL_RESOLUTION = 40  # v11 proxy.voxel_resolution (filled_volume)
CLOTH_VOXEL_SEAL_RESOLUTION = 48  # v11 visual-derived watertight reference resolution
SURFACE_FACE_BUDGET = 12000  # v11 proxy.maximum_surface_faces
FLAT_SHEET_LONGEST_SEGMENTS = 64  # v11 proxy.sheet_longest_segments
PROJECTED_LONGEST_SEGMENTS = 48
MIN_CLOTH_PROXY_FACES = 32  # v11 projection occupancy minimum
MAX_CLOTH_PROXY_COMPONENTS = 8
WELD_RELATIVE_TOLERANCE = 1.0e-5  # float32 GLB transform noise is ~1e-7 of extent


# --------------------------------------------------------------------------- #
# Loading / hygiene / quality metrics
# --------------------------------------------------------------------------- #


def load_source_mesh(source_path: Path, np: Any):
    """Load every triangle geometry in the file and concatenate."""
    import trimesh

    loaded = trimesh.load(str(source_path), force="scene", process=False)
    geometries = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
    if not geometries:
        raise RuntimeError(f"solver mesh has no faces: {source_path}")
    return trimesh.util.concatenate(geometries)


def _weld_digits(mesh: Any, np: Any) -> int:
    extent = float(max(mesh.extents))
    tolerance = max(WELD_RELATIVE_TOLERANCE * extent, 1.0e-12)
    return int(min(8, max(3, round(-math.log10(tolerance)))))


def weld_and_clean(mesh: Any, np: Any) -> tuple[Any, dict[str, Any]]:
    """Weld float32-noise duplicate vertices and drop degenerate/duplicate faces."""
    import trimesh

    cleaned = mesh.copy()
    digits = _weld_digits(cleaned, np)
    cleaned.merge_vertices(digits_vertex=digits)
    mask = np.asarray(cleaned.area_faces, dtype=float) > 0.0
    degenerate = int((~mask).sum())
    if degenerate:
        cleaned.update_faces(mask)
    if hasattr(cleaned, "nondegenerate_faces"):
        try:
            cleaned.update_faces(cleaned.nondegenerate_faces())
        except Exception:
            pass
    try:
        cleaned.update_faces(cleaned.unique_faces())
    except Exception:
        pass
    cleaned.remove_unreferenced_vertices()
    return cleaned, {"weld_digits_vertex": int(digits), "degenerate_faces_removed": degenerate}


def face_component_labels(mesh: Any, np: Any) -> Any:
    """Per-face connected-component labels via union-find on the face-adjacency
    graph.  Pure Python/numpy (no scipy/networkx graph engine) so the module
    works in the minimal CLI environment as well as the Genesis venv."""
    face_count = len(mesh.faces)
    parent = list(range(face_count))
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in adjacency:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    labels = np.empty(face_count, dtype=np.int64)
    for face in range(face_count):
        labels[face] = find(face)
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int64)


def mesh_quality(mesh: Any, np: Any) -> dict[str, Any]:
    """Scale-aware quality metrics; all counts computed on the mesh as given."""
    faces = len(mesh.faces)
    out: dict[str, Any] = {
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(faces),
        "extents": [float(x) for x in np.asarray(mesh.extents, dtype=float)],
    }
    if faces == 0:
        return out
    labels = face_component_labels(mesh, np)
    out["component_count"] = int(labels.max()) + 1 if labels.size else 0
    areas = np.bincount(labels, weights=np.asarray(mesh.area_faces, dtype=float))
    areas = np.sort(areas)[::-1]
    total_area = float(areas.sum())
    out["surface_area"] = total_area
    out["component_area_top1_fraction"] = float(areas[0] / total_area) if total_area > 0 else None
    out["components_above_1pct_area"] = int((areas > 0.01 * total_area).sum()) if total_area > 0 else 0
    out["watertight"] = bool(mesh.is_watertight)
    out["winding_consistent"] = bool(mesh.is_winding_consistent)
    if mesh.is_volume:
        out["volume"] = float(mesh.volume)
    edge_counts = collections.Counter(map(tuple, mesh.edges_sorted))
    out["boundary_edge_count"] = int(sum(1 for c in edge_counts.values() if c == 1))
    out["nonmanifold_edge_count"] = int(sum(1 for c in edge_counts.values() if c > 2))
    vertices = np.asarray(mesh.vertices, dtype=float)[np.asarray(mesh.faces, dtype=np.int64)]
    e0 = np.linalg.norm(vertices[:, 1] - vertices[:, 2], axis=1)
    e1 = np.linalg.norm(vertices[:, 0] - vertices[:, 2], axis=1)
    e2 = np.linalg.norm(vertices[:, 0] - vertices[:, 1], axis=1)
    longest = np.maximum(np.maximum(e0, e1), e2)
    area = np.maximum(np.asarray(mesh.area_faces, dtype=float), 1.0e-30)
    aspect = longest * longest / (2.0 * area * math.sqrt(3.0) / 3.0)  # equilateral == 1
    out["triangle_aspect_p95"] = float(np.quantile(aspect, 0.95))
    out["triangle_aspect_gt20_fraction"] = float((aspect > 20.0).mean())
    extents = np.asarray(mesh.extents, dtype=float)
    out["thinness_ratio"] = float(extents.min() / max(extents.max(), 1.0e-12))
    return out


# --------------------------------------------------------------------------- #
# Shared building blocks
# --------------------------------------------------------------------------- #


def checkerboard_grid_sheet(bounds_min: Any, bounds_max: Any, counts: list[int], axes: list[int], np: Any):
    """Regular grid mid-surface with checkerboard diagonal splits; port of
    v11 ``batch_pipeline.py:749-800`` (_write_flat_sheet_proxy)."""
    import trimesh

    center = 0.5 * (bounds_min + bounds_max)
    coordinates = [
        np.linspace(bounds_min[axis], bounds_max[axis], counts[index] + 1)
        for index, axis in enumerate(axes)
    ]
    vertices = []
    for second in coordinates[1]:
        for first in coordinates[0]:
            point = center.copy()
            point[axes[0]] = first
            point[axes[1]] = second
            vertices.append(point.copy())
    row = counts[0] + 1
    faces = []
    for y in range(counts[1]):
        for x in range(counts[0]):
            lower_left = y * row + x
            lower_right = lower_left + 1
            upper_left = lower_left + row
            upper_right = upper_left + 1
            if (x + y) % 2:
                faces.extend(((lower_left, lower_right, upper_left), (lower_right, upper_right, upper_left)))
            else:
                faces.extend(((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)))
    return trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int64), process=False)




def _try_fix_normals(mesh: Any) -> None:
    """Best-effort winding fix; trimesh.fix_normals needs a scipy/networkx
    graph engine which the minimal CLI environment lacks.  Solver-side use
    (PBD cloth / MPM sampling) does not depend on winding."""
    try:
        mesh.fix_normals()
    except Exception:
        pass


def flat_sheet_mid_surface(mesh: Any, np: Any, longest_segments: int = FLAT_SHEET_LONGEST_SEGMENTS):
    """cloth_sheet: thinnest axis is the sheet normal; replace source
    undulations with a clean planar grid over the silhouette bounding box."""
    bounds = np.asarray(mesh.bounds, dtype=float)
    extents = bounds[1] - bounds[0]
    thin_axis = int(np.argmin(extents))
    plane_axes = [axis for axis in range(3) if axis != thin_axis]
    longest_axis = plane_axes[int(extents[plane_axes[1]] > extents[plane_axes[0]])]
    other_axis = plane_axes[0] if plane_axes[1] == longest_axis else plane_axes[1]
    counts_map = {
        longest_axis: int(longest_segments),
        other_axis: max(12, int(round(longest_segments * extents[other_axis] / max(extents[longest_axis], 1.0e-12)))),
    }
    counts = [counts_map[axis] for axis in plane_axes]
    sheet = checkerboard_grid_sheet(bounds[0], bounds[1], counts, plane_axes, np)
    return sheet, {
        "method": "flat_mid_surface",
        "segments_by_axis": {str(axis): int(counts_map[axis]) for axis in plane_axes},
        "thin_axis": thin_axis,
    }


def _rasterize_triangles(projected: Any, faces: Any, count_u: int, count_v: int, np: Any):
    """Fill each projected triangle on a (count_v, count_u) occupancy grid.

    Pure-numpy equivalent of v11's ``cv2.fillConvexPoly`` loop so the module
    has no OpenCV dependency; the grids here are <= 96x96 so per-triangle
    barycentric fills over the bounding box are cheap and deterministic.
    """
    occupancy = np.zeros((count_v, count_u), dtype=bool)
    pixel = projected[np.asarray(faces, dtype=np.int64)].copy()
    pixel[..., 0] *= max(1, count_u - 1)
    pixel[..., 1] *= max(1, count_v - 1)
    for triangle in pixel:
        u_min = max(0, int(math.floor(triangle[:, 0].min())))
        u_max = min(count_u - 1, int(math.ceil(triangle[:, 0].max())))
        v_min = max(0, int(math.floor(triangle[:, 1].min())))
        v_max = min(count_v - 1, int(math.ceil(triangle[:, 1].max())))
        if u_min > u_max or v_min > v_max:
            continue
        us = np.arange(u_min, u_max + 1, dtype=float)
        vs = np.arange(v_min, v_max + 1, dtype=float)
        uu, vv = np.meshgrid(us, vs)
        a, b, c = triangle[0], triangle[1], triangle[2]
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1.0e-12:
            occupancy[v_min : v_max + 1, u_min : u_max + 1] = True
            continue
        w1 = ((b[1] - c[1]) * (uu - c[0]) + (c[0] - b[0]) * (vv - c[1])) / denominator
        w2 = ((c[1] - a[1]) * (uu - c[0]) + (a[0] - c[0]) * (vv - c[1])) / denominator
        w3 = 1.0 - w1 - w2
        eps = -0.5 / max(count_u, count_v)
        occupancy[v_min : v_max + 1, u_min : u_max + 1] |= (w1 >= eps) & (w2 >= eps) & (w3 >= eps)
    return occupancy


def _binary_close_2d(occupancy: Any, np: Any) -> Any:
    """3x3 binary closing (dilate then erode); fills rasterization gaps while
    preserving semantic holes larger than one grid cell.  No scipy needed."""
    padded = np.pad(occupancy, 1, mode="constant")
    dilated = np.zeros_like(occupancy, dtype=bool)
    for dv in range(3):
        for du in range(3):
            dilated |= padded[dv : dv + occupancy.shape[0], du : du + occupancy.shape[1]]
    padded = np.pad(dilated, 1, mode="constant")
    eroded = np.ones_like(occupancy, dtype=bool)
    for dv in range(3):
        for du in range(3):
            eroded &= padded[dv : dv + occupancy.shape[0], du : du + occupancy.shape[1]]
    return eroded


def projected_mid_surface(mesh: Any, np: Any, longest_segments: int = PROJECTED_LONGEST_SEGMENTS):
    """Project the (possibly body-supported) source onto its thinnest-axis
    plane, rasterize the silhouette, and rebuild a single-layer grid; port of
    v11 ``fast_visual.py:3130-3216`` (_create_projected_garment_mid_surface)."""
    import trimesh

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    normal_axis = int(np.argmin(extents))
    plane_axes = [axis for axis in range(3) if axis != normal_axis]
    longest_axis = plane_axes[int(extents[plane_axes[1]] > extents[plane_axes[0]])]
    other_axis = plane_axes[0] if plane_axes[1] == longest_axis else plane_axes[1]
    segments = {
        longest_axis: int(longest_segments),
        other_axis: max(16, int(round(longest_segments * extents[other_axis] / max(extents[longest_axis], 1.0e-12)))),
    }
    axis_u, axis_v = plane_axes
    count_u, count_v = segments[axis_u], segments[axis_v]
    projected = np.asarray(mesh.vertices, dtype=np.float64)[:, plane_axes]
    projected -= bounds[0, plane_axes]
    projected /= np.maximum(extents[plane_axes], 1.0e-12)
    occupancy = _rasterize_triangles(projected, mesh.faces, count_u, count_v, np)
    occupancy = _binary_close_2d(occupancy, np)
    coord_u = np.linspace(bounds[0, axis_u], bounds[1, axis_u], count_u + 1)
    coord_v = np.linspace(bounds[0, axis_v], bounds[1, axis_v], count_v + 1)
    center = bounds.mean(axis=0)
    vertices = []
    for v in coord_v:
        for u in coord_u:
            point = center.copy()
            point[axis_u] = u
            point[axis_v] = v
            vertices.append(point)
    row_width = count_u + 1
    faces: list[tuple[int, int, int]] = []
    for v in range(count_v):
        for u in range(count_u):
            if not occupancy[v, u]:
                continue
            lower_left = v * row_width + u
            lower_right = lower_left + 1
            upper_left = lower_left + row_width
            upper_right = upper_left + 1
            if (u + v) % 2:
                faces.extend(((lower_left, lower_right, upper_left), (lower_right, upper_right, upper_left)))
            else:
                faces.extend(((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)))
    if len(faces) < MIN_CLOTH_PROXY_FACES:
        raise ValueError("projected garment silhouette is too sparse for a cloth mid-surface")
    sheet = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces, dtype=np.int64), process=False)
    sheet.remove_unreferenced_vertices()
    _try_fix_normals(sheet)
    return sheet, {
        "method": "projected_single_layer_garment_mid_surface",
        "normal_axis": normal_axis,
        "segments_by_axis": {str(axis): int(value) for axis, value in segments.items()},
        "occupied_cell_fraction": float(occupancy.mean()),
        "limitation": "preserves the projected silhouette but not sewing panels, preformed volume, or seam stiffness",
    }


def sealed_shell_opening(mesh: Any, geometry_family: str, np: Any):
    """Cut a bounded opening into a sealed watertight shell; port of the v11
    opening surgery (``fast_visual.py:2990-3078``).

    Axis policy: garments open along the longest body axis; reinforced shells
    (hats etc.) along the thinnest; stage3 bags open at the semantic top,
    which is source +Y (stage3 assets use X/Y for the image plane and Z for
    reconstructed depth).  The cut side is the end with less surface area,
    except bags which always cut +Y max.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    centers = vertices[faces].mean(axis=1)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if geometry_family == "bag":
        axis = 1
        side = "max"
    elif geometry_family == "shell":
        axis = int(np.argmin(extents))
        side = ""
    else:
        axis = int(np.argmax(extents))
        side = ""
    lower, upper = (float(value) for value in mesh.bounds[:, axis])
    span = max(upper - lower, 1.0e-12)
    endpoint_fraction = 0.08
    endpoint_areas = {
        "min": float(areas[centers[:, axis] <= lower + endpoint_fraction * span].sum()),
        "max": float(areas[centers[:, axis] >= upper - endpoint_fraction * span].sum()),
    }
    if not side:
        side = min(endpoint_areas, key=endpoint_areas.get)
    removed = np.zeros(len(faces), dtype=bool)
    applied_fraction = 0.0
    minimum_removed_faces = 2 if len(faces) < 100 else max(4, int(round(0.002 * len(faces))))
    for fraction in (0.025, 0.04, 0.06, 0.08):
        if side == "min":
            candidate = centers[:, axis] <= lower + fraction * span
        else:
            candidate = centers[:, axis] >= upper - fraction * span
        count = int(candidate.sum())
        if minimum_removed_faces <= count <= max(2, len(faces) // 4):
            removed = candidate
            applied_fraction = fraction
            break
    if not removed.any():
        raise ValueError("sealed garment proxy could not derive a bounded semantic opening")
    mesh.update_faces(~removed)
    mesh.remove_unreferenced_vertices()
    _try_fix_normals(mesh)
    return mesh, {
        "method": "semantic_open_garment_shell_from_sealed_boundary",
        "status": "sealed_collision_volume_opened_for_cloth",
        "axis": int(axis),
        "side": side,
        "cut_fraction": applied_fraction,
        "removed_face_count": int(removed.sum()),
        "limitation": "automatic opening preserves the source silhouette but is not a recovered sewing pattern",
    }


def voxel_sealed_shell(mesh: Any, np: Any, resolution: int = CLOTH_VOXEL_SEAL_RESOLUTION):
    """Rebuild a watertight shell from a fragmented/open surface:
    voxelize -> 3x3x3 binary closing (seals sub-pitch fragment gaps) ->
    fill interior -> marching cubes.  Raises if the result is not a single
    watertight volume; the caller decides the fallback."""
    import trimesh
    from trimesh.voxel import morphology

    pitch = float(max(mesh.extents) / max(24, int(resolution)))
    voxels = mesh.voxelized(pitch)
    encoding = morphology.fill(morphology.binary_closing(voxels.encoding), method="holes")
    rebuilt = trimesh.voxel.ops.matrix_to_marching_cubes(encoding.dense, pitch=1.0)
    rebuilt.apply_transform(voxels.transform)
    rebuilt.remove_unreferenced_vertices()
    _try_fix_normals(rebuilt)
    if not rebuilt.is_watertight or not rebuilt.is_volume:
        raise ValueError("voxel seal did not produce a closed volume")
    labels = face_component_labels(rebuilt, np)
    n_bodies = int(labels.max()) + 1 if labels.size else 0
    dropped = 0
    if n_bodies > 1:
        # Satellite islands from scattered fragments: drop bodies below 1% of
        # the total area; fail only if no coherent main body survives.
        areas = np.bincount(labels, weights=np.asarray(rebuilt.area_faces, dtype=float))
        total = float(areas.sum()) or 1.0
        order = np.argsort(areas)[::-1]
        if float(areas[order[0]]) < 0.5 * total:
            raise ValueError("voxel seal produced no coherent main body")
        keep_labels = {int(label) for label in order if float(areas[label]) >= 0.01 * total}
        keep = np.asarray([int(label) in keep_labels for label in labels], dtype=bool)
        dropped = int((~keep).sum())
        rebuilt.update_faces(keep)
        rebuilt.remove_unreferenced_vertices()
        if not rebuilt.is_watertight or not rebuilt.is_volume:
            raise ValueError("voxel seal broke after dust removal")
    return rebuilt, {"voxel_pitch": float(pitch), "voxel_resolution": int(resolution), "seal_bodies": int(n_bodies), "seal_dust_faces_dropped": int(dropped)}


def filled_volume_proxy(mesh: Any, np: Any, resolution: int = VOXEL_RESOLUTION, preserve_watertight_boundary: bool = False):
    """v11 filled_volume (``fast_visual.py:2857-2904``): watertight sources may
    keep their boundary (MPM samples the interior); otherwise voxelize + fill
    + marching cubes, watertight-verified; convex hull is the fallback only."""
    pitch = float(max(mesh.extents) / max(24, int(resolution)))
    if preserve_watertight_boundary and bool(mesh.is_watertight) and bool(mesh.is_volume):
        kept = mesh.copy()
        kept.remove_unreferenced_vertices()
        _try_fix_normals(kept)
        return kept, {
            "method": "source_watertight_volume_boundary",
            "voxel_pitch_m": None,
            "voxel_resolution": None,
            "warnings": [],
            "shape_preservation_policy": "MPM samples the closed source boundary interior without rebuilding its exterior.",
        }
    warnings: list[str] = []
    method = "voxel_fill_preserving_exterior_connected_voids"
    import trimesh
    from trimesh.voxel import morphology

    try:
        voxels = mesh.voxelized(pitch)
        # Fragmented generated meshes have sub-pitch gaps between patches;
        # closing seals them so fill() does not leak (v11 sources were
        # watertight and did not need this).
        encoding = morphology.fill(morphology.binary_closing(voxels.encoding), method="holes")
        rebuilt = trimesh.voxel.ops.matrix_to_marching_cubes(encoding.dense, pitch=1.0)
        rebuilt.apply_transform(voxels.transform)
        rebuilt.remove_unreferenced_vertices()
        _try_fix_normals(rebuilt)
        if not rebuilt.is_watertight or not rebuilt.is_volume:
            raise ValueError("voxel result is not a closed volume")
        mesh = rebuilt
    except Exception as exc:
        mesh = mesh.convex_hull
        mesh.remove_unreferenced_vertices()
        _try_fix_normals(mesh)
        method = "convex_hull_fallback"
        warnings.append(f"voxel_fill_failed: {type(exc).__name__}: {exc}")
        warnings.append("convex_hull_may_remove_semantic_concavities_or_through_holes")
    return mesh, {
        "method": method,
        "voxel_pitch_m": float(pitch),
        "voxel_resolution": int(resolution),
        "warnings": warnings,
    }


def decimate_to_budget(mesh: Any, budget: int, np: Any):
    """Decimate toward ``budget`` faces; fast_simplification with rising
    aggressiveness, accepting the first pass that lands under 1.25x budget."""
    import fast_simplification
    import trimesh

    current = mesh
    for agg in (5, 7, 9):
        vertices, faces = fast_simplification.simplify(
            np.asarray(current.vertices, dtype=float),
            np.asarray(current.faces, dtype=np.int64),
            target_count=int(budget),
            agg=agg,
            lossless=False,
        )
        current = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if len(current.faces) <= int(1.25 * budget):
            break
    current.remove_unreferenced_vertices()
    return current


# --------------------------------------------------------------------------- #
# Verdict / gating
# --------------------------------------------------------------------------- #


def assess_proxy_verdict(proxy_quality: dict[str, Any], *, cloth: bool) -> tuple[str, list[str]]:
    """Classify the adapted proxy: ``ok`` / ``degraded`` (run, with recorded
    warnings) / ``unfit`` (hard proxy defect; the runner gates to
    model_limit instead of faking a physics failure).

    Gating policy (calibrated against the v12.2 baseline): only defects that
    make the simulation *uninterpretable* gate to model_limit (no faces, too
    few faces for any constraint graph, open MPM boundary, zero volume).
    Structural warnings — residual fragmentation of a fallback-kept source, a
    sealed cloth volume that will deflate — are recorded as ``degraded`` and
    still simulated: the v12.2 production path ran exactly those meshes, and
    silently gating them would manufacture regressions against the frozen
    baseline.  The recorded defect metrics are the audit trail for the
    gate-calibration study.
    """
    defects: list[str] = []
    degraded: list[str] = []
    faces = int(proxy_quality.get("face_count") or 0)
    if faces <= 0:
        defects.append("proxy_has_no_faces")
        return "unfit", defects
    if cloth:
        if faces < MIN_CLOTH_PROXY_FACES:
            defects.append(f"cloth_proxy_face_count_below_min:{faces}<{MIN_CLOTH_PROXY_FACES}")
        components = int(proxy_quality.get("component_count") or 0)
        if components > MAX_CLOTH_PROXY_COMPONENTS:
            degraded.append(f"cloth_proxy_fragmented:{components}_components")
        if proxy_quality.get("watertight") and int(proxy_quality.get("boundary_edge_count") or 0) == 0:
            degraded.append("cloth_proxy_is_sealed_volume_deflates_without_pressure_model")
    else:
        if not proxy_quality.get("watertight"):
            defects.append("mpm_proxy_not_watertight")
        if float(proxy_quality.get("volume") or 0.0) <= 0.0:
            defects.append("mpm_proxy_volume_nonpositive")
    if defects:
        return "unfit", defects
    # degraded: issues are returned as the defect list for audit; the verdict
    # field carries the severity.
    return ("degraded", degraded) if degraded else ("ok", [])


# --------------------------------------------------------------------------- #
# Top-level adaptation pipeline
# --------------------------------------------------------------------------- #


def adapt_solver_mesh(
    source_path: Path,
    scale_m: float,
    orientation: str,
    output_dir: Path,
    np: Any,
    *,
    cloth: bool,
    geometry_family: str,
) -> tuple[Path, Any, float, dict[str, Any]]:
    """v3 solver-mesh adaptation.  Returns the OBJ path, the scaled extent,
    the median edge length, and an auditable details dict (source/proxy
    quality, method, verdict, defects, warnings, limitations)."""
    import trimesh

    raw = load_source_mesh(source_path, np)
    details: dict[str, Any] = {"adaptation": "v3", "geometry_family": geometry_family}
    try:
        details["source_quality"] = mesh_quality(raw, np)
    except Exception as exc:  # metrics must never break the adaptation
        details["source_quality"] = {"error": f"{type(exc).__name__}: {exc}"}

    combined, hygiene = weld_and_clean(raw, np)
    details["hygiene"] = hygiene
    warnings: list[str] = []
    limitations: list[str] = []

    pipeline_error = None
    if cloth:
        try:
            if geometry_family == "cloth_sheet":
                combined, proxy_details = flat_sheet_mid_surface(combined, np)
                details["proxy"] = proxy_details
            elif bool(combined.is_watertight) and bool(combined.is_volume):
                try:
                    combined, opening = sealed_shell_opening(combined, geometry_family, np)
                    details["proxy"] = opening
                    warnings.append("sealed_collision_volume_converted_to_open_cloth_shell")
                except Exception as exc:
                    warnings.append(f"opening_surgery_failed: {type(exc).__name__}: {exc}")
                    combined, proxy_details = _cloth_open_fallback(combined, geometry_family, np, warnings)
                    details["proxy"] = proxy_details
            else:
                # Dominant 100k case: fragmented non-watertight generated surface.
                # Seal it into a watertight shell first (voxel closing + fill),
                # then open it semantically; never simulate the raw patch soup.
                warnings.append("source_surface_not_watertight_fragmented_voxel_seal_attempted")
                try:
                    sealed, seal_details = voxel_sealed_shell(combined, np)
                    try:
                        combined, opening = sealed_shell_opening(sealed, geometry_family, np)
                        opening = dict(opening)
                        opening["method"] = "voxel_sealed_" + opening["method"]
                        opening["seal"] = seal_details
                        details["proxy"] = opening
                        warnings.append("fragmented_source_voxel_sealed_then_opened")
                        limitations.append("voxel_seal_smooths_sub_voxel_surface_detail")
                    except Exception as exc:
                        warnings.append(f"opening_surgery_after_seal_failed: {type(exc).__name__}: {exc}")
                        combined, proxy_details = _cloth_open_fallback(sealed, geometry_family, np, warnings, seal_details)
                        details["proxy"] = proxy_details
                except Exception as exc:
                    warnings.append(f"voxel_seal_failed: {type(exc).__name__}: {exc}")
                    combined, proxy_details = _cloth_open_fallback(combined, geometry_family, np, warnings)
                    details["proxy"] = proxy_details
            if len(combined.faces) > SURFACE_FACE_BUDGET:
                try:
                    combined = decimate_to_budget(combined, SURFACE_FACE_BUDGET, np)
                    warnings.append("surface_decimated_to_face_budget_12000")
                except Exception as exc:
                    warnings.append(f"surface_decimation_failed: {type(exc).__name__}: {exc}")
        except Exception as exc:
            # The whole cloth fallback ladder is exhausted: this is a hard
            # proxy defect, not a physics failure.  Keep the welded source as
            # the (unused) export and let the verdict gate fire.
            pipeline_error = f"{type(exc).__name__}: {exc}"
            warnings.append(f"cloth_adaptation_pipeline_exhausted: {pipeline_error}")
            details["proxy"] = {"method": "none_pipeline_exhausted"}
        # Scale-before-rotate (v11 genesis_cloth.py:251-335): physical size
        # belongs to the asset, not to the world-space drop pose.
        extent = np.asarray(combined.extents, dtype=float)
        extent = np.maximum(extent, 1.0e-6)
        combined.apply_scale(float(scale_m) / float(extent.max()))
        if orientation == "corner_first":
            combined.apply_transform(trimesh.transformations.rotation_matrix(math.radians(45.0), [0.0, 1.0, 0.0]))
            combined.apply_transform(trimesh.transformations.rotation_matrix(math.radians(35.264), [1.0, 0.0, 0.0]))
    else:
        extent = np.asarray(combined.extents, dtype=float)
        extent = np.maximum(extent, 1.0e-6)
        combined.apply_scale(float(scale_m) / float(extent.max()))
        if orientation == "corner_first":
            combined.apply_transform(trimesh.transformations.rotation_matrix(math.radians(45.0), [0.0, 1.0, 0.0]))
            combined.apply_transform(trimesh.transformations.rotation_matrix(math.radians(35.264), [1.0, 0.0, 0.0]))
        if len(combined.faces) > SURFACE_FACE_BUDGET:
            combined = decimate_to_budget(combined, SURFACE_FACE_BUDGET, np)
        combined, volume_details = filled_volume_proxy(
            combined, np, preserve_watertight_boundary=(geometry_family == "bag")
        )
        details["proxy"] = volume_details
        warnings.extend(str(w) for w in volume_details.get("warnings") or [])

    bounds = combined.bounds
    center_xy = 0.5 * (bounds[0, :2] + bounds[1, :2])
    combined.apply_translation([-center_xy[0], -center_xy[1], -bounds[0, 2]])
    destination = output_dir / "solver_mesh.obj"
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.export(destination, file_type="obj")
    scaled_extent = np.asarray(combined.extents, dtype=float)
    try:
        edge_lengths = np.asarray(combined.edges_unique_length, dtype=float)
        median_edge = float(np.median(edge_lengths)) if edge_lengths.size else float(scaled_extent.max()) / 32.0
    except Exception:
        median_edge = float(scaled_extent.max()) / 32.0

    try:
        proxy_quality = mesh_quality(combined, np)
    except Exception as exc:
        proxy_quality = {"error": f"{type(exc).__name__}: {exc}"}
    details["proxy_quality"] = proxy_quality
    verdict, defects = assess_proxy_verdict(proxy_quality, cloth=cloth)
    if pipeline_error:
        defects = list(defects) + [f"cloth_adaptation_pipeline_exhausted:{pipeline_error}"]
        verdict = "unfit"
    details["verdict"] = verdict
    details["defects"] = defects
    details["warnings"] = warnings
    details["limitations"] = limitations
    details["final_geometry"] = {
        "vertex_count": int(len(combined.vertices)),
        "face_count": int(len(combined.faces)),
        "watertight": bool(combined.is_watertight),
        "median_edge_m": float(median_edge),
    }
    return destination, scaled_extent, median_edge, details


def _cloth_open_fallback(mesh: Any, geometry_family: str, np: Any, warnings: list[str], seal_details: dict | None = None):
    """Fallback ladder when a sealed/Sealed shell cannot be opened:
    garment/shell/cloth_sheet -> projected mid-surface; bag -> keep the
    sealed shell (v11 bag shape-preservation policy; the runner's verdict
    gate converts a sealed cloth proxy to an explicit model_limit rather
    than a fake physics failure)."""
    if geometry_family in ("garment", "shell", "cloth_sheet"):
        projection, proxy_details = projected_mid_surface(mesh, np)
        warnings.append("sealed_or_fragmented_shell_replaced_by_projected_mid_surface")
        return projection, proxy_details
    proxy_details = {"method": "sealed_shell_kept", "seal": seal_details}
    warnings.append("bag_opening_failed_sealed_shell_kept")
    return mesh, proxy_details
