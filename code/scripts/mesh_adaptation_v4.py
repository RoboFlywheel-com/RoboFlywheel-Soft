# SPDX-License-Identifier: Apache-2.0
"""Opt-in v12.5 repair candidate: geometry in one source frame, then pose.

No imports or mutations of Genesis state. Historical v2/v3 helpers are only
reused where their geometry algorithm is unchanged. All lengths used during
repair are in a centered source frame whose longest dimension is one.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy import ndimage

import mesh_adaptation as legacy

FACE_BUDGET = 12000
FAMILIES = {'solid', 'shell', 'garment', 'bag', 'cloth_sheet'}


class ProxyError(ValueError):
    def __init__(self, code: str, details: dict | None = None):
        super().__init__(code)
        self.details = details or {}
        self.details.update(verdict='unfit', defects=[code], adaptation='v4')


def load_scene_mesh(path: Path):
    """Bake every scene node, including repeated instances and GLTF transforms."""
    scene = trimesh.load(str(path), force='scene', process=False)
    parts = []
    for node in sorted(scene.graph.nodes_geometry):
        transform, name = scene.graph[node]
        geometry = scene.geometry[name]
        if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces):
            part = geometry.copy()
            # Bake textures before concatenating differing material types.
            try:
                part.visual = trimesh.visual.ColorVisuals(
                    mesh=part, vertex_colors=part.visual.to_color().vertex_colors)
            except (AttributeError, ValueError):
                pass
            part.apply_transform(transform)
            parts.append(part)
    if not parts:
        raise ProxyError('source_has_no_triangles')
    mesh = trimesh.util.concatenate(parts)
    if not np.isfinite(mesh.vertices).all():
        raise ProxyError('source_nonfinite_vertices')
    if mesh.faces.min() < 0 or mesh.faces.max() >= len(mesh.vertices):
        raise ProxyError('source_invalid_face_indices')
    if not np.isfinite(mesh.extents).all() or max(mesh.extents) <= 0:
        raise ProxyError('source_degenerate_bounds')
    return mesh, {'scene_geometry_count': len(scene.geometry), 'scene_instance_count': len(parts),
                  'scene_node_transforms_applied': True}


def canonicalize(mesh):
    longest = float(max(mesh.extents))
    center = mesh.bounds.mean(axis=0)
    transform = np.eye(4)
    transform[:3, :3] /= longest
    transform[:3, 3] = -center / longest
    result = mesh.copy()
    result.apply_transform(transform)
    return result, transform


def clean(mesh):
    # Physics welding must ignore UV/normal seams. Copying textured visual
    # attributes here makes trimesh keep position-identical vertices apart.
    result = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), process=False)
    before = len(result.vertices), len(result.faces)
    result.merge_vertices(digits_vertex=6, merge_tex=True, merge_norm=True)
    result.update_faces(result.nondegenerate_faces(height=1e-10))
    result.update_faces(result.unique_faces())
    result.remove_unreferenced_vertices()
    if not len(result.faces):
        raise ProxyError('source_has_no_nondegenerate_faces')
    result.fix_normals(multibody=True)
    return result, {'weld_tolerance_source_longest_ratio': 1e-6,
                    'vertices_removed': before[0] - len(result.vertices),
                    'faces_removed': before[1] - len(result.faces), 'uv_seams_ignored': True}


def quality(mesh, *, intersections=False):
    q = legacy.mesh_quality(mesh, np)
    # Legacy formula labels an equilateral triangle 2, although the comment
    # says 1. Version this metric instead of silently altering old reports.
    q['triangle_aspect_definition'] = 'longest_edge_squared / (4*sqrt(3)*area/3)'
    triangles = np.asarray(mesh.triangles)
    area = np.asarray(mesh.area_faces)
    edge2 = np.sum((triangles - np.roll(triangles, 1, axis=1)) ** 2, axis=2).max(axis=1)
    aspect = edge2 / np.maximum(4 * math.sqrt(3) * area / 3, 1e-30)
    q.update(triangle_aspect_p95=float(np.quantile(aspect, .95)),
             triangle_aspect_max=float(aspect.max()),
             triangle_aspect_gt20_fraction=float(np.mean(aspect > 20)),
             degenerate_face_count=int(np.sum(area <= 1e-14)),
             volume=float(mesh.volume) if mesh.is_volume else None,
             finite_vertices=bool(np.isfinite(mesh.vertices).all()))
    q['self_intersection'] = intersection_report(mesh) if intersections else {
        'status': 'not_checked', 'reason': 'raw_source_cost_bounded; final_proxy_checked'}
    return q


def _intersect_pairs(a, b):
    """Triangle SAT including coplanar separating axes, vectorized per pair."""
    ae = np.roll(a, -1, axis=1) - a
    be = np.roll(b, -1, axis=1) - b
    an = np.cross(ae[:, 0], ae[:, 1])
    bn = np.cross(be[:, 0], be[:, 1])
    axes = np.concatenate((an[:, None], bn[:, None],
                           np.cross(ae[:, :, None], be[:, None, :]).reshape(-1, 9, 3),
                           np.cross(an[:, None], ae), np.cross(bn[:, None], be)), axis=1)
    norm = np.linalg.norm(axes, axis=2)
    axes = axes / np.maximum(norm[:, :, None], 1e-30)
    ap = np.einsum('nvc,nac->nav', a, axes)
    bp = np.einsum('nvc,nac->nav', b, axes)
    separated = ((ap.max(axis=2) < bp.min(axis=2) - 1e-10) |
                 (bp.max(axis=2) < ap.min(axis=2) - 1e-10)) & (norm > 1e-14)
    return ~separated.any(axis=1)


def intersection_report(mesh, max_pairs=250000):
    """Check nonincident triangles; do not claim a full manifold proof.

    Adjacent faces sharing a vertex are excluded. Budget exhaustion or a
    missing spatial index is explicit and makes final proxy quality unfit.
    """
    try:
        tree = mesh.triangles_tree
        faces, triangles = np.asarray(mesh.faces), np.asarray(mesh.triangles)
        pairs = []
        triangle_bounds = np.column_stack((triangles.min(axis=1), triangles.max(axis=1)))
        for i, bounds in enumerate(triangle_bounds):
            for j in tree.intersection(bounds):
                if j > i and not np.isin(faces[i], faces[j]).any():
                    pairs.append((i, j))
                    if len(pairs) > max_pairs:
                        return {'status': 'budget_exceeded', 'candidate_pairs': len(pairs)}
        hits = []
        for start in range(0, len(pairs), 2048):
            ids = np.asarray(pairs[start:start + 2048], dtype=int)
            if ids.size:
                hit = _intersect_pairs(triangles[ids[:, 0]], triangles[ids[:, 1]])
                hits.extend(ids[hit].tolist())
        return {'status': 'checked', 'method': 'nonincident_triangle_SAT_including_coplanar',
                'candidate_pairs': len(pairs), 'intersecting_pairs': len(hits),
                'examples': hits[:8], 'adjacent_faces_excluded': True}
    except (ImportError, ModuleNotFoundError) as exc:
        return {'status': 'unavailable', 'reason': type(exc).__name__}


def padded_voxel_volume(mesh, resolution=48):
    """Seal gaps without eroding the voxel grid boundary (v3 failure)."""
    pitch = float(max(mesh.extents)) / max(24, int(resolution))
    voxels = mesh.voxelized(pitch)
    # scipy binary_closing treats outside the finite array as empty. Without
    # padding, it erases faces sitting on the AABB, and fill then leaks.
    padding = 2
    occupancy = np.pad(np.asarray(voxels.matrix, dtype=bool), padding)
    closed = ndimage.binary_closing(occupancy, structure=np.ones((3, 3, 3), dtype=bool))
    filled = ndimage.binary_fill_holes(closed)
    if not filled.any():
        raise ProxyError('voxel_reconstruction_empty')
    result = trimesh.voxel.ops.matrix_to_marching_cubes(filled, pitch=1.0)
    transform = np.asarray(voxels.transform).copy()
    transform[:3, 3] -= transform[:3, :3] @ np.full(3, padding)
    result.apply_transform(transform)
    result.fix_normals(multibody=True)
    if not result.is_watertight or not result.is_volume:
        raise ProxyError('voxel_reconstruction_not_volume')
    return result, {'method': 'padded_voxel_close_fill_marching_cubes', 'resolution': int(resolution),
                    'pitch_source_longest_ratio': pitch, 'padding_cells': padding,
                    'surface_voxels': int(occupancy.sum()), 'closed_voxels': int(closed.sum()),
                    'filled_voxels': int(filled.sum()),
                    'interior_voxels_added': int(filled.sum() - closed.sum())}


def remove_dust(mesh):
    labels = legacy.face_component_labels(mesh, np)
    areas = np.bincount(labels, weights=mesh.area_faces)
    fractions = areas / areas.sum()
    small = np.where(fractions < .001)[0]
    if len(small) and fractions[small].sum() <= .01:
        result = mesh.copy()
        result.update_faces(~np.isin(labels, small))
        result.remove_unreferenced_vertices()
        return result, {'removed_components': len(small), 'removed_area_fraction': float(fractions[small].sum())}
    return mesh, {'removed_components': 0, 'removed_area_fraction': 0.0}


def bounded_decimate(mesh):
    if len(mesh.faces) <= FACE_BUDGET:
        return mesh
    result = legacy.decimate_to_budget(mesh, FACE_BUDGET, np)
    if len(result.faces) > FACE_BUDGET:
        # The historical helper intentionally stops within 1.25x its target.
        # A lower second target enforces this version's actual hard budget.
        result = legacy.decimate_to_budget(result, int(FACE_BUDGET / 1.25), np)
    if len(result.faces) > FACE_BUDGET:
        raise ProxyError('face_budget_exceeded', {'face_count': len(result.faces), 'face_budget': FACE_BUDGET})
    return result


def assess(q, *, cloth):
    defects = []
    if not q.get('finite_vertices') or q.get('degenerate_face_count', 0):
        defects.append('nonfinite_or_degenerate_proxy')
    if q.get('nonmanifold_edge_count', 0):
        defects.append('nonmanifold_proxy_edges')
    if q.get('face_count', 0) > FACE_BUDGET:
        defects.append('face_budget_exceeded')
    if q.get('component_count') != 1:
        defects.append('disconnected_proxy_not_one_material_body')
    if q.get('triangle_aspect_max', math.inf) > 1000:
        defects.append('extreme_sliver_triangles')
    check = q.get('self_intersection', {})
    if check.get('status') != 'checked':
        defects.append('proxy_intersection_check_incomplete')
    elif check.get('intersecting_pairs'):
        defects.append('self_intersecting_proxy')
    if cloth:
        if q.get('face_count', 0) < 32:
            defects.append('surface_proxy_too_sparse')
        if not q.get('boundary_edge_count'):
            defects.append('sealed_surface_requires_membrane_model')
    elif not q.get('watertight') or not q.get('winding_consistent') or (q.get('volume') or 0) <= 0:
        defects.append('mpm_requires_oriented_closed_volume')
    return ('unfit' if defects else 'ok'), defects


def pose_transform(mesh, scale_m, orientation):
    if not math.isfinite(scale_m) or scale_m <= 0:
        raise ProxyError('invalid_solver_scale')
    transform = np.eye(4)
    transform[:3, :3] *= scale_m
    if orientation == 'corner_first':
        transform = (trimesh.transformations.rotation_matrix(math.radians(35.264), [1, 0, 0]) @
                     trimesh.transformations.rotation_matrix(math.radians(45), [0, 1, 0]) @ transform)
    elif orientation != 'flat':
        raise ProxyError('unsupported_pose')
    posed = mesh.copy()
    posed.apply_transform(transform)
    translation = [-posed.bounds[:, 0].mean(), -posed.bounds[:, 1].mean(), -posed.bounds[0, 2]]
    transform[:3, 3] += translation
    posed.apply_translation(translation)
    return posed, transform


def adapt_solver_mesh(source_path: Path, scale_m: float, orientation: str, output_dir: Path,
                      _np: Any = np, *, cloth: bool, geometry_family: str):
    details = {'adaptation': 'v4', 'geometry_family': geometry_family, 'warnings': [],
               'limitations': ['proxy_approximation_not_recovered_sewing_pattern'],
               'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest()}
    try:
        if geometry_family not in FAMILIES:
            raise ProxyError('unknown_geometry_family')
        raw, scene_info = load_scene_mesh(source_path)
        canonical, source_transform = canonicalize(raw)
        details['scene'] = scene_info
        cleaned, hygiene = clean(canonical)
        details['hygiene'] = hygiene
        details['source_quality'] = quality(cleaned)
        try:
            cleaned = bounded_decimate(cleaned)
        except ProxyError as exc:
            if str(exc) != 'face_budget_exceeded':
                raise
            # Fragmented input has a per-island simplification floor. It is
            # an intermediate voxelization input, not the final solver mesh.
            # Do not reject it before the reconstruction that removes islands.
            cleaned = legacy.decimate_to_budget(cleaned, FACE_BUDGET, np)
            if len(cleaned.faces) > 200000:
                raise ProxyError('voxel_input_resource_budget_exceeded') from exc
            details['warnings'].append('fragmented_source_exceeds_solver_face_budget_before_voxel_rebuild')
        details['pre_reconstruction_face_count'] = int(len(cleaned.faces))
        cleaned, dust = remove_dust(cleaned)
        details['source_dust'] = dust
        if cloth and geometry_family == 'cloth_sheet':
            proxy, method = legacy.projected_mid_surface(cleaned, np, longest_segments=64)
            method['method'] = 'silhouette_mid_surface'
        elif cloth:
            source_check = quality(cleaned, intersections=True)
            if (not cleaned.is_watertight and source_check['component_count'] == 1 and
                    not source_check['nonmanifold_edge_count'] and
                    source_check['triangle_aspect_max'] <= 1000 and
                    source_check['self_intersection'].get('intersecting_pairs') == 0):
                proxy, method = cleaned.copy(), {'method': 'clean_open_surface_preserved'}
            else:
                if (cleaned.is_volume and source_check['triangle_aspect_max'] <= 1000 and
                        source_check['self_intersection'].get('intersecting_pairs') == 0):
                    sealed, method = cleaned.copy(), {'method': 'source_watertight_boundary'}
                else:
                    sealed, method = padded_voxel_volume(cleaned)
                sealed, method['dust'] = remove_dust(sealed)
                try:
                    proxy, opening = legacy.sealed_shell_opening(sealed, geometry_family, np)
                    method['opening'] = opening
                except ValueError:
                    # Never substitute raw patch soup or a sealed bag on failure.
                    if geometry_family == 'bag':
                        raise ProxyError('bag_semantic_opening_unresolved')
                    proxy, projection = legacy.projected_mid_surface(cleaned, np)
                    method['projection_fallback'] = projection
                    details['warnings'].append('preformed_shell_replaced_by_projected_mid_surface')
        else:
            source_check = quality(cleaned, intersections=True)
            if (cleaned.is_volume and source_check['component_count'] == 1 and source_check['triangle_aspect_max'] <= 1000 and
                    source_check['self_intersection'].get('intersecting_pairs') == 0):
                proxy, method = cleaned.copy(), {'method': 'source_watertight_boundary'}
            else:
                proxy, method = padded_voxel_volume(cleaned)
            proxy, method['dust'] = remove_dust(proxy)
        proxy = bounded_decimate(proxy)
        proxy, final_clean = clean(proxy)
        details['proxy'] = method
        details['final_cleanup'] = final_clean
        details['canonical_proxy_quality'] = quality(proxy, intersections=True)
        verdict, defects = assess(details['canonical_proxy_quality'], cloth=cloth)
        repairable = {'extreme_sliver_triangles', 'self_intersecting_proxy', 'nonmanifold_proxy_edges'}
        if not cloth and repairable.intersection(defects):
            # One bounded geometry-only rebuild for defects introduced by
            # simplification. Preserve the failed assessment as audit evidence.
            details['pre_rebuild_quality'] = details['canonical_proxy_quality']
            proxy, repair = padded_voxel_volume(proxy, resolution=32)
            proxy = bounded_decimate(proxy)
            proxy, _ = clean(proxy)
            details['proxy']['final_topology_rebuild'] = repair
            details['canonical_proxy_quality'] = quality(proxy, intersections=True)
            verdict, defects = assess(details['canonical_proxy_quality'], cloth=cloth)
        posed, pose = pose_transform(proxy, scale_m, orientation)
        # Same source-frame geometry for drop and compression. Surgery and
        # voxelization precede the pose, so orientation cannot change volume.
        details['normalization_transform'] = (pose @ source_transform).tolist()
        details['canonical_to_solver_transform'] = pose.tolist()
        details['scale_before_rotate'] = True
        details['source_longest_dimension'] = float(max(raw.extents))
        details['solver_scale_m'] = scale_m
        details['proxy_quality'] = quality(posed)
        details['proxy_quality']['self_intersection'] = details['canonical_proxy_quality']['self_intersection']
        details.update(verdict=verdict, defects=defects)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / 'solver_mesh.obj'
        posed.export(destination)
        details['solver_mesh_sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
        details['final_geometry'] = details['proxy_quality']
        return destination, np.asarray(posed.extents), float(np.median(posed.edges_unique_length)), details
    except ProxyError as exc:
        merged = {**details, **exc.details}
        raise ProxyError(str(exc), merged) from exc
    except ValueError as exc:
        raise ProxyError('mesh_geometry_processing_failed', {**details, 'detail': str(exc)}) from exc
