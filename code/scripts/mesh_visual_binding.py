# SPDX-License-Identifier: Apache-2.0
"""Auditable source-colored visual LOD and bounded Genesis particle binding.

All visual operations happen after physical particle sampling. No visual
vertex participates in collision, material assignment, or gate thresholds.
"""
from __future__ import annotations

from itertools import permutations, product
from pathlib import Path
from types import MethodType

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import mesh_adaptation_v4 as adaptation


def alignment(visual, physical):
    """Bidirectional point-to-surface distances (not tessellation density)."""
    diagonal = max(float(np.linalg.norm(physical.extents)), 1e-12)
    distances = []
    for source, target in ((visual, physical), (physical, visual)):
        verts = np.asarray(source.vertices)
        if len(verts) > 2048:
            verts = verts[np.linspace(0, len(verts) - 1, 2048, dtype=int)]
        _, d, _ = trimesh.proximity.closest_point(target, verts)
        distances.append(float(np.quantile(d, .95)) / diagonal)
    return {'visual_to_proxy_p95_ratio': distances[0], 'proxy_to_visual_p95_ratio': distances[1],
            'maximum_bidirectional_p95': max(distances), 'passed': max(distances) <= .08,
            'threshold': .08, 'method': 'deterministic_vertex_to_triangle_surface'}


def align_source(visual, reference):
    before = alignment(visual, reference)
    if before['passed']:
        return visual, {'before': before, 'transform': np.eye(4).tolist()}
    # Axis frame correction is shape-ranked, using only proper rotations.
    # Retain uniform scale; never independently stretch axes to fake agreement.
    candidates = []
    ref_points = np.asarray(reference.vertices)[::max(1, len(reference.vertices)//2048)]
    tree = cKDTree(ref_points)
    source_points = np.asarray(visual.vertices)[::max(1, len(visual.vertices)//2048)]
    for permutation in permutations(range(3)):
        for signs in product((-1., 1.), repeat=3):
            rotation = np.zeros((3, 3))
            rotation[np.arange(3), permutation] = signs
            if np.linalg.det(rotation) < .5:
                continue
            extent = np.abs(rotation) @ visual.extents
            valid = extent > 1e-12
            scale = float(np.median(reference.extents[valid] / extent[valid]))
            transform = np.eye(4)
            transform[:3, :3] = scale * rotation
            transform[:3, 3] = reference.bounds.mean(axis=0) - transform[:3, :3] @ visual.bounds.mean(axis=0)
            points = trimesh.transform_points(source_points, transform)
            score = np.quantile(tree.query(points)[0], .95) + np.quantile(cKDTree(points).query(ref_points)[0], .95)
            candidates.append((float(score), transform))
    _, transform = min(candidates, key=lambda item: item[0])
    corrected = visual.copy()
    corrected.apply_transform(transform)
    after = alignment(corrected, reference)
    if not after['passed']:
        raise adaptation.ProxyError('visual_source_frame_mismatch', {'visual_alignment': {'before': before, 'after': after}})
    return corrected, {'before': before, 'after': after, 'transform': transform.tolist(),
                       'method': 'proper_axis_permutation_uniform_scale_translation'}


def prepare_visual(source_path: Path, collision_source_path: Path, proxy_path: Path,
                   normalization_transform, output_dir: Path):
    visual, info = adaptation.load_scene_mesh(source_path)
    correction = None
    if source_path.resolve() != collision_source_path.resolve():
        reference, _ = adaptation.load_scene_mesh(collision_source_path)
        visual, correction = align_source(visual, reference)
    visual.apply_transform(np.asarray(normalization_transform, dtype=float))
    proxy = trimesh.load(proxy_path, force='mesh', process=False)
    source_colors = np.asarray(visual.visual.to_color().vertex_colors if hasattr(visual.visual, 'to_color')
                               else visual.visual.vertex_colors)
    # Welding uses normalized coordinates so tolerance is independent of meters.
    canonical, canonical_transform = adaptation.canonicalize(visual)
    clean, _ = adaptation.clean(canonical)
    clean.apply_transform(np.linalg.inv(canonical_transform))
    component_count = int(legacy_component_count(clean))
    underresolved_surface = not proxy.is_watertight and len(clean.faces) < .25 * len(proxy.faces)
    fallback = component_count > 32 or underresolved_surface
    if fallback:
        lod = proxy.copy()
    else:
        lod = adaptation.bounded_decimate(clean)
    measured_alignment = alignment(lod, proxy)
    alignment_input = 'proxy_fallback' if fallback else 'source_lod'
    if not measured_alignment['passed']:
        lod = proxy.copy()
        fallback = True
    nearest = cKDTree(visual.vertices).query(lod.vertices)[1]
    lod.visual.vertex_colors = source_colors[nearest]
    output = output_dir / 'bound_visual_lod.obj'
    lod.export(output, file_type='obj', include_color=True)
    return output, {'scene': info, 'source_frame_correction': correction,
                    'source_welded_components': component_count,
                    'source_lod_alignment': measured_alignment,
                    'alignment_measured_on': alignment_input,
                    'underresolved_surface_visual': underresolved_surface,
                    'method': 'proxy_with_source_colors' if fallback else 'source_lod_with_baked_colors',
                    'source_geometry_fallback': fallback,
                    'limitation': 'source texture baked at vertices; proxy fallback is explicitly approximate',
                    'face_count': len(lod.faces), 'vertex_count': len(lod.vertices),
                    'normalization_transform': normalization_transform,
                    'visual_path': str(output), 'affects_physics': False}


def legacy_component_count(mesh):
    return len(np.unique(adaptation.legacy.face_component_labels(mesh, np)))


def surface_weights(vertices, particles, faces, support_count):
    if support_count < 3:
        raise adaptation.ProxyError('visual_surface_support_count_below_three')
    physical = trimesh.Trimesh(vertices=particles, faces=faces, process=False)
    closest, distances, face_ids = trimesh.proximity.closest_point(physical, vertices)
    triangle_ids = np.asarray(faces)[face_ids]
    weights3 = trimesh.triangles.points_to_barycentric(np.asarray(particles)[triangle_ids], closest)
    indices = np.repeat(triangle_ids[:, :1], support_count, axis=1)
    weights = np.zeros((len(vertices), support_count))
    indices[:, :3] = triangle_ids
    weights[:, :3] = weights3
    return indices, weights, {'method': 'nearest_triangle_barycentric', 'cross_wall_blending': False,
                              'surface_projection_distance_max_m': float(distances.max(initial=0))}


def volume_weights(vertices, particles, support_count):
    if len(particles) < support_count or support_count < 4:
        raise adaptation.ProxyError('visual_volume_support_count_unresolved')
    distances, indices = cKDTree(particles).query(vertices, k=support_count)
    supports = particles[indices]
    matrix = (supports[:, :-1] - supports[:, -1:]).swapaxes(1, 2)
    first = np.einsum('nij,nj->ni', np.linalg.pinv(matrix), vertices - supports[:, -1])
    weights = np.column_stack((first, 1 - first.sum(axis=1)))
    bad = (~np.isfinite(weights).all(axis=1)) | (np.abs(weights).sum(axis=1) > 10)
    if bad.any():
        inverse = 1 / np.maximum(distances[bad], 1e-12)
        weights[bad] = inverse / inverse.sum(axis=1, keepdims=True)
    return indices, weights, {'method': 'local_affine_with_bounded_positive_fallback',
                              'positive_fallback_vertices': int(bad.sum())}


def binding_diagnostics(vertices, particles, indices, weights):
    reconstructed = np.einsum('ni,nij->nj', weights, particles[indices])
    distance = np.linalg.norm(reconstructed - vertices, axis=1)
    diagonal = max(float(np.linalg.norm(np.ptp(particles, axis=0))), 1e-12)
    return {'partition_of_unity_error': float(np.abs(weights.sum(axis=1) - 1).max(initial=0)),
            'rest_error_p95_ratio': float(np.quantile(distance, .95)) / diagonal,
            'rest_error_max_ratio': float(distance.max(initial=0)) / diagonal,
            'weight_l1_max': float(np.abs(weights).sum(axis=1).max(initial=0)),
            'finite': bool(np.isfinite(weights).all())}


def bind_entity(gs, entity, visual_path: Path, *, cloth: bool):
    """Replace visual state only, after Genesis has sampled physical particles.

    Register weights at allocation time to avoid Genesis's O(V*P) dense
    distance matrix and its cross-wall PBD nearest-particle blending.
    """
    import genesis.utils.geom as gu

    morph = gs.morphs.Mesh(file=str(visual_path), scale=1., collision=False, convexify=False,
                           decimate=False, watertighten=None, file_meshes_are_zup=True)
    # This bypasses Scene.add_entity(), which normally resolves a surface's
    # default vis_mode. Leaving it None silently omits BOTH MPM and PBD bodies
    # from the rasterizer even though their particles continue to simulate.
    vmesh = gs.Mesh.from_morph_surface(
        morph, gs.surfaces.Default(vis_mode="visual", double_sided=True))
    if isinstance(vmesh, list):
        if len(vmesh) != 1:
            raise adaptation.ProxyError('visual_mesh_has_multiple_submeshes')
        vmesh = vmesh[0]
    body = entity.morph
    pos, quat = gu.transform_pos_quat_by_trans_quat(
        np.asarray(body.offset_pos), np.asarray(body.offset_quat), np.asarray(body.pos), np.asarray(body.quat))
    vmesh.apply_transform(gu.trans_quat_to_T(pos, quat))
    vertices = np.asarray(vmesh.verts, dtype=np.float64)
    particles = np.asarray(entity._particles, dtype=np.float64)
    support_count = int(entity.solver._n_vvert_supports)
    if cloth:
        indices, weights, info = surface_weights(vertices, particles, entity._mesh.faces, support_count)
    else:
        indices, weights, info = volume_weights(vertices, particles, support_count)
    info.update(binding_diagnostics(vertices, particles, indices, weights))
    if not info['finite'] or info['rest_error_p95_ratio'] > .08 or info['partition_of_unity_error'] > 1e-5:
        raise adaptation.ProxyError('visual_particle_binding_unresolved', {'visual_binding': info})
    entity._vmesh, entity._surface = vmesh, vmesh.surface
    entity._vverts = vertices.astype(gs.np_float)
    entity._vfaces = np.asarray(vmesh.faces, dtype=gs.np_int)

    def install(bound):
        start = int(bound._vvert_start)
        end = start + len(indices)
        field = bound.solver.vverts_info
        all_indices, all_weights = field.support_idxs.to_numpy(), field.support_weights.to_numpy()
        all_indices[start:end] = indices + int(bound._particle_start)
        all_weights[start:end] = weights
        field.support_idxs.from_numpy(all_indices)
        field.support_weights.from_numpy(all_weights)

    entity._add_vverts_to_solver = MethodType(install, entity)
    info.update(visual_vertices=len(vertices), physical_particles=len(particles),
                surface_vis_mode=entity.surface.vis_mode, affects_physics=False)
    return info
