# SPDX-License-Identifier: Apache-2.0
"""Geometry-preserving v5 adapter for manual review. No behavior-gate filtering."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import trimesh

import mesh_adaptation_v4 as base
from mesh_representation import refine_for_geometry


def decimate(mesh, budget=6500):
    return base.legacy.decimate_to_budget(mesh, budget, np) if len(mesh.faces)>budget else mesh.copy()


def adapt(source: Path, output: Path, representation, scale=.4, orientation='flat'):
    output.mkdir(parents=True, exist_ok=True)
    raw, scene = base.load_scene_mesh(source)
    representation = refine_for_geometry(representation, raw.vertices)
    canonical, source_transform = base.canonicalize(raw)
    mesh, hygiene = base.clean(canonical)
    profile = representation['profile']
    details = {'version': 'v5-review', 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'scene': scene, 'hygiene': hygiene, 'source_quality': base.quality(mesh),
               'representation': representation, 'warnings': [], 'repairs': []}
    # Decimation is an intermediate resource bound, not a reason to hide assets.
    mesh = decimate(mesh)
    mesh, dust = base.remove_dust(mesh)
    details['dust'] = dust
    if profile == 'sheet':
        proxy, method = base.legacy.projected_mid_surface(mesh, np, longest_segments=56)
        details['repairs'].append(method)
    else:
        q = base.quality(mesh)
        clean_boundary = q['nonmanifold_edge_count'] == 0 and q['triangle_aspect_max'] < 100
        proxy = None
        if profile == 'garment_cloth':
            from mesh_garment_surface import repair_surface
            proxy, surface_repair = repair_surface(mesh)
            details['repairs'].append(surface_repair)
            if proxy is None:
                details['warnings'].append('garment_surface_repair_unresolved; previous_boundary_route_retained')
        if proxy is not None:
            pass
        elif clean_boundary and (mesh.is_volume or profile not in {'filled_volume', 'sealed_membrane'}):
            proxy = mesh.copy()
            details['repairs'].append({'method': 'preserve_source_3d_boundary'})
        else:
            proxy, method = base.padded_voxel_volume(mesh, resolution=40)
            details['repairs'].append(method)
        # Open only plausible semantic caps. Never project a 3D bag, garment,
        # reinforced shell or membrane to a planar silhouette on failure.
        if profile in {'bag_shell', 'garment_shell', 'garment_cloth', 'reinforced_shell'} and proxy.is_watertight:
            family = representation['geometry_family']
            try:
                proxy, opening = base.legacy.sealed_shell_opening(proxy, 'bag' if family == 'bag' else 'garment', np)
                details['repairs'].append(opening)
            except ValueError:
                details['warnings'].append('closed_boundary_retained_without_pressure; semantic_opening_unresolved')
        proxy = decimate(proxy)
    proxy, _ = base.clean(proxy)
    details['canonical_quality'] = base.quality(proxy, intersections=True)
    # Keep disconnected material pieces visible and independently simulated.
    if details['canonical_quality']['component_count'] > 1:
        details['warnings'].append('multiple_components_retained; no artificial joining springs')
    if details['canonical_quality']['self_intersection'].get('intersecting_pairs', 0):
        details['warnings'].append('residual_proxy_intersections; review_required')
    # Generated Stage3 assets are source-Y-up. A hollow bag must not enter the
    # scene on its depth axis as though it were an already flattened sheet.
    up = np.eye(4)
    if profile != 'sheet':
        up = trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0])
    else:
        normal = np.eye(3)[int(np.argmin(proxy.extents))]
        up = trimesh.geometry.align_vectors(normal, [0, 0, 1])
    proxy.apply_transform(up)
    posed, pose = base.pose_transform(proxy, scale, orientation)
    transform = pose @ up @ source_transform
    proxy_path = output/'solver_mesh.obj'
    posed.export(proxy_path)
    details.update(normalization_transform=transform.tolist(), source_up_axis=1,
                   final_quality=base.quality(posed), physical_face_count=len(posed.faces),
                   scale_m=scale, orientation=orientation,
                   no_planar_fallback_for_3d_shell=True,
                   solver_mesh_sha256=hashlib.sha256(proxy_path.read_bytes()).hexdigest())
    return proxy_path, details
