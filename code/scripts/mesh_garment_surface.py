# SPDX-License-Identifier: Apache-2.0
"""Bounded surface-only repair for ordinary textile garments.

Never seal, fill, extrude, project, or join a garment here. A rejected geometry
candidate remains documented; the caller retains its previous boundary route.
"""
from __future__ import annotations

import numpy as np
import trimesh

import mesh_adaptation_v4 as base


def repair_surface(mesh):
    source, _ = base.clean(mesh)
    areas = np.asarray(source.area_faces)
    triangles = np.asarray(source.triangles)
    longest2 = ((triangles - np.roll(triangles, 1, axis=1))**2).sum(axis=2).max(axis=1)
    aspect = longest2 / np.maximum(4*np.sqrt(3)*areas/3, 1e-30)
    _, inverse = np.unique(np.sort(source.edges, axis=1), axis=0, return_inverse=True)
    face_edges = inverse.reshape(-1, 3)
    incidence = np.zeros(inverse.max()+1, dtype=int)
    keep = np.zeros(len(areas), dtype=bool)
    # Retain the largest incident material faces; tiny duplicate/sliver faces
    # cannot make an edge have three or more incident triangles.
    for face in np.argsort(-areas, kind='stable'):
        if aspect[face] < 100 and (incidence[face_edges[face]] < 2).all():
            keep[face] = True
            incidence[face_edges[face]] += 1
    result = source.copy()
    result.update_faces(keep)
    result.remove_unreferenced_vertices()
    # trimesh.split defaults to repairing/filling holes: explicitly disable it.
    components = list(result.split(only_watertight=False, repair=False))
    component_areas = np.array([part.area for part in components])
    dust = component_areas < .001*source.area
    if component_areas[dust].sum() <= .01*source.area:
        components = [part for part, small in zip(components, dust) if not small]
    if not components:
        return None, {'method': 'local_garment_surface_repair', 'accepted': False,
                      'reason': 'no_material_surface_retained'}
    result = trimesh.util.concatenate(components)
    result, _ = base.clean(result)
    quality = base.quality(result, intersections=True)
    retained = float(result.area/source.area)
    intersections = quality['self_intersection']
    reasons = []
    if retained < .97: reasons.append('surface_area_loss_above_3pct')
    if quality['component_count'] > 8: reasons.append('fragmented_constraint_graph')
    if quality['nonmanifold_edge_count']: reasons.append('residual_nonmanifold_edges')
    if intersections['status'] != 'checked': reasons.append('intersection_check_incomplete')
    elif intersections['intersecting_pairs'] > .02*len(result.faces):
        reasons.append('dense_surface_intersections')
    detail = {'method': 'local_garment_surface_repair', 'accepted': not reasons,
              'rejection_reasons': reasons, 'retained_area_fraction': retained,
              'quality': quality, 'volumetric_reconstruction': False,
              'new_faces_added': False, 'holes_filled': False,
              'limitation': 'source surface topology, not recovered sewing panels or a guaranteed midsurface'}
    return (None if reasons else result), detail
