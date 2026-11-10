# SPDX-License-Identifier: Apache-2.0
"""Source-color binding that survives Genesis PBD's surface visual rebuild."""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np
from PIL import Image
import trimesh

import mesh_visual_binding as previous


def color_atlas(mesh, tile=8):
    """One UV chart per triangle, with barycentric color interpolation.

    A per-vertex texel index is NOT sufficient: interpolated UVs would sample
    unrelated texels across the atlas. Separate padded face charts avoid that.
    """
    faces = np.asarray(mesh.faces)
    colors = np.asarray(mesh.visual.vertex_colors)[faces, :3].astype(float)
    columns = int(math.ceil(math.sqrt(len(faces))))
    rows = int(math.ceil(len(faces)/columns))
    image = np.zeros((rows*tile, columns*tile, 3), dtype=np.uint8)
    y, x = np.mgrid[:tile, :tile]
    b = (x-1)/(tile-3)
    c = (y-1)/(tile-3)
    weights = np.stack([1-b-c, b, c], axis=-1).clip(0)
    weights /= weights.sum(axis=-1, keepdims=True)
    uv = np.empty((len(faces), 3, 2))
    for i, color in enumerate(colors):
        ox, oy = (i % columns)*tile, (i//columns)*tile
        image[oy:oy+tile, ox:ox+tile] = np.einsum('ijk,kl->ijl', weights, color).round().astype('uint8')
        uv[i] = [[(ox+1.5)/(columns*tile), 1-(oy+1.5)/(rows*tile)],
                 [(ox+tile-1.5)/(columns*tile), 1-(oy+1.5)/(rows*tile)],
                 [(ox+1.5)/(columns*tile), 1-(oy+tile-1.5)/(rows*tile)]]
    result = trimesh.Trimesh(vertices=np.asarray(mesh.vertices)[faces].reshape(-1, 3),
                             faces=np.arange(3*len(faces)).reshape(-1, 3), process=False)
    result.visual = trimesh.visual.TextureVisuals(uv=uv.reshape(-1, 2),
        material=trimesh.visual.material.SimpleMaterial(image=Image.fromarray(image), diffuse=[255,255,255,255]))
    return result


def prepare_visual(source, collision_source, proxy, transform, output, *, cloth=False,sheet=False):
    path, info = previous.prepare_visual(source, collision_source, proxy, transform, output)
    if sheet:
        # Projecting both faces of a generated thick sheet onto one physical
        # layer creates coincident triangles. Render that layer exactly once.
        source_mesh,_=previous.adaptation.load_scene_mesh(source)
        source_mesh.apply_transform(np.asarray(transform))
        lod=trimesh.load(proxy,force='mesh',process=False)
        colors=np.asarray(source_mesh.visual.vertex_colors)
        lod.visual.vertex_colors=colors[previous.cKDTree(source_mesh.vertices).query(lod.vertices)[1]]
        lod.export(path,file_type='obj',include_color=True)
        info.update(method='single_layer_proxy_with_source_colors',source_geometry_fallback=True,
                    face_count=len(lod.faces),vertex_count=len(lod.vertices),
                    sheet_visual_policy='one double-sided physical layer; no coincident front/back triangles')
    if cloth and info['source_geometry_fallback'] and not info['underresolved_surface_visual']:
        # Small disconnected source fragments should not replace an otherwise
        # aligned outer skin with the coarse voxel surface (including its holes).
        source_mesh,_=previous.adaptation.load_scene_mesh(source)
        source_mesh.apply_transform(np.asarray(transform))
        canonical,normalization=previous.adaptation.canonicalize(source_mesh)
        clean,_=previous.adaptation.clean(canonical)
        clean,dust=previous.adaptation.remove_dust(clean)
        try:
            lod=previous.adaptation.bounded_decimate(clean)
            lod.apply_transform(np.linalg.inv(normalization))
            physical=trimesh.load(proxy,force='mesh',process=False)
            measured=previous.alignment(lod,physical)
            info['source_skin_recheck']=measured
            if measured['passed']:
                colors=np.asarray(source_mesh.visual.vertex_colors)
                lod.visual.vertex_colors=colors[previous.cKDTree(source_mesh.vertices).query(lod.vertices)[1]]
                lod.export(path,file_type='obj',include_color=True)
                info.update(method='source_lod_with_baked_colors',source_geometry_fallback=False,
                    source_lod_alignment=measured,alignment_measured_on='source_lod_after_bounded_dust_removal',
                    visual_dust=dust,face_count=len(lod.faces),vertex_count=len(lod.vertices))
        except previous.adaptation.ProxyError as exc:
            info['source_skin_recheck_error']=str(exc)
    mesh = trimesh.load(path, force='mesh', process=False)
    atlas = color_atlas(mesh)
    destination = Path(output)/'bound_visual_atlas.obj'
    atlas.export(destination)
    info.update(visual_path=str(destination), appearance='source_colors_in_per_face_UV_atlas',
                reason='PBD overwrites raw vertex colors when constructing its render surface',
                atlas_vertex_count=len(atlas.vertices), atlas_faces=len(atlas.faces),
                limitation='source texture sampled to LOD vertex colors; not full-resolution source PBR')
    return destination, info


def bind_entity(gs,entity,visual_path,*,cloth,preserve_offsets=True):
    info=previous.bind_entity(gs,entity,visual_path,cloth=cloth)
    if cloth and preserve_offsets:
        from mesh_surface_attachment_v5 import attachment_offsets,install
        vertices=np.asarray(entity._vverts,dtype=float)
        particles=np.asarray(entity._particles,dtype=float)
        indices,weights,_=previous.surface_weights(vertices,particles,entity._mesh.faces,int(entity.solver._n_vvert_supports))
        offsets,error=attachment_offsets(vertices,particles,indices,weights)
        install(entity,indices,offsets,gs)
        info.update(method='nearest_triangle_barycentric_with_corotated_visual_offset',
                    projection_rest_error_p95_ratio=info['rest_error_p95_ratio'],
                    visual_rest_error_max_m=error,
                    visual_offset_max_m=float(np.linalg.norm(offsets,axis=1).max(initial=0)),
                    offsets_affect_physics=False,
                    offset_limitation='local rigid frame follows triangles; not an affine volumetric skin model')
    return info
