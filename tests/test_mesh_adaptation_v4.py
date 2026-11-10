# SPDX-License-Identifier: Apache-2.0
"""Geometric invariants for the explicitly versioned repair candidate."""
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('scipy')
trimesh = pytest.importorskip('trimesh')
pytest.importorskip('rtree')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code/scripts'))
import mesh_adaptation_v4 as ma
import mesh_visual_binding as vb


def test_voxel_closing_keeps_aabb_faces_and_interior():
    box = trimesh.creation.box(extents=[1, .5, .25])
    result, details = ma.padded_voxel_volume(box)
    assert result.is_volume
    assert result.volume == pytest.approx(box.volume, rel=.18)
    assert details['interior_voxels_added'] > details['surface_voxels']
    assert np.allclose(result.bounds.mean(axis=0), box.bounds.mean(axis=0), atol=.015)


def test_voxel_seal_keeps_a_semantic_through_hole():
    ring = trimesh.creation.annulus(r_min=.25, r_max=.5, height=.4)
    result, _ = ma.padded_voxel_volume(ring)
    assert not result.contains([[0, 0, 0]])[0]
    assert result.contains([[.375, 0, 0]])[0]


def test_scene_instances_and_transforms_are_not_discarded(tmp_path):
    scene = trimesh.Scene()
    cube = trimesh.creation.box()
    scene.add_geometry(cube, geom_name='cube', node_name='one')
    transform = np.eye(4)
    transform[0, 3] = 3
    scene.graph.update(frame_to='two', matrix=transform, geometry='cube')
    path = tmp_path / 'instances.glb'
    path.write_bytes(scene.export(file_type='glb'))
    result, info = ma.load_scene_mesh(path)
    assert info['scene_instance_count'] == 2
    assert len(result.faces) == 2 * len(cube.faces)
    assert result.extents[0] == pytest.approx(4)


def test_physics_welds_texture_seams():
    cube = trimesh.creation.box()
    vertices = cube.vertices[cube.faces].reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.arange(2*len(vertices)).reshape(-1, 2)/100)
    result, info = ma.clean(mesh)
    assert result.is_watertight
    assert len(result.vertices) == 8
    assert info['vertices_removed'] == 28


def test_crossing_and_coplanar_triangles_are_detected():
    a = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    b = [[.2, .2, -1], [.2, .2, 1], [.8, .2, 0]]
    mesh = trimesh.Trimesh(vertices=a+b, faces=[[0, 1, 2], [3, 4, 5]], process=False)
    assert ma.intersection_report(mesh)['intersecting_pairs'] == 1
    mesh.vertices[3:] = [[.1, .1, 0], [.9, .1, 0], [.1, .9, 0]]
    assert ma.intersection_report(mesh)['intersecting_pairs'] == 1
    mesh.vertices[3:] += [2, 0, 0]
    assert ma.intersection_report(mesh)['intersecting_pairs'] == 0


def test_equilateral_aspect_is_one():
    mesh = trimesh.Trimesh(vertices=[[0,0,0], [1,0,0], [.5,np.sqrt(3)/2,0]], faces=[[0,1,2]])
    assert ma.quality(mesh)['triangle_aspect_p95'] == pytest.approx(1)


def test_pose_changes_neither_canonical_volume_nor_proxy_topology(tmp_path):
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.apply_scale([1, .7, .5])
    path = tmp_path / 'source.ply'
    mesh.export(path)
    flat = ma.adapt_solver_mesh(path, .32, 'flat', tmp_path/'flat', cloth=False, geometry_family='solid')
    drop = ma.adapt_solver_mesh(path, .32, 'corner_first', tmp_path/'drop', cloth=False, geometry_family='solid')
    assert flat[3]['verdict'] == drop[3]['verdict'] == 'ok'
    assert flat[3]['canonical_proxy_quality'] == drop[3]['canonical_proxy_quality']
    a = trimesh.load(flat[0], process=False)
    b = trimesh.load(drop[0], process=False)
    assert a.volume == pytest.approx(b.volume, rel=1e-6)
    transformed = trimesh.transform_points(mesh.vertices, np.array(drop[3]['normalization_transform']))
    assert np.max(vb.cKDTree(b.vertices).query(transformed)[0]) < 1e-6


def test_disconnected_or_sealed_cloth_is_not_accepted():
    mesh = trimesh.util.concatenate([trimesh.creation.box(), trimesh.creation.box().apply_translation([2,0,0])])
    verdict, defects = ma.assess(ma.quality(mesh, intersections=True), cloth=True)
    assert verdict == 'unfit'
    assert 'disconnected_proxy_not_one_material_body' in defects
    assert 'sealed_surface_requires_membrane_model' in defects


def test_surface_binding_does_not_mix_front_and_back_walls():
    particles = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.01],[1,0,.01],[0,1,.01]])
    vertices = np.array([[.2,.2,0],[.3,.2,.01]])
    indices, weights, _ = vb.surface_weights(vertices, particles, [[0,1,2],[3,4,5]], 4)
    assert set(indices[0]) <= {0,1,2}
    assert set(indices[1]) <= {3,4,5}
    assert vb.binding_diagnostics(vertices, particles, indices, weights)['rest_error_max_ratio'] < 1e-12


def test_volume_binding_preserves_affine_motion():
    particles = trimesh.creation.icosphere(subdivisions=1).vertices
    vertices = particles[:7] * .82
    indices, weights, _ = vb.volume_weights(vertices, particles, 18)
    transform = np.array([[1.2,.1,0],[0,.7,.2],[0,0,1.1]])
    moved = particles @ transform.T + [2,3,4]
    bound = np.einsum('ni,nij->nj', weights, moved[indices])
    np.testing.assert_allclose(bound, vertices @ transform.T + [2,3,4], atol=1e-10)


def test_visual_color_transfer_and_transform(tmp_path):
    source = trimesh.creation.icosphere(subdivisions=1)
    source.visual.vertex_colors = [255, 0, 0, 255]
    path = tmp_path/'source.ply'
    source.export(path)
    proxy_path, _, _, details = ma.adapt_solver_mesh(path, .32, 'corner_first', tmp_path,
                                                   cloth=False, geometry_family='solid')
    visual, metadata = vb.prepare_visual(path, path, proxy_path, details['normalization_transform'], tmp_path)
    lod = trimesh.load(visual, process=False)
    assert metadata['source_lod_alignment']['passed']
    assert (lod.visual.vertex_colors[:, 0] == 255).all()
    assert (lod.visual.vertex_colors[:, 1] == 0).all()


def test_coarse_closed_visual_cannot_hide_surface_folds(tmp_path):
    source = trimesh.creation.box(extents=[1,.7,.01])
    path = tmp_path/'source.ply'
    source.export(path)
    proxy, _, _, details = ma.adapt_solver_mesh(path,.32,'flat',tmp_path,
                                              cloth=True,geometry_family='cloth_sheet')
    visual, info = vb.prepare_visual(path,path,proxy,details['normalization_transform'],tmp_path)
    assert info['underresolved_surface_visual']
    assert info['source_geometry_fallback']
    assert len(trimesh.load(visual,process=False).faces) == len(trimesh.load(proxy,process=False).faces)
