# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('scipy')
trimesh=pytest.importorskip('trimesh')
pytest.importorskip('rtree')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/scripts'))
from mesh_representation import select_representation,refine_for_geometry
import mesh_adaptation_v5 as adaptation
from mesh_visual_binding_v5 import color_atlas,prepare_visual
from mesh_surface_attachment_v5 import attachment_offsets,triangle_frames
from mesh_garment_surface import repair_surface


def test_textile_garment_folds_without_changing_structural_or_rubber_profiles():
    ordinary=select_representation({'geometry_family':'garment','object_type':'Casual Pants','material_family':'textile'})
    assert ordinary['profile']=='garment_cloth'
    assert ordinary['bending_compliance'] is None
    assert not ordinary['rest_curvature']
    assert ordinary['stretch_compliance']==1e-7  # remove support, not the fabric's tensile resistance
    for name,material,profile in [('Padded Jacket','textile','reinforced_shell'),
                                  ('Peaked Cap','textile','reinforced_shell'),
                                  ('Latex Glove','rubber','garment_shell')]:
        assert select_representation({'geometry_family':'garment','object_type':name,
                                      'material_family':material})['profile']==profile


def test_nonplanar_textile_scarf_remains_3d_and_flexible():
    rep=select_representation({'geometry_family':'garment','object_type':'Scarf','material_family':'textile'})
    refined=refine_for_geometry(rep,trimesh.creation.icosphere(subdivisions=1).vertices)
    assert refined['profile']=='garment_cloth'
    assert not refined['rest_curvature']
    assert refined['semantic_geometry_conflict']['suggested_profile']=='sheet'


def test_surface_repair_preserves_open_cuff_and_adds_no_material():
    cylinder=trimesh.creation.cylinder(radius=.25,height=1,sections=32)
    cylinder.update_faces(np.abs(cylinder.face_normals[:,2])<.5)
    cylinder.remove_unreferenced_vertices()
    # A tiny third face on an existing edge emulates a generated seam defect.
    edge=cylinder.faces[0,:2]
    midpoint=cylinder.vertices[edge].mean(axis=0)+[0,0,1e-5]
    defective=trimesh.Trimesh(vertices=np.vstack([cylinder.vertices,midpoint]),
        faces=np.vstack([cylinder.faces,[*edge,len(cylinder.vertices)]]),process=False)
    repaired,info=repair_surface(defective)
    assert info['accepted']
    assert repaired is not None
    assert not repaired.is_watertight
    assert info['quality']['nonmanifold_edge_count']==0
    assert info['quality']['boundary_edge_count']==64
    assert len(repaired.faces)==len(cylinder.faces)
    np.testing.assert_allclose(repaired.extents,cylinder.extents)
    assert repaired.area==pytest.approx(cylinder.area)
    assert not info['holes_filled'] and not info['new_faces_added']


def test_surface_repair_reports_fragmentation_instead_of_joining_panels():
    parts=[]
    for i in range(9):
        part=trimesh.creation.box(extents=[.1,.1,.01]);part.apply_translation([i*.2,0,0]);parts.append(part)
    result,info=repair_surface(trimesh.util.concatenate(parts))
    assert result is None
    assert 'fragmented_constraint_graph' in info['rejection_reasons']
    assert info['quality']['component_count']==9


@pytest.mark.parametrize('family,name,profile',[
    ('bag','Backpack Fabric Body','reinforced_shell'),
    ('bag','Mud Lump','filled_volume'),
    ('bag','Paper Bag','reinforced_shell'),
    ('shell','Inflatable Mattress','sealed_membrane'),
    ('shell','Memory Foam Pillow','filled_volume'),
    ('garment','Padded Jacket','reinforced_shell'),
    ('garment','Casual Pants','garment_shell'),
    ('solid','Finger Puppet','garment_shell'),
    ('cloth_sheet','Shower Curtain','sheet'),
])
def test_representation_uses_object_semantics_before_family(family,name,profile):
    assert select_representation({'geometry_family':family,'object_type':name})['profile']==profile


def test_hollow_bag_never_becomes_a_planar_fallback(tmp_path):
    mesh=trimesh.creation.icosphere(subdivisions=2)
    mesh.apply_scale([.6,1,.35]);source=tmp_path/'source.ply';mesh.export(source)
    rep=select_representation({'geometry_family':'bag','object_type':'Backpack Fabric Body'})
    path,info=adaptation.adapt(source,tmp_path/'adapted',rep)
    result=trimesh.load(path,process=False)
    assert min(result.extents)/max(result.extents)>.2
    assert info['no_planar_fallback_for_3d_shell']
    assert result.extents[2]>result.extents[0]  # source Y becomes world Z


def test_mislabeled_strap_preserves_actual_3d_backpack(tmp_path):
    mesh=trimesh.creation.icosphere(subdivisions=2)
    mesh.apply_scale([.6,1,.35]);source=tmp_path/'source.ply';mesh.export(source)
    rep=select_representation({'geometry_family':'bag','object_type':'Backpack Shoulder Strap'})
    assert rep['profile']=='sheet'
    path,info=adaptation.adapt(source,tmp_path/'adapted',rep)
    result=trimesh.load(path,process=False)
    assert min(result.extents)/max(result.extents)>.2
    assert info['representation']['profile']=='reinforced_shell'
    assert info['representation']['semantic_geometry_conflict']['suggested_profile']=='sheet'


def test_geometry_guard_is_pose_invariant_and_respects_explicit_override():
    mesh=trimesh.creation.box(extents=[1,.15,.005])
    mesh.apply_transform(trimesh.transformations.rotation_matrix(.7,[1,2,3]))
    rep=select_representation({'geometry_family':'bag','object_type':'Backpack Shoulder Strap'})
    assert refine_for_geometry(rep,mesh.vertices)['profile']=='sheet'
    override=select_representation({'geometry_family':'bag','object_type':'Backpack Shoulder Strap',
                                    'adaptation_review':{'representation':'sheet'}})
    assert refine_for_geometry(override,trimesh.creation.box().vertices)['profile']=='sheet'


def test_uv_atlas_retains_corner_colors_after_texture_sampling():
    mesh=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0],[2,0,0],[3,0,0],[2,1,0]],
                         faces=[[0,1,2],[3,4,5]],process=False)
    colors=np.array([[255,0,0,255],[0,255,0,255],[0,0,255,255],[255,255,0,255],[0,255,255,255],[255,0,255,255]])
    mesh.visual.vertex_colors=colors
    atlas=color_atlas(mesh)
    restored=atlas.visual.to_color().vertex_colors
    np.testing.assert_allclose(restored[:,:3],colors[:,:3],atol=2)
    assert len(np.unique(restored[:,:3],axis=0))==6


def test_surface_offsets_preserve_separate_skins_and_rigid_motion():
    particles=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    skin=np.array([[.25,.25,.02],[.25,.25,-.02]])
    indices=np.array([[0,1,2,0],[0,1,2,0]])
    weights=np.array([[.5,.25,.25,0],[.5,.25,.25,0]])
    offsets,error=attachment_offsets(skin,particles,indices,weights)
    assert error<1e-14
    assert offsets[0,2]-offsets[1,2]==pytest.approx(.04)
    rotation=trimesh.transformations.rotation_matrix(.9,[1,2,3])[:3,:3]
    posed=particles@rotation.T+[.3,-.7,1.]
    frames=triangle_frames(posed[indices[:,:3]])
    restored=np.einsum('ni,nij->nj',weights,posed[indices])+np.einsum('ni,nij->nj',offsets,frames)
    np.testing.assert_allclose(restored,skin@rotation.T+[.3,-.7,1.],atol=1e-14)


def test_sheet_visual_has_one_layer_instead_of_projected_box_faces(tmp_path):
    source=trimesh.creation.box(extents=[1,1,.01])
    source.visual.vertex_colors=np.tile([220,30,60,255],(len(source.vertices),1))
    source_path=tmp_path/'source.ply';source.export(source_path)
    proxy=trimesh.Trimesh(vertices=[[-.5,-.5,0],[.5,-.5,0],[.5,.5,0],[-.5,.5,0]],
                          faces=[[0,1,2],[0,2,3]],process=False)
    proxy_path=tmp_path/'proxy.obj';proxy.export(proxy_path)
    output=tmp_path/'visual';output.mkdir()
    path,info=prepare_visual(source_path,source_path,proxy_path,np.eye(4).tolist(),output,sheet=True)
    visual=trimesh.load(path,force='mesh',process=False)
    assert len(visual.faces)==2
    assert visual.area==pytest.approx(proxy.area)
    assert info['method']=='single_layer_proxy_with_source_colors'


def test_sealed_membrane_keeps_closed_volume_across_drop_poses(tmp_path):
    mesh=trimesh.creation.icosphere(subdivisions=2)
    mesh.apply_scale([.6,1,.35]);source=tmp_path/'source.ply';mesh.export(source)
    rep=select_representation({'geometry_family':'shell','object_type':'Inflatable Mattress'})
    proxies=[]
    for pose in ('flat','corner_first'):
        path,info=adaptation.adapt(source,tmp_path/pose,rep,orientation=pose)
        result=trimesh.load(path,process=False);proxies.append(result)
        assert result.is_volume
        assert info['final_quality']['boundary_edge_count']==0
    assert proxies[0].volume==pytest.approx(proxies[1].volume,rel=1e-6)
    np.testing.assert_allclose(np.sort(proxies[0].edges_unique_length),
                               np.sort(proxies[1].edges_unique_length),rtol=1e-5,atol=1e-8)


def test_missing_source_still_writes_review_failure(tmp_path):
    from genesis_adaptation_review import run
    asset=tmp_path/'asset.json'
    asset.write_text(json.dumps({'sample_id':'missing','geometry_family':'bag','object_type':'Paper Bag'}))
    output=tmp_path/'review'
    report=run(SimpleNamespace(asset=asset,candidate=None,output=output,probe='drop',steps=10))
    assert report['status']=='review_failure'
    assert report['error']=='FileNotFoundError: source_mesh_unresolved'
    assert json.loads((output/'report.json').read_text())['simulation_executed'] is False


def test_review_index_includes_unstarted_and_failed_samples(tmp_path):
    root=tmp_path/'review';root.mkdir()
    inputs=tmp_path/'input';inputs.mkdir()
    ids=['failed','pending']
    for sid in ids:
        p=inputs/sid;p.mkdir()
        (p/'asset.json').write_text(json.dumps({'sample_id':sid,'geometry_family':'bag','object_type':'Paper Bag'}))
    (root/'manifest.json').write_text(json.dumps({'sample_ids':ids,'input':str(inputs)}))
    task=root/'failed';task.mkdir()
    (task/'result.json').write_text(json.dumps({'sample_id':'failed','asset':str(inputs/'failed/asset.json'),
        'probes':{'drop':{'status':'execution_failure','error':'timed out'}}}))
    script=Path(__file__).resolve().parents[1]/'deployment/experiments/mesh_adapt/build_representation_review.py'
    subprocess.run([sys.executable,str(script),'--root',str(root),'--previous',str(inputs)],check=True,capture_output=True)
    summary=json.loads((root/'review_summary.json').read_text())
    assert summary['tasks']==2
    assert summary['completed_tasks']==0
    assert not summary['all_source_previews']
    evidence=[json.loads(line) for line in (root/'review_evidence.jsonl').read_text().splitlines()]
    assert [row['sample_id'] for row in evidence]==ids
    assert evidence[0]['probes']['drop']['error']=='timed out'
    assert evidence[1]['probes']['source']['status']=='not_started'
