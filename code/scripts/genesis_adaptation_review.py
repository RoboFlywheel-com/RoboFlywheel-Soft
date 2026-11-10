#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""v5 representation review: show every asset, never select by behavior gates."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import traceback

import numpy as np
import trimesh

import genesis_behavior_runner as legacy
import mesh_adaptation_v4 as geometry
import mesh_adaptation_v5 as adaptation
import mesh_representation
import mesh_visual_binding_v5 as visual
import mesh_render_audit


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def numerical_policy(asset, candidate, profile, length):
    rho = min(2e4, max(5., legacy._candidate_value(asset, candidate, 'density_kg_m3')))
    young = min(1e6, max(100., legacy._candidate_value(asset, candidate, 'youngs_modulus_pa')))
    nu = min(.49, max(-.95, legacy._candidate_value(asset, candidate, 'poisson_ratio')))
    dx = length/32
    grid = max(16, int(math.ceil(1/dx)))
    dx = 1/grid
    wave = math.sqrt(young*(1-nu)/((1+nu)*(1-2*nu)*rho))
    maximum_dt = .25*dx/wave
    dt = min(.002, 256*maximum_dt)
    substeps = max(24, int(math.ceil(dt/maximum_dt)))
    if profile != 'filled_volume':
        dt, substeps = .003, 10
    return {'dt': dt, 'substeps': substeps, 'grid_density': grid, 'grid_spacing': dx,
            'particle_size': .65*dx, 'density': rho, 'youngs_modulus': young, 'poisson_ratio': nu,
            'elastic_wave_speed': wave, 'estimated_cfl': wave*dt/substeps/dx if profile=='filled_volume' else None,
            'stability_policy': 'MPM wave-speed CFL <= 0.25; PBD reference dt/substeps'}


def studio(gs, policy, length, particle_size, cloth):
    lo, hi = (-1.6*length, -1.6*length, -.4*length), (1.6*length, 1.6*length, 3*length)
    scene = gs.Scene(show_viewer=False,
        sim_options=gs.options.SimOptions(dt=policy['dt'], substeps=policy['substeps']),
        rigid_options=gs.options.RigidOptions(constraint_timeconst=.01),
        mpm_options=gs.options.MPMOptions(dt=policy['dt'], particle_size=policy['particle_size'],
            grid_density=policy['grid_density'], lower_bound=lo, upper_bound=hi),
        pbd_options=gs.options.PBDOptions(dt=policy['dt'], particle_size=particle_size,
            lower_bound=lo, upper_bound=hi, hash_grid_cell_size=max(1.25*particle_size, 3.4*length/96)),
        coupler_options=gs.options.LegacyCouplerOptions(),
        vis_options=gs.options.VisOptions(show_world_frame=False, background_color=(.93,.94,.96),
            ambient_light=(.45,.45,.45), shadow=True, plane_reflection=False,
            segmentation_level='entity',
            n_support_neighbors=4 if cloth else 18),
        renderer=gs.renderers.Rasterizer())
    scene.add_entity(gs.morphs.Plane(plane_size=(8*length,8*length)), material=gs.materials.Rigid(friction=.55,coup_friction=.55),
                     surface=gs.surfaces.Default(color=(.58,.62,.67),roughness=.85))
    return scene


def source_preview(gs, source, output, length):
    raw, _ = geometry.load_scene_mesh(source)
    canonical, transform = geometry.canonicalize(raw)
    cleaned, _ = geometry.clean(canonical)
    simple = adaptation.decimate(cleaned, budget=16000)
    colors = np.asarray(canonical.visual.vertex_colors)
    simple.visual.vertex_colors = colors[visual.previous.cKDTree(canonical.vertices).query(simple.vertices)[1]]
    simple.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0]))
    simple.apply_scale(length)
    simple.apply_translation([0,0,-simple.bounds[0,2]])
    path = output/'source_atlas.obj'
    visual.color_atlas(simple).export(path)
    policy = {'dt':.003, 'substeps':1, 'particle_size':length/30, 'grid_density':64}
    scene = studio(gs,policy,length,length/80,False)
    obj = scene.add_entity(gs.morphs.Mesh(file=str(path),fixed=True,collision=False,decimate=False,
                           convexify=False,watertighten=None,file_meshes_are_zup=True),
                           material=gs.materials.Rigid(),surface=gs.surfaces.Default(vis_mode='visual',double_sided=True))
    camera = scene.add_camera(res=(480,480),pos=(2*length,-2*length,1.5*length),
                              lookat=(0,0,.45*length),fov=38,GUI=False)
    scene.build()
    frames=[];visibility=[]
    for i in range(72):
        angle=2*np.pi*i/72
        camera.set_pose(pos=(2.2*length*np.cos(angle),2.2*length*np.sin(angle),1.5*length),lookat=(0,0,.45*length))
        rgb,_,seg,_=camera.render(rgb=True,segmentation=True)
        frames.append(rgb.copy())
        visibility.append(mesh_render_audit.visibility_sample(seg,camera._visualizer._context.seg_idxc_map,obj.idx,i,np))
    result=legacy._write_study_render(output/'render',frames,np)
    result.update(kind='source_turntable_not_simulation',simulation_executed=False)
    return result,mesh_render_audit.summarize_visibility(visibility)


def run(args):
    asset=json.loads(args.asset.read_text())
    candidate=json.loads(args.candidate.read_text()) if args.candidate else {}
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    report={'sample_id':asset['sample_id'],'object_type':asset.get('object_type'),
            'purpose':'manual mesh-to-Genesis representation review; no behavior acceptance filter',
            'probe':args.probe,'simulation_executed':False,
            'status':'preparing','warnings':[]}
    frames=[];visibility=[];trajectory=[];gs_ready=False
    try:
        representation=mesh_representation.select_representation(asset)
        report['representation']=representation
        profile=representation['profile'];cloth=profile!='filled_volume'
        resolved=legacy._resolve_mesh(asset)
        if not resolved:raise FileNotFoundError('source_mesh_unresolved')
        source=Path(resolved)
        if profile=='sheet' and representation['geometry_family']!='cloth_sheet':
            raw,_=geometry.load_scene_mesh(source)
            representation=mesh_representation.refine_for_geometry(representation,raw.vertices)
            report['representation']=representation
            profile=representation['profile'];cloth=profile!='filled_volume'
        import genesis as gs
        length=min(.5,max(.12,float(asset.get('fixed_solver_numerics',{}).get('scale_m',.4))))
        report['scale']={'solver_m':length,'semantic_m':asset.get('fixed_solver_numerics',{}).get('scale_m'),
                         'purpose':'bounded review scale; not a physical parameter estimate'}
        legacy._headless()
        if args.probe=='source':
            gs.init(backend=gs.gpu,seed=0,logging_level='warning',precision='32');gs_ready=True
            report['render'],report['render_visibility']=source_preview(gs,source,output,length)
            if not report['render_visibility']['valid']:
                raise RuntimeError(report['render_visibility']['failure_reason'])
            report['status']='source_preview'
            return report
        orientation='corner_first' if args.probe=='drop' else 'flat'
        proxy, details=adaptation.adapt(source,output,representation,length,orientation)
        report['geometry']=details
        visual_path,appearance=visual.prepare_visual(source,source,proxy,details['normalization_transform'],output,
                                                     cloth=cloth and profile!='sheet',sheet=profile=='sheet')
        report['appearance']=appearance
        mesh=trimesh.load(proxy,process=False)
        median=float(np.median(mesh.edges_unique_length))
        ps=min(.5*median,.012*length) if cloth else length/48
        policy=numerical_policy(asset,candidate,profile,length)
        report['numerics']=policy
        gs.init(backend=gs.gpu,seed=0,logging_level='warning',precision='32');gs_ready=True
        if cloth:legacy._disable_pbd_remesh()
        scene=studio(gs,policy,length,ps,cloth)
        if cloth:
            material=gs.materials.PBD.Cloth(rho=.6 if profile=='reinforced_shell' else .35,
                stretch_compliance=representation['stretch_compliance'],bending_compliance=representation['bending_compliance'] or 1.,
                stretch_relaxation=.3,bending_relaxation=.12,air_resistance=0.,
                static_friction=.55,kinetic_friction=.45)
            if profile in {'sheet','garment_cloth'}:
                scene._sim.pbd_solver._max_bending_solver_iterations=0
                report['cloth_constraints']={
                    'stretch_compliance':representation['stretch_compliance'],
                    'bending_projection':False,'volume_constraint':False,
                    'self_collision':profile!='sheet',
                    'limitation':'negligible-bending cloth approximation; not calibrated fabric stiffness'}
            if profile=='sheet':
                scene._sim.pbd_solver._kernel_solve_collision=lambda f:None
        else:
            elastic=asset.get('material_family') in {'foam','rubber','gel'} or any(
                s in str(asset.get('object_type','')).lower() for s in ('pillow','cushion','tissue','stuffed','sleeping'))
            material,report['material_clamps']=legacy._make_material(gs,'mpm_elastic' if elastic else 'mpm_elastoplastic',asset,candidate)
        clearance=1.1*ps if cloth else 2*policy['grid_spacing']
        height=.3*length if args.probe=='drop' else 0.
        soft=legacy._add_soft_body(scene,gs,proxy,material,height+clearance,adapted=True)
        report['particle_count']=len(soft._particles)
        report['binding']=visual.bind_entity(gs,soft,visual_path,cloth=cloth,preserve_offsets=profile!='sheet')
        if cloth and representation['rest_curvature']:
            from mesh_shell_constraints import ShellConstraints
            constraints=ShellConstraints(soft,sealed=profile=='sealed_membrane',scale=length)
            constraints.install();report['shell_constraints']=constraints.diagnostics
        extent=mesh.extents
        plates=[];half=extent[0]/2;gap=.08*length;thickness=.15*length
        if args.probe=='compression':
            for sign in (-1,1):
                size=(thickness,max(1.5*extent[1],.4*length),max(1.2*extent[2],.2*length))
                pos=(sign*(half+gap+thickness/2),0,size[2]/2+clearance*.2)
                plate=scene.add_entity(gs.morphs.Box(size=size,pos=pos,fixed=True),
                    material=gs.materials.Rigid(friction=.3,coup_friction=.3),
                    surface=gs.surfaces.Default(color=(.29,.35,.44),roughness=.8))
                plates.append((plate,sign,pos))
        camera=scene.add_camera(res=(480,480),pos=(.7*length,-2.5*length,1.6*length),
            lookat=(0,0,.45*length),fov=40,near=.005*length,GUI=False)
        scene.build();report['simulation_executed']=True
        initial=legacy._positions(soft,np);initial_extent=np.linalg.norm(np.ptp(initial,axis=0))
        damping=legacy._make_v2_post_step_hook(soft,np,cloth,policy['dt'],
                    1.5*ps if cloth else 1.5*policy['grid_spacing'],lateral_rate_s_inv=0 if cloth else 8.)
        steps=args.steps;stride=max(1,steps//300)
        for step in range(steps+1):
            if step%stride==0 or step==steps:
                rgb,_,seg,_=camera.render(rgb=True,segmentation=True)
                frames.append(rgb.copy())
                visibility.append(mesh_render_audit.visibility_sample(seg,camera._visualizer._context.seg_idxc_map,soft.idx,step,np))
            if step%10==0 or step==steps:
                pos=legacy._positions(soft,np);vel=legacy._velocities(soft,np,len(pos))
                extent_now=np.ptp(pos,axis=0)
                trajectory.append({'step':step,'time_s':step*policy['dt'],'extent':extent_now.tolist(),
                    'centroid':pos.mean(axis=0).tolist(),'min_z':float(pos[:,2].min()),
                    'max_speed':float(np.linalg.norm(vel,axis=1).max())})
                if not np.isfinite(pos).all() or not np.isfinite(vel).all():raise RuntimeError('nonfinite_solver_state')
                if np.linalg.norm(extent_now)<.01*initial_extent:raise RuntimeError('particles_collapsed_to_point')
                if np.linalg.norm(extent_now)>4*initial_extent:raise RuntimeError('unbounded_expansion')
            if step==steps:break
            if plates:
                t=step/steps
                fraction=0 if t<.2 else (t-.2)/.25 if t<.45 else 1 if t<.6 else 1-(t-.6)/.2 if t<.8 else 0
                travel=gap+.15*extent[0]
                for plate,sign,pos in plates:
                    plate.set_pos((pos[0]-sign*travel*fraction,pos[1],pos[2]))
            scene.step();damping()
        report['render_visibility']=mesh_render_audit.summarize_visibility(visibility)
        if not report['render_visibility']['valid']:
            raise RuntimeError(report['render_visibility']['failure_reason'])
        report['status']='simulated_for_review'
    except Exception as exc:
        report.update(status='review_failure',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(limit=8))
    finally:
        report['trajectory']=trajectory
        if frames:
            try:
                report['render']=legacy._write_study_render(output/'render',frames,np)
                report['render_visibility']=mesh_render_audit.summarize_visibility(visibility)
                report['render']['partial']=report['status']=='review_failure'
            except Exception as exc:
                report.update(status='review_failure',render_error=f'{type(exc).__name__}: {exc}')
        save(output/'report.json',report)
        if gs_ready:gs.destroy()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset',type=Path,required=True)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probe',choices=['source','drop','compression'],required=True)
    parser.add_argument('--steps',type=int,default=900)
    args=parser.parse_args()
    report=run(args)
    print(json.dumps({k:report.get(k) for k in ('sample_id','status','error','representation')},ensure_ascii=False),flush=True)
    return 0 if report['status'] in {'simulated_for_review','source_preview'} else 1


if __name__=='__main__':raise SystemExit(main())
