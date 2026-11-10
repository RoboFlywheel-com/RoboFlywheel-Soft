#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real GPU/Rasterizer regression for v4 soft-body visibility; no rescue-rate claim."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['solid', 'cloth_sheet'], required=True)
    parser.add_argument('--inject-missing-vis-mode', action='store_true',
                        help='Negative control: reproduce the historical invisible-body bug.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for key, name in [('GS_CACHE_FILE_PATH', 'cache'), ('NUMBA_CACHE_DIR', 'numba_cache'),
                      ('MPLCONFIGDIR', 'matplotlib'), ('QD_OFFLINE_CACHE_FILE_PATH', 'quadrants_cache')]:
        os.environ[key] = str(args.output/name)
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ['SOFTSIM_STUDY_RENDER_DIR'] = str(args.output/'render')
    import trimesh
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'code/scripts'))
    import genesis_behavior_runner as runner
    import mesh_visual_binding
    if args.inject_missing_vis_mode:
        original_bind = mesh_visual_binding.bind_entity

        def invisible_bind(gs, entity, *a, **kw):
            result = original_bind(gs, entity, *a, **kw)
            entity.surface.vis_mode = None
            return result
        mesh_visual_binding.bind_entity = invisible_bind
    mesh = (trimesh.creation.icosphere(subdivisions=1) if args.mode == 'solid'
            else trimesh.creation.box(extents=[1, .7, .01]))
    mesh.visual.vertex_colors = [195, 66, 26, 255]
    source = args.output/'source.ply'
    mesh.export(source)
    asset = {'sample_id': 'gpu_render_smoke_'+args.mode, 'geometry_family': args.mode,
             'resolved_artifacts': {'collision_mesh': str(source), 'visual_mesh': str(source)},
             'route': {'geometry_family': args.mode,
                       'model_family': 'mpm_elastic' if args.mode == 'solid' else 'pbd_cloth'},
             'fixed_solver_numerics': {'scale_m': .24, 'particle_size_m': .025, 'dt_s': .001,
                                       'substeps': 4, 'grid_density': 32, 'drop_height_m': .05}}
    probe = {'probe_id': 'gpu_render_smoke', 'stimulus': {'kind': 'compression', 'axis': 'x', 'orientation': 'flat'},
             'config': {'mesh_adaptation': 'v4', 'steps': 60, 'record_every': 4},
             'action': {'hold_steps': 4, 'release_steps': 4}}
    report = runner.run(asset, {}, probe, {}, args.output/'probe')
    report['validation_scope'] = 'GPU render integration only; excluded from rescue/control rates'
    report['injected_missing_vis_mode'] = args.inject_missing_vis_mode
    runner._write_report(args.output/'report.json', report)
    visibility = report.get('metrics', {}).get('render_visibility', {})
    print(json.dumps(visibility), flush=True)
    return 0 if visibility.get('valid') else 1


if __name__ == '__main__':
    raise SystemExit(main())
