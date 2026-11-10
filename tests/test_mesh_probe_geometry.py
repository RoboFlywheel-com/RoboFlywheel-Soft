# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code/scripts'))
import genesis_behavior_runner as runner
from mesh_probe_geometry import PrescribedPlateMotion, contact_coverage


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.values


class Plate:
    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=float)
    def get_pos(self):
        return Tensor(self.pos.copy())
    def set_pos(self, pos, **kwargs):
        self.pos = np.asarray(pos)
    def set_quat(self, quat, **kwargs):
        self.quat = quat
    def set_dofs_velocity(self, velocity):
        self.velocity = np.asarray(velocity)


def test_prescribed_motion_restores_pose_and_retracts_to_original_gap():
    plates = [Plate([1, 0, .1]), Plate([-1, 0, .1])]
    motion = PrescribedPlateMotion(plates, 0, .01, .2, 10, 2, 4)
    for step in range(20):
        # Deliberately disturb all axes; velocity-only driving cannot repair this.
        for plate in plates:
            plate.pos += [.03, .07, -.02]
        motion.apply(step)
        for p in plates:
            assert p.pos[1] == 0
            assert p.pos[2] == pytest.approx(.1)
            assert p.quat == [1., 0., 0., 0.]
            p.pos += .01 * p.velocity[:3]
    assert plates[0].pos[0] == pytest.approx(1)
    assert plates[1].pos[0] == pytest.approx(-1)
    assert motion.displacement(10) == pytest.approx(.2)
    assert motion.diagnostics()['response_dependent_extra_travel'] is False


def test_axial_extreme_outside_plate_footprint_is_not_contact():
    out = contact_coverage([[.45,2,0],[-.45,2,0]], [.5,-.5], 0, [.1,1,1], .5, .01)
    assert out['contacting_particles'] == 0
    inside = contact_coverage([[.45,0,.2],[-.45,0,.2]], [.5,-.5], 0, [.1,1,1], .5, .01)
    assert inside['opposed_contact']


def test_v4_is_opt_in_and_does_not_change_legacy_flags():
    assert not runner._mesh_adaptation_v4_enabled({}, {})
    assert not runner._mesh_adaptation_v4_enabled({'config': {'mesh_adaptation':'v3'}}, {})
    assert runner._mesh_adaptation_v4_enabled({}, {'mesh_adaptation':'v4'})
    assert not runner._mesh_adaptation_v3_enabled({}, {'mesh_adaptation':'v4'})


def test_proxy_rejection_keeps_auditable_evidence_and_no_material_patch():
    probe = {'required_metrics':['finite_fraction','settled']}
    report = runner._base_report({}, {}, probe, {})
    details = {'defects':['self_intersecting_proxy'], 'sampling': {'actual_particle_count':0}}
    rejected = runner._proxy_rejection(report, details, 'after_sampling')
    assert rejected['execution_status'] == 'completed'
    assert rejected['metrics']['simulation_executed'] is False
    assert rejected['diagnosis']['kind'] == 'model_limit'
    assert rejected['diagnosis']['suggested_updates'] == {}
    assert rejected['metrics']['mesh_adaptation'] == details


def test_legacy_soft_body_morph_defaults_unchanged():
    calls = []
    gs = SimpleNamespace(morphs=SimpleNamespace(Mesh=lambda **kw: calls.append(kw) or kw),
                         surfaces=SimpleNamespace(Default=lambda **kw: kw))
    scene = SimpleNamespace(add_entity=lambda *a, **kw: None)
    runner._add_soft_body(scene, gs, Path('/tmp/proxy.obj'), None, .1)
    assert calls[-1] == {'file':'/tmp/proxy.obj','pos':(0.,0.,.1),'scale':1.,'decimate':False,
                         'watertighten':None,'collision':True,'file_meshes_are_zup':True}
    runner._add_soft_body(scene, gs, Path('/tmp/proxy.obj'), None, .1, adapted=True)
    assert calls[-1]['convexify'] is False
