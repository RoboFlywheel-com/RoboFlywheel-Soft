# SPDX-License-Identifier: Apache-2.0
"""Per-entity rest-curvature and sealed-volume constraints for Genesis PBD.

Installed only by the v5 review runner. No changes to site-packages or to the
historical cloth solver. Empty bags have no pressure or global shape pinning.
"""
from __future__ import annotations

import numpy as np
import quadrants as qd
import genesis as gs


def rest_dihedrals(vertices, inner_edges):
    p = np.asarray(vertices)[np.asarray(inner_edges)]
    e = p[:, 1]-p[:, 0]
    a, b = np.cross(e, p[:, 2]-p[:, 0]), np.cross(e, p[:, 3]-p[:, 0])
    denom = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
    return np.arccos(np.clip(np.sum(a*b, axis=1)/np.maximum(denom, 1e-30), -1, 1))


@qd.data_oriented
class ShellConstraints:
    def __init__(self, entity, *, sealed=False, iterations=4, scale=.4):
        self.entity, self.solver = entity, entity.solver
        self.iterations = iterations
        self.start = int(entity._particle_start)
        self.n = len(entity._particles)
        self.edge_start = int(entity._inner_edge_start)
        self.h = len(entity._inner_edges)
        self.max_step = .02*scale
        self.area_epsilon = 1e-10*scale**2
        angles = rest_dihedrals(entity._particles, entity._inner_edges)
        self.angles = qd.field(gs.qd_float, shape=max(1, self.h))
        self.angles.from_numpy(np.asarray(angles if self.h else [0.], dtype=gs.np_float))
        self.sealed = sealed
        if sealed:
            from mesh_adaptation import face_component_labels
            labels = face_component_labels(entity._mesh.trimesh, np)
            faces = np.asarray(entity._mesh.faces)
            pts = np.asarray(entity._particles)[faces]
            volumes = np.bincount(labels, weights=np.einsum('ij,ij->i', pts[:, 0], np.cross(pts[:, 1], pts[:, 2]))/6)
            if np.any(volumes <= 1e-12):
                raise ValueError('sealed membrane requires positive oriented component volumes')
            self.nc = len(volumes)
            self.nf = len(faces)
            self.faces = qd.Vector.field(3, gs.qd_int, shape=self.nf)
            self.labels = qd.field(gs.qd_int, shape=self.nf)
            self.rest_volumes = qd.field(gs.qd_float, shape=self.nc)
            self.volumes = qd.field(gs.qd_float, shape=self.nc)
            self.denoms = qd.field(gs.qd_float, shape=self.nc)
            self.gradient = qd.Vector.field(3, gs.qd_float, shape=self.n)
            self.vertex_component = qd.field(gs.qd_int, shape=self.n)
            vc = np.zeros(self.n, dtype=gs.np_int)
            vc[faces.ravel()] = np.repeat(labels, 3)
            self.faces.from_numpy(faces.astype(gs.np_int))
            self.labels.from_numpy(labels.astype(gs.np_int))
            self.rest_volumes.from_numpy(volumes.astype(gs.np_float))
            self.vertex_component.from_numpy(vc)
        self.diagnostics = {'method': 'native_PBD_hinge_projection_with_source_rest_angles',
                            'hinges': self.h, 'nonflat_rest_hinges': int(np.sum(np.abs(angles-np.pi)>.05)),
                            'rest_angle_range_rad': [float(angles.min()), float(angles.max())] if self.h else [],
                            'global_shape_matching': False, 'sealed_volume_constraint': sealed,
                            'limitation': 'unsigned local dihedrals; sealed volume is a compliance surrogate, not gas thermodynamics'}

    def install(self):
        if len(self.solver.entities) != 1:
            raise ValueError('v5 review shell constraints require one PBD entity per scene')
        self.solver._kernel_solve_bending = self.step

    def step(self, f):
        self.bend()
        if self.sealed:
            for _ in range(2):
                self.volume_gradient()
                self.volume_project()

    @qd.kernel
    def bend(self):
        for _ in qd.static(range(self.iterations)):
            for local in range(self.h):
                ie = local+self.edge_start
                a = self.solver.inner_edges_info[ie].v1
                b = self.solver.inner_edges_info[ie].v2
                c = self.solver.inner_edges_info[ie].v3
                d = self.solver.inner_edges_info[ie].v4
                w1 = self.solver.particles[a, 0].free/self.solver.particles_info[a].mass
                w2 = self.solver.particles[b, 0].free/self.solver.particles_info[b].mass
                w3 = self.solver.particles[c, 0].free/self.solver.particles_info[c].mass
                w4 = self.solver.particles[d, 0].free/self.solver.particles_info[d].mass
                e = self.solver.particles[b, 0].pos-self.solver.particles[a, 0].pos
                p = self.solver.particles[c, 0].pos-self.solver.particles[a, 0].pos
                r = self.solver.particles[d, 0].pos-self.solver.particles[a, 0].pos
                l1, l2 = e.cross(p).norm(), e.cross(r).norm()
                if l1 > self.area_epsilon and l2 > self.area_epsilon:
                    n1, n2 = e.cross(p)/l1, e.cross(r)/l2
                    cosine = qd.math.clamp(n1.dot(n2), -1., 1.)
                    q3 = (e.cross(n2)+n1.cross(e)*cosine)/l1
                    q4 = (e.cross(n1)+n2.cross(e)*cosine)/l2
                    q2 = -(p.cross(n2)+n1.cross(p)*cosine)/l1-(r.cross(n1)+n2.cross(r)*cosine)/l2
                    q1 = -q2-q3-q4
                    denom = w1*q1.norm_sqr()+w2*q2.norm_sqr()+w3*q3.norm_sqr()+w4*q4.norm_sqr()
                    alpha = self.solver.inner_edges_info[ie].bending_compliance/self.solver._substep_dt**2
                    value = -qd.sqrt(qd.max(0., 1-cosine*cosine))*(qd.acos(cosine)-self.angles[local])
                    value *= self.solver.inner_edges_info[ie].bending_relaxation/(denom+alpha+1e-20)
                    largest = qd.max(w1*q1.norm(), w2*q2.norm(), w3*q3.norm(), w4*q4.norm())*qd.abs(value)
                    value *= qd.min(1., self.max_step/(largest+1e-20))
                    self.solver.particles[a, 0].dpos += w1*value*q1
                    self.solver.particles[b, 0].dpos += w2*value*q2
                    self.solver.particles[c, 0].dpos += w3*value*q3
                    self.solver.particles[d, 0].dpos += w4*value*q4
            for i in range(self.n):
                j = i+self.start
                delta = self.solver.particles[j, 0].dpos
                delta *= qd.min(1., self.max_step/(delta.norm()+1e-20))
                self.solver.particles[j, 0].pos += delta
                self.solver.particles[j, 0].dpos.fill(0)

    @qd.kernel
    def volume_gradient(self):
        for c in range(self.nc):
            self.volumes[c] = 0
            self.denoms[c] = 0
        for i in range(self.n):
            self.gradient[i].fill(0)
        for f in range(self.nf):
            a,b,c = self.faces[f][0], self.faces[f][1], self.faces[f][2]
            x = self.solver.particles[a+self.start, 0].pos
            y = self.solver.particles[b+self.start, 0].pos
            z = self.solver.particles[c+self.start, 0].pos
            self.volumes[self.labels[f]] += x.dot(y.cross(z))/6
            self.gradient[a] += y.cross(z)/6
            self.gradient[b] += z.cross(x)/6
            self.gradient[c] += x.cross(y)/6
        for i in range(self.n):
            component = self.vertex_component[i]
            self.gradient[i] /= self.rest_volumes[component]
            w = self.solver.particles[i+self.start, 0].free/self.solver.particles_info[i+self.start].mass
            self.denoms[component] += w*self.gradient[i].norm_sqr()

    @qd.kernel
    def volume_project(self):
        for i in range(self.n):
            component = self.vertex_component[i]
            constraint = self.volumes[component]/self.rest_volumes[component]-1
            alpha = .01/self.solver._substep_dt**2
            w = self.solver.particles[i+self.start, 0].free/self.solver.particles_info[i+self.start].mass
            delta = -w*constraint/(self.denoms[component]+alpha)*self.gradient[i]
            delta *= qd.min(1., self.max_step/(delta.norm()+1e-20))
            self.solver.particles[i+self.start, 0].pos += delta
