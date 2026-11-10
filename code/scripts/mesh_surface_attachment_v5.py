# SPDX-License-Identifier: Apache-2.0
"""Visual-only local-frame offsets for PBD surface attachments."""
from __future__ import annotations

import numpy as np


def triangle_frames(triangles):
    edge=triangles[:,1]-triangles[:,0]
    tangent=edge/np.maximum(np.linalg.norm(edge,axis=1,keepdims=True),1e-20)
    normal=np.cross(edge,triangles[:,2]-triangles[:,0])
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-20)
    return np.stack((tangent,np.cross(normal,tangent),normal),axis=1)


def attachment_offsets(vertices,particles,indices,weights):
    frames=triangle_frames(particles[indices[:,:3]])
    projected=np.einsum('ni,nij->nj',weights,particles[indices])
    offsets=np.einsum('nij,nj->ni',frames,vertices-projected)
    restored=projected+np.einsum('ni,nij->nj',offsets,frames)
    return offsets, float(np.linalg.norm(restored-vertices,axis=1).max(initial=0))


def install(entity,indices,offsets,gs):
    """Update render positions only; particle positions/velocities stay untouched."""
    import quadrants as qd

    @qd.data_oriented
    class SurfaceOffsets:
        def __init__(self):
            self.solver=entity.solver
            self.n=len(indices)
            self.start=int(entity._vvert_start)
            self.ids=qd.Vector.field(3,gs.qd_int,shape=self.n)
            self.local=qd.Vector.field(3,gs.qd_float,shape=self.n)
            self.ids.from_numpy((indices[:,:3]+int(entity._particle_start)).astype(gs.np_int))
            self.local.from_numpy(offsets.astype(gs.np_float))

        @qd.kernel
        def update(self):
            for i,b in qd.ndrange(self.n,self.solver._B):
                a=self.solver.particles[self.ids[i][0],b].pos
                edge=self.solver.particles[self.ids[i][1],b].pos-a
                third=self.solver.particles[self.ids[i][2],b].pos-a
                tangent=edge/qd.max(edge.norm(),1e-20)
                normal=edge.cross(third)
                normal/=qd.max(normal.norm(),1e-20)
                offset=self.local[i][0]*tangent+self.local[i][1]*normal.cross(tangent)+self.local[i][2]*normal
                self.solver.vverts_render[self.start+i,b].pos+=offset

    adapter=SurfaceOffsets()
    original=entity.solver._kernel_update_render_fields
    def update(f):
        original(f)
        adapter.update()
    entity.solver._kernel_update_render_fields=update
    entity._review_surface_offsets=adapter
