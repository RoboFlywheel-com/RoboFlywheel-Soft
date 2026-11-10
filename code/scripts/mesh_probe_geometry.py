# SPDX-License-Identifier: Apache-2.0
"""Geometry-driven compression stimulus and its audit, opt-in v4 only."""
from __future__ import annotations
import numpy as np


class PrescribedPlateMotion:
    """Keep the six-DOF rigid plates on their declared linear trajectories.

    Velocity-only driving lets gravity/contact rotate and translate plates
    off axis. Reassert the full pose each step, then integrate the prescribed
    velocity. There is no response-dependent parameter fitting or extra travel.
    """
    def __init__(self, plates, axis, dt, travel, compress_steps, hold_steps, release_steps):
        self.plates, self.axis, self.dt = plates, axis, dt
        self.travel = travel
        self.compress_steps, self.hold_steps, self.release_steps = compress_steps, hold_steps, release_steps
        self.origins = [p.get_pos().detach().cpu().numpy().reshape(3).copy() for p in plates]
        self.max_tracking_error = 0.

    def displacement(self, step):
        if step <= self.compress_steps:
            return self.travel * step / self.compress_steps
        if step <= self.compress_steps + self.hold_steps:
            return self.travel
        release = step - self.compress_steps - self.hold_steps
        return self.travel * max(0., 1 - release / self.release_steps)

    def apply(self, step):
        displacement = self.displacement(step)
        velocity = (self.displacement(step + 1) - displacement) / self.dt
        for sign, plate, origin in zip((1., -1.), self.plates, self.origins):
            target = origin.copy()
            target[self.axis] -= sign * displacement
            current = plate.get_pos().detach().cpu().numpy().reshape(3)
            self.max_tracking_error = max(self.max_tracking_error, float(np.linalg.norm(current-target)))
            plate.set_pos(target, zero_velocity=False)
            plate.set_quat([1., 0., 0., 0.], zero_velocity=False)
            dofs = np.zeros(6)
            dofs[self.axis] = -sign * velocity
            plate.set_dofs_velocity(dofs)

    def diagnostics(self):
        return {'method': 'prescribed_full_pose_and_axis_velocity', 'travel_per_side_m': self.travel,
                'maximum_pre_reset_tracking_error_m': self.max_tracking_error,
                'feedback_to_material_parameters': False, 'response_dependent_extra_travel': False}


def contact_coverage(positions, plate_axis, axis, plate_size, plate_center_z, eps):
    """Finite plate footprint contact; an axial extreme alone is insufficient."""
    positions = np.asarray(positions)
    if not len(positions) or not np.isfinite(positions).all():
        return {'contacting_particles': 0, 'opposed_contact': False}
    orthogonal = [a for a in range(3) if a != axis]
    footprint = np.ones(len(positions), dtype=bool)
    for a in orthogonal:
        center = plate_center_z if a == 2 else 0.
        footprint &= np.abs(positions[:, a] - center) <= .5 * plate_size[a] + eps
    contacts = []
    for sign, center in zip((1., -1.), plate_axis):
        inner = center - sign * .5 * plate_size[axis]
        contacts.append(footprint & (np.abs(positions[:, axis] - inner) <= eps))
    return {'contacting_particles': int(np.sum(contacts[0] | contacts[1])),
            'opposed_contact': bool(contacts[0].any() and contacts[1].any())}
