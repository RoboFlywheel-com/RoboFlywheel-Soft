# SPDX-License-Identifier: Apache-2.0
"""Genesis-side worker for ``softsim_genesis_runner``.

Executed as ``$GENESIS_PYTHON genesis_worker.py --job job.json --result
result.json`` by the adapter.  This file must stay self-contained: the pinned
Genesis 1.3.3 venv has no SoftSim packages installed, so only the standard
library, ``genesis`` and ``torch`` may be imported here.

Job kinds:

- ``e1_runtime``: import the mesh, initialize MPM.Elastic/ElastoPlastic,
  settle for a fixed number of steps and report the runtime contract metrics
  (finite/speed/escape/penetration), mirroring the v12 ``stability_drop_v1``
  evidence shape.
- ``e2_probe``: run a frozen compression stimulus for the candidate
  parameters and, evaluator-side, for the hidden reference parameters, then
  report only scalar behavior losses and a structured residual report.  The
  reference parameter values never appear in the result payload.
- ``e3a_grasp``: scripted position-controlled two-finger
  approach/grasp/lift-30mm/hold with the frozen continuum damage surrogate;
  reports task/damage/integrity/slip/contact/convergence measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

GENESIS_VERSION_EXPECTED = "1.3.3"

DEFAULT_NUMERICS = {
    "dt_s": 5e-4,
    "substeps": 8,
    "grid_density": 192,
    "lower_bound": [-0.21, -0.21, -0.15],
    "upper_bound": [0.21, 0.21, 0.31],
    "sampler": "pbs",
}

# v12.6 frozen cloth numerics (see configs/method_contract_v12_6.json):
# PBD stretch compliance 1e-7, bending projection off, self-collision on,
# dt=0.003s, substeps=10.  Per-row solver_numerics override these defaults.
CLOTH_DEFAULTS = {
    "dt_s": 0.003,
    "substeps": 10,
    "stretch_compliance": 1e-7,
    "bending_compliance": 1.0,  # bending projection off
}

# Legacy route names observed in pre-v12.3 bundles map onto the frozen
# vocabulary (v12.6 contract: cloth = PBD.Cloth, ipc/uipc unused).
LEGACY_SOLVER_MAP = {
    "genesis_fem_cloth_ipc": "PBD.Cloth",
    "genesis_pbd_cloth": "PBD.Cloth",
    "genesis_mpm_elastoplastic": "MPM.ElastoPlastic",
    "genesis_mpm_elastic": "MPM.Elastic",
}


def _solver_route(job: Mapping[str, Any]) -> str:
    route = job.get("route") if isinstance(job.get("route"), Mapping) else {}
    raw = " ".join(str(route.get(key) or "") for key in ("solver_adapter", "model_family")).lower()
    for legacy, canonical in LEGACY_SOLVER_MAP.items():
        if legacy in raw:
            return canonical
    if "cloth" in raw:
        return "PBD.Cloth"
    if "elastic" in raw and "elastoplastic" not in raw and "elasto_plastic" not in raw:
        return "MPM.Elastic"
    return "MPM.ElastoPlastic"


def _material(parameters: Mapping[str, Any], solver_route: str) -> Any:
    import genesis as gs

    if solver_route == "PBD.Cloth":
        return gs.materials.PBD.Cloth(
            rho=float(parameters.get("density_kg_m3", 300.0)),
            stretch_compliance=float(CLOTH_DEFAULTS["stretch_compliance"]),
            bending_compliance=float(CLOTH_DEFAULTS["bending_compliance"]),
        )
    if solver_route == "MPM.Elastic":
        return gs.materials.MPM.Elastic(
            E=float(parameters.get("youngs_modulus_pa", 50000.0)),
            nu=float(parameters.get("poisson_ratio", 0.35)),
            rho=float(parameters.get("density_kg_m3", 800.0)),
            sampler=str(DEFAULT_NUMERICS["sampler"]),
        )
    # Yield default follows the production runner (PARAM_DEFAULTS: 10000 Pa);
    # the bundle's model_family already encodes the elastic/elastoplastic
    # routing decision, no local re-routing.
    return gs.materials.MPM.ElastoPlastic(
        E=float(parameters.get("youngs_modulus_pa", 50000.0)),
        nu=float(parameters.get("poisson_ratio", 0.35)),
        rho=float(parameters.get("density_kg_m3", 800.0)),
        yield_lower=float(parameters.get("von_mises_yield_stress_pa", parameters.get("yield_stress_pa", 10000.0))),
        yield_higher=float(parameters.get("von_mises_yield_stress_pa", parameters.get("yield_stress_pa", 10000.0))),
        use_von_mises=True,
        sampler=str(DEFAULT_NUMERICS["sampler"]),
    )


def _mesh_transform(job: Mapping[str, Any]) -> tuple[Any, tuple[float, float, float], dict[str, Any]]:
    """Compute morph scale/pos from the frozen scale contract.

    Priority: ``scale.meters_per_unit`` (asset v2 metric scale) multiplies raw
    coordinates; otherwise ``scale.scale_m`` / ``numerics.scale_m`` normalizes
    the longest extent.  The mesh is centered in x/y and its lowest point is
    placed at ``numerics.drop_height_m`` above the plane.
    """
    import trimesh

    mesh = trimesh.load(str(job["mesh_path"]), force="scene", process=False)
    bounds = mesh.bounds
    extent = [float(v) for v in (bounds[1] - bounds[0]).tolist()]
    longest = max(extent) if extent else 1.0
    scale_info: dict[str, Any] = {"raw_extent_m": [round(v, 8) for v in extent]}
    scale_cfg = job.get("scale") if isinstance(job.get("scale"), Mapping) else {}
    numerics = job.get("numerics") if isinstance(job.get("numerics"), Mapping) else {}
    # Production sizing rule (v12.6 contract, routing.sizing_numerics): the
    # frozen per-asset scale_m wins; asset metric_scale is the fallback.
    target = scale_cfg.get("scale_m") or numerics.get("scale_m")
    mpu = scale_cfg.get("meters_per_unit")
    if target:
        factor = float(target) / max(longest, 1e-9)
        scale_info["scale_mode"] = "longest_extent_to_scale_m"
    elif mpu:
        factor = float(mpu)
        scale_info["scale_mode"] = "meters_per_unit"
    else:
        factor = 1.0
        scale_info["scale_mode"] = "as_authored"
    scale_info["scale_factor"] = round(factor, 10)
    drop = float(numerics.get("drop_height_m", 0.02))
    center = [(float(bounds[0][i]) + float(bounds[1][i])) / 2.0 for i in range(3)]
    pos = (-center[0] * factor, -center[1] * factor, -float(bounds[0][2]) * factor + drop)
    scale_info["drop_height_m"] = drop
    return factor, pos, scale_info

# Continuum damage/stiffness-loss surrogate bands per fragility regime.  These
# are compressive-strain onset/failure points of the surrogate law, not
# measured fracture energies.
DAMAGE_SURROGATE = {
    "low": {"onset": 0.35, "failure": 0.75},
    "medium": {"onset": 0.25, "failure": 0.60},
    "high": {"onset": 0.15, "failure": 0.45},
}


def _build_scene(job: Mapping[str, Any], parameters: Mapping[str, Any]):
    import genesis as gs

    solver_route = _solver_route(job)
    numerics = dict(DEFAULT_NUMERICS)
    if solver_route == "PBD.Cloth":
        # v12.6 contract pins cloth numerics; they supersede per-row values
        # that predate the v5 cloth fix.
        numerics["dt_s"] = CLOTH_DEFAULTS["dt_s"]
        numerics["substeps"] = CLOTH_DEFAULTS["substeps"]
        numerics["numerics_source"] = "contract_cloth_v12_6"
        row_override = {k: v for k, v in (job.get("numerics") or {}).items() if k in {"grid_density", "lower_bound", "upper_bound", "drop_height_m", "scale_m"}}
        numerics.update(row_override)
    else:
        numerics["numerics_source"] = "row_frozen"
        numerics.update({k: v for k, v in (job.get("numerics") or {}).items() if k in DEFAULT_NUMERICS or k in {"drop_height_m", "particle_size_m"}})
    factor, pos, scale_info = _mesh_transform(job)
    scaled_longest = max(scale_info["raw_extent_m"]) * factor if scale_info["raw_extent_m"] else 0.1
    # Production scene contract (genesis_behavior_runner._make_scene): domain
    # derived from the scaled asset length with margin padding, particle size
    # from the frozen per-asset numerics, gravity on the z axis.
    particle_size = float(numerics.get("particle_size_m") or 0.01)
    margin = max(0.5 * scaled_longest, 0.15, 8.0 * particle_size)
    if "lower_bound" not in (job.get("numerics") or {}):
        numerics["lower_bound"] = [-0.75 * scaled_longest - margin, -0.75 * scaled_longest - margin, -margin]
    if "upper_bound" not in (job.get("numerics") or {}):
        numerics["upper_bound"] = [0.75 * scaled_longest + margin, 0.75 * scaled_longest + margin, 2.5 * scaled_longest + margin]
    scene_kwargs: dict[str, Any] = {}
    if job.get("rigid_timeconst") is not None:
        # Kinematic grippers need the default-strength contact constraint;
        # the MPM-probe value (2*dt/substeps) lets contact stall the aperture
        # servo at the body surface (v12.5 production finding).
        scene_kwargs["rigid_options"] = gs.options.RigidOptions(constraint_timeconst=float(job["rigid_timeconst"]))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=float(numerics["dt_s"]), substeps=int(numerics["substeps"]), gravity=(0.0, 0.0, -9.81)),
        mpm_options=gs.options.MPMOptions(
            dt=float(numerics["dt_s"]),
            gravity=(0.0, 0.0, -9.81),
            particle_size=particle_size,
            grid_density=int(numerics["grid_density"]),
            lower_bound=tuple(numerics["lower_bound"]),
            upper_bound=tuple(numerics["upper_bound"]),
            enable_CPIC=False,
        ),
        coupler_options=gs.options.LegacyCouplerOptions(),
        show_viewer=False,
        **scene_kwargs,
    )
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=0.5))
    effective_route = solver_route
    entity = scene.add_entity(
        gs.morphs.Mesh(file=str(job["mesh_path"]), pos=pos, scale=factor),
        material=_material(parameters, solver_route),
    )
    numerics["solver_route"] = effective_route
    numerics["scale_info"] = scale_info
    return scene, entity, numerics


def _particle_metrics(entity, initial_pos, world: Mapping[str, Any], prev_pos: Any = None, dt_frame: float | None = None) -> dict[str, Any]:
    import torch

    pos = entity.get_particles_pos()
    vel = entity.get_particles_vel()
    finite_pos = torch.isfinite(pos).all(dim=-1)
    vel_ok = torch.isfinite(vel).all(dim=-1)
    velocity_source = "solver"
    if not bool(vel_ok.any().item()):
        # PBD entities expose NaN velocities (production runner documents this);
        # finite-difference from the previous sample instead of poisoning the
        # record with inf.
        if prev_pos is not None and dt_frame:
            vel = (pos - prev_pos) / float(dt_frame)
            velocity_source = "finite_difference"
        else:
            vel = torch.zeros_like(pos)
            velocity_source = "unavailable_pbd_nan"
        vel_ok = torch.isfinite(vel).all(dim=-1)
    # The finite gate is position-based; velocity NaN alone (PBD) is not a
    # numerical failure.
    finite = finite_pos & vel_ok if velocity_source == "solver" else finite_pos
    # torch.mean multiplies by a rounded 1/n; exact integer division keeps an
    # all-True fraction at exactly 1.0 so the finite gate is not tripped by
    # floating-point artifacts.
    finite_fraction = float(int(finite.sum().item())) / max(1, finite.numel())
    safe_pos = torch.where(torch.isfinite(pos), pos, torch.zeros_like(pos))
    speed = vel.norm(dim=-1)
    speed = torch.where(torch.isfinite(speed), speed, torch.zeros_like(speed))
    lower = torch.tensor(world["lower_bound"], dtype=safe_pos.dtype, device=safe_pos.device)
    upper = torch.tensor(world["upper_bound"], dtype=safe_pos.dtype, device=safe_pos.device)
    escaped = ((safe_pos < lower) | (safe_pos > upper)).any(dim=-1)
    extent0 = (initial_pos.max(dim=0).values - initial_pos.min(dim=0).values).clamp(min=1e-9)
    extent1 = safe_pos.max(dim=0).values - safe_pos.min(dim=0).values
    return {
        "finite_fraction": finite_fraction,
        "max_speed_m_s": float(speed.max().item()),
        "mean_speed_m_s": float(speed.mean().item()),
        "velocity_source": velocity_source,
        "escaped_fraction": float(int(escaped.sum().item())) / max(1, escaped.numel()),
        "penetration_m": float(max(0.0, -float(safe_pos[:, 2].min().item()))),
        "centroid_m": [float(v) for v in safe_pos.mean(dim=0).tolist()],
        "shape_extent_ratio": [float(v) for v in (extent1 / extent0).tolist()],
        "particle_count": int(pos.shape[0]),
        "min_z_m": float(safe_pos[:, 2].min().item()),
        "max_z_m": float(safe_pos[:, 2].max().item()),
    }


def _base_provenance(job: Mapping[str, Any], numerics: Mapping[str, Any]) -> dict[str, Any]:
    import genesis as gs

    return {
        "engine": "genesis",
        "engine_version": gs.__version__,
        "solver_route": str(numerics.get("solver_route", "MPM.ElastoPlastic")),
        "coupler": "LegacyCoupler",
        "ipc_used": False,
        "uipc_used": False,
        "dt_s": float(numerics["dt_s"]),
        "substeps": int(numerics["substeps"]),
        "grid_density": int(numerics["grid_density"]),
        "particle_size_m": float(numerics.get("particle_size_m") or 0.01),
        "sampler": str(numerics["sampler"]),
        "scale_info": numerics.get("scale_info"),
        "claim_boundary": "Genesis 1.3.3 proxy behavior evidence; not measured material ground truth",
    }


def _run_e1_runtime(job: Mapping[str, Any]) -> dict[str, Any]:
    scene, entity, numerics = _build_scene(job, job.get("parameters") or {})
    try:
        scene.build()
        initial = entity.get_particles_pos().clone()
        drop = float(numerics.get("drop_height_m", 0.02))
        dt = float(numerics["dt_s"]) * int(numerics["substeps"])
        # Cover free-fall from the frozen drop height plus a settle window.
        steps = int(job.get("steps", 0)) or max(240, int((math.sqrt(2.0 * max(drop, 0.0) / 9.8) + 0.4) / max(dt, 1e-6)))
        record_every = max(1, steps // 12)
        started = time.time()
        latest: dict[str, Any] = {}
        max_speed_seen = 0.0
        settle_speed = 0.0
        settle_from = steps - max(1, steps // 4)
        dt_sample = float(numerics["dt_s"]) * int(numerics["substeps"]) * record_every
        prev_pos = None
        for index in range(steps):
            scene.step()
            if (index + 1) % record_every == 0 or index == steps - 1:
                cur_pos = entity.get_particles_pos().clone()
                latest = _particle_metrics(entity, initial, numerics, prev_pos=prev_pos, dt_frame=dt_sample)
                prev_pos = cur_pos
                max_speed_seen = max(max_speed_seen, float(latest["max_speed_m_s"]))
                if index >= settle_from:
                    settle_speed = max(settle_speed, float(latest["max_speed_m_s"]))
        wall = time.time() - started
        latest["max_speed_m_s"] = max_speed_seen
        latest["settle_speed_m_s"] = settle_speed
        latest["gpu_seconds"] = round(wall, 6)
        latest.update(_base_provenance(job, numerics))
        return {"status": "ok", "probe_id": job.get("probe_id", "e1.runtime.stability.v1"), "metrics": latest}
    finally:
        scene.destroy()


def _compression_rollout(job: Mapping[str, Any], parameters: Mapping[str, Any], stimulus: Mapping[str, Any]) -> dict[str, Any]:
    """Quasi-static top-plate compression + release; returns behavior metrics."""
    import genesis as gs

    scene, entity, numerics = _build_scene(job, parameters)
    try:
        ratio = float(stimulus.get("compression_ratio", 0.22))
        hold_steps = int(stimulus.get("hold_steps", 28))
        release_steps = int(stimulus.get("release_steps", 28))
        press_steps = int(stimulus.get("press_steps", 40))
        plate = scene.add_entity(
            gs.morphs.Box(pos=(0.0, 0.0, 0.4), size=(0.3, 0.3, 0.01)),
            material=gs.materials.Rigid(friction=0.5),
        )
        scene.build()
        pos0 = entity.get_particles_pos()
        top0 = float(pos0[:, 2].max().item())
        bottom0 = float(pos0[:, 2].min().item())
        height0 = max(1e-9, top0 - bottom0)
        z_rest = top0 + 0.005
        z_press = top0 - ratio * height0
        plate.set_pos((0.0, 0.0, z_rest), zero_velocity=True)
        min_height = height0
        for k in range(press_steps):
            z = z_rest + (z_press - z_rest) * (k + 1) / press_steps
            plate.set_pos((0.0, 0.0, z), zero_velocity=True)
            scene.step()
            pos = entity.get_particles_pos()
            min_height = min(min_height, float(pos[:, 2].max().item() - pos[:, 2].min().item()))
        for _ in range(hold_steps):
            scene.step()
            pos = entity.get_particles_pos()
            min_height = min(min_height, float(pos[:, 2].max().item() - pos[:, 2].min().item()))
        for k in range(release_steps):
            z = z_press + (z_rest - z_press) * (k + 1) / release_steps
            plate.set_pos((0.0, 0.0, z), zero_velocity=True)
            scene.step()
        settle = int(stimulus.get("settle_steps", 60))
        prev_pos = None
        for _ in range(settle):
            prev_pos = entity.get_particles_pos().clone()
            scene.step()
        final = _particle_metrics(entity, pos0, numerics, prev_pos=prev_pos, dt_frame=float(numerics["dt_s"]) * int(numerics["substeps"]))
        recovered_height = final["max_z_m"] - final["min_z_m"]
        deformation_ratio = min_height / height0
        metrics = {
            "deformation_ratio": round(deformation_ratio, 8),
            "recovery_ratio": round(recovered_height / height0, 8),
            "settled": bool(final["max_speed_m_s"] < 0.05),
            "finite_fraction": final["finite_fraction"],
            "max_speed_m_s": final["max_speed_m_s"],
            "escaped_fraction": final["escaped_fraction"],
        }
        metrics.update(_base_provenance(job, numerics))
        return metrics
    finally:
        scene.destroy()


def _behavior_loss(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    keys = ("deformation_ratio", "recovery_ratio", "settled", "finite_fraction")
    total = 0.0
    residual: dict[str, Any] = {}
    for key in keys:
        c = candidate.get(key)
        r = reference.get(key)
        if isinstance(c, bool) or isinstance(r, bool):
            diff = 0.0 if bool(c) == bool(r) else 1.0
        else:
            diff = abs(float(c) - float(r)) / max(abs(float(r)), 1e-6)
            diff = min(1.0, diff)
        residual[key] = {"candidate": c, "reference": r}
        total += diff
    return total / len(keys), residual


def _run_e2_probe(job: Mapping[str, Any]) -> dict[str, Any]:
    stimulus = dict(job.get("stimulus") or {"kind": "compression", "compression_ratio": 0.22})
    candidate_metrics = _compression_rollout(job, job.get("candidate_parameters") or {}, stimulus)
    reference_metrics = _compression_rollout(job, job.get("evaluator_parameters") or {}, stimulus)
    loss, residual = _behavior_loss(candidate_metrics, reference_metrics)
    return {
        "status": "ok",
        "probe_id": job.get("probe_id"),
        "metrics": {
            "behavior_loss": round(loss, 8),
            "finite_fraction": candidate_metrics["finite_fraction"],
            "candidate_metrics": {key: candidate_metrics[key] for key in ("deformation_ratio", "recovery_ratio", "settled")},
            "residual_report": residual,
            "engine": candidate_metrics["engine"],
            "engine_version": candidate_metrics["engine_version"],
            "solver_route": candidate_metrics["solver_route"],
            "coupler": candidate_metrics["coupler"],
            "ipc_used": False,
            "uipc_used": False,
        },
    }


def _grasp_episode(job: Mapping[str, Any], closure_ratio: float, *, reactive_slip: bool = False) -> dict[str, Any]:
    """One scripted approach/grasp/lift-30mm/hold episode at a fixed closure.

    ``reactive_slip=True`` runs the aperture-reactive controller: when the
    object centroid sags behind the finger plane during lift/hold, the grip
    tightens in bounded steps (width/kinematic slip response; no force stream
    exists under the frozen coupler, so force-reactive control is out of
    contract).
    """
    import genesis as gs

    regime = str(job.get("fragility_regime", "medium"))
    band = DAMAGE_SURROGATE.get(regime, DAMAGE_SURROGATE["medium"])
    scene, entity, numerics = _build_scene(job, job.get("parameters") or {})
    try:
        finger_size = (0.04, 0.08, 0.04)
        left = scene.add_entity(gs.morphs.Box(pos=(-0.2, 0.0, 0.06), size=finger_size), material=gs.materials.Rigid(friction=2.0))
        right = scene.add_entity(gs.morphs.Box(pos=(0.2, 0.0, 0.06), size=finger_size), material=gs.materials.Rigid(friction=2.0))
        scene.build()
        # Let the object come to rest before measuring the grasp frame: the
        # frozen drop height in row numerics targets the drop probe, and the
        # grasp pose must be derived from the *settled* geometry.
        settle_steps = int(job.get("settle_steps", 120))
        for _ in range(settle_steps):
            scene.step()
        pos0 = entity.get_particles_pos()
        cx0 = pos0.mean(dim=0)
        min0 = pos0.min(dim=0).values
        max0 = pos0.max(dim=0).values
        width0 = float(max0[0] - min0[0])
        height0 = max(1e-9, float(max0[2] - min0[2]))
        y_depth0 = float(max0[1] - min0[1])
        z_grasp = float(min0[2]) + 0.5 * height0
        closure_ratio = min(0.6, max(0.05, float(closure_ratio)))
        gap0 = width0 * (1.0 - closure_ratio) + finger_size[0]
        steps_per_phase = int(job.get("steps_per_phase", 240))
        lift_m = 0.03
        hold_steps = int(job.get("hold_steps", 120))
        initial = pos0.clone()
        contact_steps = 0
        max_compression = 0.0
        max_strain = 0.0
        continuum_peak = [0.0]
        observe_count = [0]
        slip_max = 0.0
        lift_gain = 0.0
        held = True
        reactive_tightenings = 0
        z_lift_ref: list[float] = [float("nan")]

        # Velocity servo with *actual-position* feedback: a set_dofs_velocity
        # command only persists one substep (dt), so recompute from the real
        # finger pose every step; this also holds z against gravity.
        dt_sub = float(numerics["dt_s"])
        y_center = float(cx0[1])
        gap_state = {"gap": gap0}
        demo_path = job.get("demo_log")
        demo_handle = open(str(demo_path), "w", encoding="utf-8") if demo_path else None
        demo_step = {"n": 0}

        def _demo_log(phase: str, target_z: float, prev_gap: float) -> None:
            if demo_handle is None:
                return
            pos = entity.get_particles_pos()
            n = pos.shape[0]
            stride = max(1, n // 1024)
            cloud = pos[::stride][:1024].detach().cpu().tolist()
            centroid = pos.mean(dim=0)
            fl = left.get_pos(); fr = right.get_pos()
            ee = [(float(fl[i]) + float(fr[i])) / 2.0 for i in range(3)]
            row = {
                "step": demo_step["n"],
                "phase": phase,
                "pointcloud": cloud,
                "centroid_m": [float(v) for v in centroid],
                "ee_pose": ee + [1.0, 0.0, 0.0, 0.0],
                "gripper_width_m": float(fr[0] - fl[0]) - finger_size[0],
                "action_7d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(prev_gap - gap_state["gap"])],
            }
            demo_handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            demo_step["n"] += 1

        def _fingers(xl: float, xr: float, z: float) -> None:
            for finger, tx in ((left, xl), (right, xr)):
                cur = finger.get_pos()
                targets = (tx, y_center, z)
                vel = [max(-0.5, min(0.5, (targets[i] - float(cur[i])) / dt_sub)) for i in range(3)]
                finger.set_dofs_velocity(vel + [0.0, 0.0, 0.0])

        def _observe(in_hold: bool, finger_z: float, reactive_armed: bool = False) -> None:
            nonlocal contact_steps, max_compression, max_strain, slip_max, held, reactive_tightenings
            observe_count[0] += 1
            pos = entity.get_particles_pos()
            centroid = pos.mean(dim=0)
            cur_min = pos.min(dim=0).values
            cur_max = pos.max(dim=0).values
            cur_width = float(cur_max[0] - cur_min[0])
            if cur_width < width0:
                max_compression = max(max_compression, (width0 - cur_width) / max(width0, 1e-9))
            # Damage strain proxy: x-compression OR bulge growth in y/z — under
            # hard squeezes the soft body bulges instead of shrinking on x, and
            # a width-only proxy saturates (v1 full-run non-monotonicity).
            bulge_y = max(0.0, float(cur_max[1] - cur_min[1]) / max(y_depth0, 1e-9) - 1.0)
            bulge_z = max(0.0, float(cur_max[2] - cur_min[2]) / max(height0, 1e-9) - 1.0)
            max_strain = max(max_strain, max_compression, bulge_y, bulge_z)
            if in_hold or finger_z > z_grasp:
                pass
            finger_inner_l = float(left.get_pos()[0]) + finger_size[0] / 2
            finger_inner_r = float(right.get_pos()[0]) - finger_size[0] / 2
            near_l = (pos[:, 0] - finger_inner_l).abs() < 0.004
            near_r = (pos[:, 0] - finger_inner_r).abs() < 0.004
            in_contact_now = bool((near_l | near_r).any().item())
            if in_contact_now:
                contact_steps += 1
            # Contact-gated: the squeeze channel must not mix with slip/drop
            # impact strain (v2 sweep showed low-force cases spike on falls).
            if observe_count[0] % 20 == 0 and in_contact_now:
                try:
                    import torch as _torch
                    st = entity.get_state()
                    F_t = _torch.as_tensor(st.F, dtype=_torch.float32)
                    if F_t.ndim == 4:
                        F_t = F_t[0]
                    eye = _torch.eye(3, dtype=F_t.dtype, device=F_t.device)
                    strain_i = (F_t - eye).norm(dim=(-1, -2))
                    # Robust aggregate: contact-edge particles always spike the
                    # max; the 95th percentile tracks the bulk squeeze load.
                    q95 = float(strain_i.quantile(0.95).item())
                    continuum_peak[0] = max(continuum_peak[0], q95)
                except Exception:
                    pass
            # Aperture-reactive slip response: the object sagging >3mm behind
            # the finger plane is a kinematic slip signature; tighten the grip
            # by 2% of width per trigger (bounded, no force readout).
            # Tightening floor: never exceed 40% beyond the starting closure;
            # over-gripping squeezes the soft body out (measured in rehearsal).
            if reactive_slip and reactive_armed and float(finger_z - centroid[2]) > 0.003 and gap_state["gap"] > gap0 * 0.6:
                gap_state["gap"] = max(gap0 * 0.6, gap_state["gap"] - 0.02 * width0)
                reactive_tightenings += 1
            if in_hold:
                slip_max = max(slip_max, abs(float(centroid[0] - cx0[0])) + abs(float(centroid[1] - cx0[1])))
                if float(centroid[2] - z_lift_ref[0]) < lift_m - 0.005:
                    held = False

        # approach: interpolate both fingers to the grasp pose beside the object
        start_l, start_r = -0.2, 0.2
        standoff_l, standoff_r = -(width0 / 2 + finger_size[0] / 2 + 0.01), (width0 / 2 + finger_size[0] / 2 + 0.01)
        for k in range(steps_per_phase):
            t = (k + 1) / steps_per_phase
            _fingers(start_l + (standoff_l - start_l) * t, start_r + (standoff_r - start_r) * t, z_grasp)
            scene.step()
            _observe(False, z_grasp)
            _demo_log("approach", z_grasp, gap_state["gap"])
        # squeeze to the commanded closure gap, then let contact settle
        for k in range(steps_per_phase):
            t = (k + 1) / steps_per_phase
            gap_now = gap_state["gap"]
            prev_gap = gap_state["gap"]
            _fingers(standoff_l + (-gap_now / 2 - standoff_l) * t, standoff_r + (gap_now / 2 - standoff_r) * t, z_grasp)
            scene.step()
            _observe(False, z_grasp)
            _demo_log("squeeze", z_grasp, prev_gap)
        for _ in range(max(30, steps_per_phase // 4)):
            scene.step()
            _observe(False, z_grasp)
        # lift (slow, quasi-static); reference frame = centroid at lift start
        z_lift_ref[0] = float(entity.get_particles_pos().mean(dim=0)[2])
        for k in range(steps_per_phase * 2):
            t = (k + 1) / (steps_per_phase * 2)
            gap_now = gap_state["gap"]
            prev_gap = gap_state["gap"]
            _fingers(-gap_now / 2, gap_now / 2, z_grasp + lift_m * t)
            scene.step()
            _observe(False, z_grasp + lift_m * t, True)
            lift_gain = max(lift_gain, float(entity.get_particles_pos().mean(dim=0)[2] - z_lift_ref[0]))
            _demo_log("lift", z_grasp + lift_m * t, prev_gap)
        # hold
        for _ in range(hold_steps):
            gap_now = gap_state["gap"]
            prev_gap = gap_state["gap"]
            _fingers(-gap_now / 2, gap_now / 2, z_grasp + lift_m)
            scene.step()
            _observe(True, z_grasp + lift_m, True)
            _demo_log("hold", z_grasp + lift_m, prev_gap)
        final = _particle_metrics(entity, initial, numerics)
        # Damage driver: continuum internal strain from the per-particle
        # deformation gradient F (evaluator-side state, never policy-visible).
        # ||F-I||_F is monotone in applied load by construction — the v1 full
        # run showed aperture/extent proxies saturate on stiff solids.  PBD
        # cloth has no F and falls back to the aperture/extent proxy.
        inner_sep = float(right.get_pos()[0] - left.get_pos()[0]) - finger_size[0]
        aperture_strain = max(0.0, (width0 - inner_sep) / max(width0, 1e-9))
        continuum_strain = None
        try:
            state = entity.get_state()
            F = state.F
            import torch as _torch

            F_t = _torch.as_tensor(F, dtype=_torch.float32)
            if F_t.ndim == 4:
                F_t = F_t[0]
            eye = _torch.eye(3, dtype=F_t.dtype, device=F_t.device)
            strain_i = (F_t - eye).norm(dim=(-1, -2))
            continuum_strain = float(strain_i.mean().item())
        except Exception:
            continuum_strain = None
        continuum_mean = continuum_strain
        continuum_max = continuum_peak[0] if continuum_peak[0] > 0 else continuum_strain
        if continuum_max is not None:
            damage_strain = continuum_max
        else:
            damage_strain = max(aperture_strain, max_strain)
        damage = min(1.0, max(0.0, (damage_strain - band["onset"]) / max(1e-9, band["failure"] - band["onset"]))) if damage_strain > band["onset"] else 0.0
        integrity = 1.0 - damage
        slip_norm = slip_max / max(width0, 1e-9)
        lifted = lift_gain >= lift_m * 0.8
        task_success = bool(lifted and held and final["finite_fraction"] >= 1.0)
        damage_failure = bool(damage >= 0.25)
        metrics = {
            "task_success": task_success,
            "grasp_success": contact_steps > 0,
            "lift_success": bool(lifted),
            "hold_success": bool(lifted and held),
            "damage_failure": damage_failure,
            "damage_fraction": round(damage, 8),
            "integrity": round(integrity, 8),
            "slip": round(slip_norm, 8),
            "contact_count": int(contact_steps),
            "convergence": bool(final["finite_fraction"] >= 1.0 and final["max_speed_m_s"] < 5.0),
            "finite_fraction": final["finite_fraction"],
            "max_compression_ratio": round(max_compression, 8),
            "max_strain": round(max_strain, 8),
            "aperture_strain": round(aperture_strain, 8),
            "damage_strain": round(damage_strain, 8),
            "continuum_strain_F": round(continuum_mean, 8) if continuum_mean is not None else None,
            "continuum_strain_F_q95": round(continuum_max, 8) if continuum_max is not None else None,
            "retained_lifted_mass_fraction": round(1.0 - damage * 0.8, 8),
            "residual_set": round(damage * 0.02, 8),
            "drop_or_slip": bool(not held or slip_norm >= 0.18),
            "peak_contact_force_n": None,
            "contact_force_status": "not_available_rigid_mpm_coupler",
            "damage_model_id": "TofuDamage.continuum_damage_stiffness_loss_surrogate",
            "closure_ratio": round(closure_ratio, 8),
            "reactive_tightenings": int(reactive_tightenings),
            "lift_gain_m": round(lift_gain, 8),
            "z_lift_ref_m": round(float(z_lift_ref[0]), 8),
            "fragility_regime": regime,
        }
        if not task_success:
            metrics["failure_reason"] = "damage_failure" if damage_failure else ("slip_before_hold" if not held else "lift_not_achieved")
        metrics.update(_base_provenance(job, numerics))
        if demo_handle is not None:
            demo_handle.close()
            metrics["demo_log"] = str(demo_path)
            metrics["demo_steps"] = demo_step["n"]
        return metrics
    finally:
        scene.destroy()


# Feed-forward commanded-force priors by material family (draft cache; the
# frozen DeliGrasp-style cache is produced offline and injected via
# SOFTSIM_DELIGRASP_PRIOR_CACHE).  Explicitly NOT closed-loop DeliGrasp: the
# rigid-MPM force stream is unavailable, so no reactive force channel exists.
_DELIGRASP_FAMILY_PRIOR_N = {
    "soft_food": 8.0,
    "foam": 25.0,
    "textile": 12.0,
    "gel": 6.0,
}


def _deligrasp_prior_force(job: Mapping[str, Any]) -> tuple[float, str]:
    import os as _os

    cache_path = _os.environ.get("SOFTSIM_DELIGRASP_PRIOR_CACHE")
    if cache_path and Path(cache_path).is_file():
        for line in Path(cache_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("sample_id")) == str(job.get("sample_id")):
                return float(row["force_n"]), "frozen_cache"
    route = job.get("route") if isinstance(job.get("route"), Mapping) else {}
    family = str(route.get("material_family") or "")
    if family in _DELIGRASP_FAMILY_PRIOR_N:
        return _DELIGRASP_FAMILY_PRIOR_N[family], "family_table_draft"
    return 15.0, "default_draft"


def _run_e3a_grasp(job: Mapping[str, Any]) -> dict[str, Any]:
    """Track-C baseline dispatch for the scripted grasp-lift-hold task."""
    force_n = float(job.get("force_n", 15.0))
    method = str(job.get("method_id") or "fixed")
    # Commanded force -> closure depth through the frozen surrogate stiffness
    # mapping (documented draft; no measured force stream exists yet).
    closure_ratio = min(0.6, max(0.2, force_n / 60.0))
    provenance: dict[str, Any] = {"baseline_track": "C", "method_id": method}

    if method in {"fixed", "scripted_contact_regulation", "fixed_low", "fixed_high"}:
        metrics = _grasp_episode(job, closure_ratio)
        metrics["commanded_force_n"] = force_n
    elif method == "reactive_slip":
        metrics = _grasp_episode(job, closure_ratio, reactive_slip=True)
        metrics["commanded_force_n"] = force_n
        provenance["controller"] = "aperture_reactive_width_slip"
    elif method == "deligrasp_prior":
        prior_force, cache_source = _deligrasp_prior_force(job)
        metrics = _grasp_episode(job, min(0.6, max(0.2, prior_force / 60.0)))
        metrics["commanded_force_n"] = prior_force
        provenance["controller"] = "deligrasp_style_feedforward_prior"
        provenance["prior_cache_source"] = cache_source
        provenance["deligrasp_note"] = "DeliGrasp-style adaptation; feed-forward only (force gate not passed)"
    elif method == "privileged_oracle":
        # Teacher/oracle: internal closure sweep with privileged damage readout.
        # Never enters the deployable-policy ranking.
        sweep = [0.1, 0.2, 0.3, 0.45, 0.6]
        trials = []
        full_trials: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for closure in sweep:
            trial = _grasp_episode(job, closure)
            full_trials.append(trial)
            trials.append({
                "closure_ratio": closure,
                "task_success": trial["task_success"],
                "damage_fraction": trial["damage_fraction"],
                "intact_success": bool(trial["task_success"] and trial["damage_fraction"] < 0.25),
            })
            if trial["task_success"] and trial["damage_fraction"] < 0.25:
                if best is None or trial["damage_fraction"] < best["damage_fraction"]:
                    best = trial
        if best is None:
            # No intact success exists for this object: report the least-damage
            # full trial verbatim (its task_success is almost surely False —
            # that is the honest "no safe grip exists" oracle verdict).
            best = min(full_trials, key=lambda t: t["damage_fraction"])
        metrics = dict(best)
        metrics["commanded_force_n"] = force_n
        metrics["oracle_sweep"] = trials
        provenance["controller"] = "privileged_oracle_closure_sweep"
        provenance["privileged_observation_used"] = True
    else:
        raise ValueError(f"unknown E3-A baseline method: {method}")
    metrics["baseline_provenance"] = provenance
    return {"status": "ok", "probe_id": "e3a.grasp_lift_hold.v1", "metrics": metrics}



def _probe_common_metrics(scene, entity, numerics, initial, steps, record_every=24):
    """Settle/dynamics metrics shared by the E1-C task probes."""
    latest: dict[str, Any] = {}
    dt_sample = float(numerics["dt_s"]) * int(numerics["substeps"]) * record_every
    prev_pos = None
    for index in range(steps):
        scene.step()
        if (index + 1) % record_every == 0 or index == steps - 1:
            cur_pos = entity.get_particles_pos().clone()
            latest = _particle_metrics(entity, initial, numerics, prev_pos=prev_pos, dt_frame=dt_sample)
            prev_pos = cur_pos
    return latest


def _run_task_probe_drape(job: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the asset onto a narrow bar; pass = drapes and settles (no slide-off/explosion)."""
    import genesis as gs

    scene, entity, numerics = _build_scene(job, job.get("parameters") or {})
    try:
        scale_info = numerics.get("scale_info") or {}
        length = max(scale_info.get("raw_extent_m", [0.1])) * float(scale_info.get("scale_factor", 1.0))
        bar = scene.add_entity(
            gs.morphs.Box(pos=(0.0, 0.0, 0.25 * length), size=(max(0.2 * length, 0.01), 2.0 * length, 0.02 * length)),
            material=gs.materials.Rigid(friction=0.6),
        )
        scene.build()
        initial = entity.get_particles_pos().clone()
        latest = _probe_common_metrics(scene, entity, numerics, initial, int(job.get("steps", 600)))
        centroid_z = float(latest.get("centroid_m", [0, 0, 0])[2])
        draped = centroid_z > 0.1 * length  # still on/around the bar, not slid to the floor plane
        task_pass = bool(latest["finite_fraction"] >= 1.0 and latest["escaped_fraction"] <= 0.01 and draped and latest["max_speed_m_s"] < 2.0)
        metrics = dict(latest)
        metrics.update({"task_pass": task_pass, "draped_on_bar": draped, "bar_height_m": 0.25 * length})
        return {"status": "ok", "probe_id": job.get("probe_id", "e1.task.drape.v1"), "metrics": metrics}
    finally:
        scene.destroy()


def _run_task_probe_push(job: Mapping[str, Any]) -> dict[str, Any]:
    """Quasi-static lateral push (5% length/s); pass = tracks the pusher without topple/explosion."""
    import genesis as gs

    scene, entity, numerics = _build_scene(job, job.get("parameters") or {})
    try:
        scale_info = numerics.get("scale_info") or {}
        length = max(scale_info.get("raw_extent_m", [0.1])) * float(scale_info.get("scale_factor", 1.0))
        plate = scene.add_entity(
            gs.morphs.Box(pos=(-(0.75 * length), 0.0, 0.5 * length), size=(0.02 * length, 0.8 * length, 0.6 * length)),
            material=gs.materials.Rigid(friction=0.4, coup_friction=0.4),
        )
        scene.build()
        settle = int(job.get("settle_steps", 120))
        for _ in range(settle):
            scene.step()
        initial = entity.get_particles_pos().clone()
        pos0 = entity.get_particles_pos()
        c0 = pos0.mean(dim=0)
        dt_sub = float(numerics["dt_s"])
        push_dist = 0.5 * length
        start_x = -0.75 * length
        target_x = start_x + push_dist
        steps = int(job.get("steps", 600))
        latest: dict[str, Any] = {}
        record_every = max(1, steps // 24)
        dt_sample = dt_sub * int(numerics["substeps"]) * record_every
        prev_pos = None
        for index in range(steps):
            cur = plate.get_pos()
            # actual-position feedback servo toward the push target
            vel = max(-0.5, min(0.5, (target_x - float(cur[0])) / dt_sub))
            plate.set_dofs_velocity([vel, 0.0, 0.0, 0.0, 0.0, 0.0])
            scene.step()
            if (index + 1) % record_every == 0 or index == steps - 1:
                cur_pos = entity.get_particles_pos().clone()
                latest = _particle_metrics(entity, initial, numerics, prev_pos=prev_pos, dt_frame=dt_sample)
                prev_pos = cur_pos
        c1 = entity.get_particles_pos().mean(dim=0)
        displacement = float(c1[0] - c0[0])
        tracked = displacement >= 0.2 * push_dist
        upright = float(c1[2]) >= 0.5 * float(c0[2])
        task_pass = bool(latest["finite_fraction"] >= 1.0 and tracked and upright and latest["escaped_fraction"] <= 0.01)
        metrics = dict(latest)
        metrics.update({"task_pass": task_pass, "push_displacement_m": round(displacement, 8), "tracked_pusher": tracked, "upright_after_push": upright,
                        "plate_x_final_m": round(float(plate.get_pos()[0]), 8), "plate_target_x_m": round(target_x, 8), "object_x_left_m": round(float(pos0[:, 0].min()), 8)})
        return {"status": "ok", "probe_id": job.get("probe_id", "e1.task.push.v1"), "metrics": metrics}
    finally:
        scene.destroy()


def _run_task_probe_place(job: Mapping[str, Any]) -> dict[str, Any]:
    """Lift 30mm, translate 40% of length laterally, lower, release; pass = placed upright, no drop scatter."""
    import genesis as gs

    scene, entity, numerics = _build_scene(job, job.get("parameters") or {})
    try:
        scale_info = numerics.get("scale_info") or {}
        length = max(scale_info.get("raw_extent_m", [0.1])) * float(scale_info.get("scale_factor", 1.0))
        finger_size = (0.04, 0.08, 0.04)
        left = scene.add_entity(gs.morphs.Box(pos=(-0.2, 0.0, 0.06), size=finger_size), material=gs.materials.Rigid(friction=2.0))
        right = scene.add_entity(gs.morphs.Box(pos=(0.2, 0.0, 0.06), size=finger_size), material=gs.materials.Rigid(friction=2.0))
        scene.build()
        settle = int(job.get("settle_steps", 120))
        for _ in range(settle):
            scene.step()
        pos0 = entity.get_particles_pos()
        cx0 = pos0.mean(dim=0)
        min0 = pos0.min(dim=0).values
        max0 = pos0.max(dim=0).values
        width0 = float(max0[0] - min0[0])
        height0 = max(1e-9, float(max0[2] - min0[2]))
        z_grasp = float(min0[2]) + 0.5 * height0
        gap = width0 * 0.7 + finger_size[0]
        y_center = float(cx0[1])
        dt_sub = float(numerics["dt_s"])

        def fingers(xl, xr, x_shift, z):
            for finger, tx in ((left, xl + x_shift), (right, xr + x_shift)):
                cur = finger.get_pos()
                vel = [max(-0.5, min(0.5, (t - float(cur[i])) / dt_sub)) for i, t in enumerate((tx, y_center, z))]
                finger.set_dofs_velocity(vel + [0.0, 0.0, 0.0])

        initial = pos0.clone()
        steps_per_phase = int(job.get("steps_per_phase", 240))
        latest: dict[str, Any] = {}
        record_every = 24
        step_count = 0

        prev_pos = None

        def tick():
            nonlocal latest, step_count, prev_pos
            prev_pos = entity.get_particles_pos().clone()
            scene.step()
            step_count += 1
            if step_count % record_every == 0:
                latest = _particle_metrics(entity, initial, numerics, prev_pos=prev_pos, dt_frame=float(numerics["dt_s"]) * int(numerics["substeps"]) * record_every)

        for k in range(steps_per_phase):
            t = (k + 1) / steps_per_phase
            fingers(-0.2 + (-(gap / 2) - (-0.2)) * t, 0.2 + ((gap / 2) - 0.2) * t, 0.0, z_grasp)
            tick()
        for k in range(steps_per_phase):
            t = (k + 1) / steps_per_phase
            fingers(-gap / 2, gap / 2, 0.0, z_grasp + 0.03 * t)
            tick()
        translate = 0.4 * length
        for k in range(steps_per_phase):
            t = (k + 1) / steps_per_phase
            fingers(-gap / 2, gap / 2, translate * t, z_grasp + 0.03)
            tick()
        for k in range(steps_per_phase // 2):
            t = (k + 1) / (steps_per_phase // 2)
            fingers(-gap / 2, gap / 2, translate, z_grasp + 0.03 * (1 - t) + 0.005 * t)
            tick()
        # release: open fingers
        for k in range(steps_per_phase // 2):
            t = (k + 1) / (steps_per_phase // 2)
            fingers(-gap / 2 - 0.05 * t, gap / 2 + 0.05 * t, translate, z_grasp + 0.005)
            tick()
        for _ in range(steps_per_phase // 2):
            tick()
        latest = latest or _particle_metrics(entity, initial, numerics, prev_pos=prev_pos, dt_frame=float(numerics["dt_s"]) * int(numerics["substeps"]))
        final_pos = entity.get_particles_pos()
        c1 = final_pos.mean(dim=0)
        placed_x = float(c1[0])
        placed_z = float(c1[2])
        on_target = abs(placed_x - translate) < 0.25 * length
        down = placed_z < float(cx0[2]) + 0.01
        upright = (float(final_pos[:, 2].max()) - float(final_pos[:, 2].min())) <= 1.5 * height0
        task_pass = bool(latest["finite_fraction"] >= 1.0 and on_target and down and upright and latest["escaped_fraction"] <= 0.01)
        metrics = dict(latest)
        metrics.update({"task_pass": task_pass, "placed_x_m": round(placed_x, 8), "target_x_m": round(translate, 8), "on_target": on_target, "down_after_release": down, "upright_after_place": upright})
        return {"status": "ok", "probe_id": job.get("probe_id", "e1.task.place.v1"), "metrics": metrics}
    finally:
        scene.destroy()


def _run_task_probe_pinch_lift(job: Mapping[str, Any]) -> dict[str, Any]:
    """Pinch-lift reuses the E3-A grasp kernel at mid force (fixed closure)."""
    metrics = _grasp_episode(job, 0.45)
    task_pass = bool(metrics["task_success"] and metrics["finite_fraction"] >= 1.0)
    out = dict(metrics)
    out["task_pass"] = task_pass
    return {"status": "ok", "probe_id": job.get("probe_id", "e1.task.pinch_lift.v1"), "metrics": out}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    result_path = Path(args.result)
    try:
        import genesis as gs

        if gs.__version__ != GENESIS_VERSION_EXPECTED:
            raise RuntimeError(f"genesis version {gs.__version__} != frozen {GENESIS_VERSION_EXPECTED}")
        gs.init(backend=gs.gpu, logging_level="warning", seed=int(job.get("seed", 0)))
        kind = str(job.get("kind"))
        if kind == "e1_runtime":
            result = _run_e1_runtime(job)
        elif kind == "e2_probe":
            result = _run_e2_probe(job)
        elif kind == "e3a_grasp":
            result = _run_e3a_grasp(job)
        elif kind == "e1_task_drape":
            result = _run_task_probe_drape(job)
        elif kind == "e1_task_push":
            result = _run_task_probe_push(job)
        elif kind == "e1_task_place":
            result = _run_task_probe_place(job)
        elif kind == "e1_task_pinch_lift":
            result = _run_task_probe_pinch_lift(job)
        else:
            raise ValueError(f"unknown job kind: {kind}")
    except Exception as exc:  # noqa: BLE001 - failures must land as result JSON
        result = {"status": "error", "error_type": type(exc).__name__, "detail": traceback.format_exc(limit=5)}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
