#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Genesis behavior-contract runner for the SoftSim physical agent.

This runner executes ONE real Genesis MPM simulation per invocation and emits a
``softsim.probe_report`` (schema_version 2) with an explicit behavior contract:

    finite_fraction / contact_fraction / settled / deformation_ratio
    (/ recovery_ratio for compression) / loss / behavior_pass / diagnosis

The metrics describe *simulated soft-body behavior under a fixed numerical
contract*, not measured material ground truth.  The report always carries
``limitations: [..., "behavior_contract_not_material_ground_truth"]``.

CLI (invoked by ``code/scripts/genesis_probe.py`` via ``GENESIS_PYTHON``)::

    python genesis_behavior_runner.py \
        --asset-json A --candidate-json C --probe-json P \
        --output-dir O --report-json R [--simulator-config-json S]

Exit codes: 0 = simulation completed (regardless of behavior pass/fail),
2 = probe not executed (unsupported stimulus), 3 = failure (asset missing,
engine crash).  The report file is written in every case.

v12.3-candidate changes are strictly opt-in: set
``behavior_contract_version: "v12.3"`` in the probe config or simulator config.
Without it the behavior is v12.2 exactly.  v12.3 changes (measured in
deployment/experiments/settle_study): not_settled diagnosis suggests
yield_stress_pa down-adjustment instead of the Genesis-MPM-noop
plastic_viscosity_pa_s; cloth/PBD substeps are clamped to <= 8.

Study-render sidecar (opt-in): when the environment variable
``SOFTSIM_STUDY_RENDER_DIR`` is non-empty, the simulation is additionally
rendered through a fixed Genesis camera and ``clip.mp4`` + ``poster.png``
are written into that directory after the run.  Physics, solver numerics,
metrics and the report are untouched; with the variable unset the runner is
byte-for-byte identical to the numeric-only behavior.

v12.4-candidate mesh adaptation is strictly opt-in: set
``mesh_adaptation: "v2"`` in the probe config or simulator config.  Without
it the solver-mesh path is byte-for-byte v12.2/v12.3.  v2 ports the measured
fast-visual-v11 mechanisms (reference/genesis_mesh_adaptation.md):

* MPM filled_volume: voxelize (pitch = max_extent/40) + fill + marching
  cubes, watertight-verified; the unconditional convex hull becomes the
  fallback only, with warning
  ``convex_hull_may_remove_semantic_concavities_or_through_holes``.
* Cloth: cloth_sheet -> flat grid mid-surface; sealed garment/bag/shell ->
  bounded opening surgery (cut fractions 0.025-0.08, bounded-count check);
  projected mid-surface as the garment/shell fallback.  No decimation of the
  generated proxy beyond the 12000-face budget; no initial ripples.
* Cloth numerics: dt=0.003 s, substeps=10 (v11-stable; PBD degenerates at
  substeps>=16 per the settle study, and v12.3's <=8 clamp predates the v11
  dt=0.003/10 configuration).
* Post-step hooks (the settle-study showed naive every-step velocity
  scaling is non-monotonic and disturbs contact; the v11 mechanisms are
  contact-gated and/or center-of-mass preserving instead):
  - MPM: InternalVelocityRelaxation (damps only the deviatoric part of
    particle velocities, preserves CoM) + ContactNormalDissipation (after
    first contact, damps only the upward CoM component; free fall
    untouched).
  - Cloth: post-contact exponential velocity damping (starts only after
    first floor contact).

v12.5-candidate mesh adaptation is likewise strictly opt-in: set
``mesh_adaptation: "v3"`` in the probe config or simulator config.  v3 lives
in the sibling module ``mesh_adaptation.py`` and adds over v2:

* Source hygiene for all routes: float32-aware vertex welding, degenerate /
  duplicate face removal, fragmentation measurement (generated GLBs arrive
  as thousands of disjoint patches that exact welding cannot fuse).
* Cloth route never simulates the raw fragmented surface: non-watertight
  sources are voxel-sealed (voxelize -> binary closing -> fill -> marching
  cubes) into a watertight shell and then opened with the bounded semantic
  opening surgery; the projected mid-surface remains the fallback.
* MPM route keeps the v11 filled_volume policy, including
  ``preserve_watertight_boundary`` for watertight bag sources.
* Every adaptation emits an auditable proxy quality report
  (``metrics.mesh_adaptation.proxy_quality``: watertight, components,
  boundary edges, area/volume, aspect ratio, thinness) and a verdict.
  ``unfit`` proxies are gated to diagnosis ``model_limit`` with mechanism
  ``mesh_proxy_unfit`` instead of producing a fake physics failure.
* MPM post-step hook adds contact-gated lateral CoM dissipation alongside
  the v11 upward-only CND (the v12.4 experiment showed compression
  lateral oscillation has no dissipation channel otherwise).

v12.5-repair1-candidate is independently opt-in with ``mesh_adaptation: "v4"``.
It bakes scene instances, repairs in a common source frame with padded voxel
closing, enforces final proxy topology, binds a source-colored visual LOD,
and prescribes full compression plate poses. Numerical policies inherited
from v3 remain explicit; behavior thresholds are unchanged. See REPAIR_REPORT_ZH.md.

v12.6-candidate is independently opt-in with ``mesh_adaptation: "v5"``.  It
integrates the v5 representation layer (mesh_adaptation_delivery_20260922):
``mesh_representation.select_representation`` (object name takes precedence
over the coarse dataset family) drives the solver route — filled_volume ->
MPM, all surface representations -> PBD cloth — overriding the prior
model_family routing (evidence in ``metrics.mesh_adaptation.route_override``).
Mesh preparation goes through ``mesh_adaptation_v5.adapt`` (source-frame
hygiene, bounded garment surface repair, semantic opening surgery, Y-up ->
Z-up).  Cloth constraints follow the per-representation profiles
(stretch_compliance 1e-7; bending projection off for sheet/garment_cloth;
source-rest-dihedral shell constraints via ``mesh_shell_constraints`` for
garment_shell/bag_shell/reinforced_shell/sealed_membrane; self-collision kept
for non-sheet profiles with a domain-covering hash grid).  v5 implies the v3
solver behaviors (MPM scale cap, lateral dissipation, cloth plate fixes) but
not the v4 visual binding / plate prescription.  MPM numerics are unchanged.

Human-calibrated gate overrides (gate_study, 10 experts) are opt-in via
``gate_overrides`` in the probe config or simulator config (probe config
wins); without any override the gating is byte-for-byte the default:

* ``settle_gate: "off"`` demotes the tail/peak<=0.05 settle ratio from a hard
  gate to a diagnostic (37/38 above-threshold clips were human-judged
  settled).  settled/peak/tail metrics and the not_settled diagnosis are
  still produced; the settle term is zeroed in the loss so the elastic-budget
  progress rule is not inflated.  Applies to both drop and compression.
* ``drop_deformation_band`` overrides the MPM drop deformation band
  (calibration: upper 0.9 -> 0.98, aligned with the cloth band).
* ``recovery_band_mpm`` / ``recovery_band_cloth`` override the compression
  recovery bands (calibration: lower 0.84 was systematically too strict;
  v12.6 uses [0.45, 1.16] / [0.0, 1.16]).

Effective gates are echoed in metrics (``settle_gate``,
``gate_calibration_ref``; the bands are echoed per-probe as before).

v12.6 also fixes compression-plate engagement (measured on the p0
supplemental assets), opt-in v3+ only: plates are placed from the placed
entity's measured particle AABB, driven with the v4 prescribed full-pose
schedule (the velocity servo under-tracks at ~50% even before contact), MPM
compression scenes use CPIC particle-level rigid boundaries, the MPM grid is
refined so small bodies span >= 16 cells (grid_density is per-meter), and the
compression-axis extent is measured with a 1% trim (corner particles pinched
out between plate and floor are a contact artifact that otherwise pins the
min/max extent).  The default (v12.2) path is byte-for-byte unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import sys
import traceback
from pathlib import Path
from typing import Any

SCHEMA_NAME = "softsim.probe_report"
SCHEMA_VERSION = 2
BACKEND = "genesis_behavior"

# Mirrors code/src/softsim_physical_agent/routing.py ALLOWED_PHYSICAL_PARAMETERS.
ALLOWED_PHYSICAL_PARAMETERS = frozenset(
    {
        "density_kg_m3",
        "youngs_modulus_pa",
        "poisson_ratio",
        "yield_stress_pa",
        "yield_lower",
        "yield_higher",
        "plastic_viscosity_pa_s",
        "tensile_cohesion_pa",
        "friction_static",
        "friction_dynamic",
    }
)

# Physical parameter defaults when neither candidate nor prior supplies a value.
PARAM_DEFAULTS = {
    "density_kg_m3": 600.0,
    "youngs_modulus_pa": 50000.0,
    "poisson_ratio": 0.35,
    "plastic_viscosity_pa_s": 0.0,
    "yield_stress_pa": 10000.0,
    "friction_static": 0.4,
}

FIXED_NUMERIC_DEFAULTS = {
    "particle_size_m": 0.01,
    "dt_s": 0.001,
    "substeps": 16,
    "grid_density": 64.0,
    "scale_m": 0.32,
    "seed": 17,
    "drop_height_m": 0.2,
    "proxy_mode": "alpha-wrap",
}

BASE_LIMITATIONS = [
    "behavior_contract_not_material_ground_truth",
    "metric_scale_from_fixed_solver_numerics_scale_m",
    "surface_mesh_decimated_and_voxelized_by_genesis",
    "explicit_mpm_contract_is_resolution_and_dt_dependent",
]

UNSUPPORTED_TOKENS = ("cloth", "pbd", "shell")

# Opt-in user-study render sidecar.  When this environment variable is
# non-empty, the run is also rendered to <dir>/clip.mp4 + <dir>/poster.png.
STUDY_RENDER_ENV = "SOFTSIM_STUDY_RENDER_DIR"


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _number(value: Any, default: float) -> float:
    if isinstance(value, dict):
        value = value.get("value")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _prior_value(asset: dict[str, Any], name: str) -> float | None:
    prior = asset.get("prior") or {}
    for item in prior.get("parameters") or []:
        if isinstance(item, dict) and str(item.get("name")) == name:
            return _number(item.get("value"), float("nan"))
    values = (prior.get("solver_adapter") or {}).get("parameters") or {}
    if name in values:
        return _number(values[name], float("nan"))
    return None


def _candidate_value(asset: dict[str, Any], candidate: dict[str, Any], name: str) -> float:
    """Candidate wins over prior; both fall back to PARAM_DEFAULTS."""
    raw = candidate.get(name)
    if raw is not None:
        return _number(raw, PARAM_DEFAULTS.get(name, 0.0))
    prior_value = _prior_value(asset, name)
    if prior_value is not None and math.isfinite(prior_value):
        return prior_value
    return PARAM_DEFAULTS.get(name, 0.0)


def _fixed_numerics(asset: dict[str, Any], simulator_config: dict[str, Any]) -> dict[str, Any]:
    """Solver numerics come from the task-level fixed contract only.

    Precedence: simulator-config ``fixed_solver_numerics`` (the formal task
    contract) > asset-level ``fixed_solver_numerics`` > defaults.  Probe config
    never overrides solver numerics; it only supplies steps/record_every/seed_offset.
    """
    merged: dict[str, Any] = dict(FIXED_NUMERIC_DEFAULTS)
    for source in (
        (asset.get("fixed_solver_numerics") or {}),
        ((asset.get("prior") or {}).get("fixed_solver_numerics") or {}),
        (simulator_config.get("fixed_solver_numerics") or {}),
    ):
        if not isinstance(source, dict):
            continue
        for name in FIXED_NUMERIC_DEFAULTS:
            if name in source:
                merged[name] = _number(source[name], merged[name]) if name != "proxy_mode" else str(source[name])
    merged["particle_size_m"] = min(max(merged["particle_size_m"], 0.001), 0.05)
    merged["dt_s"] = min(max(merged["dt_s"], 1.0e-5), 3.0e-3)
    merged["substeps"] = int(min(max(merged["substeps"], 4), 64))
    merged["grid_density"] = float(min(max(merged["grid_density"], 32.0), 128.0))
    merged["scale_m"] = min(max(merged["scale_m"], 0.02), 2.0)
    merged["seed"] = int(merged["seed"])
    merged["drop_height_m"] = min(max(merged["drop_height_m"], 0.01), 2.0)
    return merged


def _fixed_drop_height(asset: dict[str, Any], simulator_config: dict[str, Any]) -> float | None:
    """drop_height_m only when a fixed_numerics source explicitly defines it."""
    for source in (
        simulator_config.get("fixed_solver_numerics"),
        (asset.get("prior") or {}).get("fixed_solver_numerics"),
        asset.get("fixed_solver_numerics"),
    ):
        if isinstance(source, dict) and "drop_height_m" in source:
            value = _number(source["drop_height_m"], float("nan"))
            if math.isfinite(value) and value > 0:
                return value
    return None


def _resolve_mesh(asset: dict[str, Any]) -> str | None:
    """resolved_artifacts absolute path first, then artifact_refs via env roots."""
    resolved = asset.get("resolved_artifacts") or {}
    for role in ("collision_mesh", "visual_mesh"):
        raw = resolved.get(role)
        if isinstance(raw, str) and Path(raw).is_absolute() and Path(raw).is_file():
            return raw
    refs = asset.get("artifact_refs") or {}
    for role in ("collision_mesh", "visual_mesh"):
        ref = refs.get(role) or {}
        if not isinstance(ref, dict):
            continue
        raw_path = ref.get("path")
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if path.is_absolute():
            if path.is_file():
                return str(path)
            continue
        alias = str(ref.get("root_alias") or "source")
        env_name = "SOFTSIM_ROOT_" + re.sub(r"[^A-Za-z0-9]", "_", alias).upper()
        root = os.environ.get(env_name) or os.environ.get("SOFTSIM_ASSET_ROOT")
        if root:
            candidate = Path(root) / str(path)
            if candidate.is_file():
                return str(candidate)
    return None


def _route_text(asset: dict[str, Any]) -> str:
    route = asset.get("route") if isinstance(asset.get("route"), dict) else {}
    prior = asset.get("prior") if isinstance(asset.get("prior"), dict) else {}
    parts = []
    for container in (route, prior):
        for key in ("model_family", "solver_adapter", "material_family", "geometry_family", "representation", "proxy_mode"):
            value = container.get(key)
            if isinstance(value, str):
                parts.append(value)
        adapter = container.get("solver_adapter")
        if isinstance(adapter, dict) and isinstance(adapter.get("solver"), str):
            parts.append(adapter["solver"])
    return " ".join(parts).lower()


def _route_supported(asset: dict[str, Any]) -> tuple[bool, str]:
    """Route gate.  Supported: MPM volumetric routes and cloth/shell/PBD/FEM
    sheet routes (executed as real PBD.Cloth; the legacy ``*_fem_cloth_ipc``
    label is mapped to PBD.Cloth per the frozen method contract).  Only
    routes with explicit damage/fracture requirements remain unsupported.
    """
    text = _route_text(asset)
    if any(token in text for token in ("damage", "fracture")):
        return False, "unsupported"
    if "mpm" in text:
        return True, "mpm_elastoplastic" if ("elastoplastic" in text or "elasto_plastic" in text) else "mpm_elastic"
    if any(token in text for token in ("cloth", "pbd", "fem", "shell")):
        return True, "pbd_cloth"
    if "elastoplastic" in text or "elasto_plastic" in text:
        return True, "mpm_elastoplastic"
    if "elastic" in text or not text.strip():
        return True, "mpm_elastic"
    return False, "unsupported"


# --------------------------------------------------------------------------- #
# Report scaffolding
# --------------------------------------------------------------------------- #


def _base_report(asset: dict[str, Any], candidate: dict[str, Any], probe: dict[str, Any], simulator_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "backend": BACKEND,
        "sample_id": asset.get("sample_id"),
        "probe": probe,
        "probe_id": probe.get("probe_id"),
        "candidate": candidate,
        "simulator_config": simulator_config,
        "execution_status": "not_executed",
        "behavior_pass": False,
        "passed": False,
        "explicit_behavior_evidence": False,
        "metrics": {},
        "diagnosis": {"kind": "unsupported_probe", "suggested_updates": {}},
        "outputs": {},
        "limitations": list(BASE_LIMITATIONS),
        "provenance": {"hostname": socket.gethostname()},
    }


def _sentinel_metrics(probe: dict[str, Any]) -> dict[str, Any]:
    """Non-empty metrics so routing-level rejections still carry explicit evidence."""
    required = probe.get("required_metrics") if isinstance(probe.get("required_metrics"), list) else []
    metrics: dict[str, Any] = {}
    for name in required:
        metrics[str(name)] = False if str(name) == "settled" else 0.0
    metrics["route_supported"] = 0.0
    return metrics


# --------------------------------------------------------------------------- #
# Diagnosis (bounded, whitelist-only parameter suggestions)
# --------------------------------------------------------------------------- #


def _v123_enabled(probe: dict[str, Any], simulator_config: dict[str, Any]) -> bool:
    """v12.3-candidate contract is strictly opt-in: a probe config or simulator
    config must carry ``behavior_contract_version: "v12.3"``.  Otherwise the
    runner is byte-for-byte v12.2 behavior."""
    for container in ((probe.get("config") or {}), simulator_config):
        if str(container.get("behavior_contract_version") or "") == "v12.3":
            return True
    return False


# Human-calibrated gate overrides (deployment/experiments/gate_study, 10 raters).
# All keys are opt-in through ``gate_overrides`` in the probe config or the
# simulator config (probe config wins, same style as the other switches).
# Without any override the runner is byte-for-byte the default behavior.
GATE_CALIBRATION_REF = "gate_study_manifest.json@694a504470f60b62"


def _gate_overrides(probe: dict[str, Any], simulator_config: dict[str, Any]) -> dict[str, Any]:
    for container in ((probe.get("config") or {}), simulator_config):
        overrides = container.get("gate_overrides")
        if isinstance(overrides, dict) and overrides:
            return overrides
    return {}


def _gate_band(overrides: dict[str, Any], key: str, default: list[float]) -> list[float]:
    raw = overrides.get(key)
    if (
        isinstance(raw, list)
        and len(raw) == 2
        and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in raw)
        and float(raw[0]) <= float(raw[1])
    ):
        return [float(raw[0]), float(raw[1])]
    return default


def _settle_hard_gate(overrides: dict[str, Any]) -> bool:
    """Human-calibration (gate_study): 37/38 clips above the tail/peak 0.05
    line were judged settled by all 10 experts, so the settle ratio is
    demoted from a hard gate to a diagnostic when ``settle_gate: "off"``.
    The metrics and the not_settled diagnosis keep being produced either way."""
    return str(overrides.get("settle_gate") or "on") != "off"


def _bounded_delta(candidate_value: float, relative: float, seed_absolute: float) -> float:
    """Single-parameter relative step bounded to <= 50% of the current value."""
    if candidate_value > 0:
        delta = relative * candidate_value
        limit = 0.5 * candidate_value
        return max(-limit, min(limit, delta))
    return seed_absolute if relative > 0 else 0.0


def _diagnose(
    metrics: dict[str, Any],
    candidate: dict[str, Any],
    asset: dict[str, Any],
    probe_kind: str,
    v123: bool = False,
) -> dict[str, Any]:
    if metrics.get("behavior_pass") is True:
        return {"kind": "pass", "suggested_updates": {}, "mechanism": "none"}

    youngs = _candidate_value(asset, candidate, "youngs_modulus_pa")
    updates: dict[str, float] = {}

    if float(metrics.get("finite_fraction", 0.0)) < 1.0:
        updates["youngs_modulus_pa"] = _bounded_delta(youngs, 0.5, 0.0)
        return {"kind": "unstable_or_escaped", "suggested_updates": updates, "mechanism": "nonfinite_state"}

    if metrics.get("ground_penetration"):
        updates["youngs_modulus_pa"] = _bounded_delta(youngs, 0.5, 0.0)
        return {"kind": "ground_penetration", "suggested_updates": updates, "mechanism": "particles_below_floor"}

    if float(metrics.get("contact_fraction", 0.0)) <= 0.0:
        return {"kind": "no_contact", "suggested_updates": {}, "mechanism": "no_floor_or_plate_contact"}

    if not metrics.get("settled"):
        if v123:
            # v12.3: measured on pilot-v2 settle failures (settle_study) that the
            # only effective dissipation channel in Genesis MPM is plastic yield
            # work; plastic_viscosity_pa_s never reaches the solver (Genesis 1.3.3
            # ElastoPlastic has no such field) and is a no-op suggestion.
            yield_base = _candidate_value(asset, candidate, "yield_stress_pa")
            updates["yield_stress_pa"] = _bounded_delta(yield_base, -0.5, 0.0)
            return {
                "kind": "parameter_correctable",
                "suggested_updates": updates,
                "mechanism": "not_settled_persistent_motion",
                "notes": [
                    "plastic_viscosity_pa_s is a no-op in Genesis 1.3.3 MPM (no solver field); yield_stress_pa reduction is the measured dissipation channel",
                ],
            }
        viscosity = _candidate_value(asset, candidate, "plastic_viscosity_pa_s")
        updates["plastic_viscosity_pa_s"] = _bounded_delta(viscosity, 0.5, 20.0)
        # Persistent bounce with no viscosity channel left means E is too high.
        if viscosity <= 0:
            updates["youngs_modulus_pa"] = _bounded_delta(youngs, -0.3, 0.0)
        return {"kind": "parameter_correctable", "suggested_updates": updates, "mechanism": "not_settled_persistent_motion"}

    band = metrics.get("deformation_band") or [0.02, 0.7]
    deformation = float(metrics.get("deformation_ratio", 0.0))
    if deformation > float(band[1]):
        updates["youngs_modulus_pa"] = _bounded_delta(youngs, 0.5, 0.0)
        return {"kind": "parameter_correctable", "suggested_updates": updates, "mechanism": "deformation_above_band"}
    if deformation < float(band[0]):
        updates["youngs_modulus_pa"] = _bounded_delta(youngs, -0.5, 0.0)
        return {"kind": "parameter_correctable", "suggested_updates": updates, "mechanism": "deformation_below_band"}

    if probe_kind == "compression":
        recovery = metrics.get("recovery_ratio")
        rec_band = metrics.get("recovery_band") or [0.0, 1.05]
        if recovery is not None and not (float(rec_band[0]) <= float(recovery) <= float(rec_band[1])):
            yield_stress = _candidate_value(asset, candidate, "yield_stress_pa")
            if "yield_stress_pa" in candidate:
                updates["yield_stress_pa"] = _bounded_delta(yield_stress, 0.5, 0.0)
            else:
                updates["youngs_modulus_pa"] = _bounded_delta(youngs, 0.5, 0.0)
            return {"kind": "parameter_correctable", "suggested_updates": updates, "mechanism": "recovery_outside_band"}

    updates["youngs_modulus_pa"] = _bounded_delta(youngs, 0.25, 0.0)
    return {"kind": "parameter_correctable", "suggested_updates": updates, "mechanism": "behavior_band_mismatch"}


# --------------------------------------------------------------------------- #
# Genesis scene construction (headless, numeric-only build)
# --------------------------------------------------------------------------- #


def _headless() -> None:
    os.environ.setdefault("PYGLET_HEADLESS", "1")
    os.environ.setdefault("PYGLET_HEADLESS_DEVICE", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


# --------------------------------------------------------------------------- #
# Study render sidecar (SOFTSIM_STUDY_RENDER_DIR-gated, pure bypass)
# --------------------------------------------------------------------------- #


def _study_render_dir() -> Path | None:
    raw = (os.environ.get(STUDY_RENDER_ENV) or "").strip()
    return Path(raw) if raw else None


def _study_vis_options(gs: Any) -> Any:
    """Light, even-lit background so the (darker) floor plane and the object
    silhouette stay readable for human raters.  Physics unaffected."""
    return gs.options.VisOptions(
        background_color=(0.93, 0.93, 0.95),
        ambient_light=(0.35, 0.35, 0.35),
        shadow=True,
        plane_reflection=False,
    )


def _study_camera_pose(
    probe_kind: str,
    length: float,
    extent: Any,
    drop_top_z: float | None = None,
    axis_index: int = 0,
    plate_outer_half_span: float | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Fixed 3/4 side view; the floor plane and the full motion stay in frame.

    drop: the frame covers the release height down to the floor, object roughly
    centered.  compression: aimed at the compression zone, the compression axis
    appears horizontal so plate contact and squash are visible.
    """
    fov_half_tan = math.tan(math.radians(30.0) / 2.0)  # camera fov = 30
    ex, ey, ez = (float(extent[0]), float(extent[1]), float(extent[2]))
    if probe_kind == "drop":
        top_z = float(drop_top_z if drop_top_z is not None else ez)
        # Bounding-sphere framing: the body (radius r_obj) must stay in frame
        # anywhere along the release-to-floor trajectory, whatever its pose.
        r_obj = 0.5 * math.sqrt(ex * ex + ey * ey + ez * ez)
        lookat = (0.0, 0.0, 0.5 * top_z)
        cover = 0.5 * top_z + r_obj + 0.05 * length
        dist = max(cover / fov_half_tan, 0.25 * length, 0.05)
        rel = (1.0, -1.0, 0.6)
    else:
        lookat = (0.0, 0.0, 0.5 * ez)
        span_h = float(plate_outer_half_span) if plate_outer_half_span else 0.5 * float(extent[axis_index])
        cover = max(span_h, 0.9 * ez) + 0.25 * length
        dist = max(cover / fov_half_tan, 0.25 * length, 0.05)
        rel = (0.35, -1.0, 0.7) if axis_index == 0 else (1.0, -0.35, 0.7)
    norm = math.sqrt(sum(c * c for c in rel))
    pos = tuple(lookat[i] + dist * rel[i] / norm for i in range(3))
    return pos, lookat


def _write_study_render(render_dir: Path, frames: list[Any], np: Any) -> dict[str, Any]:
    """Encode recorded frames to clip.mp4 (H.264) and dump a poster frame.

    fps is chosen so the clip duration lands in [4, 8] s.  The poster is the
    ~60% progress frame (post-contact, near maximum deformation for drops).
    """
    import imageio.v2 as imageio

    if not frames:
        raise RuntimeError("study render produced zero frames")
    render_dir.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    fps = min(120.0, max(12.0, n / 6.0))
    clip_path = render_dir / "clip.mp4"
    writer = imageio.get_writer(
        str(clip_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()
    poster_idx = min(n - 1, int(round(0.6 * (n - 1))))
    poster_path = render_dir / "poster.png"
    imageio.imwrite(str(poster_path), np.asarray(frames[poster_idx], dtype=np.uint8))
    return {
        "clip_path": str(clip_path),
        "poster_path": str(poster_path),
        "n_frames": n,
        "fps": fps,
        "duration_s": n / fps,
        "poster_frame_index": poster_idx,
    }


def _prepare_solver_mesh(
    source_path: Path,
    scale_m: float,
    orientation: str,
    max_faces: int,
    output_dir: Path,
    np: Any,
    close_volume: bool = True,
) -> tuple[Path, Any]:
    """Load, scale, orient, decimate, (optionally) convex-hull, export OBJ.

    ``close_volume`` is required for MPM volume sampling but must stay off for
    cloth surfaces.  Returns the OBJ path, the *scaled* axis-aligned extent
    (pre-translation), and the median edge length (PBD collision radius must
    stay below the vertex spacing).
    """
    import fast_simplification
    import trimesh

    loaded = trimesh.load(str(source_path), force="scene", process=False)
    geometries = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
    if not geometries:
        raise RuntimeError(f"solver mesh has no faces: {source_path}")
    combined = trimesh.util.concatenate(geometries)
    extent = np.asarray(combined.extents, dtype=float)
    extent = np.maximum(extent, 1.0e-6)
    scale = float(scale_m) / float(extent.max())
    combined.apply_scale(scale)
    if orientation == "corner_first":
        # Deterministic corner-down pose: align the bbox body diagonal with +z.
        combined.apply_transform(trimesh.transformations.rotation_matrix(math.radians(45.0), [0.0, 1.0, 0.0]))
        combined.apply_transform(trimesh.transformations.rotation_matrix(math.radians(35.264), [1.0, 0.0, 0.0]))
    if len(combined.faces) > max_faces:
        vertices, faces = fast_simplification.simplify(
            np.asarray(combined.vertices, dtype=float),
            np.asarray(combined.faces, dtype=np.int64),
            target_count=int(max_faces),
            agg=2,
            lossless=False,
        )
        combined = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if close_volume:
        try:
            hull = combined.convex_hull
            if len(hull.faces) >= 4 and len(hull.vertices) >= 4:
                combined = hull
        except Exception:
            pass
    bounds = combined.bounds
    center_xy = 0.5 * (bounds[0, :2] + bounds[1, :2])
    combined.apply_translation([-center_xy[0], -center_xy[1], -bounds[0, 2]])
    destination = output_dir / "solver_mesh.obj"
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.export(destination, file_type="obj")
    scaled_extent = combined.extents
    try:
        edge_lengths = np.asarray(combined.edges_unique_length, dtype=float)
        median_edge = float(np.median(edge_lengths)) if edge_lengths.size else float(scaled_extent.max()) / 32.0
    except Exception:
        median_edge = float(np.asarray(scaled_extent, dtype=float).max()) / 32.0
    return destination, np.asarray(scaled_extent, dtype=float), median_edge


# --------------------------------------------------------------------------- #
# v12.4-candidate mesh adaptation ("mesh_adaptation": "v2", strictly opt-in)
#
# Ports of the measured fast-visual-v11 mechanisms (see the module docstring
# and reference/genesis_mesh_adaptation.md).  Nothing in this section runs
# unless the probe/simulator config opts in.
# --------------------------------------------------------------------------- #

V2_VOXEL_RESOLUTION = 40  # v11 proxy.voxel_resolution
V2_SURFACE_FACE_BUDGET = 12000  # v11 proxy.maximum_surface_faces
V2_FLAT_SHEET_LONGEST_SEGMENTS = 64  # v11 proxy.sheet_longest_segments
V2_PROJECTED_LONGEST_SEGMENTS = 48
# v11 InternalVelocityRelaxation (genesis_materials.py:179): damps only the
# deviatoric (CoM-relative) part of particle velocities.  Rate derives from
# relaxation_time_s / equilibrium fraction like v11's damped-solid profiles.
V2_IVR_RELAXATION_TIME_S = 0.25
V2_IVR_EQUILIBRIUM_FRACTION = 0.5
# v11 ContactNormalDissipation (genesis_materials.py:228): after first floor
# contact, damps only the upward CoM component; profile values span 64-144/s
# with windows 0.24-0.36 s.
V2_CND_RATE_S_INV = 64.0
V2_CND_IMPACT_WINDOW_S = 0.24
# v11 cloth post-contact velocity damping (genesis_cloth.py:879-887); profile
# values span 8-28/s, garments use 8.0.
V2_CLOTH_POST_CONTACT_DAMPING_RATE_S_INV = 8.0
# v12.5: contact-gated lateral (x/y) CoM dissipation for MPM.  The v12.4
# experiment showed compression-release lateral oscillation never settles
# because CND only damps the upward CoM component; floor/plate contact
# friction is the physical channel this surrogates.  Contact-gated only
# (no impact window): free fall and ballistic rebound stay untouched.
V3_CND_LATERAL_RATE_S_INV = 16.0
# v12.5 post-build proxy gates: an MPM volume sampled with too few particles
# cannot resolve the behavior contract (thin volumes sample to <2 particles
# across); this is a proxy/numerics mismatch, gated as model_limit, not a
# physics failure.
V3_MIN_MPM_PARTICLES = 128
V3_MIN_CLOTH_VERTICES = 32
# v12.5: MPM solver-scale cap (m).  Measured on Genesis 1.3.3 with the
# contract-fixed grid_density=64: clean icosphere drops rest at +0.010m
# (scale 0.5) but sink -0.057/-0.42/-0.66m at scales 0.64/1.4/1.9 regardless
# of particle_size — a grid-coupling resolution limit, not a mesh defect.
V3_MPM_SOLVER_SCALE_CAP_M = 0.5
V3_MPM_MIN_CELLS_PER_BODY = 16.0


def _mesh_adaptation_v2_enabled(probe: dict[str, Any], simulator_config: dict[str, Any]) -> bool:
    for container in ((probe.get("config") or {}), simulator_config):
        if str(container.get("mesh_adaptation") or "") == "v2":
            return True
    return False


def _mesh_adaptation_v3_enabled(probe: dict[str, Any], simulator_config: dict[str, Any]) -> bool:
    for container in ((probe.get("config") or {}), simulator_config):
        if str(container.get("mesh_adaptation") or "") == "v3":
            return True
    return False


def _mesh_adaptation_v4_enabled(probe: dict[str, Any], simulator_config: dict[str, Any]) -> bool:
    return any(str(container.get("mesh_adaptation") or "") == "v4"
               for container in ((probe.get("config") or {}), simulator_config))


def _mesh_adaptation_v5_enabled(probe: dict[str, Any], simulator_config: dict[str, Any]) -> bool:
    return any(str(container.get("mesh_adaptation") or "") == "v5"
               for container in ((probe.get("config") or {}), simulator_config))


def _proxy_rejection(report: dict[str, Any], details: dict[str, Any], phase: str,
                     contract: str = "softsim.genesis_proxy.v12.5-repair1-candidate") -> dict[str, Any]:
    report.update(execution_status="completed", behavior_pass=False, passed=False,
                  explicit_behavior_evidence=True)
    report["metrics"] = {**_sentinel_metrics(report["probe"]), "mesh_adaptation": details,
                         "proxy_gate": phase, "simulation_executed": False, "route_supported": 1.0,
                         "adaptation_contract": contract}
    report["diagnosis"] = {"kind": "model_limit", "suggested_updates": {},
                           "mechanism": "mesh_proxy_unfit", "defects": details.get("defects", []),
                           "failure_attribution": "proxy_defect", "phase": phase}
    report["limitations"].append("v4_proxy_rejection_is_not_executed_physics")
    return report


def _geometry_family(asset: dict[str, Any]) -> str:
    route = asset.get("route") if isinstance(asset.get("route"), dict) else {}
    prior = asset.get("prior") if isinstance(asset.get("prior"), dict) else {}
    for container in (route, asset, prior):
        value = container.get("geometry_family")
        if isinstance(value, str) and value:
            return value.lower()
    return "unknown"


def _load_source_mesh(source_path: Path, np: Any):
    import trimesh

    loaded = trimesh.load(str(source_path), force="scene", process=False)
    geometries = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
    if not geometries:
        raise RuntimeError(f"solver mesh has no faces: {source_path}")
    return trimesh.util.concatenate(geometries)


def _checkerboard_grid_sheet(
    bounds_min: Any,
    bounds_max: Any,
    counts: list[int],
    axes: list[int],
    np: Any,
):
    """Regular grid mid-surface with checkerboard diagonal splits; port of
    batch_pipeline.py:749-800 (_write_flat_sheet_proxy)."""
    import trimesh

    center = 0.5 * (bounds_min + bounds_max)
    coordinates = [
        np.linspace(bounds_min[axis], bounds_max[axis], counts[index] + 1)
        for index, axis in enumerate(axes)
    ]
    vertices = []
    for second in coordinates[1]:
        for first in coordinates[0]:
            point = center.copy()
            point[axes[0]] = first
            point[axes[1]] = second
            vertices.append(point.copy())
    row = counts[0] + 1
    faces = []
    for y in range(counts[1]):
        for x in range(counts[0]):
            lower_left = y * row + x
            lower_right = lower_left + 1
            upper_left = lower_left + row
            upper_right = upper_left + 1
            if (x + y) % 2:
                faces.extend(
                    ((lower_left, lower_right, upper_left), (lower_right, upper_right, upper_left))
                )
            else:
                faces.extend(
                    ((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left))
                )
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def _flat_sheet_mid_surface(mesh: Any, np: Any, longest_segments: int = V2_FLAT_SHEET_LONGEST_SEGMENTS):
    """cloth_sheet: thinnest axis is the sheet normal; replace the source
    undulations with a clean planar grid over the silhouette bounding box."""
    bounds = np.asarray(mesh.bounds, dtype=float)
    extents = bounds[1] - bounds[0]
    thin_axis = int(np.argmin(extents))
    plane_axes = [axis for axis in range(3) if axis != thin_axis]
    longest_axis = plane_axes[int(extents[plane_axes[1]] > extents[plane_axes[0]])]
    other_axis = plane_axes[0] if plane_axes[1] == longest_axis else plane_axes[1]
    counts_map = {
        longest_axis: int(longest_segments),
        other_axis: max(
            12,
            int(round(longest_segments * extents[other_axis] / max(extents[longest_axis], 1.0e-12))),
        ),
    }
    counts = [counts_map[axis] for axis in plane_axes]
    sheet = _checkerboard_grid_sheet(bounds[0], bounds[1], counts, plane_axes, np)
    return sheet, {
        "method": "flat_mid_surface",
        "segments_by_axis": {str(axis): int(counts_map[axis]) for axis in plane_axes},
        "thin_axis": thin_axis,
    }


def _projected_mid_surface(mesh: Any, np: Any, longest_segments: int = V2_PROJECTED_LONGEST_SEGMENTS):
    """Project the (possibly body-supported) source onto its thinnest-axis
    plane, rasterize the silhouette, and rebuild a single-layer grid; port of
    fast_visual.py:3130-3216 (_create_projected_garment_mid_surface)."""
    import cv2

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    normal_axis = int(np.argmin(extents))
    plane_axes = [axis for axis in range(3) if axis != normal_axis]
    longest_axis = plane_axes[int(extents[plane_axes[1]] > extents[plane_axes[0]])]
    other_axis = plane_axes[0] if plane_axes[1] == longest_axis else plane_axes[1]
    segments = {
        longest_axis: int(longest_segments),
        other_axis: max(
            16,
            int(round(longest_segments * extents[other_axis] / max(extents[longest_axis], 1.0e-12))),
        ),
    }
    axis_u, axis_v = plane_axes
    count_u, count_v = segments[axis_u], segments[axis_v]
    projected = np.asarray(mesh.vertices, dtype=np.float64)[:, plane_axes]
    projected -= bounds[0, plane_axes]
    projected /= np.maximum(extents[plane_axes], 1.0e-12)
    pixel_triangles = projected[np.asarray(mesh.faces, dtype=np.int64)].copy()
    pixel_triangles[..., 0] *= max(1, count_u - 1)
    pixel_triangles[..., 1] *= max(1, count_v - 1)
    occupancy = np.zeros((count_v, count_u), dtype=np.uint8)
    for triangle in pixel_triangles:
        cv2.fillConvexPoly(occupancy, np.rint(triangle).astype(np.int32), 1)
    # Close rasterization gaps at the grid scale while preserving larger
    # semantic holes.
    occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    coord_u = np.linspace(bounds[0, axis_u], bounds[1, axis_u], count_u + 1)
    coord_v = np.linspace(bounds[0, axis_v], bounds[1, axis_v], count_v + 1)
    center = bounds.mean(axis=0)
    vertices = []
    for v in coord_v:
        for u in coord_u:
            point = center.copy()
            point[axis_u] = u
            point[axis_v] = v
            vertices.append(point)
    row_width = count_u + 1
    faces: list[tuple[int, int, int]] = []
    for v in range(count_v):
        for u in range(count_u):
            if not occupancy[v, u]:
                continue
            lower_left = v * row_width + u
            lower_right = lower_left + 1
            upper_left = lower_left + row_width
            upper_right = upper_left + 1
            if (u + v) % 2:
                faces.extend(
                    ((lower_left, lower_right, upper_left), (lower_right, upper_right, upper_left))
                )
            else:
                faces.extend(
                    ((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left))
                )
    if len(faces) < 32:
        raise ValueError("projected garment silhouette is too sparse for a cloth mid-surface")
    import trimesh

    sheet = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    sheet.remove_unreferenced_vertices()
    sheet.fix_normals()
    return sheet, {
        "method": "projected_single_layer_garment_mid_surface",
        "normal_axis": normal_axis,
        "segments_by_axis": {str(axis): int(value) for axis, value in segments.items()},
        "occupied_cell_fraction": float(occupancy.mean()),
        "limitation": "preserves the projected silhouette but not sewing panels, preformed volume, or seam stiffness",
    }


def _sealed_shell_opening(mesh: Any, geometry_family: str, np: Any):
    """Cut a bounded opening into a sealed watertight shell; port of the
    opening surgery in fast_visual.py:2990-3078 (_create_surface_proxy).

    Axis policy: garments open along the longest body axis; reinforced shells
    (hats etc.) along the thinnest; stage3 bags open at the semantic top,
    which is source +Y (stage3 assets use X/Y for the image plane and Z for
    reconstructed depth).  The cut side is the end with less surface area,
    except bags which always cut +Y max.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    centers = vertices[faces].mean(axis=1)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if geometry_family == "bag":
        axis = 1
        side = "max"
    elif geometry_family == "shell":
        axis = int(np.argmin(extents))
        side = ""
    else:
        axis = int(np.argmax(extents))
        side = ""
    lower, upper = (float(value) for value in mesh.bounds[:, axis])
    span = max(upper - lower, 1.0e-12)
    endpoint_fraction = 0.08
    endpoint_areas = {
        "min": float(areas[centers[:, axis] <= lower + endpoint_fraction * span].sum()),
        "max": float(areas[centers[:, axis] >= upper - endpoint_fraction * span].sum()),
    }
    if not side:
        side = min(endpoint_areas, key=endpoint_areas.get)
    removed = np.zeros(len(faces), dtype=bool)
    applied_fraction = 0.0
    minimum_removed_faces = 2 if len(faces) < 100 else max(4, int(round(0.002 * len(faces))))
    for fraction in (0.025, 0.04, 0.06, 0.08):
        if side == "min":
            candidate = centers[:, axis] <= lower + fraction * span
        else:
            candidate = centers[:, axis] >= upper - fraction * span
        count = int(candidate.sum())
        if minimum_removed_faces <= count <= max(2, len(faces) // 4):
            removed = candidate
            applied_fraction = fraction
            break
    if not removed.any():
        raise ValueError("sealed garment proxy could not derive a bounded semantic opening")
    mesh.update_faces(~removed)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh, {
        "method": "semantic_open_garment_shell_from_sealed_boundary",
        "status": "sealed_collision_volume_opened_for_cloth",
        "axis": int(axis),
        "side": side,
        "cut_fraction": applied_fraction,
        "removed_face_count": int(removed.sum()),
        "limitation": "automatic opening preserves the source silhouette but is not a recovered sewing pattern",
    }


def _voxel_filled_volume_proxy(mesh: Any, np: Any, resolution: int = V2_VOXEL_RESOLUTION):
    """filled_volume: voxelize + fill + marching cubes, watertight-verified;
    convex hull is the fallback only.  Port of fast_visual.py:2857-2904
    (_create_filled_proxy)."""
    import trimesh

    pitch = float(max(mesh.extents) / max(24, int(resolution)))
    warnings: list[str] = []
    method = "voxel_fill_preserving_exterior_connected_voids"
    try:
        voxels = mesh.voxelized(pitch).fill()
        rebuilt = trimesh.voxel.ops.matrix_to_marching_cubes(voxels.matrix, pitch=1.0)
        rebuilt.apply_transform(voxels.transform)
        rebuilt.remove_unreferenced_vertices()
        rebuilt.fix_normals()
        if not rebuilt.is_watertight or not rebuilt.is_volume:
            raise ValueError("voxel result is not a closed volume")
        mesh = rebuilt
    except Exception as exc:
        mesh = mesh.convex_hull
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        method = "convex_hull_fallback"
        warnings.append(f"voxel_fill_failed: {type(exc).__name__}: {exc}")
        warnings.append("convex_hull_may_remove_semantic_concavities_or_through_holes")
    return mesh, {
        "method": method,
        "voxel_pitch_m": float(pitch),
        "voxel_resolution": int(resolution),
        "warnings": warnings,
    }


def _decimate_to_budget(mesh: Any, budget: int, np: Any):
    """Decimate toward ``budget`` faces; fast_simplification with rising
    aggressiveness, accepting the first pass that lands under 1.25x budget."""
    import fast_simplification
    import trimesh

    current = mesh
    for agg in (5, 7, 9):
        vertices, faces = fast_simplification.simplify(
            np.asarray(current.vertices, dtype=float),
            np.asarray(current.faces, dtype=np.int64),
            target_count=int(budget),
            agg=agg,
            lossless=False,
        )
        current = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if len(current.faces) <= int(1.25 * budget):
            break
    current.remove_unreferenced_vertices()
    return current


def _prepare_solver_mesh_v2(
    source_path: Path,
    scale_m: float,
    orientation: str,
    output_dir: Path,
    np: Any,
    cloth: bool,
    geometry_family: str,
) -> tuple[Path, Any, float, dict[str, Any]]:
    """v12.4-candidate solver-mesh adaptation.  Returns the OBJ path, the
    scaled extent, the median edge length, and an auditable details dict."""
    import trimesh

    combined = _load_source_mesh(source_path, np)
    details: dict[str, Any] = {"adaptation": "v2", "geometry_family": geometry_family}

    if cloth:
        surgery_warnings: list[str] = []
        if geometry_family == "cloth_sheet":
            combined, sheet_details = _flat_sheet_mid_surface(combined, np)
            details["proxy"] = sheet_details
        elif bool(combined.is_watertight):
            try:
                combined, opening = _sealed_shell_opening(combined, geometry_family, np)
                details["proxy"] = opening
                surgery_warnings.append("sealed_collision_volume_converted_to_open_cloth_shell")
            except Exception as exc:
                surgery_warnings.append(f"opening_surgery_failed: {type(exc).__name__}: {exc}")
                if geometry_family in ("garment", "shell", "cloth_sheet"):
                    combined, projection = _projected_mid_surface(combined, np)
                    details["proxy"] = projection
                    surgery_warnings.append("sealed_or_body_supported_shell_replaced_by_projected_mid_surface")
                else:
                    # Bags are never projected to a single layer (v11 bag
                    # shape-preservation policy); keep the sealed shell.
                    details["proxy"] = {"method": "source_boundary_surface"}
                    surgery_warnings.append("bag_opening_failed_sealed_shell_kept")
        else:
            details["proxy"] = {"method": "source_boundary_surface"}
        if len(combined.faces) > V2_SURFACE_FACE_BUDGET:
            try:
                combined = _decimate_to_budget(combined, V2_SURFACE_FACE_BUDGET, np)
                surgery_warnings.append("surface_decimated_to_face_budget_12000")
            except Exception as exc:
                surgery_warnings.append(f"surface_decimation_failed: {type(exc).__name__}: {exc}")
        details["warnings"] = surgery_warnings
        # Scale-before-rotate (v11 genesis_cloth.py:251-335): the physical
        # size belongs to the asset, not to the world-space drop pose.
        extent = np.asarray(combined.extents, dtype=float)
        extent = np.maximum(extent, 1.0e-6)
        combined.apply_scale(float(scale_m) / float(extent.max()))
        if orientation == "corner_first":
            combined.apply_transform(trimesh.transformations.rotation_matrix(math.radians(45.0), [0.0, 1.0, 0.0]))
            combined.apply_transform(trimesh.transformations.rotation_matrix(math.radians(35.264), [1.0, 0.0, 0.0]))
    else:
        extent = np.asarray(combined.extents, dtype=float)
        extent = np.maximum(extent, 1.0e-6)
        combined.apply_scale(float(scale_m) / float(extent.max()))
        if orientation == "corner_first":
            combined.apply_transform(trimesh.transformations.rotation_matrix(math.radians(45.0), [0.0, 1.0, 0.0]))
            combined.apply_transform(trimesh.transformations.rotation_matrix(math.radians(35.264), [1.0, 0.0, 0.0]))
        if len(combined.faces) > V2_SURFACE_FACE_BUDGET:
            # Bound the voxelization cost; the surface budget is far above the
            # marching-cubes output at resolution 40.
            combined = _decimate_to_budget(combined, V2_SURFACE_FACE_BUDGET, np)
        combined, volume_details = _voxel_filled_volume_proxy(combined, np)
        details["proxy"] = volume_details

    bounds = combined.bounds
    center_xy = 0.5 * (bounds[0, :2] + bounds[1, :2])
    combined.apply_translation([-center_xy[0], -center_xy[1], -bounds[0, 2]])
    destination = output_dir / "solver_mesh.obj"
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.export(destination, file_type="obj")
    scaled_extent = np.asarray(combined.extents, dtype=float)
    details["final_geometry"] = {
        "vertex_count": int(len(combined.vertices)),
        "face_count": int(len(combined.faces)),
        "watertight": bool(combined.is_watertight),
    }
    try:
        edge_lengths = np.asarray(combined.edges_unique_length, dtype=float)
        median_edge = float(np.median(edge_lengths)) if edge_lengths.size else float(scaled_extent.max()) / 32.0
    except Exception:
        median_edge = float(scaled_extent.max()) / 32.0
    return destination, scaled_extent, median_edge, details


def _make_v2_post_step_hook(
    soft: Any,
    np: Any,
    cloth: bool,
    dt_s: float,
    contact_height_m: float,
    lateral_rate_s_inv: float = 0.0,
):
    """v12.4 stepping hooks; None when v2/v3 is off.  All mechanisms are
    contact-gated or CoM-preserving (unlike the naive every-step velocity
    scaling the settle study rejected).  ``lateral_rate_s_inv`` > 0 (v12.5)
    additionally damps the lateral CoM components while in contact."""

    if cloth:
        rate = V2_CLOTH_POST_CONTACT_DAMPING_RATE_S_INV
        attenuation = math.exp(-rate * dt_s)
        state = {"active": False, "damped_steps": 0}

        def cloth_hook() -> None:
            pos = _positions(soft, np)
            if pos.size == 0 or not np.isfinite(pos).all():
                return
            if not state["active"]:
                if float(pos[:, 2].min()) <= contact_height_m:
                    state["active"] = True
            if not state["active"]:
                return
            vel = _velocities(soft, np, len(pos))
            if vel.shape != pos.shape or not np.isfinite(vel).all():
                return
            soft.set_particles_vel(vel * attenuation)
            state["damped_steps"] += 1

        cloth_hook.state = state  # type: ignore[attr-defined]
        cloth_hook.rate_s_inv = rate  # type: ignore[attr-defined]
        return cloth_hook

    ivr_rate = (1.0 - V2_IVR_EQUILIBRIUM_FRACTION) / V2_IVR_RELAXATION_TIME_S
    ivr_attenuation = math.exp(-ivr_rate * dt_s)
    cnd_attenuation = math.exp(-V2_CND_RATE_S_INV * dt_s)
    lateral_attenuation = math.exp(-lateral_rate_s_inv * dt_s)
    state = {"first_contact_step": None, "steps": 0, "upward_damped_steps": 0, "lateral_damped_steps": 0}

    def mpm_hook() -> None:
        pos = _positions(soft, np)
        if pos.size == 0 or not np.isfinite(pos).all():
            return
        vel = _velocities(soft, np, len(pos))
        if vel.shape != pos.shape or not np.isfinite(vel).all():
            return
        state["steps"] += 1
        # InternalVelocityRelaxation: damp only the deviatoric part.
        com = vel.mean(axis=0, keepdims=True)
        adjusted = com + ivr_attenuation * (vel - com)
        # ContactNormalDissipation: after first contact (plus the short
        # impact window that covers the ballistic rebound), damp only the
        # upward CoM component.
        in_contact = float(pos[:, 2].min()) <= contact_height_m
        if in_contact and state["first_contact_step"] is None:
            state["first_contact_step"] = state["steps"]
        in_window = bool(
            state["first_contact_step"] is not None
            and (state["steps"] - state["first_contact_step"]) * dt_s <= V2_CND_IMPACT_WINDOW_S
        )
        if in_contact or in_window:
            upward = max(float(com[0, 2]), 0.0)
            if upward > 0.0:
                adjusted[:, 2] -= (1.0 - cnd_attenuation) * upward
                state["upward_damped_steps"] += 1
        # v12.5: contact-gated lateral CoM dissipation (contact-friction
        # surrogate).  Only while actually touching; no impact window, so
        # free fall and rebound flight keep their lateral momentum.
        if in_contact and lateral_rate_s_inv > 0.0:
            lateral = np.hypot(com[0, 0], com[0, 1])
            if lateral > 0.0:
                adjusted[:, 0] -= (1.0 - lateral_attenuation) * com[0, 0]
                adjusted[:, 1] -= (1.0 - lateral_attenuation) * com[0, 1]
                state["lateral_damped_steps"] += 1
        soft.set_particles_vel(adjusted)

    mpm_hook.state = state  # type: ignore[attr-defined]
    mpm_hook.ivr_rate_s_inv = ivr_rate  # type: ignore[attr-defined]
    mpm_hook.cnd_rate_s_inv = V2_CND_RATE_S_INV  # type: ignore[attr-defined]
    mpm_hook.lateral_rate_s_inv = lateral_rate_s_inv  # type: ignore[attr-defined]
    return mpm_hook



def _build_numeric_scene(scene: Any) -> None:
    """Build solver state without the EGL rasterizer (see v12 worker)."""
    scene._parallelize(0, (0.0, 0.0), 1, True)
    scene._sim.build()
    scene._reset()
    scene._is_built = True
    scene._sim.step()
    scene._reset()


def _to_numpy(value: Any, np: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    return np.asarray(value)


def _positions(entity: Any, np: Any) -> Any:
    for name in ("get_particles_pos", "get_particles_position", "get_pos"):
        getter = getattr(entity, name, None)
        if not callable(getter):
            continue
        try:
            value = _to_numpy(getter(), np)
        except TypeError:
            value = _to_numpy(getter(relative=False), np)
        while value.ndim > 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim == 2 and value.shape[-1] == 3:
            return value.astype(float, copy=False)
        if value.ndim == 1 and value.size == 3:
            return value.reshape(1, 3).astype(float, copy=False)
    return np.empty((0, 3), dtype=float)


def _velocities(entity: Any, np: Any, count: int) -> Any:
    for name in ("get_particles_vel", "get_particles_velocity"):
        getter = getattr(entity, name, None)
        if not callable(getter):
            continue
        try:
            value = _to_numpy(getter(), np)
        except TypeError:
            value = _to_numpy(getter(relative=False), np)
        while value.ndim > 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim == 2 and value.shape[-1] == 3:
            return value.astype(float, copy=False)
    return np.zeros((count, 3), dtype=float)


def _make_scene(gs: Any, fixed: dict[str, Any], length: float, cloth: bool = False, pbd_particle_size: float | None = None, vis_options: Any = None, v3_cloth: bool = False, pbd_hash_cell_size: float | None = None, mpm_cpic: bool = False):
    margin = max(0.5 * length, 0.15, 8.0 * fixed["particle_size_m"])
    world_lo = (-0.75 * length - margin, -0.75 * length - margin, -margin)
    world_hi = (0.75 * length + margin, 0.75 * length + margin, 2.5 * length + margin)
    dt = fixed["dt_s"]
    # Rigid constraint timeconst: the v12.2 choice (2*dt/substeps) makes the
    # kinematic-plate velocity servo soft against contact resistance, and the
    # v12.5 bisect showed cloth compression plates then stop at the body
    # surface instead of engaging (garment_000933 class).  For v3 cloth scenes
    # keep the Genesis default (0.01); MPM and all non-v3 paths are unchanged.
    rigid_timeconst = 0.01 if v3_cloth else max(2.0 * dt / fixed["substeps"], 1.0e-4)
    pbd_particle = pbd_particle_size if pbd_particle_size else max(0.005 * length, 1.0e-3)
    pbd_extra: dict[str, Any] = {}
    if pbd_hash_cell_size is not None:
        # v12.6/v5 keeps PBD self-collision for non-sheet profiles; the hash
        # cell must cover the domain within the 150^3 grid clamp, otherwise
        # distant particles alias into one slot (the historical reason the
        # runner disabled self-collision entirely).
        pbd_extra["hash_grid_cell_size"] = max(1.25 * pbd_particle, float(pbd_hash_cell_size))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=fixed["substeps"], gravity=(0.0, 0.0, -9.81)),
        mpm_options=gs.options.MPMOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.81),
            particle_size=fixed["particle_size_m"],
            grid_density=fixed["grid_density"],
            lower_bound=world_lo,
            upper_bound=world_hi,
            # CPIC (compatible particle-in-cell) gives particle-level rigid
            # boundary tracking.  It is required for the kinematic compression
            # plates to actually transfer force into small MPM bodies
            # (p0_memory_foam_cushion: with CPIC off the plates pass their
            # full travel while the body extent never shrinks).  v3+ only, and
            # only for the compression scene; the drop boundary behavior is
            # unchanged.
            enable_CPIC=mpm_cpic,
        ),
        pbd_options=gs.options.PBDOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.81),
            # PBD self-collision pushes apart any pair closer than
            # particle_size whose rest distance is smaller; it must stay well
            # below the mesh vertex spacing or the sheet inflates/jitters.
            particle_size=pbd_particle,
            lower_bound=world_lo,
            upper_bound=world_hi,
            **pbd_extra,
        ) if cloth else None,
        rigid_options=gs.options.RigidOptions(
            gravity=(0.0, 0.0, -9.81),
            constraint_timeconst=rigid_timeconst,
        ),
        coupler_options=gs.options.LegacyCouplerOptions(),
        show_viewer=False,
        vis_options=vis_options,
        renderer=gs.renderers.Rasterizer(),
    )
    return scene


def _make_material(gs: Any, mode: str, asset: dict[str, Any], candidate: dict[str, Any]):
    E = min(1.0e6, max(100.0, _candidate_value(asset, candidate, "youngs_modulus_pa")))
    nu = min(0.49, max(-0.95, _candidate_value(asset, candidate, "poisson_ratio")))
    rho = min(2.0e4, max(5.0, _candidate_value(asset, candidate, "density_kg_m3")))
    requested_E = _candidate_value(asset, candidate, "youngs_modulus_pa")
    if mode == "mpm_elastoplastic":
        yield_stress = max(50.0, _candidate_value(asset, candidate, "yield_stress_pa"))
        material = gs.materials.MPM.ElastoPlastic(
            E=E, nu=nu, rho=rho, sampler="regular", von_mises_yield_stress=yield_stress
        )
    else:
        material = gs.materials.MPM.Elastic(E=E, nu=nu, rho=rho, sampler="regular", model="corotation")
    clamps = {"youngs_modulus_pa": {"requested": float(requested_E), "effective": float(E)}}
    return material, clamps


def _prior_param(asset: dict[str, Any], name: str, default: float) -> float:
    value = _prior_value(asset, name)
    if value is not None and math.isfinite(value) and value > 0:
        return value
    pv = (asset.get("prior") or {}).get("physical_values") or {}
    if isinstance(pv, dict) and name in pv:
        return _number(pv[name], default)
    return default


def _make_cloth_material(gs: Any, asset: dict[str, Any], candidate: dict[str, Any]):
    """Map whitelisted physical parameters onto PBD.Cloth.

    Auditable mapping (echoed in report.parameter_mapping):
      rho_area  = density_kg_m3 * thickness_m   (thickness is a prior-fixed
                  geometry attribute, never candidate-tunable)
      stretch_compliance = 1 / (E * thickness)  [m/N], clamped
      bending_compliance = 1 / bending_stiffness (prior-fixed), clamped
      static/kinetic friction from friction_static / friction_dynamic
    """
    E = min(1.0e9, max(1.0e3, _candidate_value(asset, candidate, "youngs_modulus_pa")))
    rho = min(2.0e4, max(5.0, _candidate_value(asset, candidate, "density_kg_m3")))
    thickness = min(1.0e-2, max(1.0e-5, _prior_param(asset, "thickness_m", 1.0e-3)))
    bending_stiffness = max(1.0e-8, _prior_param(asset, "bending_stiffness", 1.0e-3))
    friction = min(2.0, max(0.01, _candidate_value(asset, candidate, "friction_static")))
    kinetic = _candidate_value(asset, candidate, "friction_dynamic")
    if kinetic <= 0:
        kinetic = 0.9 * friction
    kinetic = min(2.0, max(0.01, kinetic))
    surface_density = rho * thickness
    stretch_compliance = min(1.0e-1, max(1.0e-7, 1.0 / (E * thickness)))
    bending_compliance = min(1.0e3, max(1.0e-4, 1.0 / bending_stiffness))
    material = gs.materials.PBD.Cloth(
        rho=surface_density,
        static_friction=friction,
        kinetic_friction=kinetic,
        bending_compliance=bending_compliance,
        stretch_compliance=stretch_compliance,
        # PBD air_resistance is a quadratic drag divided by *per-particle*
        # mass; with realistic thin-cloth mass it pins the sheet to a
        # terminal velocity far below free fall.  Keep it zero.
        air_resistance=0.0,
    )
    mapping = {
        "surface_density_kg_m2": {"value": float(surface_density), "formula": "density_kg_m3 * thickness_m"},
        "thickness_m": {"value": float(thickness), "source": "prior_fixed_geometry"},
        "stretch_compliance_m_n": {"value": float(stretch_compliance), "formula": "1 / (youngs_modulus_pa * thickness_m)"},
        "bending_compliance_rad_n": {"value": None, "note": "bending disabled: genesis 1.3.3 PBD bending assumes a flat rest angle per inner edge and is unstable on curved meshes"},
        "static_friction": {"value": float(friction), "formula": "friction_static"},
        "kinetic_friction": {"value": float(kinetic), "formula": "friction_dynamic or 0.9 * friction_static"},
    }
    return material, mapping


def _make_cloth_material_v5(gs: Any, asset: dict[str, Any], candidate: dict[str, Any], representation: dict[str, Any]):
    """v12.6/v5: cloth parameters come from the representation profile table
    (mesh_representation.py / genesis_adaptation_review.py recipe), not from
    the candidate compliance mapping.  Friction stays candidate-driven
    (production whitelist).  Bending compliance is only exercised through the
    source-rest-dihedral ShellConstraints kernel for rest_curvature profiles;
    for sheet/garment_cloth the bending projection is off and the material
    value is inert (iterations are zeroed)."""
    profile = str(representation.get("profile") or "")
    surface_density = 0.6 if profile == "reinforced_shell" else 0.35
    stretch_compliance = float(representation.get("stretch_compliance") or 1.0e-7)
    bending_raw = representation.get("bending_compliance")
    bending_compliance = float(bending_raw) if bending_raw is not None else 1.0
    friction = min(2.0, max(0.01, _candidate_value(asset, candidate, "friction_static")))
    kinetic = _candidate_value(asset, candidate, "friction_dynamic")
    if kinetic <= 0:
        kinetic = 0.9 * friction
    kinetic = min(2.0, max(0.01, kinetic))
    material = gs.materials.PBD.Cloth(
        rho=surface_density,
        static_friction=friction,
        kinetic_friction=kinetic,
        stretch_compliance=stretch_compliance,
        bending_compliance=bending_compliance,
        stretch_relaxation=0.3,
        bending_relaxation=0.12,
        air_resistance=0.0,
    )
    mapping = {
        "source": "v5_representation_profile",
        "profile": profile,
        "surface_density_kg_m2": {"value": float(surface_density), "source": "v5_review_recipe (0.6 reinforced_shell else 0.35)"},
        "stretch_compliance_m_n": {"value": float(stretch_compliance), "source": "v5 representation (1e-7)"},
        "bending_compliance_rad_n": {
            "value": None if bending_raw is None else float(bending_compliance),
            "source": "v5 representation profile; applied via ShellConstraints source-rest-dihedral kernel"
            if bending_raw is not None else "bending projection disabled for sheet/garment_cloth",
        },
        "stretch_relaxation": {"value": 0.3, "note": "solver projection strength, not a material property"},
        "bending_relaxation": {"value": 0.12, "note": "solver projection strength, not a material property"},
        "static_friction": {"value": float(friction), "formula": "friction_static"},
        "kinetic_friction": {"value": float(kinetic), "formula": "friction_dynamic or 0.9 * friction_static"},
    }
    return material, mapping


def _disable_pbd_remesh() -> None:
    """Skip Genesis' pymeshlab-based remesh; the solver mesh is pre-bounded."""
    from genesis.engine.mesh import Mesh

    if getattr(Mesh, "_softsim_fast_remesh", False):
        return

    def fast_remesh(self: Any, edge_len_abs: Any = None, edge_len_ratio: float = 0.01, fix: bool = True) -> None:
        self._invalidate_geometry_cache()
        self.clear_visuals()

    Mesh.remesh = fast_remesh
    Mesh._softsim_fast_remesh = True


def _add_soft_body(scene: Any, gs: Any, mesh_path: Path, material: Any, z_offset: float, adapted: bool = False):
    morph = gs.morphs.Mesh(
        file=str(mesh_path),
        pos=(0.0, 0.0, float(z_offset)),
        scale=1.0,
        decimate=False,
        watertighten=None,
        collision=True,
        file_meshes_are_zup=True,
        **({"convexify": False} if adapted else {}),
    )
    return scene.add_entity(
        morph,
        material=material,
        surface=gs.surfaces.Default(color=(0.68, 0.32, 0.18), roughness=0.62, double_sided=True),
    )


# --------------------------------------------------------------------------- #
# Metric reduction from recorded trajectories
# --------------------------------------------------------------------------- #


def _reduce_records(records: list[dict[str, Any]], np: Any) -> dict[str, Any]:
    """Common per-record reduction: finite flags, speeds, bbox, min_z."""
    out = {
        "n_records": len(records),
        "finite_flags": [],
        "mean_speeds": [],
        "min_z": [],
        "z_extent": [],
        "axis_extent": [],
        "centroids": [],
    }
    for record in records:
        pos = record["pos"]
        vel = record["vel"]
        finite = bool(np.isfinite(pos).all() and np.isfinite(vel).all()) if pos.size else False
        out["finite_flags"].append(finite)
        if finite and pos.size:
            speeds = np.linalg.norm(vel, axis=1) if vel.shape == pos.shape else np.zeros(len(pos))
            out["mean_speeds"].append(float(np.mean(speeds)))
            out["min_z"].append(float(pos[:, 2].min()))
            out["z_extent"].append(float(np.ptp(pos[:, 2])))
            out["axis_extent"].append(np.ptp(pos, axis=0))
            out["centroids"].append(pos.mean(axis=0))
        else:
            out["mean_speeds"].append(float("inf"))
            out["min_z"].append(float("nan"))
            out["z_extent"].append(float("nan"))
            out["axis_extent"].append(np.full(3, np.nan))
            out["centroids"].append(np.full(3, np.nan))
    return out


def _centroid_speeds(centroids: list[Any], dt_record: float, np: Any) -> list[float]:
    speeds = []
    previous = None
    for centroid in centroids:
        if previous is None or not np.isfinite(centroid).all() or not np.isfinite(previous).all():
            speeds.append(float("nan") if previous is not None else 0.0)
        else:
            speeds.append(float(np.linalg.norm(centroid - previous)) / max(dt_record, 1.0e-9))
        previous = centroid
    return speeds


def _settled(mean_speeds: list[float]) -> tuple[bool, float, float]:
    finite_speeds = [s for s in mean_speeds if math.isfinite(s)]
    if len(finite_speeds) < 5:
        return False, 0.0, float("inf") if not finite_speeds else finite_speeds[-1]
    tail_count = max(3, int(math.ceil(0.2 * len(finite_speeds))))
    tail = finite_speeds[-tail_count:]
    peak = max(finite_speeds)
    tail_mean = sum(tail) / len(tail)
    settled = peak > 0.0 and tail_mean <= 0.05 * peak
    return bool(settled), float(peak), float(tail_mean)


def _loss_from_components(components: list[float]) -> float:
    if not components:
        return 1.0
    return float(sum(c * c for c in components) / len(components))


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #


def _trajectory_summary(records: list[dict[str, Any]], np: Any, extra_keys: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    summary = []
    for record in records:
        pos = record["pos"]
        vel = record["vel"]
        if pos.size and np.isfinite(pos).all():
            speeds = np.linalg.norm(vel, axis=1) if vel.shape == pos.shape and np.isfinite(vel).all() else None
            entry = {
                "step": record["step"],
                "mean_speed_m_s": float(np.mean(speeds)) if speeds is not None else None,
                "min_z_m": float(pos[:, 2].min()),
                "extent_m": [float(x) for x in np.ptp(pos, axis=0)],
                "centroid_m": [float(x) for x in pos.mean(axis=0)],
            }
        else:
            entry = {"step": record["step"], "mean_speed_m_s": None, "min_z_m": None, "extent_m": None, "centroid_m": None}
        if record.get("phase"):
            entry["phase"] = record["phase"]
        for key in extra_keys:
            if key in record:
                entry[key] = record[key]
        summary.append(entry)
    return summary


def _run_drop(gs: Any, np: Any, scene: Any, soft: Any, probe: dict[str, Any], fixed: dict[str, Any], tolerance: float, step_fn: Any, contact_eps: float, penetration_base: float, deformation_band: list[float], settle_hard_gate: bool = True) -> dict[str, Any]:
    config = probe.get("config") or {}
    steps = max(20, int(_number(config.get("steps"), 400)))
    record_every = max(1, int(_number(config.get("record_every"), 4)))
    particle_size = fixed["particle_size_m"]

    initial = _positions(soft, np)
    if initial.size == 0:
        raise RuntimeError("Genesis entity exposed no particle positions after build")
    h0 = float(np.ptp(initial[:, 2]))
    if h0 <= 0:
        raise RuntimeError("degenerate initial geometry (zero z-extent)")

    records: list[dict[str, Any]] = []
    prev_pos = None
    record_dt = record_every * fixed["dt_s"]
    for step in range(steps):
        step_fn()
        if (step + 1) % record_every == 0 or step == steps - 1:
            pos = _positions(soft, np)
            vel = _velocities(soft, np, len(pos))
            if pos.size == 0 or not np.isfinite(vel).all():
                # PBD entities do not maintain a readable velocity buffer
                # (get_particles_vel returns NaN); fall back to finite
                # differences of recorded positions.
                vel = (pos - prev_pos) / record_dt if prev_pos is not None and prev_pos.shape == pos.shape else np.zeros_like(pos)
            prev_pos = pos
            records.append({"step": step + 1, "pos": pos, "vel": vel})

    reduced = _reduce_records(records, np)
    n = max(reduced["n_records"], 1)
    finite_flags = reduced["finite_flags"]
    finite_fraction = sum(1 for f in finite_flags if f) / n

    # All thresholds are relative to the asset/numerics scale, never absolute.
    # MPM: LegacyCoupler keeps the lowest particle center ~1 grid cell
    # (dx = 1/grid_density) above the floor.  Cloth (PBD): scale-relative band.
    # Both are computed by the caller and echoed in the metrics.
    contact_flags = [
        bool(math.isfinite(z) and z <= contact_eps) for z in reduced["min_z"]
    ]
    contact_fraction = sum(contact_flags) / n

    # Settle is judged on bulk (centroid) motion: undamped explicit MPM keeps
    # low-level internal particle jitter long after the body is macroscopically
    # at rest.  Mean particle speed is still reported for diagnosis.
    record_dt = record_every * fixed["dt_s"]
    centroid_speeds = _centroid_speeds(reduced["centroids"], record_dt, np)
    settled, peak_speed, tail_speed = _settled(centroid_speeds)
    _, mean_peak_speed, mean_tail_speed = _settled(reduced["mean_speeds"])

    contact_heights = [h for h, c in zip(reduced["z_extent"], contact_flags) if c and math.isfinite(h)]
    min_contact_height = min(contact_heights) if contact_heights else h0
    deformation_ratio = 1.0 - min_contact_height / h0

    finite_min_z = [z for z in reduced["min_z"] if math.isfinite(z)]
    min_z_ever = min(finite_min_z) if finite_min_z else float("nan")
    penetration_margin = max(penetration_base, 0.02 * h0)
    ground_penetration = bool(math.isfinite(min_z_ever) and min_z_ever < -penetration_margin)

    loss_components = [
        1.0 - finite_fraction,
        max(0.0, 0.5 - contact_fraction) / 0.5,
        # settle_gate=off (human-calibrated): the settle term must not inflate
        # the loss the elastic-budget progress rule reads.
        0.0 if (settled or not settle_hard_gate) else 1.0,
        max(0.0, deformation_band[0] - deformation_ratio, deformation_ratio - deformation_band[1])
        / (deformation_band[1] - deformation_band[0]),
        1.0 if ground_penetration else 0.0,
    ]
    loss = _loss_from_components(loss_components)

    behavior_pass = bool(
        finite_fraction >= 1.0
        and contact_fraction > 0.0
        and (settled or not settle_hard_gate)
        and deformation_band[0] <= deformation_ratio <= deformation_band[1]
        and not ground_penetration
    )
    return {
        "behavior_pass": behavior_pass,
        "finite_fraction": float(finite_fraction),
        "contact_fraction": float(contact_fraction),
        "settled": bool(settled),
        "deformation_ratio": float(deformation_ratio),
        "deformation_band": deformation_band,
        "ground_penetration": ground_penetration,
        "min_z_ever_m": float(min_z_ever) if math.isfinite(min_z_ever) else None,
        "min_contact_height_m": float(min_contact_height),
        "initial_height_m": float(h0),
        "contact_eps_m": float(contact_eps),
        "penetration_margin_m": float(penetration_margin),
        "settled_basis": "centroid_speed",
        "peak_centroid_speed_m_s": peak_speed,
        "tail_centroid_speed_m_s": tail_speed,
        "peak_mean_speed_m_s": mean_peak_speed,
        "tail_mean_speed_m_s": mean_tail_speed,
        "loss": float(loss),
        "particle_count": int(len(initial)),
        "recorded_steps": reduced["n_records"],
        "executed_steps": steps,
        "record_every": record_every,
        "stimulus_kind": "drop",
        "trajectory": _trajectory_summary(records, np),
    }


def _run_compression(
    gs: Any,
    np: Any,
    scene: Any,
    soft: Any,
    plates: list[Any],
    probe: dict[str, Any],
    fixed: dict[str, Any],
    tolerance: float,
    axis_index: int,
    compression_ratio: float,
    plate_travel: float,
    plate_gap0: float,
    plate_thickness: float,
    step_fn: Any,
    contact_eps: float,
    penetration_base: float,
    deformation_band_min: float,
    recovery_band: list[float],
    recovery_min_reference_m: float = 0.0,
    servo_on: bool = False,
    prescribed_motion: bool = False,
    plate_geometry: dict[str, Any] | None = None,
    settle_hard_gate: bool = True,
    extent_trim_fraction: float = 0.0,
) -> dict[str, Any]:
    config = probe.get("config") or {}
    action = probe.get("action") or {}
    steps = max(60, int(_number(config.get("steps"), 560)))
    record_every = max(1, int(_number(config.get("record_every"), 8)))
    hold_steps = max(1, int(_number(action.get("hold_steps"), 28)))
    release_steps = max(1, int(_number(action.get("release_steps"), 28)))
    particle_size = fixed["particle_size_m"]
    dt = fixed["dt_s"]

    compress_steps = max(40, int(0.35 * steps))
    if compress_steps + hold_steps + release_steps >= steps:
        compress_steps = max(20, steps - hold_steps - release_steps - 20)

    approach_speed = plate_travel / (compress_steps * dt)
    retract_speed = plate_travel / (release_steps * dt)
    motion = None
    if prescribed_motion:
        from mesh_probe_geometry import PrescribedPlateMotion
        motion = PrescribedPlateMotion(plates, axis_index, dt, plate_travel,
                                       compress_steps, hold_steps, release_steps)

    initial = _positions(soft, np)
    if initial.size == 0:
        raise RuntimeError("Genesis entity exposed no particle positions after build")
    h0 = float(np.ptp(initial[:, axis_index]))
    if h0 <= 0:
        raise RuntimeError("degenerate initial geometry along compression axis")

    # v12.5 engagement servo: the fixed kinematic schedule assumes the body
    # stays put and deforms with the closing gap.  Soft bodies instead squash
    # out / spread / escape (the v12.4 "plates never engaged" failure class):
    # the gap closes by the commanded amount while the body axis extent never
    # shrinks.  Under v3 the compress phase servos on the MEASURED body axis
    # extent: plates keep advancing while the body has not reached the
    # commanded compression, and stop (hold) once it has.  A gap floor keeps
    # the plates from sandwiching to nothing; if the floor is hit first the
    # shortfall is recorded in compression_ratio_achieved_by_plates.  Contract
    # thresholds are unchanged; the stimulus just honestly chases its own
    # commanded ratio.
    servo_target_extent = (1.0 - compression_ratio) * h0
    servo_gap_floor = max(0.3 * h0, 2.0 * particle_size)
    servo_active = bool(servo_on)
    servo_state = {"target_extent_m": servo_target_extent, "gap_floor_m": servo_gap_floor, "engaged_steps": 0, "capped": False}
    initial_plate_gap: float | None = None
    if servo_active and plates:
        plate_positions = [float(np.asarray(_to_numpy(p.get_pos(), np)).ravel()[axis_index]) for p in plates]
        if len(plate_positions) == 2 and all(math.isfinite(p) for p in plate_positions):
            initial_plate_gap = abs(plate_positions[0] - plate_positions[1])
            servo_state["initial_gap_m"] = initial_plate_gap

    axis_vel = [0.0, 0.0, 0.0]
    records: list[dict[str, Any]] = []
    prev_pos = None
    current_pos = initial
    for step in range(steps):
        if step < compress_steps:
            speed = approach_speed
            if servo_active:
                extent_now = float(np.ptp(current_pos[:, axis_index])) if current_pos.size else h0
                gap_now = None
                if records and records[-1].get("plate_axis_m"):
                    gap_now = abs(records[-1]["plate_axis_m"][0] - records[-1]["plate_axis_m"][1])
                if extent_now <= servo_target_extent:
                    speed = 0.0  # commanded compression achieved on the body
                elif gap_now is not None and gap_now <= servo_gap_floor:
                    speed = 0.0
                    servo_state["capped"] = True
                else:
                    servo_state["engaged_steps"] += 1
        elif step < compress_steps + hold_steps:
            speed = 0.0
        elif step < compress_steps + hold_steps + release_steps:
            speed = -retract_speed
            # v12.5: with the engagement servo the plates may be deeper than
            # the nominal schedule; keep retracting until the gap is back to
            # its initial value so the settle phase is genuinely plate-free.
            if servo_active and initial_plate_gap is not None:
                gap_now = None
                if records and records[-1].get("plate_axis_m"):
                    gap_now = abs(records[-1]["plate_axis_m"][0] - records[-1]["plate_axis_m"][1])
                if gap_now is not None and gap_now >= initial_plate_gap:
                    speed = 0.0
        else:
            speed = 0.0
        if motion is not None:
            motion.apply(step)
        else:
            for sign, plate in zip((1.0, -1.0), plates):
                velocity = list(axis_vel)
                velocity[axis_index] = -sign * speed  # left plate (+axis side) moves inward
                velocity.extend([0.0, 0.0, 0.0])
                plate.set_dofs_velocity(velocity)
        step_fn()
        pos_now = _positions(soft, np)
        if pos_now.size and np.isfinite(pos_now).all():
            current_pos = pos_now
        if (step + 1) % record_every == 0 or step == steps - 1:
            pos = pos_now
            vel = _velocities(soft, np, len(pos))
            if pos.size == 0 or not np.isfinite(vel).all():
                # PBD entities expose NaN velocities; finite-difference them.
                record_dt = record_every * dt
                vel = (pos - prev_pos) / record_dt if prev_pos is not None and prev_pos.shape == pos.shape else np.zeros_like(pos)
            prev_pos = pos
            phase = (
                "compress" if step < compress_steps
                else "hold" if step < compress_steps + hold_steps
                else "release" if step < compress_steps + hold_steps + release_steps
                else "settle"
            )
            plate_axis = []
            for plate in plates:
                plate_pos = _to_numpy(plate.get_pos(), np)
                plate_axis.append(float(np.asarray(plate_pos).ravel()[axis_index]))
            records.append({"step": step + 1, "pos": pos, "vel": vel, "phase": phase, "plate_axis_m": plate_axis})

    reduced = _reduce_records(records, np)
    n = max(reduced["n_records"], 1)
    finite_fraction = sum(1 for f in reduced["finite_flags"] if f) / n

    # Plate contact uses the measured plate positions of each record: contact
    # when the body's axis extremes reach either plate's current inner face.
    # contact_eps is scale-relative and computed by the caller.
    half_thickness = 0.5 * plate_thickness
    contact_flags = []
    for record in records:
        pos = record["pos"]
        plate_axis = record.get("plate_axis_m")
        if pos.size == 0 or not np.isfinite(pos).all() or not plate_axis:
            contact_flags.append(False)
            continue
        inner_hi = plate_axis[0] - half_thickness  # plate on the +axis side
        inner_lo = plate_axis[1] + half_thickness
        hi = float(pos[:, axis_index].max())
        lo = float(pos[:, axis_index].min())
        contact_flags.append(bool(hi >= inner_hi - contact_eps or lo <= inner_lo + contact_eps))
    contact_fraction = sum(contact_flags) / n

    # Compression settle is judged on bulk (centroid) motion after release:
    # undamped explicit MPM keeps low-level internal jitter for a long time,
    # which is not the "is the body at rest" question this probe asks.
    record_dt = record_every * dt
    centroid_speeds = _centroid_speeds(reduced["centroids"], record_dt, np)
    settled, peak_speed, tail_speed = _settled(centroid_speeds)
    mean_settled, mean_peak_speed, mean_tail_speed = _settled(reduced["mean_speeds"])

    def axis_extent_of(record: dict[str, Any]) -> float:
        pos = record["pos"]
        if pos.size == 0 or not np.isfinite(pos).all():
            return float("nan")
        if extent_trim_fraction > 0.0:
            # v12.6: min/max of a multi-thousand-particle cloud is pinned by a
            # handful of corner particles pinched out between the plate and
            # the floor (measured: 20/6138 particles moved the cushion extent
            # from the squeezed bulk 0.049 to 0.065).  The trimmed range
            # measures the bulk body the contract asks about.
            lo, hi = np.quantile(pos[:, axis_index], [extent_trim_fraction, 1.0 - extent_trim_fraction])
            return float(hi - lo)
        return float(np.ptp(pos[:, axis_index]))

    compress_extents = [axis_extent_of(r) for r in records if r["phase"] in {"compress", "hold"}]
    compress_extents = [e for e in compress_extents if math.isfinite(e)]
    min_axis_extent = float(min(compress_extents)) if compress_extents else h0
    deformation_ratio = (h0 - min_axis_extent) / h0

    settle_extents = [axis_extent_of(r) for r in records if r["phase"] == "settle"]
    settle_extents = [e for e in settle_extents if math.isfinite(e)]
    final_pool = settle_extents[-max(2, len(settle_extents) // 3):] if settle_extents else []
    final_axis_extent = float(np.mean(final_pool)) if final_pool else float("nan")

    # Recovery is judged on the VERTICAL (z) height relative to the resting
    # pre-contact baseline, centered at 1.0: lateral compression squashes the
    # body vertically and a healthy rebound returns to ~its original height.
    # (The previous definition referenced the max-compression height, which
    # mis-flagged exactly that healthy rebound as a failure; it is retained as
    # recovery_vs_compressed for audit.  The axis-extent recovery is kept as
    # recovery_axis_extent.)
    z_series = reduced["z_extent"]
    pre_pool = [z for z, c in zip(z_series, contact_flags) if not c and math.isfinite(z)]
    z_pre = float(np.mean(pre_pool)) if pre_pool else float(np.ptp(initial[:, 2]))
    # v12.5: a perfectly flat mid-surface proxy has z_pre == 0, which makes the
    # height-based recovery undefined (NaN auto-fail).  Fall back to the
    # compression-axis extent recovery (already measured) and record the
    # reference switch explicitly.  Thresholds are unchanged.
    recovery_reference = "vertical_height"
    if recovery_min_reference_m > 0.0 and not (math.isfinite(z_pre) and z_pre > recovery_min_reference_m):
        recovery_reference = "compression_axis_extent"
    compress_z = [z for z, r in zip(z_series, records) if r["phase"] in {"compress", "hold"} and math.isfinite(z)]
    z_min_compressed = float(min(compress_z)) if compress_z else float("nan")
    settle_z = [z for z, r in zip(z_series, records) if r["phase"] == "settle" and math.isfinite(z)]
    z_final_pool = settle_z[-max(2, len(settle_z) // 3):] if settle_z else []
    z_final = float(np.mean(z_final_pool)) if z_final_pool else float("nan")
    recovery_ratio = z_final / z_pre if math.isfinite(z_final) and z_pre > 0 else float("nan")
    recovery_vs_compressed = (
        z_final / z_min_compressed
        if math.isfinite(z_final) and math.isfinite(z_min_compressed) and z_min_compressed > 0
        else float("nan")
    )
    compression_flattening = (
        z_min_compressed / z_pre if math.isfinite(z_min_compressed) and z_pre > 0 else float("nan")
    )
    recovery_axis_extent = final_axis_extent / h0 if math.isfinite(final_axis_extent) else float("nan")
    if recovery_reference == "compression_axis_extent":
        recovery_ratio = recovery_axis_extent

    finite_min_z = [z for z in reduced["min_z"] if math.isfinite(z)]
    min_z_ever = min(finite_min_z) if finite_min_z else float("nan")
    penetration_margin = max(penetration_base, 0.02 * h0)
    ground_penetration = bool(math.isfinite(min_z_ever) and min_z_ever < -penetration_margin)

    # Achieved compression from measured plate travel (contact resistance can
    # slow the plates below their commanded kinematic travel).
    plate_axis_series = [r.get("plate_axis_m") for r in records if r.get("phase") in {"compress", "hold"} and r.get("plate_axis_m")]
    if plate_axis_series:
        gaps = [abs(p[0] - p[1]) for p in plate_axis_series]
        initial_gap = abs(records[0]["plate_axis_m"][0] - records[0]["plate_axis_m"][1]) if records[0].get("plate_axis_m") else None
        achieved_compression = (initial_gap - min(gaps)) / h0 if initial_gap else float("nan")
    else:
        achieved_compression = float("nan")

    deformation_band = [deformation_band_min, 0.95]
    # MPM solids are elastic: recovery is symmetric around 1.0.  Cloth under
    # PBD has no bending restoring force (see limitations), so crumpling and
    # staying low is normal cloth behavior; the contract only rejects
    # over-recovery (inflation past the pre-contact height).
    recovery_ok = bool(math.isfinite(recovery_ratio) and recovery_band[0] <= recovery_ratio <= recovery_band[1])
    loss_components = [
        1.0 - finite_fraction,
        max(0.0, 0.5 - contact_fraction) / 0.5,
        # settle_gate=off (human-calibrated): keep the settle term out of the
        # loss the elastic-budget progress rule reads.
        0.0 if (settled or not settle_hard_gate) else 1.0,
        max(0.0, deformation_band[0] - deformation_ratio, deformation_ratio - deformation_band[1])
        / (deformation_band[1] - deformation_band[0]),
        0.0 if recovery_ok else max(0.0, recovery_band[0] - recovery_ratio, recovery_ratio - recovery_band[1]) / max(recovery_band[1] - recovery_band[0], 1.0e-6) if math.isfinite(recovery_ratio) else 1.0,
        1.0 if ground_penetration else 0.0,
    ]
    loss = _loss_from_components(loss_components)

    behavior_pass = bool(
        finite_fraction >= 1.0
        and contact_fraction > 0.0
        and (settled or not settle_hard_gate)
        and deformation_band[0] <= deformation_ratio <= deformation_band[1]
        and recovery_ok
        and not ground_penetration
    )
    result = {
        "behavior_pass": behavior_pass,
        "finite_fraction": float(finite_fraction),
        "contact_fraction": float(contact_fraction),
        "settled": bool(settled),
        "deformation_ratio": float(deformation_ratio),
        "deformation_band": deformation_band,
        "recovery_ratio": float(recovery_ratio) if math.isfinite(recovery_ratio) else None,
        "recovery_band": recovery_band,
        "recovery_vs_compressed": float(recovery_vs_compressed) if math.isfinite(recovery_vs_compressed) else None,
        "compression_flattening": float(compression_flattening) if math.isfinite(compression_flattening) else None,
        "recovery_axis_extent": float(recovery_axis_extent) if math.isfinite(recovery_axis_extent) else None,
        "recovery_reference": recovery_reference,
        "height_pre_contact_m": float(z_pre),
        "height_min_compressed_m": float(z_min_compressed) if math.isfinite(z_min_compressed) else None,
        "height_after_release_m": float(z_final) if math.isfinite(z_final) else None,
        "ground_penetration": ground_penetration,
        "min_z_ever_m": float(min_z_ever) if math.isfinite(min_z_ever) else None,
        "initial_axis_extent_m": float(h0),
        "min_axis_extent_m": float(min_axis_extent),
        "final_axis_extent_m": float(final_axis_extent) if math.isfinite(final_axis_extent) else None,
        "compression_axis": "xyz"[axis_index],
        "compression_ratio_commanded": float(compression_ratio),
        "compression_ratio_achieved_by_plates": float(achieved_compression) if math.isfinite(achieved_compression) else None,
        "engagement_servo": servo_state if servo_on else None,
        "peak_centroid_speed_m_s": peak_speed,
        "tail_centroid_speed_m_s": tail_speed,
        "peak_mean_speed_m_s": mean_peak_speed,
        "tail_mean_speed_m_s": mean_tail_speed,
        "settled_basis": "centroid_speed",
        "contact_eps_m": float(contact_eps),
        "penetration_margin_m": float(penetration_margin),
        "loss": float(loss),
        "particle_count": int(len(initial)),
        "recorded_steps": reduced["n_records"],
        "executed_steps": steps,
        "record_every": record_every,
        "phase_steps": {"compress": compress_steps, "hold": hold_steps, "release": release_steps},
        "stimulus_kind": "compression",
        "trajectory": _trajectory_summary(records, np, extra_keys=("plate_axis_m",)),
    }
    if motion is not None:
        from mesh_probe_geometry import contact_coverage
        coverage = [contact_coverage(r["pos"], r["plate_axis_m"], axis_index,
                                    plate_geometry["plate_size_m"], plate_geometry["plate_center_z_m"], contact_eps)
                    for r in records if r["phase"] in {"compress", "hold"}]
        result["probe_geometry_audit"] = {
            **motion.diagnostics(),
            "finite_footprint_contact_records": sum(c["contacting_particles"] > 0 for c in coverage),
            "opposed_contact_records": sum(c["opposed_contact"] for c in coverage),
            "maximum_contacting_particles": max((c["contacting_particles"] for c in coverage), default=0),
            "evaluated_records": len(coverage),
        }
        if not result["probe_geometry_audit"]["finite_footprint_contact_records"]:
            result["behavior_pass"] = False
            result["loss"] = max(result["loss"], 1.0)
    return result


# --------------------------------------------------------------------------- #
# Main run
# --------------------------------------------------------------------------- #


def run(asset: dict[str, Any], candidate: dict[str, Any], probe: dict[str, Any], simulator_config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import numpy as np
    import genesis as gs

    report = _base_report(asset, candidate, probe, simulator_config)
    stimulus = probe.get("stimulus") or {}
    probe_kind = str(stimulus.get("kind") or "")
    tolerance = _number(probe.get("tolerance"), 0.05)

    if probe_kind not in {"drop", "compression"}:
        report["diagnosis"] = {
            "kind": "unsupported_probe",
            "suggested_updates": {},
            "detail": f"stimulus kind {probe_kind!r} not implemented by genesis_behavior runner",
        }
        report["metrics"] = _sentinel_metrics(probe)
        report["explicit_behavior_evidence"] = True
        return report

    mesh = _resolve_mesh(asset)
    if not mesh:
        report["execution_status"] = "failed"
        report["diagnosis"] = {
            "kind": "asset_missing",
            "suggested_updates": {},
            "detail": "no collision_mesh/visual_mesh resolved from resolved_artifacts or artifact_refs",
        }
        report["metrics"] = _sentinel_metrics(probe)
        report["explicit_behavior_evidence"] = True
        return report

    supported, mode = _route_supported(asset)
    route = asset.get("route") if isinstance(asset.get("route"), dict) else {}
    report["provenance"]["route"] = {k: route.get(k) for k in ("geometry_family", "material_family", "model_family", "solver_adapter", "proxy_mode")}
    if not supported:
        report["execution_status"] = "completed"
        report["behavior_pass"] = False
        report["passed"] = False
        report["explicit_behavior_evidence"] = True
        report["metrics"] = _sentinel_metrics(probe)
        report["diagnosis"] = {
            "kind": "unsupported_model",
            "suggested_updates": {},
            "detail": "route requires a model class this runner does not implement (e.g. damage/fracture); refusing to fake behavior",
        }
        report["limitations"] = BASE_LIMITATIONS + ["route_not_executed_unsupported_model"]
        return report

    v123 = _v123_enabled(probe, simulator_config)
    # Human-calibrated gate overrides (gate_study, 10 raters): all opt-in.
    gate_overrides = _gate_overrides(probe, simulator_config)
    settle_hard_gate = _settle_hard_gate(gate_overrides)
    v5 = _mesh_adaptation_v5_enabled(probe, simulator_config)
    v4 = _mesh_adaptation_v4_enabled(probe, simulator_config) and not v5
    v3 = _mesh_adaptation_v3_enabled(probe, simulator_config) or v4 or v5
    v2 = _mesh_adaptation_v2_enabled(probe, simulator_config) or v3
    geometry_family = _geometry_family(asset)

    # v12.6: the v5 representation (object name takes precedence over the
    # coarse dataset family) drives the solver route, overriding the prior
    # model_family routing.  The override is recorded as evidence.
    v5_representation: dict[str, Any] | None = None
    route_override: dict[str, Any] | None = None
    if v5:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import mesh_representation

        v5_representation = mesh_representation.select_representation(asset)
        v5_profile = str(v5_representation["profile"])
        v5_cloth = v5_profile != "filled_volume"
        prior_mode = mode
        if v5_cloth:
            mode = "pbd_cloth"
        elif not mode.startswith("mpm"):
            mode = "mpm_elastoplastic"
        route_override = {
            "prior_model_family": route.get("model_family"),
            "prior_solver_adapter": route.get("solver_adapter"),
            "prior_mode": prior_mode,
            "v5_profile": v5_profile,
            "v5_solver": v5_representation["solver"],
            "v5_reason": v5_representation.get("reason"),
            "overridden": prior_mode != mode,
        }

    cloth = mode == "pbd_cloth"

    _headless()
    study_dir = _study_render_dir()
    fixed = _fixed_numerics(asset, simulator_config)
    substeps_clamped_from = None
    if cloth and v123 and int(fixed["substeps"]) > 8:
        # v12.3: measured that PBD cloth integration degenerates at
        # substeps >= 16 (gravity increment lost / frozen or overshooting);
        # substeps <= 8 fall correctly.  MPM path is unaffected.
        substeps_clamped_from = int(fixed["substeps"])
        fixed["substeps"] = 8
    v2_cloth_numerics_from = None
    if cloth and v2:
        # v12.4: the v11-validated PBD configuration is dt=0.003 s with 10
        # substeps (genesis_cloth.py:338-391).  It supersedes the v12.3
        # substeps clamp, which was measured at dt=0.001.
        v2_cloth_numerics_from = {"dt_s": float(fixed["dt_s"]), "substeps": int(fixed["substeps"])}
        fixed["dt_s"] = 0.003
        fixed["substeps"] = 10
    mpm_solver_scale_capped_from = None
    if v3 and not cloth and float(fixed["scale_m"]) > V3_MPM_SOLVER_SCALE_CAP_M:
        # v12.5 solver-scale normalization (the v11 pipeline's
        # solver_longest_dimension_m idea): measured on Genesis 1.3.3 that at
        # the contract-fixed grid_density=64 (dx=15.6mm), MPM bodies larger
        # than ~0.5m sink through the grid-coupled floor regardless of
        # particle size (clean icosphere: min_z +0.010 at 0.5m, -0.057 at
        # 0.64m, -0.42 at 1.4m, -0.66 at 1.9m).  Simulate at the capped solver
        # scale with the scale/24 particle ratio preserved; all contract
        # thresholds are scale-relative so the behavior contract is unchanged.
        # The semantic metric scale (prior) stays in metrics.scale_m.
        mpm_solver_scale_capped_from = float(fixed["scale_m"])
        ratio = V3_MPM_SOLVER_SCALE_CAP_M / mpm_solver_scale_capped_from
        fixed["scale_m"] = V3_MPM_SOLVER_SCALE_CAP_M
        fixed["particle_size_m"] = float(fixed["particle_size_m"]) * ratio
        # drop_height_m: the contract rule min(0.5*scale, 0.25) saturates at
        # 0.25 for every scale >= 0.5, so the capped scene keeps the same
        # absolute drop height as the uncapped one.
    mpm_grid_refined_from = None
    if v3 and not cloth and float(fixed["scale_m"]) * float(fixed["grid_density"]) < V3_MPM_MIN_CELLS_PER_BODY:
        # v12.6 small-asset grid refinement (mirror of the v3 scale cap):
        # grid_density is cells per METER, so at the contract-fixed 64 a body
        # smaller than ~0.25m spans < 16 grid cells and the kinematic
        # compression plates cannot resolve a 0.22 squeeze through the
        # grid-coupled boundary (measured on p0_memory_foam_cushion: plates
        # close their full travel while the body extent never shrinks).
        # Refine the grid so the body spans >= 16 cells; contract thresholds
        # are scale-relative and unchanged.
        mpm_grid_refined_from = float(fixed["grid_density"])
        fixed["grid_density"] = float(math.ceil(V3_MPM_MIN_CELLS_PER_BODY / float(fixed["scale_m"])))
    config = probe.get("config") or {}
    seed = int(fixed["seed"]) + int(_number(config.get("seed_offset"), 0))
    length = float(fixed["scale_m"])
    orientation = str(stimulus.get("orientation") or "flat")
    grid_dx = 1.0 / float(fixed["grid_density"])
    compression_ratio = min(0.6, max(0.02, _number(stimulus.get("compression_ratio"), 0.22)))

    # Scale-relative thresholds.  MPM: LegacyCoupler keeps particle centers
    # ~1 grid cell off boundaries.  Cloth (PBD): particle-per-vertex contact
    # is tighter, so a smaller scale-relative band suffices.
    if cloth:
        contact_eps = max(0.02 * length, 0.5 * grid_dx)
        penetration_base = 0.02 * length
        drop_deformation_band = [0.0, 0.98]  # cloth legitimately deforms a lot
        compression_deformation_min = 0.0
        # PBD cloth has no bending restoring force: it crumples and stays low,
        # which is normal.  Only reject over-recovery (inflation past the
        # pre-contact height), the actual PBD failure mode.
        recovery_band = [0.0, 1.0 + 2.0 * tolerance]
        # Human-calibrated: explicit cloth recovery band (v12.6 [0.0, 1.16]
        # makes the current default explicit; recovery lower 0.84 was
        # systematically too strict — 56/59 sub-0.84 cases human-accepted).
        recovery_band = _gate_band(gate_overrides, "recovery_band_cloth", recovery_band)
    else:
        contact_eps = max(3.5 * fixed["particle_size_m"], 1.25 * grid_dx)
        penetration_base = fixed["particle_size_m"]
        drop_deformation_band = [0.005, 0.9]
        compression_deformation_min = max(0.02, 0.3 * compression_ratio)
        recovery_band = [max(0.0, 1.0 - 2.0 * tolerance), 1.0 + 2.0 * tolerance]
        # Human-calibrated: drop deformation upper 0.9 -> 0.98 (humans accept
        # to 0.97, aligned with the cloth band); recovery lower 0.84 -> 0.45.
        drop_deformation_band = _gate_band(gate_overrides, "drop_deformation_band", drop_deformation_band)
        recovery_band = _gate_band(gate_overrides, "recovery_band_mpm", recovery_band)

    mesh_adaptation_details: dict[str, Any] | None = None
    visual_path = None
    if v5:
        # v12.6: v5 representation-driven adaptation (mesh_adaptation_v5 on
        # the v4 geometry base).  Quality/修复记录全部进 metrics；v5 不做
        # v4 的 pre-simulation unfit 门控（审核语义是"全部执行并记录"），
        # 只有 ProxyError 级别的硬失败才门控。
        import mesh_adaptation_v5 as _mesh_adaptation_v5
        import mesh_adaptation_v4 as _mesh_adaptation_v4

        try:
            mesh_path, v5_details = _mesh_adaptation_v5.adapt(
                Path(mesh), output_dir, v5_representation, length, orientation)
        except _mesh_adaptation_v4.ProxyError as exc:
            return _proxy_rejection(report, {"adaptation": "v5", **exc.details}, "before_simulation",
                                    contract="softsim.genesis_proxy.v12.6-candidate")
        import trimesh as _trimesh

        posed_proxy = _trimesh.load(str(mesh_path), process=False)
        scaled_extent = np.asarray(posed_proxy.extents, dtype=float)
        try:
            edge_lengths = np.asarray(posed_proxy.edges_unique_length, dtype=float)
            median_edge = float(np.median(edge_lengths)) if edge_lengths.size else float(scaled_extent.max()) / 32.0
        except Exception:
            median_edge = float(scaled_extent.max()) / 32.0
        # adapt() 内部的 refine_for_geometry 可能把名字暗示的 sheet 精化回
        # 3D 壳（仍是 PBD 路径）；约束决策用精化后的表示。
        v5_representation = v5_details.get("representation") or v5_representation
        mesh_adaptation_details = {"adaptation": "v5", "route_override": route_override, **v5_details}
    elif v3:
        # v12.5: production adaptation module (sibling file, lazy import so
        # the default/v2 paths never touch it).
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        if v4:
            import mesh_adaptation_v4 as _mesh_adaptation
            import mesh_visual_binding
            try:
                mesh_path, scaled_extent, median_edge, mesh_adaptation_details = _mesh_adaptation.adapt_solver_mesh(
                    Path(mesh), length, orientation, output_dir, np, cloth=cloth, geometry_family=geometry_family)
                if mesh_adaptation_details["verdict"] == "unfit":
                    return _proxy_rejection(report, mesh_adaptation_details, "before_simulation")
                visual_asset = {
                    "resolved_artifacts": {"visual_mesh": (asset.get("resolved_artifacts") or {}).get("visual_mesh")},
                    "artifact_refs": {"visual_mesh": (asset.get("artifact_refs") or {}).get("visual_mesh")},
                }
                visual_source = _resolve_mesh(visual_asset) or mesh
                visual_path, visual_details = mesh_visual_binding.prepare_visual(
                    Path(visual_source), Path(mesh), mesh_path,
                    mesh_adaptation_details["normalization_transform"], output_dir)
                mesh_adaptation_details["visual_preparation"] = visual_details
            except _mesh_adaptation.ProxyError as exc:
                return _proxy_rejection(report, {**(mesh_adaptation_details or {}), **exc.details}, "before_simulation")
        else:
            import mesh_adaptation as _mesh_adaptation
            mesh_path, scaled_extent, median_edge, mesh_adaptation_details = _mesh_adaptation.adapt_solver_mesh(
                Path(mesh), length, orientation, output_dir, np,
                cloth=cloth, geometry_family=geometry_family,
            )
        if mesh_adaptation_details.get("verdict") == "unfit":
            # Hard proxy defect: gate to model_limit with explicit evidence
            # instead of spending solver time on a mesh that cannot produce
            # meaningful physics (and instead of faking a physics failure).
            report["execution_status"] = "completed"
            report["behavior_pass"] = False
            report["passed"] = False
            report["explicit_behavior_evidence"] = True
            report["metrics"] = {
                **_sentinel_metrics(probe),
                "mesh_adaptation": mesh_adaptation_details,
                "proxy_gate": "unfit",
            }
            report["diagnosis"] = {
                "kind": "model_limit",
                "suggested_updates": {},
                "mechanism": "mesh_proxy_unfit",
                "defects": list(mesh_adaptation_details.get("defects") or []),
                "detail": "adapted solver proxy failed the pre-simulation quality gate; see metrics.mesh_adaptation",
            }
            report["limitations"] = list(BASE_LIMITATIONS) + [
                "v3_mesh_adaptation_module",
                "proxy_unfit_gated_before_simulation",
            ] + [str(w) for w in (mesh_adaptation_details.get("warnings") or [])]
            return report
    elif v2:
        mesh_path, scaled_extent, median_edge, mesh_adaptation_details = _prepare_solver_mesh_v2(
            Path(mesh), length, orientation, output_dir, np,
            cloth=cloth, geometry_family=geometry_family,
        )
    else:
        mesh_path, scaled_extent, median_edge = _prepare_solver_mesh(
            Path(mesh), length, orientation, 4000 if cloth else 2500, output_dir, np,
            close_volume=not cloth,
        )
    if cloth:
        _disable_pbd_remesh()
    friction = min(2.0, max(0.01, _candidate_value(asset, candidate, "friction_static")))

    pbd_ps = min(0.5 * median_edge, 0.01 * length) if cloth else None

    gs.init(backend=gs.gpu, logging_level="warning", seed=seed, precision="32")
    try:
        vis_options = _study_vis_options(gs) if study_dir is not None else None
        if v4:
            vis_options = vis_options or gs.options.VisOptions()
            vis_options.n_support_neighbors = 4 if cloth else 18
        # v12.6/v5 cloth profiles: sheet/garment_cloth run with the bending
        # projection off (as before); rest-curvature profiles (garment_shell,
        # bag_shell, reinforced_shell, sealed_membrane) instead keep bending
        # iterations and install the v5 source-rest-dihedral ShellConstraints
        # kernel after the entity is created.  v5 keeps PBD self-collision
        # for non-sheet profiles with a domain-covering hash grid (the
        # 150^3-cell aliasing that motivated disabling it is avoided by the
        # explicit hash_grid_cell_size); the flat sheet keeps it disabled.
        v5_profile = str((v5_representation or {}).get("profile") or "")
        v5_shell_profile = bool(v5 and cloth and (v5_representation or {}).get("rest_curvature"))
        v5_keep_self_collision = bool(v5 and cloth and v5_profile != "sheet")
        scene = _make_scene(
            gs, fixed, length, cloth=cloth, pbd_particle_size=pbd_ps,
            vis_options=vis_options,
            v3_cloth=bool(v3 and cloth),
            pbd_hash_cell_size=(3.4 * length / 96.0) if v5_keep_self_collision else None,
            mpm_cpic=bool(v3 and not cloth and probe_kind == "compression"),
        )
        if cloth:
            # Genesis 1.3.3's PBD bending constraint hard-codes a flat (pi)
            # rest angle for every inner edge; on curved assets (bags,
            # garments, shells) it applies a persistent flattening force and
            # the sheet inflates/oscillates.  PBDOptions validates iterations
            # > 0, so disable bending directly on the solver before build.
            # (v12.5 re-tested enabling it with the v11 compliance cap: drop
            # probes destabilized — garment/shell lost floor contact entirely
            # — so bending stays disabled in all versions.)
            # v12.6/v5 rest-curvature profiles are the exception: bending is
            # applied through the v5 ShellConstraints kernel, which projects
            # toward the SOURCE rest dihedrals instead of the flat angle.
            if not v5_shell_profile:
                scene._sim.pbd_solver._max_bending_solver_iterations = 0
            # PBD self-collision uses a spatial hash whose grid is clamped to
            # 150^3 cells; with cloth-scale particle radii the cells cover less
            # than the domain, so distant particles alias into one slot and get
            # pushed apart until the sheet explodes to the world bounds.
            # Rigid-boundary contact (floor/plates) is a separate kernel and
            # stays enabled.  v12.6/v5 non-sheet profiles keep self-collision
            # (domain-covering hash grid, see _make_scene).
            if not v5_keep_self_collision:
                scene._sim.pbd_solver._kernel_solve_collision = lambda f: None
        # Surface colors are cosmetic only (friction lives in the material);
        # study renders use a light studio floor to match the light background.
        floor_color = (0.70, 0.72, 0.76) if study_dir is not None else (0.18, 0.20, 0.24)
        scene.add_entity(
            gs.morphs.Plane(plane_size=(max(5.0, 6.0 * length), max(5.0, 6.0 * length))),
            material=gs.materials.Rigid(friction=friction, coup_friction=friction),
            surface=gs.surfaces.Default(color=floor_color, roughness=0.9),
        )
        parameter_mapping: dict[str, Any] = {}
        if cloth:
            if v5:
                material, parameter_mapping = _make_cloth_material_v5(gs, asset, candidate, v5_representation)
            else:
                material, parameter_mapping = _make_cloth_material(gs, asset, candidate)
            clamps: dict[str, Any] = {}
        else:
            material, clamps = _make_material(gs, mode, asset, candidate)

        plates: list[Any] = []
        plate_meta: dict[str, Any] = {}
        if probe_kind == "drop":
            # drop_height_m is a task-fixed solver numeric: it wins over the
            # probe stimulus whenever any fixed_numerics source defines it.
            fixed_height = _fixed_drop_height(asset, simulator_config)
            if fixed_height is not None:
                height = fixed_height
                height_source = "fixed_solver_numerics"
            else:
                height = _number(stimulus.get("height_m"), float(fixed["drop_height_m"]))
                height_source = "probe_stimulus"
            height = min(max(height, 0.5 * fixed["particle_size_m"]), 2.0 * length)
            z_offset = height + 0.5 * fixed["particle_size_m"]
            soft = _add_soft_body(scene, gs, mesh_path, material, z_offset, adapted=v4)
            plate_meta = {"drop_height_m": float(height), "drop_height_source": height_source}
            study_framing = {"drop_top_z": z_offset + float(scaled_extent[2]), "axis_index": 0, "plate_outer_half_span": None}
        else:
            axis = str(stimulus.get("axis") or "x").lower()
            axis_index = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
            soft = _add_soft_body(scene, gs, mesh_path, material, 0.5 * fixed["particle_size_m"], adapted=v4)
            extent = scaled_extent
            axis_center = 0.0
            if v3:
                # v12.6: plates are placed from the PLACED entity's measured
                # particle AABB (world coordinates), not from the proxy bbox
                # extent.  The proxy extent and the sampled particle span can
                # differ (MPM sampler inset, multi-geometry padding), and a
                # wrong half-width places the plates off the body.
                body_particles = np.asarray(soft._particles, dtype=float)
                if body_particles.size:
                    body_lo = body_particles.min(axis=0)
                    body_hi = body_particles.max(axis=0)
                    extent = np.asarray(body_hi - body_lo, dtype=float)
                    axis_center = float(0.5 * (body_lo[axis_index] + body_hi[axis_index]))
            half = 0.5 * float(extent[axis_index])
            plate_gap0 = 1.0 * fixed["particle_size_m"]
            # Travel per side must cover the initial standoff gap as well as
            # the commanded compression, otherwise the plates stop short and
            # the measured deformation collapses to the gap size.
            plate_travel = 0.5 * compression_ratio * float(extent[axis_index]) + plate_gap0
            # Plate thickness must span >= 2 MPM grid cells: thinner plates
            # tunnel through the grid-coupled boundary at small asset scales.
            thickness = max(0.15 * length, 2.0 * fixed["particle_size_m"], 2.0 * grid_dx)
            plate_size = [0.0, 0.0, 0.0]
            for i in range(3):
                plate_size[i] = thickness if i == axis_index else max(1.6 * float(extent[i]), thickness)
            if axis_index == 2:
                raise RuntimeError("vertical (z) compression not supported; use lateral axis x or y")
            # Lift the plates slightly off the floor so floor friction does not
            # resist their commanded motion; they still cover the body's height.
            # v12.5 cloth: the MPM-scale lift (0.5*particle_size_m ~= scale/48)
            # floats the plates ABOVE the cloth contact shell (~pbd_ps), so
            # plates swept over flat sheets without ever touching them (the
            # v12.4 "probe never reached the body" failure).  The cloth lift
            # stays inside the PBD contact radius.
            plate_lift = 0.25 * float(pbd_ps) if (v3 and cloth) else 0.5 * fixed["particle_size_m"]
            plate_center_z = 0.5 * plate_size[2] + plate_lift
            for sign in (1.0, -1.0):
                center = [0.0, 0.0, plate_center_z]
                center[axis_index] = axis_center + sign * (half + plate_gap0 + 0.5 * thickness)
                # v12.5 cloth: the plates are kinematically driven; with the
                # contact-shell-height lift the floor friction pins them in
                # place (measured: plates never traveled on 5cm cloth assets).
                # Give the plates zero friction so only the body's contact
                # resists the commanded motion; compression transfer is
                # normal-force dominated.
                plate_friction = 0.01 if (v3 and cloth) else friction  # genesis validates friction >= 0.01
                plate = scene.add_entity(
                    gs.morphs.Box(size=tuple(plate_size), pos=tuple(center)),
                    material=gs.materials.Rigid(friction=plate_friction, coup_friction=plate_friction),
                    surface=gs.surfaces.Default(
                        color=(0.42, 0.46, 0.52) if study_dir is not None else (0.3, 0.35, 0.4),
                        roughness=0.8,
                    ),
                )
                plates.append(plate)
            plate_meta = {
                "compression_axis": axis,
                "compression_ratio": float(compression_ratio),
                "plate_travel_m": float(plate_travel),
                "plate_gap0_m": float(plate_gap0),
                "plate_size_m": [float(x) for x in plate_size],
            }
            if v3:
                plate_meta["plate_center_z_m"] = float(plate_center_z)
                plate_meta["plate_axis_center_m"] = float(axis_center)
                plate_meta["plate_placement"] = "measured_particle_aabb"
            study_framing = {
                "drop_top_z": None,
                "axis_index": axis_index,
                "plate_outer_half_span": half + plate_gap0 + thickness,
            }

        v5_shell_constraint_diagnostics: dict[str, Any] | None = None
        if v5_shell_profile:
            # v12.6/v5: source-rest-dihedral hinge projection (and the sealed
            # volume compliance surrogate for sealed_membrane) replaces the
            # Genesis flat-rest bending kernel.  Installed before build; the
            # kernel reads the material's per-edge bending compliance.
            from mesh_shell_constraints import ShellConstraints

            try:
                v5_constraints = ShellConstraints(
                    soft, sealed=(v5_profile == "sealed_membrane"), scale=length)
                v5_constraints.install()
                v5_shell_constraint_diagnostics = v5_constraints.diagnostics
            except Exception as exc:
                # Never fake a shell constraint: fall back to the bending-off
                # cloth approximation with explicit evidence.
                scene._sim.pbd_solver._max_bending_solver_iterations = 0
                v5_shell_constraint_diagnostics = {
                    "installed": False,
                    "fallback": "bending_projection_disabled",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if mesh_adaptation_details is not None and v5:
            mesh_adaptation_details["shell_constraints"] = v5_shell_constraint_diagnostics
            mesh_adaptation_details["self_collision_kept"] = v5_keep_self_collision

        if v4:
            sampled = np.asarray(soft._particles)
            mesh_adaptation_details["sampling"] = {
                "actual_particle_count": int(len(sampled)), "sampling_completed": True,
                "method": "surface_vertices" if cloth else "Genesis_regular_volume_sampler",
                "particle_size_m": float(pbd_ps if cloth else fixed["particle_size_m"]),
                "sampled_extents_m": np.ptp(sampled, axis=0).tolist() if len(sampled) else [],
            }
            if not cloth:
                minimum_span = min(mesh_adaptation_details["canonical_proxy_quality"]["extents"]) * length
                mesh_adaptation_details["sampling"]["minimum_thickness_in_particle_spacings"] = minimum_span / fixed["particle_size_m"]
                if minimum_span < 2 * fixed["particle_size_m"]:
                    mesh_adaptation_details.update(verdict="unfit", defects=["mpm_thickness_below_two_particle_spacings"])
                    return _proxy_rejection(report, mesh_adaptation_details, "after_sampling")
            floor_count = V3_MIN_CLOTH_VERTICES if cloth else V3_MIN_MPM_PARTICLES
            if len(sampled) < floor_count:
                mesh_adaptation_details.update(verdict="unfit", defects=["sampled_particle_count_below_minimum"])
                return _proxy_rejection(report, mesh_adaptation_details, "after_sampling")
            try:
                mesh_adaptation_details["visual_binding"] = mesh_visual_binding.bind_entity(
                    gs, soft, visual_path, cloth=cloth)
            except _mesh_adaptation.ProxyError as exc:
                return _proxy_rejection(report, {**mesh_adaptation_details, **exc.details}, "visual_binding")

        post_step_hook = None
        if v2:
            # Contact-gated / CoM-preserving damping hooks (v11 mechanisms;
            # see _make_v2_post_step_hook).  Contact height mirrors the
            # metric contact band (MPM) or the PBD particle radius (cloth).
            # v12.5 adds contact-gated lateral CoM dissipation for MPM.
            hook_contact_height = 1.25 * float(pbd_ps) if cloth else float(contact_eps)
            post_step_hook = _make_v2_post_step_hook(
                soft, np, cloth, float(fixed["dt_s"]), hook_contact_height,
                lateral_rate_s_inv=0.0 if (cloth or not v3) else V3_CND_LATERAL_RATE_S_INV,
            )

        study_frames: list[Any] = []
        study_visibility: list[dict[str, Any]] = []
        if study_dir is not None:
            if v4 or v5:
                import mesh_render_audit
            cam_pos, cam_lookat = _study_camera_pose(
                probe_kind, length, scaled_extent, **study_framing,
            )
            study_camera = scene.add_camera(
                res=(320, 320), pos=cam_pos, lookat=cam_lookat, fov=30.0, GUI=False,
            )
            scene.build()
            base_step = scene.step

            def step_fn() -> None:
                base_step()
                if post_step_hook is not None:
                    post_step_hook()
                frame_index = len(study_frames)
                audit_frame = (v4 or v5) and frame_index % 30 == 0
                rgb, _, segmentation, _ = study_camera.render(rgb=True, segmentation=audit_frame)
                study_frames.append(rgb)
                if audit_frame:
                    study_visibility.append(mesh_render_audit.visibility_sample(
                        segmentation, study_camera._visualizer._context.seg_idxc_map,
                        soft.idx, frame_index, np))
        else:
            _build_numeric_scene(scene)
            base_step = scene._sim.step  # numeric-only build has no Scene.FPS_tracker

            def step_fn() -> None:
                base_step()
                if post_step_hook is not None:
                    post_step_hook()

        # v12.5 post-build proxy gate: if the adapted proxy sampled to too
        # few particles/vertices, the behavior contract is unresolvable;
        # gate to model_limit with explicit evidence instead of faking a
        # physics outcome.
        post_build_defect = None
        if v3:
            n_built = len(_positions(soft, np))
            floor_count = V3_MIN_CLOTH_VERTICES if cloth else V3_MIN_MPM_PARTICLES
            if n_built < floor_count:
                kind = "cloth_proxy_vertices" if cloth else "mpm_sampled_particles"
                post_build_defect = f"{kind}_below_min:{n_built}<{floor_count}"
        if post_build_defect is not None:
            metrics = _sentinel_metrics(probe)
            metrics["proxy_gate"] = "unfit_post_build"
        elif probe_kind == "drop":
            metrics = _run_drop(
                gs, np, scene, soft, probe, fixed, tolerance, step_fn,
                contact_eps, penetration_base, drop_deformation_band,
                settle_hard_gate=settle_hard_gate,
            )
        else:
            metrics = _run_compression(
                gs, np, scene, soft, plates, probe, fixed, tolerance,
                axis_index, compression_ratio, plate_meta["plate_travel_m"], plate_meta["plate_gap0_m"],
                plate_meta["plate_size_m"][axis_index],
                step_fn,
                contact_eps, penetration_base, compression_deformation_min,
                recovery_band,
                # v12.5: degenerate recovery reference for flat proxies
                # (z_pre == 0 makes the height-based ratio undefined).
                # v12.6: the v4-proven prescribed full-pose plate drive is
                # used for every adaptation opt-in (v3+): the velocity servo
                # under-tracks at ~50% of commanded speed even before contact
                # (measured on p0_memory_foam_cushion), so the plates never
                # delivered the commanded compression.
                recovery_min_reference_m=(max(2.0 * float(pbd_ps or 0.0), 0.005 * length) if v3 else 0.0),
                servo_on=v3 and not v4,
                prescribed_motion=v3,
                plate_geometry=plate_meta if v3 else None,
                settle_hard_gate=settle_hard_gate,
                # v12.6: trimmed compression-axis extent (corner pinch-out
                # particles are a contact artifact, not body extent).
                extent_trim_fraction=0.01 if v3 else 0.0,
            )
        if study_dir is not None and post_build_defect is None:
            render_meta = _write_study_render(study_dir, study_frames, np)
            render_meta["camera"] = {"pos": list(cam_pos), "lookat": list(cam_lookat), "fov": 30.0, "res": [320, 320]}
            if v4 or v5:
                visibility = mesh_render_audit.summarize_visibility(study_visibility)
                render_meta["soft_body_visibility"] = visibility
                metrics["render_visibility"] = visibility
                if not visibility["valid"]:
                    print(f"Study render validation failed: {visibility['failure_reason']}", file=sys.stderr)
            _write_report(study_dir / "study_render.json", render_meta)
    except Exception as exc:
        if v4 and isinstance(exc, _mesh_adaptation.ProxyError):
            return _proxy_rejection(report, {**mesh_adaptation_details, **exc.details}, "scene_adaptation")
        if v4 and "Entity has zero particles" in str(exc):
            mesh_adaptation_details.update(verdict="unfit", defects=["genesis_volume_sampler_empty"])
            mesh_adaptation_details["sampling"] = {"actual_particle_count": 0, "sampling_completed": True}
            return _proxy_rejection(report, mesh_adaptation_details, "after_sampling")
        raise
    finally:
        try:
            gs.destroy()
        except Exception:
            pass

    if post_build_defect is not None:
        metrics.update(
            {
                "engine": "genesis",
                "engine_version": getattr(gs, "__version__", "unknown"),
                "backend": "gpu",
                "mode": mode,
                "scale_m": float(length),
                "mesh_path": str(mesh),
                "solver_mesh_path": str(mesh_path),
                "mesh_adaptation": mesh_adaptation_details,
            }
        )
        report["execution_status"] = "completed"
        report["behavior_pass"] = False
        report["passed"] = False
        report["explicit_behavior_evidence"] = True
        report["metrics"] = metrics
        report["diagnosis"] = {
            "kind": "model_limit",
            "suggested_updates": {},
            "mechanism": "mesh_proxy_unfit_post_build",
            "defects": [post_build_defect],
            "detail": "adapted solver proxy sampled below the minimum resolution for the fixed solver numerics; see metrics.mesh_adaptation",
        }
        report["limitations"] = list(BASE_LIMITATIONS) + [
            "v3_mesh_adaptation_module",
            "proxy_unfit_gated_after_scene_build",
        ]
        return report

    behavior_pass = bool(metrics.get("behavior_pass"))
    metrics.update(
        {
            "engine": "genesis",
            "engine_version": getattr(gs, "__version__", "unknown"),
            "backend": "gpu",
            "mode": mode,
            "coupler": "legacy",
            "ipc_used": False,
            "uipc_used": False,
            "dt_s": float(fixed["dt_s"]),
            "substeps": int(fixed["substeps"]),
            "particle_size_m": "not_applicable" if cloth else float(fixed["particle_size_m"]),
            "grid_density": "not_applicable" if cloth else float(fixed["grid_density"]),
            "scale_m": float(length),
            "seed": int(seed),
            "mesh_path": str(mesh),
            "solver_mesh_path": str(mesh_path),
            "solver_parameter_clamps": clamps,
            "tolerance": float(tolerance),
        }
    )
    if cloth:
        metrics["pbd_particle_size_m"] = float(pbd_ps)
        metrics["solver_mesh_median_edge_m"] = float(median_edge)
    if substeps_clamped_from is not None:
        metrics["substeps_effective"] = int(fixed["substeps"])
        metrics["substeps_clamped_from"] = int(substeps_clamped_from)
    if mpm_solver_scale_capped_from is not None:
        metrics["solver_scale_m"] = float(fixed["scale_m"])
        metrics["solver_scale_capped_from_m"] = float(mpm_solver_scale_capped_from)
        metrics["semantic_scale_m"] = float(mpm_solver_scale_capped_from)
    if mpm_grid_refined_from is not None:
        metrics["grid_density_effective"] = float(fixed["grid_density"])
        metrics["grid_density_refined_from"] = float(mpm_grid_refined_from)
    if v3 and probe_kind == "compression":
        metrics["plate_drive"] = "prescribed_full_pose"
        metrics["deformation_extent_trim_fraction"] = 0.01
        if not cloth:
            metrics["mpm_cpic"] = True
    if v2:
        metrics["mesh_adaptation"] = mesh_adaptation_details
        if v2_cloth_numerics_from is not None:
            metrics["cloth_numerics_v2_from"] = v2_cloth_numerics_from
        if post_step_hook is not None:
            hook_diag: dict[str, Any] = {
                "state": getattr(post_step_hook, "state", {}),
            }
            if cloth:
                hook_diag["post_contact_damping_rate_s_inv"] = post_step_hook.rate_s_inv  # type: ignore[attr-defined]
            else:
                hook_diag["internal_velocity_relaxation_rate_s_inv"] = post_step_hook.ivr_rate_s_inv  # type: ignore[attr-defined]
                hook_diag["contact_normal_damping_rate_s_inv"] = post_step_hook.cnd_rate_s_inv  # type: ignore[attr-defined]
                if v3:
                    hook_diag["contact_lateral_damping_rate_s_inv"] = post_step_hook.lateral_rate_s_inv  # type: ignore[attr-defined]
            metrics["v2_post_step_hook"] = hook_diag
    metrics.update(plate_meta)
    if gate_overrides:
        # Echo the effective gates only when overrides are active; the default
        # report stays byte-for-byte.  The effective deformation_band /
        # recovery_band are already echoed by the probe metrics above.
        metrics["settle_gate"] = "on" if settle_hard_gate else "off"
        metrics["gate_calibration_ref"] = GATE_CALIBRATION_REF

    report["execution_status"] = "completed"
    report["behavior_pass"] = behavior_pass
    report["passed"] = behavior_pass
    report["explicit_behavior_evidence"] = True
    report["metrics"] = metrics
    report["diagnosis"] = _diagnose(metrics, candidate, asset, probe_kind, v123=v123)
    if v4:
        metrics["adaptation_contract"] = "softsim.genesis_proxy.v12.5-repair1-candidate"
        metrics["simulation_executed"] = True
        metrics["semantic_scale_m"] = float(mpm_solver_scale_capped_from or length)
        if not behavior_pass:
            probe_audit = metrics.get("probe_geometry_audit") or {}
            if probe_kind == "compression" and probe_audit.get("finite_footprint_contact_records") == 0:
                report["diagnosis"].update(kind="model_limit", suggested_updates={},
                    failure_attribution="probe_mismatch", mechanism="finite_plate_footprint_missed_body")
            else:
                report["diagnosis"]["failure_attribution"] = "material_prior_or_model_limit_unresolved"
            if report["diagnosis"].get("mechanism") == "not_settled_persistent_motion":
                report["diagnosis"].update(kind="model_limit", suggested_updates={},
                    failure_attribution="model_or_solver_limit_unresolved",
                    notes=["Residual motion under the fixed window does not identify material viscosity or plastic yield; no no-op viscosity patch or settle-driven plasticity change is proposed."])
        report["limitations"].extend([
            "v4_padded_voxel_reconstruction_source_frame_before_pose",
            "v4_visual_binding_does_not_affect_particle_sampling",
            "v4_prescribed_plate_pose_changes_stimulus_not_behavior_thresholds",
        ])
    if v5:
        metrics["adaptation_contract"] = "softsim.genesis_proxy.v12.6-candidate"
        report["limitations"] = list(report.get("limitations") or BASE_LIMITATIONS) + [
            "v5_representation_driven_route_override",
            "v5_mesh_adaptation_review_layer_integrated",
            "v2_damping_hooks_are_solver_surrogates_not_material_properties",
            "v5_representation_is_semantic_approximation_not_material_identification",
        ]
        if v2_cloth_numerics_from is not None:
            report["limitations"].append("v2_cloth_dt_3ms_substeps_10_v11_configuration")
        if mpm_solver_scale_capped_from is not None:
            report["limitations"].append("v3_mpm_solver_scale_normalized_grid_coupling_resolution_limit")
        if not cloth:
            report["limitations"].append("v3_contact_lateral_dissipation_is_contact_friction_surrogate")
        if cloth and v5_shell_profile:
            report["limitations"].append("v5_shell_constraints_source_rest_dihedrals_unsigned_local")
        if cloth and v5_keep_self_collision:
            report["limitations"].append("v5_self_collision_kept_domain_covering_hash_grid")
        v5_details = mesh_adaptation_details or {}
        for key in ("warnings", "limitations"):
            for value in v5_details.get(key) or []:
                report["limitations"].append(str(value))
        if v5_details.get("shell_constraints") and not v5_details["shell_constraints"].get("installed", True):
            report["limitations"].append("v5_shell_constraints_install_failed_bending_disabled")
    report["outputs"] = {}
    if v123:
        report["limitations"] = list(report.get("limitations") or BASE_LIMITATIONS) + [
            "v123_yield_stress_settle_channel_measured",
            "v123_plastic_viscosity_noop_genesis_mpm",
        ]
        if substeps_clamped_from is not None:
            report["limitations"].append("v123_cloth_substeps_capped_le_8")
    if v2 and not v3:
        report["limitations"] = list(report.get("limitations") or BASE_LIMITATIONS) + [
            "v2_mesh_adaptation_ported_from_fast_visual_v11",
            "v2_damping_hooks_are_solver_surrogates_not_material_properties",
        ]
        if not cloth:
            report["limitations"].append("v2_voxel_filled_volume_proxy_convex_hull_is_fallback_only")
        if v2_cloth_numerics_from is not None:
            report["limitations"].append("v2_cloth_dt_3ms_substeps_10_v11_configuration")
        proxy_warnings = ((mesh_adaptation_details or {}).get("proxy") or {}).get("warnings")
        if proxy_warnings:
            report["limitations"].extend(str(w) for w in proxy_warnings)
    if v3 and probe_kind == "compression":
        # v12.6: engagement fixes measured on the p0 supplemental assets.
        report["limitations"].append("v3_compression_plates_prescribed_full_pose")
        report["limitations"].append("v3_compression_plates_placed_from_measured_particle_aabb")
        report["limitations"].append("v3_compression_axis_extent_trimmed_p1_p99_corner_pinchout")
        if not cloth:
            report["limitations"].append("v3_compression_mpm_cpic_particle_level_rigid_boundary")
    if v3 and mpm_grid_refined_from is not None:
        report["limitations"].append("v3_mpm_grid_refined_small_asset_min_16_cells_per_body")
    if v3 and not v4 and not v5:
        report["limitations"] = list(report.get("limitations") or BASE_LIMITATIONS) + [
            "v3_mesh_adaptation_module",
            "v2_damping_hooks_are_solver_surrogates_not_material_properties",
        ]
        if not cloth:
            report["limitations"].append("v3_voxel_filled_volume_proxy_convex_hull_is_fallback_only")
        else:
            report["limitations"].append("v3_fragmented_cloth_sources_voxel_sealed_or_projected")
        if not cloth and v3:
            report["limitations"].append("v3_contact_lateral_dissipation_is_contact_friction_surrogate")
        if mpm_solver_scale_capped_from is not None:
            report["limitations"].append("v3_mpm_solver_scale_normalized_grid_coupling_resolution_limit")
        if metrics.get("recovery_reference") == "compression_axis_extent":
            report["limitations"].append("v3_recovery_reference_axis_extent_for_flat_proxy")
        if v3 and cloth:
            report["limitations"].append("v3_cloth_rigid_timeconst_genesis_default_for_plate_engagement")
            if probe_kind == "compression":
                report["limitations"].append("v3_cloth_plate_lift_within_contact_shell")
        if v2_cloth_numerics_from is not None:
            report["limitations"].append("v2_cloth_dt_3ms_substeps_10_v11_configuration")
        v3_warnings = (mesh_adaptation_details or {}).get("warnings")
        if v3_warnings:
            report["limitations"].extend(str(w) for w in v3_warnings)
        v3_limitations = (mesh_adaptation_details or {}).get("limitations")
        if v3_limitations:
            report["limitations"].extend(str(w) for w in v3_limitations)
    if v4:
        report["limitations"] = [value for value in report["limitations"]
                                 if value != "surface_mesh_decimated_and_voxelized_by_genesis"]
        report["limitations"].extend([
            "v4_proxy_preprocessed_without_Genesis_geometry_repair",
            "v4_inherits_v3_solver_scale_cap_and_contact_dissipation",
            "v4_inherits_v3_flat_proxy_recovery_reference",
        ])
        report["limitations"].extend(mesh_adaptation_details.get("limitations", []))
        report["limitations"].extend(mesh_adaptation_details.get("warnings", []))
    if cloth:
        report["parameter_mapping"] = parameter_mapping
        report["limitations"] = list(report.get("limitations") or BASE_LIMITATIONS) + [
            "cloth_route_executed_as_pbd_cloth",
            "legacy_model_name_genesis_fem_cloth_ipc_mapped_to_pbd_cloth",
            "ipc_not_used_per_method_contract",
            "cloth_parameter_mapping_is_order_of_magnitude_compliance",
            "cloth_recovery_band_onesided_crumple_is_normal",
        ]
        if not v5_keep_self_collision:
            report["limitations"].append("pbd_self_collision_disabled_hash_grid_wrap")
        if not v5_shell_profile:
            report["limitations"].append("pbd_bending_disabled_flat_rest_assumption")
    report["provenance"].update({"model_mode": mode, "fixed_solver_numerics": fixed})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="SoftSim Genesis behavior-contract runner (MPM)")
    parser.add_argument("--asset-json", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--probe-json", required=True)
    parser.add_argument("--simulator-config-json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    asset = _read_json(args.asset_json)
    candidate = _read_json(args.candidate_json)
    probe = _read_json(args.probe_json)
    simulator_config = _read_json(args.simulator_config_json) if args.simulator_config_json else {}

    try:
        report = run(asset, candidate, probe, simulator_config, Path(args.output_dir))
        # Exit-code contract: 0 whenever the simulation executed (completed),
        # even when the behavior contract fails — behavior_pass lives in the
        # report.  Non-zero is reserved for true execution failures
        # (crash/asset_missing -> 3) and not_executed probes (-> 2).
        status = str(report.get("execution_status"))
        if status == "completed":
            code = 0
        elif status == "not_executed":
            code = 2
        else:
            code = 3
    except Exception as exc:
        report = _base_report(asset, candidate, probe, simulator_config)
        report["execution_status"] = "failed"
        report["behavior_pass"] = False
        report["passed"] = False
        report["explicit_behavior_evidence"] = True
        report["metrics"] = _sentinel_metrics(probe)
        report["diagnosis"] = {
            "kind": "numerical_failure",
            "suggested_updates": {},
            "detail": f"{type(exc).__name__}: {exc}",
        }
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback_tail"] = traceback.format_exc(limit=10)
        code = 3
    _write_report(report_path, report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
