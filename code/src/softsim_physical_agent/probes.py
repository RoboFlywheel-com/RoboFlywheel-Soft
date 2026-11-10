# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from .io import content_hash
from .models import ProbeSpec


DEFAULT_CALIBRATION = {
    "probe_id": "calibration.drop.corner_first.v1",
    "stimulus": {"kind": "drop", "height_m": 0.18, "orientation": "corner_first", "floor": "rigid_plane"},
    "action": {"kind": "release", "gripper": "none", "contact_schedule": "single_impact"},
    "config": {"steps": 48, "dt_s": 0.001, "record_every": 4, "seed_offset": 11},
    "tolerance": 0.05,
    "required_metrics": ["loss", "contact_fraction", "settled", "deformation_ratio"],
}

DEFAULT_HELDOUT = {
    "probe_id": "heldout.compression.lateral.v1",
    "stimulus": {"kind": "compression", "compression_ratio": 0.22, "axis": "x", "rate": "quasi_static"},
    "action": {"kind": "compress_release", "hold_steps": 8, "release_steps": 8},
    "config": {"steps": 64, "dt_s": 0.0008, "record_every": 4, "seed_offset": 97},
    "tolerance": 0.08,
    "required_metrics": ["loss", "contact_fraction", "settled", "recovery_ratio"],
}


def build_probe_pair(config: dict[str, Any]) -> tuple[ProbeSpec, ProbeSpec]:
    calibration = ProbeSpec.from_dict(config.get("calibration_probe", DEFAULT_CALIBRATION), "calibration.v1")
    heldout = ProbeSpec.from_dict(config.get("heldout_probe", DEFAULT_HELDOUT), "heldout.v1")
    if calibration.probe_id == heldout.probe_id:
        raise ValueError("calibration and held-out probe_id must differ")
    if probe_hash(calibration) == probe_hash(heldout):
        raise ValueError("calibration and held-out probe hashes must differ")
    if component_hash(calibration, "stimulus") == component_hash(heldout, "stimulus"):
        raise ValueError("calibration and held-out stimulus hashes must differ")
    if component_hash(calibration, "action") == component_hash(heldout, "action"):
        raise ValueError("calibration and held-out action hashes must differ")
    if component_hash(calibration, "config") == component_hash(heldout, "config"):
        raise ValueError("calibration and held-out config hashes must differ")
    return calibration, heldout


def probe_hash(probe: ProbeSpec) -> str:
    return content_hash(probe.as_dict())


def component_hash(probe: ProbeSpec, component: str) -> str:
    if component not in {"stimulus", "action", "config"}:
        raise ValueError(component)
    return content_hash(getattr(probe, component))


def probe_record(probe: ProbeSpec) -> dict[str, Any]:
    return {
        **probe.as_dict(),
        "probe_hash": probe_hash(probe),
        "stimulus_hash": component_hash(probe, "stimulus"),
        "action_hash": component_hash(probe, "action"),
        "config_hash": component_hash(probe, "config"),
    }

