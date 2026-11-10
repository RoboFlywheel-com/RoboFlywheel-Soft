# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Mapping

from .io import content_hash, portable_value


CANONICAL_NUMERIC_NAMES = {
    "density": "density_kg_m3",
    "density_kg_m3": "density_kg_m3",
    "youngs_modulus": "youngs_modulus_pa",
    "youngs_modulus_pa": "youngs_modulus_pa",
    "poisson_ratio": "poisson_ratio",
    "particle_size": "particle_size_m",
    "particle_size_m": "particle_size_m",
    "dt": "dt_s",
    "time_step_s": "dt_s",
    "dt_s": "dt_s",
    "drop_height": "drop_height_m",
    "drop_height_m": "drop_height_m",
    "scale": "scale_m",
    "asset_size": "scale_m",
}

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

# Absolute safety bounds for whitelisted parameters.  A proposer may introduce
# an allowed parameter that the prior did not carry (e.g. damping for a
# not-settled diagnosis); these bounds keep such additions bounded even when
# the prior supplies no valid_range for them.
DEFAULT_PARAMETER_BOUNDS = {
    "density_kg_m3": (1.0, 30000.0),
    "youngs_modulus_pa": (1.0, 1.0e10),
    "poisson_ratio": (-0.49, 0.499),
    "yield_stress_pa": (0.0, 1.0e10),
    "yield_lower": (0.0, 1.0e10),
    "yield_higher": (0.0, 1.0e10),
    "plastic_viscosity_pa_s": (0.0, 1.0e6),
    "tensile_cohesion_pa": (0.0, 1.0e10),
    "friction_static": (0.0, 2.0),
    "friction_dynamic": (0.0, 2.0),
}


def prior_scale_m(prior: Mapping[str, Any]) -> float:
    """Metric scale from the VLM prior's longest dimension (never measured)."""

    for item in prior.get("parameters") or []:
        if isinstance(item, Mapping) and str(item.get("name")) == "longest_dimension_m":
            raw = item.get("value")
            try:
                value = float(raw.get("value") if isinstance(raw, Mapping) else raw)
            except (TypeError, ValueError):
                break
            if value == value and abs(value) != float("inf") and value > 0:
                return value
    return 0.32


def sizing_numerics(prior: Mapping[str, Any]) -> dict[str, Any]:
    """Frozen per-asset solver numerics derived from the prior metric scale.

    Single authoritative implementation of the v12.2 sizing rule; shared by the
    local qualification task builder and the cloud worker entry so both paths
    produce identical fixed_solver_numerics for the same prior.
    """

    scale = min(max(prior_scale_m(prior), 0.02), 2.0)
    return {
        "particle_size_m": round(scale / 24.0, 6),
        "dt_s": 0.001,
        "substeps": 16,
        "grid_density": 64.0,
        "scale_m": round(scale, 6),
        "seed": 17,
        "drop_height_m": round(min(0.5 * scale, 0.25), 6),
        "proxy_mode": "alpha-wrap",
    }

FIXED_NUMERIC_NAMES = frozenset(
    {
        "particle_size_m",
        "dt_s",
        "substeps",
        "grid_density",
        "scale_m",
        "seed",
        "drop_height_m",
        "alpha_wrap_fraction",
        "alpha_wrap_offset_fraction",
        "proxy_mode",
    }
)


def canonical_name(name: str) -> str:
    return CANONICAL_NUMERIC_NAMES.get(str(name), str(name))


def _value(value: Any, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get("value", default)
    return value if value is not None else default


def physical_values(bundle: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in bundle.get("parameters") or []:
        if not isinstance(item, Mapping) or "name" not in item:
            continue
        raw = _value(item.get("value"))
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw == raw and abs(float(raw)) != float("inf"):
            values[canonical_name(str(item["name"]))] = float(raw)
    adapter = bundle.get("solver_adapter") or {}
    if isinstance(adapter, Mapping):
        params = adapter.get("parameters") or {}
        if isinstance(params, Mapping):
            for name, raw in params.items():
                raw = _value(raw)
                if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw == raw and abs(float(raw)) != float("inf"):
                    values.setdefault(canonical_name(str(name)), float(raw))
    for container_key in ("physical_parameters", "candidate"):
        container = bundle.get(container_key)
        if isinstance(container, Mapping):
            nested = container.get("parameters") if container_key == "candidate" else container
            if isinstance(nested, Mapping):
                for name, raw in nested.items():
                    raw = _value(raw)
                    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw == raw and abs(float(raw)) != float("inf"):
                        values.setdefault(canonical_name(str(name)), float(raw))
    return values


def fixed_numerics(bundle: Mapping[str, Any], route: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Extract solver numerics once; these are never candidate-tunable."""

    result: dict[str, Any] = {}
    numerics = bundle.get("solver_numerics") or {}
    if isinstance(numerics, Mapping):
        for raw_name, raw in numerics.items():
            name = canonical_name(str(raw_name))
            value = _value(raw)
            if value is not None:
                result[name] = value
    # Legacy physical_properties.mpm records sometimes sit under solver_adapter.
    adapter = bundle.get("solver_adapter") or {}
    if isinstance(adapter, Mapping):
        for raw_name, raw in (adapter.get("numerics") or {}).items():
            name = canonical_name(str(raw_name))
            result.setdefault(name, _value(raw))
    if route:
        for name, value in route.get("fixed_solver_numerics", {}).items():
            result.setdefault(name, value)
    defaults: dict[str, Any] = sizing_numerics(bundle)
    for name, value in defaults.items():
        result.setdefault(name, value)
    # Normalize integral fields without losing their audit value.
    for name in ("substeps", "grid_density", "seed"):
        try:
            result[name] = int(result[name]) if name != "grid_density" else float(result[name])
        except (TypeError, ValueError):
            raise ValueError(f"invalid fixed solver numeric: {name}={result[name]!r}")
    return result


def build_route(row: Mapping[str, Any], prior: Mapping[str, Any]) -> dict[str, Any]:
    existing = prior.get("route") if isinstance(prior.get("route"), Mapping) else {}
    family = str(row.get("geometry_family") or row.get("geometry_family_id") or prior.get("geometry_family") or existing.get("geometry_family") or "unknown")
    material = str(row.get("material_family") or row.get("physical_profile") or prior.get("material_family") or existing.get("material_family") or "generic")
    representation = str(prior.get("representation") or existing.get("representation") or ("surface_shell" if family in {"cloth_sheet", "garment", "bag", "shell"} else "watertight_volume"))
    adapter = prior.get("solver_adapter") if isinstance(prior.get("solver_adapter"), Mapping) else {}
    model = str(prior.get("model_family") or existing.get("model_family") or adapter.get("solver") or "MPM.Elastic")
    solver = str(adapter.get("solver") or existing.get("solver_adapter") or model)
    proxy_contract = prior.get("proxy_contract") if isinstance(prior.get("proxy_contract"), Mapping) else {}
    proxy = str(proxy_contract.get("mode") or existing.get("proxy_mode") or adapter.get("proxy_mode") or "alpha-wrap")
    route = {
        "route_id": f"genesis_proxy.{family}.{material}",
        "geometry_family": family,
        "material_family": material,
        "representation": representation,
        "model_family": model,
        "solver_adapter": solver,
        "proxy_mode": proxy,
        "capability_scope": "simulator_grounded_behavior_contract",
    }
    route["route_hash"] = content_hash(route)
    return route


def validate_patch(patch: Mapping[str, Any], allowed: set[str] | frozenset[str]) -> dict[str, float]:
    unknown = set(str(name) for name in patch) - set(allowed)
    if unknown:
        raise ValueError(f"candidate patch changes non-physical parameters: {sorted(unknown)}")
    result: dict[str, float] = {}
    for name, raw in patch.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw != raw or abs(float(raw)) == float("inf"):
            raise ValueError(f"candidate parameter {name} must be numeric")
        result[str(name)] = float(raw)
    return result


def canonical_prior(row: Mapping[str, Any], prior: Mapping[str, Any], route: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy/staging priors while preserving source provenance."""

    import copy

    result = portable_value(copy.deepcopy(dict(prior)))
    result.setdefault("schema_name", "softsim.physical_prior")
    result.setdefault("schema_version", 2)
    raw_sample = row.get("sample_id") or prior.get("sample_id")
    if not raw_sample:
        raise ValueError("prior/task has no sample_id")
    result["sample_id"] = str(raw_sample)
    result["asset_revision_id"] = str(row.get("asset_revision_id") or prior.get("asset_revision_id") or content_hash({"sample_id": result["sample_id"], "prior": prior}))
    result.setdefault("status", "predicted_prior")
    result["route"] = dict(route)
    result["physical_values"] = physical_values(result)
    result["fixed_solver_numerics"] = fixed_numerics(result, route)
    result.setdefault("provenance", {})
    result["provenance"].setdefault("is_ground_truth", False)
    result["provenance"].setdefault("prior_only", True)
    return result
