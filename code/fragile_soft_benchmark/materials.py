# SPDX-License-Identifier: Apache-2.0
"""Explicit material capability registry and deterministic constructors.

The registry is intentionally conservative.  A profile that asks for a model
we cannot execute is returned as ``unsupported`` (or raises in strict mode); it
is never silently changed to elastic material.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .hashing import sha256_json


class UnsupportedMaterialModel(ValueError):
    """Raised when a requested constitutive model is not implemented."""


class InvalidMaterialProfile(ValueError):
    """Raised for malformed or physically invalid material parameters."""


class TofuDamage:
    """Continuum stiffness-loss surrogate used by the tofu fixture.

    This intentionally exposes no crack topology or fracture-energy state.  It
    is a small backend-neutral law; a Genesis adapter may map its parameters to
    an MPM material implementation without changing the public contract.
    """

    model_id = "tofu_damage"
    explicit_fracture = False
    surrogate = True

    def __init__(self, **parameters: float) -> None:
        self.parameters = {str(key): float(value) for key, value in parameters.items()}
        _validate(self.model_id, self.parameters, tuple(MODEL_REGISTRY[self.model_id]["required"]))

    def retained_stiffness(self, compressive_strain: float) -> float:
        return tofu_damage_response(compressive_strain, self.parameters)

    def damage_fraction(self, compressive_strain: float) -> float:
        return 1.0 - self.retained_stiffness(compressive_strain)


@dataclass(frozen=True)
class MaterialConstruction:
    model_id: str
    implementation: str
    support_status: str
    surrogate: bool
    explicit_fracture: bool
    parameters: dict[str, float]
    source_profile_id: str | None
    parameter_hash: str
    unsupported_reason: str | None = None
    requested_model_id: str | None = None
    capability_scope: str | None = None
    surrogate_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "material_model_id": self.model_id,
            "implementation": self.implementation,
            "support_status": self.support_status,
            "surrogate": self.surrogate,
            "explicit_fracture": self.explicit_fracture,
            "parameters": dict(self.parameters),
            "source_profile_id": self.source_profile_id,
            "parameter_hash": self.parameter_hash,
            "requested_model_id": self.requested_model_id or self.model_id,
            "capability_scope": self.capability_scope,
            "surrogate_reason": self.surrogate_reason,
        }
        if self.unsupported_reason:
            result["unsupported_reason"] = self.unsupported_reason
        return result


# Aliases preserve old P0 labels while giving every output one canonical ID.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "tofu_damage": {
        "implementation": "TofuDamage",
        "surrogate": True,
        "explicit_fracture": False,
        "required": (
            "youngs_modulus_pa",
            "poisson_ratio",
            "density_kg_m3",
            "damage_start_compressive_strain",
            "damage_complete_compressive_strain",
            "residual_stiffness_fraction",
        ),
    },
    "plastic": {
        "implementation": "MPM.ElastoPlastic",
        "surrogate": False,
        "explicit_fracture": False,
        "required": ("youngs_modulus_pa", "poisson_ratio", "density_kg_m3", "yield_lower", "yield_higher"),
    },
    "foam": {
        "implementation": "FoamCompressionSet",
        "surrogate": True,
        "explicit_fracture": False,
        "required": ("youngs_modulus_pa", "poisson_ratio", "density_kg_m3", "yield_lower", "yield_higher"),
    },
    "gel": {
        "implementation": "GelElasticSurrogate",
        "surrogate": True,
        "explicit_fracture": False,
        "required": ("youngs_modulus_pa", "poisson_ratio", "density_kg_m3", "relaxation_time_s"),
    },
    "elastic": {
        "implementation": "MPM.Elastic",
        "surrogate": False,
        "explicit_fracture": False,
        "required": ("youngs_modulus_pa", "poisson_ratio", "density_kg_m3"),
    },
    "effective_elastic_bulk_surrogate": {
        "implementation": "MPM.Elastic",
        "surrogate": True,
        "explicit_fracture": False,
        "required": ("youngs_modulus_pa", "poisson_ratio", "density_kg_m3"),
        "surrogate_reason": "effective bulk response; no pressure, leakage, shell, or grain state",
    },
}

ALIASES = {
    "damage": "tofu_damage",
    "tofu": "tofu_damage",
    "compression_set": "foam",
    "foam_slow": "foam",
    "foam_elastic": "elastic",
    "gel_block": "gel",
    "pbd_elastic": "elastic",
    "elastic_bulk": "effective_elastic_bulk_surrogate",
}

UNSUPPORTED_MODELS: dict[str, str] = {
    "granular_equivalent": "P0 effective filled-body label; explicit shell plus grains is not implemented",
    "discrete_granular": "discrete grain coupling is outside the local constructor",
    "pressure_shell": "pressure state, leakage, and thin-shell solver are not implemented",
    "inflatable_shell": "pressure state, leakage, and thin-shell solver are not implemented",
    "viscoelastic": "time-dependent constitutive integration is not implemented",
    "fracture": "explicit fracture/cohesive topology is not implemented",
    "cohesive_fracture": "explicit fracture/cohesive topology is not implemented",
}


def canonical_model_id(model: str) -> str:
    key = str(model).strip().lower().replace("-", "_")
    return ALIASES.get(key, key)


def _numeric_parameters(profile: Mapping[str, Any]) -> dict[str, float]:
    params: dict[str, float] = {}
    for key, value in profile.items():
        if key in {"model", "model_id", "profile_id", "id", "name", "name_en", "name_zh"}:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            params[str(key)] = float(value)
    # Permit a nested ``parameters`` object used by newer records.
    nested = profile.get("parameters")
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                params[str(key)] = float(value)
    return params


def _validate(model_id: str, params: Mapping[str, float], required: tuple[str, ...]) -> None:
    missing = [key for key in required if key not in params]
    if missing:
        raise InvalidMaterialProfile(f"{model_id}: missing parameters: {', '.join(missing)}")
    for key, value in params.items():
        if not (value == value and abs(value) != float("inf")):
            raise InvalidMaterialProfile(f"{model_id}: non-finite parameter {key}")
    if params["youngs_modulus_pa"] <= 0 or params["density_kg_m3"] <= 0:
        raise InvalidMaterialProfile(f"{model_id}: E and density must be positive")
    nu = params["poisson_ratio"]
    if not -0.5 < nu < 0.5:
        raise InvalidMaterialProfile(f"{model_id}: poisson_ratio must be in (-0.5, 0.5)")
    if model_id == "tofu_damage":
        start = params["damage_start_compressive_strain"]
        complete = params["damage_complete_compressive_strain"]
        if start <= 0 or complete <= start:
            raise InvalidMaterialProfile("tofu_damage: complete onset must exceed start onset")
        residual = params["residual_stiffness_fraction"]
        if not 0 <= residual <= 1:
            raise InvalidMaterialProfile("tofu_damage: residual stiffness must be in [0, 1]")
    if model_id in {"plastic", "foam"} and (
        params["yield_lower"] <= 0 or params["yield_higher"] <= 0
    ):
        raise InvalidMaterialProfile(f"{model_id}: yield values must be positive")
    if model_id == "gel" and params.get("relaxation_time_s", 0.0) <= 0:
        raise InvalidMaterialProfile("gel: relaxation_time_s must be positive")


def construct_material(
    profile: Mapping[str, Any],
    *,
    strict: bool = True,
    model_override: str | None = None,
    requested_model_id: str | None = None,
    capability_scope: str | None = None,
) -> MaterialConstruction:
    """Construct a canonical material descriptor from a P0/new profile.

    ``strict=False`` is useful to produce an explicit rejected record for an
    unsupported P0 profile.  It still returns ``support_status=unsupported``;
    it never substitutes another constitutive law.
    """

    raw_model = model_override or profile.get("model_id") or profile.get("model")
    if not raw_model:
        raise InvalidMaterialProfile("material profile has no model/model_id")
    raw_key = str(raw_model).strip().lower().replace("-", "_")
    model_id = canonical_model_id(raw_key)
    params = _numeric_parameters(profile)
    source_profile_id = profile.get("profile_id") or profile.get("id")
    requested = requested_model_id or raw_model
    if model_id not in MODEL_REGISTRY:
        reason = UNSUPPORTED_MODELS.get(model_id, f"unknown constitutive model: {raw_model}")
        if strict:
            raise UnsupportedMaterialModel(reason)
        return MaterialConstruction(
            model_id=model_id,
            implementation="UNSUPPORTED",
            support_status="unsupported",
            surrogate=True,
            explicit_fracture=False,
            parameters=params,
            source_profile_id=str(source_profile_id) if source_profile_id else None,
            parameter_hash=sha256_json({"model_id": model_id, "parameters": params}),
            unsupported_reason=reason,
            requested_model_id=str(requested),
            capability_scope=capability_scope,
        )
    spec = MODEL_REGISTRY[model_id]
    try:
        _validate(model_id, params, tuple(spec["required"]))
    except InvalidMaterialProfile as exc:
        if strict:
            raise
        return MaterialConstruction(
            model_id=model_id,
            implementation="INVALID_PROFILE",
            support_status="invalid",
            surrogate=bool(spec["surrogate"]),
            explicit_fracture=False,
            parameters=params,
            source_profile_id=str(source_profile_id) if source_profile_id else None,
            parameter_hash=sha256_json({"model_id": model_id, "parameters": params}),
            unsupported_reason=str(exc),
            requested_model_id=str(requested),
            capability_scope=capability_scope,
        )
    return MaterialConstruction(
        model_id=model_id,
        implementation=str(spec["implementation"]),
        support_status="supported_surrogate" if bool(spec["surrogate"]) else "supported",
        surrogate=bool(spec["surrogate"]),
        explicit_fracture=bool(spec["explicit_fracture"]),
        parameters=params,
        source_profile_id=str(source_profile_id) if source_profile_id else None,
        parameter_hash=sha256_json({"model_id": model_id, "parameters": params}),
        requested_model_id=str(requested),
        capability_scope=capability_scope,
        surrogate_reason=spec.get("surrogate_reason"),
    )


def tofu_damage_response(compressive_strain: float, parameters: Mapping[str, float]) -> float:
    """Return retained stiffness for the P0 TofuDamage surrogate in [0, 1]."""

    start = float(parameters["damage_start_compressive_strain"])
    complete = float(parameters["damage_complete_compressive_strain"])
    residual = float(parameters["residual_stiffness_fraction"])
    if compressive_strain <= start:
        return 1.0
    if compressive_strain >= complete:
        return residual
    alpha = (compressive_strain - start) / (complete - start)
    return max(residual, 1.0 - alpha)
