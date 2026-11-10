# SPDX-License-Identifier: Apache-2.0
from fragile_soft_benchmark.materials import InvalidMaterialProfile, UnsupportedMaterialModel, construct_material, tofu_damage_response


def test_tofu_damage_is_explicit_surrogate_and_monotone():
    profile = {
        "model": "damage",
        "profile_id": "P-FOOD-CRUSH",
        "youngs_modulus_pa": 28000,
        "poisson_ratio": 0.32,
        "density_kg_m3": 950,
        "damage_start_compressive_strain": 0.052,
        "damage_complete_compressive_strain": 0.0565,
        "residual_stiffness_fraction": 0.004,
    }
    material = construct_material(profile)
    assert material.model_id == "tofu_damage"
    assert material.surrogate is True
    assert material.explicit_fracture is False
    values = [tofu_damage_response(x, material.parameters) for x in (0.0, 0.053, 0.057)]
    assert values[0] >= values[1] >= values[2]


def test_unsupported_model_never_falls_back():
    profile = {"model": "granular_equivalent", "youngs_modulus_pa": 1, "poisson_ratio": 0, "density_kg_m3": 1}
    try:
        construct_material(profile)
    except UnsupportedMaterialModel:
        pass
    else:
        raise AssertionError("unsupported model was accepted")
    rejected = construct_material(profile, strict=False)
    assert rejected.support_status == "unsupported"
    assert rejected.implementation == "UNSUPPORTED"


def test_invalid_profile_is_explicitly_rejected_in_non_strict_mode():
    profile = {"model": "damage", "youngs_modulus_pa": -1, "poisson_ratio": 0.3, "density_kg_m3": 950, "damage_start_compressive_strain": 0.05, "damage_complete_compressive_strain": 0.06, "residual_stiffness_fraction": 0.01}
    try:
        construct_material(profile)
    except InvalidMaterialProfile:
        pass
    else:
        raise AssertionError("invalid profile was accepted")
    rejected = construct_material(profile, strict=False)
    assert rejected.support_status == "invalid"
    assert rejected.implementation == "INVALID_PROFILE"


def test_inflatable_effective_bulk_surrogate_is_not_pressure_shell():
    material = construct_material(
        {"model": "effective_elastic_bulk_surrogate", "profile_id": "P-SHELL-INFLATABLE", "youngs_modulus_pa": 74340, "poisson_ratio": 0.4, "density_kg_m3": 533},
        requested_model_id="pressure_shell",
        capability_scope="pressure_shell",
    )
    assert material.support_status == "supported_surrogate"
    assert material.requested_model_id == "pressure_shell"
    assert material.explicit_fracture is False
    assert "pressure" in (material.surrogate_reason or "")
