# SPDX-License-Identifier: Apache-2.0
from fragile_soft_benchmark.difficulty import construct_difficulty
from fragile_soft_benchmark.scenario import PHASES, immutable_contract_hashes, make_scenario


def test_scenario_phase_order_and_contract_hashes():
    scenario = make_scenario()
    assert tuple(item["name"] for item in scenario["phases"]) == PHASES
    hashes = immutable_contract_hashes(scenario)
    assert all(value.startswith("sha256:") for value in hashes.values())


def test_difficulty_changes_registered_parameters_only():
    scenario = make_scenario()
    hashes = immutable_contract_hashes(scenario)
    material = {"material_model_id": "tofu_damage", "parameters": {"damage_start_compressive_strain": 0.05, "damage_complete_compressive_strain": 0.08, "residual_stiffness_fraction": 0.01}}
    result = construct_difficulty(material, {"low": 5, "medium": 10, "high": 20}, difficulty_id="d07", geometry_revision_id="geo.r000", **{key: hashes[key] for key in ("scenario_hash", "action_protocol_hash", "evaluator_hash")})
    assert result["force_schedule"]["medium"] == 10
    assert result["immutable_contract"]["evaluator_hash"] == hashes["evaluator_hash"]
    assert set(result["changed_parameters"]) <= {"damage_start_compressive_strain", "damage_complete_compressive_strain", "residual_stiffness_fraction"}
    try:
        construct_difficulty(material, {"low": 1}, difficulty_id="d00", geometry_revision_id="geo", **{key: hashes[key] for key in ("scenario_hash", "action_protocol_hash", "evaluator_hash")}, registry={"d00": {"geometry_scale": 2}})
    except ValueError:
        pass
    else:
        raise AssertionError("unregistered difficulty axis was accepted")
