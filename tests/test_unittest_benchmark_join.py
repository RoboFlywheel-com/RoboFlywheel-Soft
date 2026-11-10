# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest
from pathlib import Path

from fragile_soft_benchmark.adapter import P0Adapter
from fragile_soft_benchmark.builder import build_instances
from fragile_soft_benchmark.materials import InvalidMaterialProfile, construct_material
from fragile_soft_benchmark.mock_backend import run_mock_episode
from fragile_soft_benchmark.qualification import make_default_qualification_instances, run_qualification


FIXTURE = Path(__file__).parent / "fixtures" / "p0_legacy"


class BenchmarkJoinAndMaterialTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = P0Adapter.from_files(
            FIXTURE / "manifest.json",
            FIXTURE / "material_profiles.json",
            summary_json_path=FIXTURE / "summary.json",
            summary_csv_path=FIXTURE / "summary.csv",
            source_root=FIXTURE,
        )

    def test_unjoined_is_provisional_and_verified_join_is_exact(self) -> None:
        config = {"canonical_groups": [{"canonical_group_id": "soft_food_form", "source_asset_id": "tofu", "split": "calibration"}], "geometry_revisions_per_group": 1, "difficulty_ids": ["d00"]}
        provisional = build_instances(self.adapter, config)
        self.assertIsNone(provisional[0]["physical_bundle_id"])
        self.assertEqual(provisional[0]["physical_join_status"], "unjoined")
        joined = build_instances(
            self.adapter,
            config,
            physical_index=[{"sample_id": "sample-0", "asset_revision_id": "sample-0-r1", "physical_bundle_id": "pb-real", "status": "verified"}],
            physical_join_map=[{"benchmark_instance_id": "fsb-soft_food_form-r000-d00", "physical_bundle_id": "pb-real"}],
            require_physical_join=True,
        )
        self.assertTrue(joined[0]["formal_join_ready"])
        self.assertEqual(joined[0]["physical_bundle_id"], "pb-real")

    def test_formal_scope_requires_physical_join_and_external_backend(self) -> None:
        instances = make_default_qualification_instances(self.adapter, {"qualification_revisions_per_group": 1, "qualification_difficulty_ids": ["d00", "d02", "d05"]})

        def runner(instance, **kwargs):
            record = run_mock_episode(instance, **kwargs)
            record["backend_id"] = "fixture.formal.v1"
            return record

        result = run_qualification(instances, formal_scope=True, backend_id="fixture.formal.v1", episode_runner=runner)
        self.assertEqual(result["report"]["status"], "rejected")
        self.assertFalse(result["report"]["e3_a_eligible"])

    def test_invalid_material_does_not_fall_back(self) -> None:
        profile = {"model": "damage", "youngs_modulus_pa": -1, "poisson_ratio": 0.3, "density_kg_m3": 950, "damage_start_compressive_strain": 0.05, "damage_complete_compressive_strain": 0.06, "residual_stiffness_fraction": 0.01}
        with self.assertRaises(InvalidMaterialProfile):
            construct_material(profile)
        self.assertEqual(construct_material(profile, strict=False).support_status, "invalid")

    def test_runner_exception_keeps_failed_metric_record(self) -> None:
        instances = make_default_qualification_instances(self.adapter, {"qualification_revisions_per_group": 1, "qualification_difficulty_ids": ["d00", "d02", "d05"]})

        def broken_runner(instance, **kwargs):
            raise RuntimeError("simulator unavailable")

        result = run_qualification(instances, formal_scope=True, backend_id="fixture.formal.v1", episode_runner=broken_runner)
        self.assertEqual(result["report"]["status"], "rejected")
        self.assertTrue(result["metrics"])
        self.assertIn("backend_failure", result["metrics"][0]["failure_codes"])


if __name__ == "__main__":
    unittest.main()
