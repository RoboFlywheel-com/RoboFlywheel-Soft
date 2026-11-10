# SPDX-License-Identifier: Apache-2.0
"""Planner asset-geometry join: physical rows must reach the executor with a
resolvable, hashed geometry reference from the asset ledger (v12.6 path)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from softsim_formal.io import read_jsonl
from softsim_formal.planner import plan_run

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RELEASE = ROOT / "tests" / "fixtures" / "formal_release_v2" / "data"


class PlannerAssetGeometryJoinTest(unittest.TestCase):
    def test_e1_tasks_carry_asset_geometry_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = plan_run(
                "e1-runtime",
                output_dir=tmp,
                config={"promotion_stage": "fixture", "count": 4, "methods": ["contract_probe"], "seeds": [0]},
                release_root=FIXTURE_RELEASE,
                count=4,
                shard_count=1,
            )
            self.assertEqual(result.status, "planned")
            tasks = list(read_jsonl(Path(tmp) / "task_manifest.jsonl"))
            self.assertEqual(len(tasks), 4)
            for task in tasks:
                row = task.get("source_row") or {}
                ref = row.get("geometry_ref") or {}
                self.assertEqual(ref.get("root_alias"), "fixture")
                self.assertTrue(str(ref.get("relative_path", "")).endswith("geometry.proxy"))
                self.assertTrue(str(ref.get("sha256", "")).startswith("sha256:"))
                scale = row.get("metric_scale") or {}
                self.assertGreater(float(scale.get("meters_per_unit", 0)), 0.0)

    def test_e2_tasks_carry_geometry_and_stay_hidden_target_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = plan_run(
                "e2-recovery",
                output_dir=tmp,
                config={"count": 2, "methods": ["ruletune"], "seeds": [0], "candidate_budget": 4},
                release_root=FIXTURE_RELEASE,
                count=2,
                shard_count=1,
            )
            self.assertEqual(result.status, "planned")
            tasks = list(read_jsonl(Path(tmp) / "task_manifest.jsonl"))
            self.assertTrue(tasks)
            for task in tasks:
                row = task.get("source_row") or {}
                self.assertIn("geometry_ref", row)
                self.assertNotIn("hidden_target", task)
                self.assertNotIn("hidden_target", row)
                self.assertEqual(task.get("target_access_policy"), "evaluator_sidecar_required")


if __name__ == "__main__":
    unittest.main()
