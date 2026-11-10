# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fragile_soft_benchmark.adapter import P0Adapter
from fragile_soft_benchmark.qualification import make_default_qualification_instances, run_qualification
from fragile_soft_benchmark.validation import validate_output_dir


class FragileSoftContractTest(unittest.TestCase):
    def test_local_contract(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "p0_legacy"
        adapter = P0Adapter.from_files(
            fixture / "manifest.json",
            fixture / "material_profiles.json",
            summary_json_path=fixture / "summary.json",
            summary_csv_path=fixture / "summary.csv",
            source_root=fixture,
        )
        instances = make_default_qualification_instances(
            adapter,
            {"qualification_revisions_per_group": 1, "qualification_difficulty_ids": ["d00", "d02", "d05"]},
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_qualification(instances, output_dir=directory)
            self.assertEqual(result["report"]["status"], "qualified")
            self.assertTrue(validate_output_dir(directory, expected_count=12)["valid"])
            metric = json.loads((Path(directory) / "metric_records.jsonl").read_text().splitlines()[0])
            self.assertIn("safe_force_window", metric["metrics"])


if __name__ == "__main__":
    unittest.main()
