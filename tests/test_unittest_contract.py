# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from softsim_physical_agent.agent import build_tasks, run_task
from softsim_physical_agent.cli import _fixture_task
from softsim_physical_agent.io import read_json
from softsim_physical_agent.simulators import MockSimulator


class PhysicalAgentContractTest(unittest.TestCase):
    def _config(self):
        return {
            "backend": "mock",
            "proposer_backend": "ruletune",
            "max_iterations": 3,
            "update_gain": 0.8,
            "calibration_probe": {"probe_id": "calibration.unittest", "stimulus": {"kind": "drop"}, "action": {"kind": "release"}, "config": {"steps": 4}},
            "heldout_probe": {"probe_id": "heldout.unittest", "stimulus": {"kind": "compression"}, "action": {"kind": "compress_release"}, "config": {"steps": 5}},
        }

    def test_first_pass_writes_complete_task_contract(self):
        config = self._config()
        task = build_tasks([_fixture_task("unittest", {"density_kg_m3": 600.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30})], config)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = run_task(task, root, config, MockSimulator(config), force=True)
            self.assertEqual(marker["status"], "accepted_first_pass")
            task_dir = root / "tasks" / "unittest"
            for relative in ("prior.json", "physics.json", "agent_trace.jsonl", "heldout/report.json", "SUCCESS.json"):
                self.assertTrue((task_dir / relative).is_file(), relative)

    def test_model_limit_is_not_silently_dropped(self):
        config = self._config()
        task = build_tasks([_fixture_task("unittest-limit", {"density_kg_m3": 700.0}, behavior="model_limit")], config)[0]
        with tempfile.TemporaryDirectory() as temporary:
            marker = run_task(task, Path(temporary), config, MockSimulator(config), force=True)
            self.assertEqual(marker["status"], "rejected_model_limit")
            self.assertEqual(read_json(Path(temporary) / "tasks" / "unittest-limit" / "physics.json")["status"], "rejected_model_limit")


if __name__ == "__main__":
    unittest.main()

