# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ConfigAndInstallTest(unittest.TestCase):
    def test_method_contract_vocabulary(self) -> None:
        value = json.loads((ROOT / "configs" / "formal_method_contract.json").read_text(encoding="utf-8"))
        self.assertEqual(value["genesis_version"], "1.3.3")
        self.assertEqual(value["solver"]["cloth"], "PBD.Cloth")
        self.assertEqual(value["solver"]["coupler"], "LegacyCoupler")
        self.assertFalse(value["ipc_used"])
        self.assertFalse(value["uipc_used"])
        self.assertFalse(value["release_gate"]["accepted_first_pass_is_correction_success"])

    def test_clean_install_target_imports_without_pythonpath(self) -> None:
        # The command intentionally uses the environment's interpreter and a
        # temporary target directory; no repository PYTHONPATH is injected.
        probe = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True, check=False)
        if probe.returncode != 0:
            self.skipTest("test interpreter has no pip; install smoke is covered by the deployment run")
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--target", directory, str(ROOT / "code")], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            smoke = subprocess.run([sys.executable, "-c", "from pathlib import Path; import softsim_formal, fragile_soft_benchmark, softsim_physical_agent, softsim_genesis_runner, softsim_tools; assert (Path(softsim_tools.__file__).parent / 'genesis_probe.py').is_file(); assert (softsim_genesis_runner._REPO_SCRIPTS / 'genesis_behavior_runner.py').is_file(); print(softsim_formal.__version__)"], cwd=directory, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": directory}, capture_output=True, text=True, check=False)
            self.assertEqual(smoke.returncode, 0, smoke.stderr)

    def test_scheduler_command_is_explicit_and_deterministic(self) -> None:
        from softsim_formal.scheduler import SlurmArraySpec
        command = SlurmArraySpec("runs/e1", 4, episode_executor="softsim_genesis_runner:run_episode", target_provider="softsim_targets:get_target").command()
        self.assertIn("--array=0-3", command)
        self.assertIn("softsim-exp run", command)
        self.assertIn("--episode-executor softsim_genesis_runner:run_episode", command)
        self.assertIn("--target-provider softsim_targets:get_target", command)
        self.assertIn("--process-isolation", command)


if __name__ == "__main__":
    unittest.main()
