# SPDX-License-Identifier: Apache-2.0
"""E2 evaluator-side target sidecar contract tests."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import softsim_targets

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RELEASE = ROOT / "tests" / "fixtures" / "formal_release_v2" / "data"


class TargetSidecarTest(unittest.TestCase):
    def _plan(self, tmp: str) -> Path:
        """Minimal E2-shaped frozen plan: handle + inline prior per task."""
        out = Path(tmp) / "plan"
        out.mkdir(parents=True, exist_ok=True)
        from softsim_formal.canonical import fingerprint

        rows = []
        for index in range(2):
            sample = f"sample-{index:04d}"
            rows.append({
                "task_id": fingerprint({"task": index}),
                "sample_id": sample,
                "hidden_target_handle": fingerprint({"handle": sample}),
                "prior_parameters": {"youngs_modulus_pa": 50000.0 + index * 1000.0, "density_kg_m3": 600.0, "poisson_ratio": 0.3},
            })
        with open(out / "task_manifest.jsonl", "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return out

    def test_build_and_provide_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir = self._plan(tmp)
            sidecar = Path(tmp) / "sidecar.jsonl"
            info = softsim_targets.build_sidecar(plan_dir, seed=7, output=sidecar)
            self.assertGreater(info["rows"], 0)
            mode = os.stat(sidecar).st_mode & 0o777
            self.assertEqual(mode, 0o600)
            os.environ[softsim_targets.SIDECAR_ENV] = str(sidecar)
            try:
                softsim_targets._cache.clear()
                for task in (json.loads(line) for line in open(plan_dir / "task_manifest.jsonl")):
                    target = softsim_targets.get_target(task)
                    self.assertIsInstance(target, dict)
                    self.assertTrue(target)
                    prior = task["prior_parameters"]
                    # Hidden target must differ from the prior in at least one
                    # parameter (otherwise the recovery problem is trivial).
                    self.assertTrue(any(abs(float(target[k]) - float(prior[k])) > 1e-9 for k in prior))
            finally:
                os.environ.pop(softsim_targets.SIDECAR_ENV, None)
                softsim_targets._cache.clear()

    def test_missing_env_and_unknown_handle_fail_loudly(self) -> None:
        os.environ.pop(softsim_targets.SIDECAR_ENV, None)
        with self.assertRaises(softsim_targets.TargetSidecarError):
            softsim_targets.get_target({"hidden_target_handle": "sha256:0" * 8})
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = Path(tmp) / "s.jsonl"
            sidecar.write_text(json.dumps({"hidden_target_handle": "sha256:" + "a" * 64, "parameters": {"x": 1.0}}) + "\n")
            os.environ[softsim_targets.SIDECAR_ENV] = str(sidecar)
            try:
                softsim_targets._cache.clear()
                with self.assertRaises(softsim_targets.TargetSidecarError):
                    softsim_targets.get_target({"hidden_target_handle": "sha256:" + "b" * 64})
            finally:
                os.environ.pop(softsim_targets.SIDECAR_ENV, None)
                softsim_targets._cache.clear()

    def test_sidecar_deterministic_per_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan_dir = self._plan(tmp)
            a = softsim_targets.build_sidecar(plan_dir, seed=7, output=Path(tmp) / "a.jsonl")
            b = softsim_targets.build_sidecar(plan_dir, seed=7, output=Path(tmp) / "b.jsonl")
            c = softsim_targets.build_sidecar(plan_dir, seed=8, output=Path(tmp) / "c.jsonl")
            self.assertEqual(Path(a["sidecar"]).read_bytes(), Path(b["sidecar"]).read_bytes())
            self.assertNotEqual(Path(a["sidecar"]).read_bytes(), Path(c["sidecar"]).read_bytes())


if __name__ == "__main__":
    unittest.main()
