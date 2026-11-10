# SPDX-License-Identifier: Apache-2.0
"""Regression: every E3-A terminal failure path must pass the schema gate.

E3-A pilot v1 crashed because a prefetch-failure record (cloud launcher entry) was
E1-shaped.  These tests pin the contract for both the runner fallback and the
entry's prefetch-failure writer.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from softsim_formal.canonical import fingerprint
from softsim_formal.records import make_episode_record, validate_record
from softsim_formal.runner import terminal_failure_fields
from softsim_formal.schema import validate_episode_record
from softsim_formal.e3a import validate_episode_metrics
from softsim_formal.records import make_run_manifest, make_task_spec


def _e3a_task():
    ph = fingerprint({"p": "e3a-fail"})
    manifest = make_run_manifest(
        run_id="run-e3a-fail",
        experiment_id="e3a",
        protocol_hash=ph,
        expected_task_count=1,
        task_manifest_sha256=fingerprint(["x"]),
        config_hash=fingerprint({"c": 1}),
        code_hash=fingerprint({"k": 1}),
        population="e3a_fragile_soft_episodes",
        denominator_scope="planned_tasks_including_failures",
    )
    task = make_task_spec(
        experiment_id="e3a",
        protocol_hash=ph,
        population="e3a_fragile_soft_episodes",
        sample_id="s",
        asset_revision_id="r",
        physical_bundle_id="pb",
        canonical_strata={"family": "solid", "canonical_id": "x", "split": "test"},
    )
    return manifest, task


class E3ATerminalFailureContractTest(unittest.TestCase):
    def test_terminal_failure_fields_e3a_complete(self) -> None:
        manifest, task = _e3a_task()
        outcomes, measurements = terminal_failure_fields("e3a", "prefetch_failed", "test")
        record = make_episode_record(
            run_manifest=manifest,
            task=task,
            execution_status="failed",
            outcome_status="failure",
            failure_domain="infrastructure",
            failure_code="prefetch_failed",
            metrics={"finite_fraction": 0.0},
            outcomes=outcomes,
            measurements=measurements,
        )
        validate_record(record)
        validate_episode_metrics(record)

    def test_terminal_failure_fields_e1_shape_unchanged(self) -> None:
        manifest, task = _e3a_task()
        manifest["experiment_id"] = "e1-runtime"
        task["experiment_id"] = "e1-runtime"
        outcomes, measurements = terminal_failure_fields("e1-runtime", "prefetch_failed", "test")
        record = make_episode_record(
            run_manifest=manifest,
            task=task,
            execution_status="failed",
            outcome_status="failure",
            failure_domain="infrastructure",
            failure_code="prefetch_failed",
            metrics={"finite_fraction": 0.0},
            outcomes={"contract_pass": False, "runtime_pass": False},
            measurements=measurements,
        )
        validate_record(record)

    def test_entry_prefetch_failure_writer_e3a(self) -> None:
        import sys

        entry_dir = Path(__file__).resolve().parents[1] / "deployment" / "cloud_e1"
        if not (entry_dir / "entry_e1_gate_oss.py").is_file():
            self.skipTest("internal cloud launcher is excluded from the source release")
        sys.path.insert(0, str(entry_dir))
        import entry_e1_gate_oss as entry

        manifest, task = _e3a_task()
        with tempfile.TemporaryDirectory() as tmp:
            entry._write_prefetch_failure(Path(tmp), manifest, task, "bundle_artifact_missing", "simulated")
            import json

            record = json.loads((Path(tmp) / "tasks" / str(task["task_id"]) / "attempt-000" / "episode.json").read_text())
        validate_episode_record(record)
        validate_episode_metrics(record)
        self.assertEqual(record["failure_code"], "bundle_artifact_missing")
        self.assertFalse(record["outcomes"]["task_success"])


if __name__ == "__main__":
    unittest.main()
