# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from softsim_formal.aggregate import aggregate_run
from softsim_formal.canonical import canonical_dumps, fingerprint, sha256_file
from softsim_formal.e2 import HeldoutAccessError, HiddenTargetGuard, validate_trace_isolation
from softsim_formal.e3a import validate_episode_metrics, validate_observation
from softsim_formal.planner import plan_run
from softsim_formal.release import BlockedRelease, ReleaseError, ReleaseReader
from softsim_formal.render import check_render, render_run
from softsim_formal.runner import run_all_shards, status_run


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "formal_release_v2" / "data"


class FormalReleaseReaderTest(unittest.TestCase):
    def test_missing_physical_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "asset_release_v2").mkdir(parents=True)
            with self.assertRaises(BlockedRelease) as raised:
                ReleaseReader(root).read()
            self.assertEqual(raised.exception.code, "blocked_missing_physical_release")

    def test_fixture_release_keeps_population_states(self) -> None:
        bundle = ReleaseReader(FIXTURE).read()
        self.assertEqual(len(bundle.physical), 12)
        self.assertEqual(len(bundle.verified_physical), 10)
        self.assertEqual(len(bundle.rejected_physical), 1)
        self.assertEqual(len(bundle.task_failures), 1)

    def test_asset_without_metric_scale_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            shutil.copytree(FIXTURE, root)
            ledger = root / "asset_release_v2" / "asset_ledger.jsonl"
            rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
            rows[0].pop("metric_scale", None)
            rows[0]["record_sha256"] = fingerprint({key: value for key, value in rows[0].items() if key != "record_sha256"})
            ledger.write_text("".join(canonical_dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifest = root / "asset_release_v2" / "release_manifest.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["ledger_sha256"] = sha256_file(ledger)
            manifest.write_text(canonical_dumps(value) + "\n", encoding="utf-8")
            with self.assertRaises(ReleaseError) as raised:
                ReleaseReader(root).read()
            self.assertEqual(raised.exception.code, "asset_row_invalid")

    def test_predicted_bundle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(FIXTURE, root / "data")
            work = root / "data"
            index = work / "physical_release_v12" / "physical_bundle_index.jsonl"
            rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
            rows[0]["status"] = "predicted"
            rows[0]["record_sha256"] = fingerprint({key: value for key, value in rows[0].items() if key != "record_sha256"})
            index.write_text("".join(canonical_dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifest = work / "physical_release_v12" / "physical_release_manifest.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["index_sha256"] = sha256_file(index)
            manifest.write_text(canonical_dumps(value) + "\n", encoding="utf-8")
            with self.assertRaises(ReleaseError) as raised:
                ReleaseReader(work).read()
            self.assertEqual(raised.exception.code, "predicted_bundle")

    def test_verified_bundle_without_trace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(FIXTURE, root / "data")
            work = root / "data"
            index = work / "physical_release_v12" / "physical_bundle_index.jsonl"
            rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
            rows[0].pop("agent_trace_ref", None)
            rows[0]["record_sha256"] = fingerprint({key: value for key, value in rows[0].items() if key != "record_sha256"})
            index.write_text("".join(canonical_dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifest = work / "physical_release_v12" / "physical_release_manifest.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["index_sha256"] = sha256_file(index)
            manifest.write_text(canonical_dumps(value) + "\n", encoding="utf-8")
            with self.assertRaises(ReleaseError) as raised:
                ReleaseReader(work).read()
            self.assertIn(raised.exception.code, {"trace_missing", "physical_row_invalid"})

    def test_unjoined_benchmark_is_rehashed_after_explicit_physical_join(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            shutil.copytree(FIXTURE, root)
            benchmark_path = root / "fragile_soft_benchmark" / "instance_manifest.jsonl"
            rows = [json.loads(line) for line in benchmark_path.read_text(encoding="utf-8").splitlines()]
            mappings = []
            for row in rows:
                row.pop("instance_hash")
                mappings.append({
                    "benchmark_instance_id": row["benchmark_instance_id"],
                    "physical_release_id": "softsim-100k-agent-genesis-v12-fixture",
                    "physical_bundle_id": row.pop("physical_bundle_id"),
                })
                for key in (
                    "sample_id", "asset_revision_id", "physical_bundle_status",
                    "physical_join_status", "formal_join_ready", "physical_release_id",
                ):
                    row.pop(key, None)
                row["instance_hash"] = fingerprint(row)
            benchmark_path.write_text(
                "".join(canonical_dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            physical_manifest_path = root / "physical_release_v12" / "physical_release_manifest.json"
            physical_manifest = json.loads(physical_manifest_path.read_text(encoding="utf-8"))
            physical_manifest["benchmark_manifest_sha256"] = sha256_file(benchmark_path)
            physical_manifest_path.write_text(canonical_dumps(physical_manifest) + "\n", encoding="utf-8")
            join_path = root / "fragile_soft_benchmark" / "benchmark_physical_join.jsonl"
            join_path.write_text(
                "".join(canonical_dumps(row) + "\n" for row in mappings),
                encoding="utf-8",
            )
            bundle = ReleaseReader(root).read()
            self.assertEqual(len(bundle.benchmark_manifest), len(rows))
            self.assertTrue(bundle.benchmark_join_report["all_joined_verified"])
            self.assertEqual(bundle.benchmark_join_report["schema_version"], "fragile-soft.physical-join.v1")
            self.assertTrue(all(row["physical_join_status"] == "joined_verified" for row in bundle.benchmark_manifest))
            self.assertTrue(all(row.get("source_instance_hash") for row in bundle.benchmark_manifest))

    def test_join_map_from_another_physical_release_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            shutil.copytree(FIXTURE, root)
            join_path = root / "fragile_soft_benchmark" / "benchmark_physical_join.jsonl"
            join_path.write_text(
                canonical_dumps({
                    "benchmark_instance_id": "fsb-fixture-r000-d00",
                    "physical_release_id": "foreign-physical-release",
                    "physical_bundle_id": "pb-fixture-00000",
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ReleaseError) as raised:
                ReleaseReader(root).read()
            self.assertEqual(raised.exception.code, "benchmark_physical_join_mismatch")

    def test_symlinked_authority_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            shutil.copytree(FIXTURE, root)
            target = root / "physical_release_v12" / "physical_release_manifest.json"
            backup = root / "physical_release_v12" / "physical_release_manifest.real.json"
            target.rename(backup)
            target.symlink_to(backup.name)
            with self.assertRaises(ReleaseError) as raised:
                ReleaseReader(root).read()
            self.assertEqual(raised.exception.code, "symlink_input")

    def test_symlinked_logical_root_binding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            shutil.copytree(FIXTURE, root)
            target = root / "artifacts"
            link = Path(directory) / "artifact-link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ReleaseError) as raised:
                ReleaseReader(root, root_bindings={"fixture": link}).read()
            self.assertEqual(raised.exception.code, "symlink_input")


class FormalPipelineTest(unittest.TestCase):
    def test_plan_run_resume_aggregate_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            planned = plan_run("e3a", output_dir=run_dir, fixture=True, count=12, shard_count=3)
            self.assertEqual(planned.status, "planned")
            first = run_all_shards(run_dir, workers=2)
            self.assertEqual(first["status"], "completed")
            before = status_run(run_dir)
            self.assertEqual(before["completed_task_count"], 12)
            # A second pass must reuse terminal markers and preserve attempt 000.
            second = run_all_shards(run_dir, workers=2)
            self.assertEqual(second["status"], "completed")
            attempts = list((run_dir / "tasks").glob("*/attempt-*"))
            self.assertTrue(attempts)
            self.assertFalse(any(path.name == "attempt-001" for path in attempts))
            merged = aggregate_run(run_dir)
            self.assertTrue(merged["summary"]["complete"])
            self.assertEqual(merged["summary"]["planned_denominator"], 12)
            rendered = render_run(run_dir, output_dir=run_dir / "rendered")
            self.assertEqual(rendered["denominator"]["planned"], 12)
            self.assertTrue(check_render(run_dir / "rendered")["valid"])
            header = (run_dir / "rendered" / "summary.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("source_records_sha256", header)
            self.assertIn("planned_denominator", header)
            self.assertTrue((run_dir / "rendered" / "figure_data.csv").is_file())
            rendered_rows = json.loads((run_dir / "rendered" / "summary.json").read_text(encoding="utf-8"))["rows"]
            self.assertTrue(rendered_rows)
            self.assertTrue(any(row.get("damage_fraction_value") is not None for row in rendered_rows))

    def test_process_isolation_executes_and_preserves_terminal_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            plan_run("e1-runtime", output_dir=run_dir, fixture=True, count=1)
            result = run_all_shards(run_dir, process_isolation=True, python_executable=sys.executable)
            self.assertEqual(result["status"], "completed")
            row = json.loads((run_dir / "shard_records" / "shard-0000.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["attempt_id"], "attempt-000")

    def test_fingerprint_change_makes_terminal_result_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            plan_run("e1-runtime", output_dir=run_dir, fixture=True, count=2)
            run_all_shards(run_dir)
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["config_hash"] = fingerprint({"changed": True})
            manifest_path.write_text(canonical_dumps(manifest) + "\n", encoding="utf-8")
            run_all_shards(run_dir)
            attempts = list((run_dir / "tasks").glob("*/attempt-*"))
            self.assertTrue(any(path.name == "attempt-001" for path in attempts))

    def test_contract_schema_change_changes_package_code_hash(self) -> None:
        from softsim_formal.planner import package_code_hash
        schema = ROOT / "code" / "contracts" / "v1" / "asset_release.schema.json"
        original = schema.read_text(encoding="utf-8")
        before = package_code_hash()
        # An installed wheel is immutable and intentionally does not inspect
        # the checkout's external contract directory.  Source-tree builds do,
        # so a schema edit there must invalidate the hash.
        planner_path = Path(__import__("softsim_formal.planner", fromlist=["__file__"]).__file__).resolve()
        source_layout = planner_path == (ROOT / "code" / "src" / "softsim_formal" / "planner.py").resolve()
        if not source_layout:
            self.assertTrue(before.startswith("sha256:") and len(before) == 71)
            return
        try:
            schema.write_text(original + "\n", encoding="utf-8")
            after = package_code_hash()
        finally:
            schema.write_text(original, encoding="utf-8")
        self.assertNotEqual(before, after)

    def test_aggregate_rejects_record_from_another_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            plan_run("e1-runtime", output_dir=run_dir, fixture=True, count=1)
            run_all_shards(run_dir)
            shard = run_dir / "shard_records" / "shard-0000.jsonl"
            row = json.loads(shard.read_text(encoding="utf-8").splitlines()[0])
            row["run_id"] = "run-foreign"
            row["record_sha256"] = fingerprint({key: value for key, value in row.items() if key != "record_sha256"})
            shard.write_text(canonical_dumps(row) + "\n", encoding="utf-8")
            shard_summary_path = run_dir / "shard_records" / "shard-0000.json"
            shard_summary = json.loads(shard_summary_path.read_text(encoding="utf-8"))
            shard_summary["record_file_sha256"] = sha256_file(shard)
            shard_summary_path.write_text(canonical_dumps(shard_summary) + "\n", encoding="utf-8")
            aggregate_run(run_dir)
            summary = json.loads((run_dir / "aggregate_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["invalid_records"])
            self.assertFalse(summary["complete"])

    def test_e1_failure_and_e3a_failure_retain_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            plan_run("e3a", output_dir=run_dir, fixture=True, count=10)
            run_all_shards(run_dir)
            aggregate_run(run_dir)
            rows = [json.loads(line) for line in (run_dir / "canonical" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
            failures = [row for row in rows if row["outcome_status"] == "failure"]
            self.assertTrue(failures)
            for row in failures:
                validate_episode_metrics(row)
                self.assertIn("damage_fraction", row["measurements"])

    def test_e2_first_pass_is_not_counted_as_correction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan_run("e2-recovery", output_dir=directory, fixture=True, count=4)
            run_all_shards(directory)
            aggregate_run(directory)
            rows = [json.loads(line) for line in (Path(directory) / "canonical" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
            for row in rows:
                measurements = row["measurements"]
                if measurements.get("accepted_first_pass"):
                    self.assertFalse(measurements.get("correction_success"))
                if measurements.get("correction_success"):
                    self.assertGreater(int(measurements.get("candidate_count", 0)), 1)


class IsolationAndContractTest(unittest.TestCase):
    def test_heldout_guard_and_trace_isolation(self) -> None:
        guard = HiddenTargetGuard({"loss": 1.0}, {"loss": 0.1})
        with self.assertRaises(HeldoutAccessError):
            guard.read_heldout()
        guard.freeze()
        self.assertEqual(guard.read_heldout()["loss"], 0.1)
        validate_trace_isolation([
            {"phase": "calibration", "heldout_feedback_used": False},
            {"phase": "candidate_frozen", "heldout_feedback_used": False},
            {"phase": "heldout_verification", "feedback_allowed": False, "heldout_feedback_used": False},
        ])

    def test_privileged_observation_is_rejected(self) -> None:
        validate_observation({"rgb": "fixture", "gripper_width": 0.04})
        with self.assertRaises(ValueError):
            validate_observation({"rgb": "fixture", "damage_state": 0.2})


if __name__ == "__main__":
    unittest.main()
