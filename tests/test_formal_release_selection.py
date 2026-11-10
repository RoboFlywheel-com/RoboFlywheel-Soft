# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import tempfile
import unittest
import json
import shutil
from pathlib import Path

from softsim_formal.planner import plan_run
from softsim_formal.runner import run_all_shards
from softsim_formal.aggregate import aggregate_run
from softsim_formal.records import make_episode_record
from softsim_formal.canonical import canonical_dumps, fingerprint, sha256_file


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "formal_release_v2" / "data"


class ReleaseSelectionTest(unittest.TestCase):
    def test_e2_plan_excludes_nonverified_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = plan_run("e2-recovery", output_dir=directory, release_root=FIXTURE, count=0)
            self.assertEqual(result.status, "planned")
            self.assertEqual(result.task_count, 10)
            self.assertEqual(result.manifest["physical_execution_population"], 12)
            self.assertEqual(result.manifest["verified_physical_count"], 10)

    def test_e1_plan_keeps_terminal_physical_denominator_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = plan_run("e1-runtime", output_dir=directory, release_root=FIXTURE, count=0)
            self.assertEqual(result.status, "planned")
            self.assertEqual(result.task_count, 12)
            run_all_shards(directory)
            aggregate = aggregate_run(directory)
            self.assertEqual(aggregate["summary"]["planned_denominator"], 12)
            self.assertEqual(aggregate["summary"]["observed_records"], 12)
            self.assertEqual(aggregate["summary"]["invalid_record_count"], 12)
            self.assertIn("real_executor_missing", aggregate["summary"]["failure_code_counts"])

    def test_e3a_plan_requires_promoted_qualified_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = plan_run("e3a", output_dir=directory, release_root=FIXTURE, count=0)
            self.assertEqual(result.status, "planned")
            self.assertEqual(result.task_count, 6)

    def test_e3a_plan_accepts_join_map_outside_release_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "release"
            shutil.copytree(FIXTURE, root)
            joined = root / "fragile_soft_benchmark" / "instance_manifest.jsonl"
            rows = [json.loads(line) for line in joined.read_text(encoding="utf-8").splitlines()]
            mappings = []
            for row in rows:
                mappings.append({"benchmark_instance_id": row["benchmark_instance_id"], "physical_release_id": row["physical_release_id"], "physical_bundle_id": row["physical_bundle_id"]})
                for key in ("sample_id", "asset_revision_id", "physical_bundle_id", "physical_bundle_status", "physical_join_status", "formal_join_ready", "physical_release_id"):
                    row.pop(key, None)
                row.pop("instance_hash", None)
                row["instance_hash"] = fingerprint(row)
            joined.write_text("".join(canonical_dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifest_path = root / "physical_release_v12" / "physical_release_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["benchmark_manifest_sha256"] = sha256_file(joined)
            manifest_path.write_text(canonical_dumps(manifest) + "\n", encoding="utf-8")
            external_map = Path(directory) / "join-map.jsonl"
            external_map.write_text("".join(canonical_dumps(row) + "\n" for row in mappings), encoding="utf-8")
            output = Path(directory) / "run"
            result = plan_run("e3a", output_dir=output, release_root=root, benchmark_join_map=external_map, count=1)
            self.assertEqual(result.status, "planned")
            self.assertEqual(result.task_count, 1)
            self.assertTrue((output / "benchmark_join_report.json").is_file())
            self.assertEqual(result.manifest["benchmark_join_report_ref"], "benchmark_join_report.json")

    def test_production_e2_manifest_contains_no_hidden_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = plan_run("e2-recovery", output_dir=directory, release_root=FIXTURE, count=1)
            self.assertEqual(result.status, "planned")
            rows = [json.loads(line) for line in (Path(directory) / "task_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertNotIn("hidden_target", rows[0])
            self.assertEqual(rows[0]["target_access_policy"], "evaluator_sidecar_required")
            self.assertIn("hidden_target_handle", rows[0])

    def test_disabled_external_methods_stay_out_of_comparison_population(self) -> None:
        config = json.loads((ROOT / "configs" / "e2_formal.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            result = plan_run("e2-recovery", output_dir=directory, config=config, fixture=True, count=1)
            self.assertEqual(result.status, "planned")
            methods = {json.loads(line)["method_id"] for line in (Path(directory) / "task_manifest.jsonl").read_text(encoding="utf-8").splitlines()}
            self.assertNotIn("gp_bo_disabled_optional", methods)
            self.assertNotIn("external_policy_baseline", methods)
            self.assertIn("gp_bo_disabled_optional", result.manifest["disabled_methods"])

    def test_production_e2_without_target_provider_is_explicitly_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = plan_run("e2-recovery", output_dir=directory, release_root=FIXTURE, count=1)
            self.assertEqual(result.status, "planned")
            run_all_shards(directory)
            aggregate = aggregate_run(directory)
            rows = [json.loads(line) for line in (Path(directory) / "canonical" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["failure_code"], "hidden_target_provider_missing")
            self.assertFalse(aggregate["completion"]["publishable"])

    def test_production_e1_without_real_executor_is_blocked_not_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = plan_run("e1-runtime", output_dir=directory, release_root=FIXTURE, count=1)
            self.assertEqual(result.status, "planned")
            # Isolate the blocked/invalid publication gate from the fixture's
            # synthetic flag: even a nominally production manifest must fail.
            manifest_path = Path(directory) / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["synthetic"] = False
            manifest_path.write_text(canonical_dumps(manifest) + "\n", encoding="utf-8")
            run_all_shards(directory)
            aggregate = aggregate_run(directory)
            rows = [json.loads(line) for line in (Path(directory) / "canonical" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["failure_code"], "real_executor_missing")
            self.assertEqual(aggregate["summary"]["blocked_record_count"], 1)
            self.assertEqual(aggregate["summary"]["invalid_record_count"], 1)
            self.assertFalse(aggregate["completion"]["publishable"])

    def test_injected_executor_is_the_only_production_execution_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = plan_run("e1-runtime", output_dir=directory, release_root=FIXTURE, count=1)
            self.assertEqual(result.status, "planned")
            def executor(task, run_manifest, attempt_dir, **kwargs):
                return make_episode_record(
                    run_manifest=run_manifest,
                    task=task,
                    execution_status="completed",
                    outcome_status="success",
                    failure_domain="none",
                    metrics={"real_executor_used": True},
                    measurements={"backend": "fixture-injected"},
                )
            run_all_shards(directory, episode_executor=executor)
            aggregate_run(directory)
            rows = [json.loads(line) for line in (Path(directory) / "canonical" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(rows[0]["metrics"]["real_executor_used"])
            self.assertNotIn("synthetic_formula_used", rows[0]["measurements"])

    def test_injected_e2_executor_cannot_bypass_heldout_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = plan_run("e2-recovery", output_dir=directory, release_root=FIXTURE, count=1)
            self.assertEqual(result.status, "planned")

            def leaking_executor(task, run_manifest, attempt_dir, **kwargs):
                record = make_episode_record(
                    run_manifest=run_manifest,
                    task=task,
                    execution_status="completed",
                    outcome_status="success",
                    failure_domain="none",
                    metrics={"real_executor_used": True},
                    measurements={
                        "backend": "fixture-injected",
                        "prior_parameters": {"youngs_modulus_pa": 1.0},
                        "final_parameters": {"youngs_modulus_pa": 2.0},
                        "candidate_count": 2,
                    },
                )
                return record, [
                    {"phase": "calibration", "heldout_feedback_used": False},
                    {"phase": "heldout_verification", "feedback_allowed": False},
                    {"phase": "candidate_frozen", "feedback_allowed": False},
                ], []

            run_all_shards(
                directory,
                target_provider=lambda task: {"youngs_modulus_pa": 60000.0},
                episode_executor=leaking_executor,
            )
            aggregate_run(directory)
            rows = [json.loads(line) for line in (Path(directory) / "canonical" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["failure_code"], "heldout_isolation_violation")
            self.assertFalse(rows[0]["measurements"].get("synthetic_formula_used", False))

    def test_injected_e2_executor_cannot_return_target_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = plan_run("e2-recovery", output_dir=directory, release_root=FIXTURE, count=1)
            self.assertEqual(result.status, "planned")

            def leaking_executor(task, run_manifest, attempt_dir, **kwargs):
                record = make_episode_record(
                    run_manifest=run_manifest,
                    task=task,
                    execution_status="completed",
                    outcome_status="success",
                    failure_domain="none",
                    metrics={"real_executor_used": True},
                    measurements={
                        "prior_parameters": {"youngs_modulus_pa": 1.0},
                        "final_parameters": {"youngs_modulus_pa": 2.0},
                        "candidate_count": 2,
                        "target_parameters": {"youngs_modulus_pa": 3.0},
                    },
                )
                return record, [
                    {"phase": "calibration", "heldout_feedback_used": False},
                    {"phase": "candidate_frozen", "feedback_allowed": False},
                    {"phase": "heldout_verification", "feedback_allowed": False},
                ], []

            run_all_shards(
                directory,
                target_provider=lambda task: {"youngs_modulus_pa": 60000.0},
                episode_executor=leaking_executor,
            )
            aggregate_run(directory)
            rows = [json.loads(line) for line in (Path(directory) / "canonical" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["failure_code"], "hidden_target_leak_detected")

    def test_e3a_executor_exception_retains_damage_and_integrity_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = plan_run("e3a", output_dir=directory, release_root=FIXTURE, count=1)
            self.assertEqual(result.status, "planned")

            def broken_executor(task, run_manifest, attempt_dir, **kwargs):
                raise RuntimeError("simulator unavailable")

            run_all_shards(directory, episode_executor=broken_executor)
            aggregate_run(directory)
            rows = [json.loads(line) for line in (Path(directory) / "canonical" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["failure_code"], "worker_exception")
            self.assertIn("damage_fraction", rows[0]["measurements"])
            self.assertIn("integrity", rows[0]["measurements"])
