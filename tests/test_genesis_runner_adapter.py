# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the Genesis episode-executor adapter (no GPU required).

The Genesis runtime paths are exercised separately by the single-asset smoke;
here we pin the failure/blocked semantics and the hidden-target boundary.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import softsim_genesis_runner as sgr
from softsim_formal.canonical import fingerprint
from softsim_formal.e3a import validate_episode_metrics
from softsim_formal.records import make_run_manifest, make_task_spec, validate_record

ROOT = Path(__file__).resolve().parents[1]

HIDDEN_KEYS = {"hidden_target", "target", "ground_truth", "groundtruth", "oracle_target", "oracle_parameters", "true_parameters", "heldout_target", "secret_target"}


def _manifest(experiment_id: str) -> dict:
    return make_run_manifest(
        run_id="run-adaptertest",
        experiment_id=experiment_id,
        protocol_hash=fingerprint({"protocol": experiment_id}),
        expected_task_count=1,
        task_manifest_sha256=fingerprint([]),
        config_hash=fingerprint({"cfg": 1}),
        code_hash=fingerprint({"code": 1}),
        population=f"{experiment_id}_population",
        denominator_scope="planned_tasks_including_failures",
    )


def _task(experiment_id: str, **extra) -> dict:
    spec = make_task_spec(
        experiment_id=experiment_id,
        protocol_hash=fingerprint({"protocol": experiment_id}),
        population=f"{experiment_id}_population",
        sample_id="sample-x",
        asset_revision_id="sample-x-r001",
        physical_bundle_id="pb-x",
        method_id=extra.pop("method_id", "contract_probe"),
        extra=extra,
    )
    return spec


def _assert_no_hidden_payload(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in {"target_handle", "hidden_target_handle", "target_access_policy", "target_provider", "target_provider_required"}:
                continue
            assert normalized not in HIDDEN_KEYS, f"hidden key leaked: {key}"
            assert not normalized.startswith("target_"), f"hidden key leaked: {key}"
            assert not normalized.endswith("_target"), f"hidden key leaked: {key}"
            _assert_no_hidden_payload(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_hidden_payload(item)


class GenesisRunnerAdapterTest(unittest.TestCase):
    def test_missing_mesh_yields_terminal_failure_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest("e1-runtime")
            task = _task("e1-runtime")
            record = sgr.run_episode(task, manifest, Path(tmp) / "attempt-000")
            validate_record(record)
            self.assertEqual(record["execution_status"], "failed")
            self.assertEqual(record["failure_domain"], "infrastructure")
            self.assertEqual(record["failure_code"], "mesh_unresolved")
            self.assertFalse(record["metrics"]["synthetic_formula_used"] if "synthetic_formula_used" in record["metrics"] else False)

    def test_e2_without_target_provider_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest("e2-recovery")
            task = _task("e2-recovery", method_id="ruletune", prior_parameters={"youngs_modulus_pa": 50000.0})
            record, steps, calls = sgr.run_episode(task, manifest, Path(tmp) / "attempt-000", target_provider=None)
            validate_record(record)
            self.assertEqual(record["outcome_status"], "blocked")
            self.assertEqual(record["failure_code"], "hidden_target_provider_missing")
            self.assertEqual(steps, [])
            self.assertEqual(calls, [])

    def test_e3a_failure_path_keeps_required_measurements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest("e3a")
            task = _task("e3a", force_n=15.0, fragility_regime="medium")
            record = sgr.run_episode(task, manifest, Path(tmp) / "attempt-000")
            validate_record(record)
            validate_episode_metrics(record)
            self.assertEqual(record["failure_code"], "mesh_unresolved")
            self.assertIn("damage_fraction", record["measurements"])
            self.assertIn("integrity", record["measurements"])
            self.assertIsInstance(record["measurements"]["safe_force_window"], dict)

    def test_e2_trace_phases_and_no_hidden_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mesh = Path(tmp) / "mesh.obj"
            mesh.write_text("v 0 0 0\n", encoding="utf-8")
            manifest = _manifest("e2-recovery")
            task = _task(
                "e2-recovery",
                method_id="ruletune",
                prior_parameters={"youngs_modulus_pa": 50000.0, "density_kg_m3": 800.0},
                candidate_budget=4,
                hidden_target_handle=fingerprint({"handle": 1}),
                target_access_policy="evaluator_sidecar_required",
                source_row={"solver_mesh_path": str(mesh)},
            )
            provider = mock.Mock(return_value={"youngs_modulus_pa": 60000.0, "density_kg_m3": 900.0})
            fake_probe = {
                "status": "ok",
                "report": {
                    "execution_status": "completed",
                    "explicit_behavior_evidence": True,
                    "behavior_pass": True,
                    "metrics": {"finite_fraction": 1.0, "contact_fraction": 0.9, "settled": True, "deformation_ratio": 0.8, "recovery_ratio": 0.95},
                },
            }
            with mock.patch.object(sgr, "_run_bridge_probe", return_value=fake_probe):
                record, steps, calls = sgr.run_episode(task, manifest, Path(tmp) / "attempt-000", target_provider=provider)
            validate_record(record)
            provider.assert_called_once()
            phases = [step.get("phase") for step in steps]
            self.assertIn("calibration", phases)
            self.assertIn("candidate_frozen", phases)
            self.assertIn("heldout_verification", phases)
            self.assertLess(phases.index("candidate_frozen"), phases.index("heldout_verification"))
            for payload in (record, steps, calls):
                _assert_no_hidden_payload(payload)
            self.assertEqual(record["measurements"]["candidate_count"], 1)
            self.assertTrue(record["measurements"]["accepted_first_pass"])
            self.assertFalse(record["measurements"]["correction_success"])
            self.assertEqual(record["agent_status"], "accepted_first_pass")

    def test_executor_contract_required_fields(self) -> None:
        contract = json.loads((ROOT / "configs" / "formal_executor_contract.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _manifest("e1-runtime")
            task = _task("e1-runtime")
            record = sgr.run_episode(task, manifest, Path(tmp) / "attempt-000")
            for field in contract["required_record_fields"]:
                self.assertIn(field, record, field)
            e3a_record = sgr.run_episode(_task("e3a", force_n=15.0), _manifest("e3a"), Path(tmp) / "attempt-001")
            for field in contract["e3a_required_measurements"]:
                self.assertIn(field, e3a_record["measurements"], field)

    def test_worker_module_is_stdlib_parseable(self) -> None:
        import py_compile

        py_compile.compile(str(ROOT / "code" / "src" / "softsim_genesis_runner" / "genesis_worker.py"), doraise=True)


class RunnerTraceGateTest(unittest.TestCase):
    """E2 trace-completeness gate must not relabel infrastructure failures."""

    def _run_with_executor(self, executor) -> dict:
        from softsim_formal.runner import _execute_task

        manifest = _manifest("e2-recovery")
        manifest["real_executor_required"] = True
        task = _task(
            "e2-recovery",
            method_id="ruletune",
            prior_parameters={"youngs_modulus_pa": 50000.0},
            hidden_target_handle=fingerprint({"handle": 1}),
            target_access_policy="evaluator_sidecar_required",
        )
        with tempfile.TemporaryDirectory() as tmp:
            record, steps, calls = _execute_task(
                task,
                manifest,
                Path(tmp),
                target_provider=lambda t: {"youngs_modulus_pa": 60000.0},
                episode_executor=executor,
            )
        return record

    def test_infra_failure_record_keeps_infrastructure_taxonomy(self) -> None:
        def executor(task, run_manifest, attempt_dir, target_provider=None):
            return sgr._failure_record(
                task,
                run_manifest,
                failure_code="GenesisWorkerCrash",
                failure_domain="infrastructure",
                detail="simulated crash",
                experiment_id="e2-recovery",
            ), [], []

        record = self._run_with_executor(executor)
        self.assertEqual(record["execution_status"], "failed")
        self.assertEqual(record["failure_domain"], "infrastructure")
        self.assertEqual(record["failure_code"], "GenesisWorkerCrash")

    def test_completed_record_without_trace_phases_is_blocked(self) -> None:
        def executor(task, run_manifest, attempt_dir, target_provider=None):
            from softsim_formal.records import make_episode_record

            return make_episode_record(
                run_manifest=run_manifest,
                task=task,
                execution_status="completed",
                outcome_status="success",
                failure_domain="none",
                metrics={"candidate_count": 1},
                measurements={"prior_parameters": {}, "final_parameters": {}, "candidate_count": 1},
            ), [], []

        record = self._run_with_executor(executor)
        self.assertEqual(record["outcome_status"], "blocked")
        self.assertEqual(record["failure_code"], "e2_trace_incomplete")


class E2MethodEngineTest(unittest.TestCase):
    """Method-dispatch semantics of the production E2 adapter (bridge mocked)."""

    def _run(self, method: str, budget: int, losses: list[float]):
        with tempfile.TemporaryDirectory() as tmp:
            mesh = Path(tmp) / "mesh.obj"
            mesh.write_text("v 0 0 0\n", encoding="utf-8")
            manifest = _manifest("e2-recovery")
            task = _task(
                "e2-recovery",
                method_id=method,
                prior_parameters={"youngs_modulus_pa": 50000.0, "density_kg_m3": 800.0, "poisson_ratio": 0.3},
                candidate_budget=budget,
                hidden_target_handle=fingerprint({"handle": method}),
                target_access_policy="evaluator_sidecar_required",
                source_row={"solver_mesh_path": str(mesh)},
            )
            queue = list(losses)
            target_params = {"youngs_modulus_pa": 60000.0, "density_kg_m3": 900.0, "poisson_ratio": 0.35}

            def fake_bridge(attempt_dir, *, label, mesh_path, parameters, route, numerics, probe, sample_id, timeout_s=None):
                # Evaluator-side reference rollouts get the target params and a
                # fixed deformation of 1.0; candidate rollouts pop the scripted
                # deformation.  behavior loss = mean metric distance.
                if dict(parameters) == target_params:
                    deform = 1.0
                else:
                    deform = queue.pop(0) if queue else 0.5
                return {
                    "status": "ok",
                    "report": {
                        "execution_status": "completed",
                        "explicit_behavior_evidence": True,
                        "behavior_pass": deform >= 0.99,
                        "metrics": {"finite_fraction": 1.0, "contact_fraction": 0.9, "settled": True, "deformation_ratio": deform, "recovery_ratio": 0.9},
                    },
                }

            with mock.patch.object(sgr, "_run_bridge_probe", side_effect=fake_bridge):
                record, steps, calls = sgr.run_episode(task, manifest, Path(tmp) / "attempt-000", target_provider=lambda t: dict(target_params))
            validate_record(record)
            return record, steps, calls

    def test_fixed_profile_spends_exactly_one_evaluation(self) -> None:
        record, steps, calls = self._run("fixed_profile", 8, [0.5])
        self.assertEqual(record["measurements"]["candidate_count"], 1)
        self.assertEqual([s["phase"] for s in steps], ["calibration", "candidate_frozen", "heldout_verification"])

    def test_sobol_respects_budget_and_is_deterministic(self) -> None:
        record_a, steps_a, _ = self._run("seeded_sobol", 4, [0.5] * 16)
        record_b, steps_b, _ = self._run("seeded_sobol", 4, [0.5] * 16)
        self.assertEqual(record_a["measurements"]["candidate_count"], 4)
        params_a = [s["candidate_parameters"] for s in steps_a if s["phase"] == "calibration"]
        self.assertEqual(len(params_a), 4)
        self.assertEqual(params_a, [s["candidate_parameters"] for s in steps_b if s["phase"] == "calibration"])
        self.assertEqual(params_a[0], record_a["measurements"]["prior_parameters"])

    def test_full_agent_abstains_on_flat_response(self) -> None:
        record, steps, _ = self._run("full_numeric_agent", 8, [0.5] * 32)
        self.assertEqual(record["agent_status"], "uncertain_human_escalation")
        self.assertEqual(record["failure_code"], "abstain_non_identifiable")
        self.assertEqual(steps[-1]["phase"], "abstain")
        phases = [s["phase"] for s in steps]
        self.assertNotIn("heldout_verification", phases)

    def test_abstain_trace_passes_runner_gate(self) -> None:
        from softsim_formal.runner import _validate_external_e2_trace

        record, steps, calls = self._run("full_numeric_agent", 8, [0.5] * 32)
        violation = _validate_external_e2_trace({"task_id": record["task_id"]}, record, steps, calls, require_trace=True)
        self.assertIsNone(violation)

    def test_ruletune_stops_on_calibration_pass(self) -> None:
        # candidate deformation 0.99 vs reference 1.0 -> loss 0.002 <= tolerance
        record, steps, _ = self._run("ruletune", 8, [0.99] * 32)
        self.assertEqual(record["measurements"]["candidate_count"], 1)
        self.assertTrue(record["measurements"]["accepted_first_pass"])
        self.assertFalse(record["measurements"]["correction_success"])

    def test_no_hidden_payload_in_any_output(self) -> None:
        record, steps, calls = self._run("seeded_sobol", 2, [0.5] * 8)
        for payload in (record, steps, calls):
            _assert_no_hidden_payload(payload)


if __name__ == "__main__":
    unittest.main()
