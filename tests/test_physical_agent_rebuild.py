# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

try:
    import pytest
except ImportError:  # pragma: no cover - keeps unittest discovery dependency-free.
    class _PytestCompat:
        class MonkeyPatch:  # type-only compatibility for deferred annotations.
            pass

        @staticmethod
        def skip(reason: str):
            raise unittest.SkipTest(reason)

    pytest = _PytestCompat()

from softsim_physical_agent.agent import build_tasks, run_task
from softsim_physical_agent.aggregate import aggregate_run
from softsim_physical_agent.audit import build_golden_reference_audit
from softsim_physical_agent.cli import _fixture_task
from softsim_physical_agent.input_adapter import load_legacy_assets
from softsim_physical_agent.io import content_hash, file_sha256, read_json
from softsim_physical_agent.models import (
    ACCEPTED_FIRST_PASS,
    CORRECTED_THEN_ACCEPTED,
    REJECTED_MODEL_LIMIT,
    REJECTED_NUMERICAL_FAILURE,
    SIM_EXECUTED_UNVERIFIED,
    UNCERTAIN_HUMAN_ESCALATION,
)
from softsim_physical_agent.proposers import QwenProposer
from softsim_physical_agent.simulators import CommandSimulator, MockSimulator


def _config(**overrides):
    config = {
        "backend": "mock",
        "proposer_backend": "ruletune",
        "max_iterations": 5,
        "update_gain": 0.8,
        "calibration_probe": {
            "probe_id": "calibration.test.drop",
            "stimulus": {"kind": "drop", "height_m": 0.12},
            "action": {"kind": "release", "axis": "z"},
            "config": {"steps": 12, "dt_s": 0.001, "seed_offset": 1},
            "tolerance": 0.04,
            "required_metrics": ["loss", "contact_fraction", "settled"],
        },
        "heldout_probe": {
            "probe_id": "heldout.test.compression",
            "stimulus": {"kind": "compression", "ratio": 0.2, "axis": "x"},
            "action": {"kind": "compress_release", "hold_steps": 3},
            "config": {"steps": 14, "dt_s": 0.0008, "seed_offset": 2},
            "tolerance": 0.08,
            "required_metrics": ["loss", "contact_fraction", "settled"],
        },
    }
    config.update(overrides)
    return config


def _task(sample_id: str, target: dict[str, float], behavior: str = "normal"):
    return build_tasks([_fixture_task(sample_id, target, behavior=behavior)], _config())[0]


def test_first_pass_and_corrected_statuses(tmp_path: Path):
    config = _config()
    first = _task("first", {"density_kg_m3": 600.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30})
    corrected = _task("corrected", {"density_kg_m3": 840.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30})
    first_marker = run_task(first, tmp_path, config, MockSimulator(config), force=True)
    corrected_marker = run_task(corrected, tmp_path, config, MockSimulator(config), force=True)
    assert first_marker["status"] == ACCEPTED_FIRST_PASS
    assert corrected_marker["status"] == CORRECTED_THEN_ACCEPTED
    trace = (tmp_path / "tasks" / "corrected" / "agent_trace.jsonl").read_text(encoding="utf-8").splitlines()
    calibration_events = [json.loads(line) for line in trace if json.loads(line).get("event") == "calibration_probe"]
    assert len(calibration_events) >= 2
    assert calibration_events[0]["candidate_hash"] != calibration_events[-1]["candidate_hash"]


def test_budget_exhaustion_is_retained_and_has_heldout_record(tmp_path: Path):
    config = _config(max_iterations=1)
    task = _task("budget", {"density_kg_m3": 900.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30})
    marker = run_task(task, tmp_path, config, MockSimulator(config), force=True)
    assert marker["status"] == REJECTED_MODEL_LIMIT
    task_dir = tmp_path / "tasks" / "budget"
    assert (task_dir / "physics.json").is_file()
    heldout = read_json(task_dir / "heldout" / "report.json")
    assert heldout["execution_status"] == "not_executed"
    assert (task_dir / "SUCCESS.json").is_file()


def test_calibration_and_heldout_hashes_are_independent(tmp_path: Path):
    config = _config()
    task = _task("hashes", {"density_kg_m3": 600.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30})
    marker = run_task(task, tmp_path, config, MockSimulator(config), force=True)
    physics = read_json(tmp_path / "tasks" / "hashes" / marker["physics_ref"])
    calibration = physics["verification"]["calibration"]
    heldout = physics["verification"]["heldout"]
    for key in ("probe_hash", "stimulus_hash", "action_hash", "config_hash"):
        assert calibration[key] != heldout[key]
    assert physics["verification"]["heldout_feedback_used"] is False


def test_config_change_invalidates_success_and_preserves_history(tmp_path: Path):
    config_a = _config(update_gain=0.5)
    task = _task("resume", {"density_kg_m3": 600.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30})
    first = run_task(task, tmp_path, config_a, MockSimulator(config_a), force=True)
    config_b = _config(update_gain=0.9)
    second = run_task(task, tmp_path, config_b, MockSimulator(config_b), force=False)
    assert first["config_hash"] != second["config_hash"]
    history = tmp_path / "tasks" / "resume" / "history"
    assert history.is_dir()
    assert any(path.name == "SUCCESS.json" for path in history.rglob("SUCCESS.json"))


def test_aggregate_rewrites_trace_ref_and_marks_mock_non_publishable(tmp_path: Path):
    config = _config()
    task = _task("aggregate", {"density_kg_m3": 600.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30})
    run_task(task, tmp_path, config, MockSimulator(config), force=True)
    manifest = aggregate_run(tmp_path, expected_count=1, release_id="fixture-release")
    assert manifest["publishable"] is False
    index = [json.loads(line) for line in (tmp_path / "physical_bundle_index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert index[0]["trace_ref"] == "bundles/aggregate/agent_trace.jsonl"
    assert (tmp_path / index[0]["trace_ref"]).is_file()
    bundle = read_json(tmp_path / index[0]["bundle"]["path"])
    assert bundle["agent"]["trace_ref"] == index[0]["trace_ref"]
    assert file_sha256(tmp_path / index[0]["bundle"]["path"]) == index[0]["bundle"]["sha256"]
    assert bundle["record_sha256"] == content_hash({key: value for key, value in bundle.items() if key != "record_sha256"})
    trace_event = json.loads((tmp_path / index[0]["trace_ref"]).read_text(encoding="utf-8").splitlines()[0])
    assert trace_event["report_ref"].startswith("bundles/aggregate/")
    assert bundle["agent"]["trace_sha256"] == file_sha256(tmp_path / index[0]["trace_ref"])
    assert manifest["source_sha256"] == bundle["agent_source_sha256"]


def test_elastic_budget_extends_only_on_progress(tmp_path: Path):
    """A converging task earns extra iterations; a stalled task stops at base."""

    # Converging: target far from prior, gain 0.5 halves the error each round.
    converging = _config(max_iterations=2, max_iterations_cap=6, budget_step=2, update_gain=0.5)
    task = _task("elastic-win", {"density_kg_m3": 1600.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30})
    marker = run_task(task, tmp_path, converging, MockSimulator(converging), force=True)
    assert marker["status"] == CORRECTED_THEN_ACCEPTED
    trace = [json.loads(line) for line in (tmp_path / "tasks" / "elastic-win" / "agent_trace.jsonl").read_text().splitlines()]
    escalations = [event for event in trace if event.get("event") == "budget_escalation"]
    assert escalations, "converging task must record at least one budget_escalation"
    physics = read_json(tmp_path / "tasks" / "elastic-win" / marker["physics_ref"])
    assert physics["agent"]["budget_final"] > 2
    assert physics["agent"]["budget_cap"] == 6

    # Stalled: tiny gain keeps improving below the 5% gate -> no escalation.
    stalled = _config(max_iterations=2, max_iterations_cap=6, budget_step=2, update_gain=0.01)
    task2 = _task("elastic-stall", {"density_kg_m3": 1600.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30})
    marker2 = run_task(task2, tmp_path, stalled, MockSimulator(stalled), force=True)
    assert marker2["status"] == REJECTED_MODEL_LIMIT
    trace2 = [json.loads(line) for line in (tmp_path / "tasks" / "elastic-stall" / "agent_trace.jsonl").read_text().splitlines()]
    assert not [event for event in trace2 if event.get("event") == "budget_escalation"]
    assert len([event for event in trace2 if event.get("event") == "calibration_probe"]) == 2


def test_proposal_may_introduce_absent_whitelisted_parameter(tmp_path: Path):
    """A bounded proposal for an allowed parameter missing from the prior (e.g.
    damping for a not-settled diagnosis) must reach the next candidate instead
    of being silently clamped away."""

    from softsim_physical_agent.models import Proposal

    class AddViscosityProposer:
        def propose(self, **kwargs):
            candidate = kwargs["candidate"]
            if "plastic_viscosity_pa_s" not in candidate:
                return Proposal("proposed", patch={"plastic_viscosity_pa_s": 20.0}, rule_id="test.add_viscosity")
            # Second iteration: move density the rest of the way so the mock accepts.
            return Proposal("proposed", patch={"density_kg_m3": 840.0}, rule_id="test.settle_density")

    config = _config(max_iterations=4)
    task = _task("add-param", {"density_kg_m3": 840.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30})
    marker = run_task(task, tmp_path, config, MockSimulator(config), AddViscosityProposer(), force=True)
    assert marker["status"] == CORRECTED_THEN_ACCEPTED
    physics = read_json(tmp_path / "tasks" / "add-param" / marker["physics_ref"])
    assert physics["candidate"]["parameters"].get("plastic_viscosity_pa_s") == 20.0
    trace = [json.loads(line) for line in (tmp_path / "tasks" / "add-param" / "agent_trace.jsonl").read_text().splitlines()]
    proposals = [event for event in trace if event.get("event") == "candidate_proposal"]
    assert proposals[0]["patch"] == {"plastic_viscosity_pa_s": 20.0}


def test_aggregate_maps_five_agent_statuses_to_three_formal_statuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Regression: the release index keeps the 3-state formal vocabulary in
    ``status`` and the 5-state agent decision in ``agent_status``; a rejected or
    uncertain task must make the gate fire on the formal status, and
    ``accepted_first_pass`` must never be counted as a correction success."""

    for name in ("DASHSCOPE_API_KEY", "QWEN_API_KEY", "QWEN_VL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config = _config()
    runs = {
        "fp": _task("map-first", {"density_kg_m3": 600.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30}),
        "corr": _task("map-corrected", {"density_kg_m3": 840.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30}),
        "limit": _task("map-limit", {"density_kg_m3": 600.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30}, behavior="model_limit"),
        "numfail": _task("map-numfail", {"density_kg_m3": 600.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30}, behavior="numerical_failure"),
    }
    markers = {key: run_task(task, tmp_path, config, MockSimulator(config), force=True) for key, task in runs.items()}
    assert markers["fp"]["status"] == ACCEPTED_FIRST_PASS
    assert markers["corr"]["status"] == CORRECTED_THEN_ACCEPTED
    assert markers["limit"]["status"] == REJECTED_MODEL_LIMIT
    assert markers["numfail"]["status"] == REJECTED_NUMERICAL_FAILURE
    # An unavailable proposer on a failing calibration is an escalation, not a reject.
    qwen_config = _config(proposer_backend="qwen", max_iterations=3)
    uncertain_task = _task("map-uncertain", {"density_kg_m3": 900.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30})
    uncertain_marker = run_task(uncertain_task, tmp_path, qwen_config, MockSimulator(qwen_config), force=True)
    assert uncertain_marker["status"] == UNCERTAIN_HUMAN_ESCALATION

    manifest = aggregate_run(tmp_path, expected_count=5, release_id="status-mapping-regression")
    assert manifest["status_counts"] == {"verified": 2, "rejected": 2, "task_failure": 1}
    assert manifest["agent_status_counts"] == {
        ACCEPTED_FIRST_PASS: 1,
        CORRECTED_THEN_ACCEPTED: 1,
        REJECTED_MODEL_LIMIT: 1,
        REJECTED_NUMERICAL_FAILURE: 1,
        UNCERTAIN_HUMAN_ESCALATION: 1,
    }
    assert manifest["publishable"] is False
    assert "rejected_or_uncertain_tasks_present" in manifest["non_publishable_reasons"]
    rows = [json.loads(line) for line in (tmp_path / "physical_bundle_index.jsonl").read_text(encoding="utf-8").splitlines()]
    by_sample = {row["sample_id"]: row for row in rows}
    expected_mapping = {
        "map-first": ("verified", ACCEPTED_FIRST_PASS),
        "map-corrected": ("verified", CORRECTED_THEN_ACCEPTED),
        "map-limit": ("rejected", REJECTED_MODEL_LIMIT),
        "map-numfail": ("task_failure", REJECTED_NUMERICAL_FAILURE),
        "map-uncertain": ("rejected", UNCERTAIN_HUMAN_ESCALATION),
    }
    assert set(by_sample) == set(expected_mapping)
    for sample_id, (formal, agent_status) in expected_mapping.items():
        row = by_sample[sample_id]
        assert row["status"] == formal
        assert row["agent_status"] == agent_status
        if formal == "verified":
            assert "failure_code" not in row
        else:
            assert row["failure_code"]
    # accepted_first_pass is prior-compatible only; it is not a correction success.
    first_bundle = read_json(tmp_path / by_sample["map-first"]["bundle"]["path"])
    corrected_bundle = read_json(tmp_path / by_sample["map-corrected"]["bundle"]["path"])
    assert first_bundle["prior_compatible"] is True
    assert first_bundle["correction_applied"] is False
    assert corrected_bundle["correction_applied"] is True


def test_aggregate_all_formally_verified_does_not_add_rejected_gate(tmp_path: Path):
    class GenesisLikeMock(MockSimulator):
        backend_name = "genesis"

    config = _config()
    task = _task("production-like", {"density_kg_m3": 600.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30})
    task["source_kind"] = "server_asset_release"
    task["prior"].setdefault("provenance", {})["synthetic_fixture"] = False
    run_task(task, tmp_path, config, GenesisLikeMock(config), force=True)
    manifest = aggregate_run(tmp_path, expected_count=1, release_id="production-like-release")
    assert manifest["status_counts"]["verified"] == 1
    coverage = read_json(tmp_path / "summaries" / "coverage.json")
    assert coverage["status_counts"]["verified"] == 1
    assert "rejected_or_uncertain_tasks_present" not in manifest["non_publishable_reasons"]
    assert manifest["publishable"] is True


def test_command_backend_rejects_status_pass_without_behavior_metrics(tmp_path: Path):
    helper = tmp_path / "status_only.py"
    helper.write_text(
        "import json,sys\n"
        "from pathlib import Path\n"
        "args=sys.argv\n"
        "report=Path(args[args.index('--report')+1])\n"
        "report.write_text(json.dumps({'status':'pass'}))\n",
        encoding="utf-8",
    )
    config = {"backend": "command", "command": ["python3", str(helper), "--report", "{report_json}"]}
    simulator = CommandSimulator(config)
    probe = __import__("softsim_physical_agent.probes", fromlist=["build_probe_pair"]).build_probe_pair(_config())[0]
    result = simulator.run({"sample_id": "command"}, {"density_kg_m3": 1.0}, probe, tmp_path / "probe", simulator_config={"seed": 1})
    report = read_json(tmp_path / "probe" / "report.json")
    assert result.passed is False
    assert report["execution_status"] == SIM_EXECUTED_UNVERIFIED
    assert report["explicit_behavior_evidence"] is False


def test_command_backend_preserves_unexecuted_runner_status(tmp_path: Path):
    helper = tmp_path / "unexecuted.py"
    helper.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "p=Path(sys.argv[sys.argv.index('--report')+1]); p.write_text(json.dumps({'execution_status':'not_executed','behavior_pass':False,'metrics':{},'diagnosis':{'kind':'behavior_unverified'}})); raise SystemExit(2)\n",
        encoding="utf-8",
    )
    config = {"backend": "command", "command": ["python3", str(helper), "--report", "{report_json}"]}
    simulator = CommandSimulator(config)
    probe = __import__("softsim_physical_agent.probes", fromlist=["build_probe_pair"]).build_probe_pair(_config())[0]
    result = simulator.run({"sample_id": "command"}, {"density_kg_m3": 1.0}, probe, tmp_path / "probe", simulator_config={"seed": 1})
    assert result.execution_status == "not_executed"


def test_behavior_metrics_alias_is_accepted_only_with_explicit_diagnosis(tmp_path: Path):
    helper = tmp_path / "behavior_metrics.py"
    helper.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "p=Path(sys.argv[sys.argv.index('--report')+1]); p.write_text(json.dumps({'behavior_pass':True,'behavior_metrics':{'loss':0,'contact_fraction':1,'settled':True},'diagnosis':{'kind':'pass','suggested_updates':{}}}))\n",
        encoding="utf-8",
    )
    config = {"backend": "command", "command": ["python3", str(helper), "--report", "{report_json}"]}
    simulator = CommandSimulator(config)
    probe = __import__("softsim_physical_agent.probes", fromlist=["build_probe_pair"]).build_probe_pair(_config())[0]
    result = simulator.run({"sample_id": "alias"}, {"density_kg_m3": 1.0}, probe, tmp_path / "probe", simulator_config={"seed": 1})
    assert result.passed is True
    assert result.explicit_behavior_evidence is True


def test_command_unverified_task_is_uncertain_and_keeps_all_artifacts(tmp_path: Path):
    helper = tmp_path / "status_only_task.py"
    helper.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "p=Path(sys.argv[sys.argv.index('--report')+1]); p.write_text(json.dumps({'status':'pass'}))\n",
        encoding="utf-8",
    )
    config = _config(backend="command", command=["python3", str(helper), "--report", "{report_json}"], max_iterations=1)
    task = _task("command-task", {"density_kg_m3": 600.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30})
    marker = run_task(task, tmp_path / "run", config, CommandSimulator(config), force=True)
    assert marker["status"] == UNCERTAIN_HUMAN_ESCALATION
    assert read_json(tmp_path / "run" / "tasks" / "command-task" / "candidate_00" / "report.json")["execution_status"] == SIM_EXECUTED_UNVERIFIED


def test_genesis_bridge_does_not_reuse_drop_runner_for_compression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import subprocess
    asset = tmp_path / "asset.json"
    candidate = tmp_path / "candidate.json"
    probe = tmp_path / "probe.json"
    output = tmp_path / "bridge-report.json"
    asset.write_text(json.dumps({"sample_id": "bridge"}), encoding="utf-8")
    candidate.write_text(json.dumps({"density_kg_m3": 600.0}), encoding="utf-8")
    probe.write_text(json.dumps({"probe_id": "heldout", "stimulus": {"kind": "compression"}, "action": {"kind": "compress"}, "config": {"steps": 2}}), encoding="utf-8")
    monkeypatch.delenv("SOFTSIM_GENESIS_BEHAVIOR_RUNNER", raising=False)
    monkeypatch.delenv("GENESIS_MPM_DROP_RUNNER", raising=False)
    command = ["python3", "code/scripts/genesis_probe.py", "--asset-json", str(asset), "--candidate-json", str(candidate), "--probe-json", str(probe), "--output-dir", str(tmp_path / "out"), "--report-json", str(output)]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    assert result.returncode != 0
    report = read_json(output)
    assert report["execution_status"] == "not_executed"
    assert report["diagnosis"]["kind"] == "behavior_unverified"


def test_qwen_missing_key_is_structured_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    proposal = QwenProposer({}).propose()
    assert proposal.status == "proposer_unavailable"
    assert proposal.details["transport_called"] is False


def test_qwen_command_backend_emits_structured_proposal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    helper = tmp_path / "qwen_stub.py"
    helper.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "out=Path(sys.argv[sys.argv.index('--response')+1]); out.write_text(json.dumps({'patch': {'density_kg_m3': 700}, 'rationale': 'stub'}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "local-test-only")
    proposer = __import__("softsim_physical_agent.proposers", fromlist=["QwenProposer"]).QwenProposer({"command": ["python3", str(helper), "--response", "{response_json}"]})
    proposal = proposer.propose(prior={}, candidate={"density_kg_m3": 600.0}, calibration=None)
    assert proposal.status == "proposed"
    assert proposal.patch == {"density_kg_m3": 700.0}


def test_legacy_input_preserves_identity_and_relative_hashes(tmp_path: Path):
    asset = tmp_path / "assets" / "sample_a"
    (asset / "collision").mkdir(parents=True)
    (asset / "visual").mkdir(parents=True)
    (asset / "metadata.json").write_text(json.dumps({"sample_id": "sample_a", "geometry_family": "solid", "object_type": "block"}), encoding="utf-8")
    (asset / "physical_properties.json").write_text(json.dumps({"profile": "foam", "mpm": {"density": 200, "youngs_modulus": 1000, "poisson_ratio": 0.3, "particle_size": 0.01, "dt": 0.001, "substeps": 4, "grid_density": 32}}), encoding="utf-8")
    (asset / "collision" / "collision.obj").write_text("v 0 0 0\n", encoding="utf-8")
    rows = load_legacy_assets(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["sample_id"] == "sample_a"
    assert row["asset_revision_id"].startswith("sha256:")
    ref = row["artifact_refs"]["collision_mesh"]
    assert not Path(ref["path"]).is_absolute()
    assert ref["sha256"].startswith("sha256:")


def test_golden_reference_audit_covers_five_families(tmp_path: Path):
    stage3 = Path(os.environ.get("SOFTSIM_STAGE3_ROOT", ""))
    if not (stage3 / "manifests" / "delivery_manifest.jsonl").is_file():
        pytest.skip("stage3 reference is not mounted")
    audit = build_golden_reference_audit(stage3, tmp_path / "golden.json")
    assert audit["asset_count"] == 150
    assert set(audit["geometry_family_counts"]) == {"solid", "cloth_sheet", "shell", "garment", "bag"}
    assert audit["selected_count"] == 5
    assert all(item["old_behavior_evidence"] == "sim_executed_unverified" for item in audit["selected_assets"])


def test_stage3_manifest_resolves_assets_relative_to_release_root():
    from softsim_physical_agent.input_adapter import load_stage3_manifest

    root = Path(os.environ.get("SOFTSIM_STAGE3_ROOT", ""))
    manifest = root / "manifests" / "delivery_manifest.jsonl"
    if not manifest.is_file():
        return
    row = load_stage3_manifest(manifest)[0]
    assert row["artifact_refs"]["collision_mesh"]["exists"] is True
    assert not Path(row["artifact_refs"]["collision_mesh"]["path"]).is_absolute()


def test_build_tasks_does_not_persist_machine_absolute_metadata_paths():
    row = _fixture_task("portable", {"density_kg_m3": 600.0, "youngs_modulus_pa": 50000.0, "poisson_ratio": 0.30})
    row["metadata"] = {"visual_lod": {"output_files": ["/machine/output/visual.obj"]}}
    task = build_tasks([row], _config())[0]
    assert not "/machine/" in json.dumps(task)
