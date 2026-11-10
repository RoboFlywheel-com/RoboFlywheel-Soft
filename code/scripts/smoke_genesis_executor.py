#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Single-asset local smoke for the Genesis episode-executor adapter.

Runs ``softsim_genesis_runner:run_episode`` against one sample physical bundle
(the window-1 v12 sample) through the pinned Genesis 1.3.3 venv
(``GENESIS_PYTHON``).  The bundle is not a frozen release, so every artifact
produced here is explicitly marked ``synthetic: true`` and must not be cited
as a formal number.

Usage::

    GENESIS_PYTHON=/path/to/genesis-env/bin/python \
    PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=0 \
    python3 code/scripts/smoke_genesis_executor.py \
        --sample-dir ./data/sample-bundle \
        --experiment e1 --output ./runs/genesis-smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from softsim_formal.canonical import fingerprint, read_json
from softsim_formal.io import atomic_write_json
from softsim_formal.records import make_run_manifest, make_task_spec, validate_record
from softsim_genesis_runner import run_episode


def _parameters(sample_dir: Path) -> dict[str, float]:
    physics = read_json(sample_dir / "physics.json")
    params = physics.get("solver_adapter", {}).get("parameters") or {}
    return {str(k): float(v) for k, v in params.items() if isinstance(v, (int, float))}


def _physics(sample_dir: Path) -> dict:
    path = sample_dir / "physics.json"
    return read_json(path) if path.is_file() else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--mesh", help="explicit mesh path; default: solver_mesh.obj under the sample dir")
    parser.add_argument("--method-id", default="softsim.genesis_proxy.v12.6")
    parser.add_argument("--experiment", choices=("e1", "e3a"), default="e1")
    parser.add_argument("--force-n", type=float, default=15.0)
    parser.add_argument("--fragility-regime", default="medium")
    parser.add_argument("--method", default=None, help="E3-A baseline method_id (fixed/reactive_slip/deligrasp_prior/privileged_oracle)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sample_dir = Path(args.sample_dir).resolve()
    mesh = Path(args.mesh).resolve() if args.mesh else None
    if mesh is None:
        mesh = sample_dir / "heldout" / "genesis" / "solver_mesh.obj"
    if not mesh.is_file():
        mesh = sample_dir / "candidate_00" / "calibration" / "genesis" / "solver_mesh.obj"
    if not mesh.is_file():
        print(json.dumps({"status": "error", "message": f"no solver_mesh.obj under {sample_dir}"}))
        return 2
    physics = _physics(sample_dir)
    output = Path(args.output).resolve()
    attempt_dir = output / "attempt-000"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = "e1-runtime" if args.experiment == "e1" else "e3a"
    protocol_hash = fingerprint({"protocol": "smoke", "experiment": experiment_id, "synthetic": True})
    manifest = make_run_manifest(
        run_id="run-genesis-executor-smoke",
        experiment_id=experiment_id,
        protocol_hash=protocol_hash,
        expected_task_count=1,
        task_manifest_sha256=fingerprint(["smoke"]),
        config_hash=fingerprint({"smoke": True, "experiment": args.experiment}),
        code_hash=fingerprint({"code": "worktree-ws3"}),
        population=f"{experiment_id}_smoke_synthetic",
        denominator_scope="planned_tasks_including_failures",
    )
    manifest["synthetic"] = True
    route = physics.get("route") if isinstance(physics.get("route"), dict) else {}
    numerics = physics.get("solver_numerics") if isinstance(physics.get("solver_numerics"), dict) else {}
    extra: dict = {"source_row": {
        "solver_mesh_path": mesh.as_posix(),
        "solver_adapter": {"parameters": _parameters(sample_dir)},
        "route": route,
        "solver_numerics": numerics,
    }}
    if args.experiment == "e3a":
        extra.update({"force_n": args.force_n, "fragility_regime": args.fragility_regime, "damage_model_id": "TofuDamage.continuum_damage_stiffness_loss_surrogate"})
        if args.method:
            extra["method_override"] = args.method
    success_path = sample_dir / "SUCCESS.json"
    sample_id = read_json(success_path).get("sample_id", sample_dir.name) if success_path.is_file() else (physics.get("sample_id") or sample_dir.name)
    task = make_task_spec(
        experiment_id=experiment_id,
        protocol_hash=protocol_hash,
        population=f"{experiment_id}_smoke_synthetic",
        sample_id=sample_id,
        asset_revision_id="smoke-revision",
        physical_bundle_id="smoke-bundle",
        method_id=(args.method if args.method else args.method_id),
        canonical_strata={"family": str(route.get("geometry_family") or "solid"), "canonical_id": "smoke", "split": "test"},
        extra=extra,
    )
    result = run_episode(task, manifest, attempt_dir)
    record = result[0] if isinstance(result, (list, tuple)) else result
    validate_record(record)
    record["synthetic"] = True
    record["smoke_evidence_level"] = "synthetic_unfrozen_sample_bundle_not_a_formal_number"
    from softsim_formal.records import record_hash

    record["record_sha256"] = record_hash(record)
    atomic_write_json(output / "smoke_episode_record.json", record)
    summary = {
        "status": "ok",
        "synthetic": True,
        "experiment": args.experiment,
        "execution_status": record["execution_status"],
        "outcome_status": record["outcome_status"],
        "failure_domain": record["failure_domain"],
        "metrics": record.get("metrics"),
        "record": (output / "smoke_episode_record.json").as_posix(),
        "attempt_dir": attempt_dir.as_posix(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True))
    return 0 if record["execution_status"] in {"completed", "failed", "invalid"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
