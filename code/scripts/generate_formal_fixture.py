# SPDX-License-Identifier: Apache-2.0
"""Generate the tiny hashed release used by local formal-runner tests.

The fixture is synthetic and explicitly non-production.  Keeping generation in
source makes it possible to rebuild it byte-for-byte after a clean checkout.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from softsim_formal.canonical import canonical_dumps, fingerprint, sha256_file


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_dumps(row) + "\n" for row in rows), encoding="utf-8")


def row_hash(row: dict[str, Any]) -> str:
    return fingerprint({key: value for key, value in row.items() if key != "record_sha256"})


def build(root: Path, count: int = 12) -> Path:
    if root.exists():
        shutil.rmtree(root)
    asset_dir = root / "asset_release_v2"
    physical_dir = root / "physical_release_v12"
    benchmark_dir = root / "fragile_soft_benchmark"
    artifacts = root / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    method = {
        "schema_name": "softsim.method_contract",
        "schema_version": 1,
        "method_id": "softsim.genesis_proxy.v12.1",
        "genesis_version": "1.3.3",
        "solver": {"cloth": "PBD.Cloth", "volume": "MPM.Elastic/MPM.ElastoPlastic", "coupler": "LegacyCoupler"},
        "ipc_used": False,
        "uipc_used": False,
        "proxy": True,
        "simulation_calibrated_definition": "behavior contract pass only; not real-material ground truth",
        "synthetic": True,
    }
    write_json(root / "method_contract.json", method)
    method_hash = sha256_file(root / "method_contract.json")
    assets: list[dict[str, Any]] = []
    physical: list[dict[str, Any]] = []
    benchmark: list[dict[str, Any]] = []
    for index in range(count):
        sample = f"fixture-{index:05d}"
        revision = f"{sample}-r002"
        geometry = artifacts / sample / "geometry.proxy"
        geometry.parent.mkdir(parents=True, exist_ok=True)
        geometry.write_bytes((f"synthetic geometry {sample}\n").encode("ascii"))
        geometry_ref = {"root_alias": "fixture", "relative_path": geometry.relative_to(root).as_posix(), "sha256": sha256_file(geometry)}
        asset = {"sample_id": sample, "asset_revision_id": revision, "dataset_release_id": "softsim-100k-assets-v2-fixture", "status": "released", "geometry_ref": geometry_ref, "metric_scale": {"meters_per_unit": 0.12, "unit": "m", "source": "synthetic_fixture_proxy"}, "geometry_family": ("solid", "cloth", "elastoplastic")[index % 3], "canonical_id": f"canonical-{index % 4:02d}"}
        asset["record_sha256"] = row_hash(asset)
        assets.append(asset)
        status = "task_failure" if index == count - 1 else ("rejected" if index == count - 2 else "verified")
        row: dict[str, Any] = {"sample_id": sample, "asset_revision_id": revision, "dataset_release_id": "softsim-100k-agent-genesis-v12-fixture", "physical_bundle_id": f"pb-fixture-{index:05d}", "status": status, "method_contract_hash": method_hash, "solver": "Genesis 1.3.3", "solver_numerics": {"dt_s": 0.001, "substeps": 8, "particle_size_m": 0.01, "grid_density": 64, "scale_m": 0.12, "seed": index, "proxy_mode": "genesis_proxy"}}
        # Every terminal physical row, including rejected/task_failure, keeps
        # an agent trace.  Verified rows additionally require both evidence
        # reports; rejected rows carry a failure-only trace.
        trace_payload = "calibration_only\nheldout_verification\n" if status == "verified" else "terminal_failure\n"
        trace_items = [("agent_trace_ref", "agent_trace.jsonl", trace_payload)]
        if status == "verified":
            trace_items.extend((("calibration_evidence_ref", "calibration.json", "{\"behavior_pass\":true}\n"), ("heldout_evidence_ref", "heldout.json", "{\"behavior_pass\":true}\n")))
        for kind, filename, payload in trace_items:
                artifact = artifacts / sample / filename
                artifact.write_text(payload, encoding="utf-8")
                row[kind] = {"root_alias": "fixture", "relative_path": artifact.relative_to(root).as_posix(), "sha256": sha256_file(artifact)}
        if status == "verified":
            row["verification_state"] = "verified_behavior_contract"
        else:
            row["failure_code"] = "fixture_rejected" if status == "rejected" else "fixture_task_failure"
            row["retriable"] = False
        row["record_sha256"] = row_hash(row)
        physical.append(row)
        if index < min(6, count):
            instance = {"benchmark_instance_id": f"fsb-fixture-r{index:03d}-d{index % 3:02d}", "sample_id": sample, "asset_revision_id": revision, "physical_bundle_id": row["physical_bundle_id"], "provisional_physical_bundle_id": row["physical_bundle_id"], "physical_bundle_status": status, "physical_join_status": "joined_verified" if status == "verified" else "joined_nonverified", "formal_join_ready": status == "verified", "physical_release_id": "softsim-100k-agent-genesis-v12-fixture", "geometry_revision_id": revision, "canonical_group_id": f"fixture-family-{index % 3}", "split": "hidden_test" if index % 2 else "calibration", "qualification_status": "qualified", "formal_e3_a_manifest": True, "damage_model_id": "TofuDamage.continuum_damage_stiffness_loss_surrogate", "fragility_regime": ("low", "medium", "high")[index % 3]}
            instance["instance_hash"] = fingerprint(instance)
            benchmark.append(instance)
    ledger_path = asset_dir / "asset_ledger.jsonl"
    index_path = physical_dir / "physical_bundle_index.jsonl"
    benchmark_path = benchmark_dir / "instance_manifest.jsonl"
    write_jsonl(ledger_path, assets)
    write_jsonl(index_path, physical)
    write_jsonl(benchmark_path, benchmark)
    write_json(asset_dir / "release_manifest.json", {"schema_name": "softsim.asset_release_manifest", "schema_version": 2, "release_id": "softsim-100k-assets-v2-fixture", "publishable": True, "synthetic": True, "population_count": count, "ledger_sha256": sha256_file(ledger_path), "logical_roots": {"fixture": "."}})
    write_json(physical_dir / "physical_release_manifest.json", {"schema_name": "softsim.physical_release_manifest", "schema_version": 12, "release_id": "softsim-100k-agent-genesis-v12-fixture", "publishable": True, "synthetic": True, "population_count": count, "verified_count": max(0, count - 2), "rejected_count": 1 if count >= 2 else 0, "task_failure_count": 1 if count >= 1 else 0, "index_sha256": sha256_file(index_path), "benchmark_manifest_sha256": sha256_file(benchmark_path), "method_contract_sha256": method_hash, "logical_roots": {"fixture": "."}})
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    print(build(args.output, args.count).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
