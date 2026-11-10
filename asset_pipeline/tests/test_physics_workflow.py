# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from softsim_asset_pipeline.fixtures import create_ledger_fixture
from softsim_asset_pipeline.io import iter_jsonl, read_json
from softsim_asset_pipeline.physics import (
    aggregate_physics_release,
    plan_physics,
    prepare_physics,
    run_physics_shard,
)


def test_mock_physics_pipeline_partial_retry_and_resume(tmp_path: Path) -> None:
    fixture = create_ledger_fixture(tmp_path / "fixture", 20)
    run_root = tmp_path / "fixture/run"
    plan = plan_physics(
        asset_ledger=Path(fixture["asset_ledger"]),
        output_root=run_root,
        config_path=Path(fixture["config"]),
        legacy_src=None,
        num_shards=1,
        expected_ready_count=20,
    )
    assert plan["geometry_ready_count"] == 20
    prepared = prepare_physics(
        run_root=run_root,
        shard_index=0,
        roots={},
        workers=4,
        allow_metadata_only=True,
    )
    assert prepared["failed"] == 0
    first = run_physics_shard(
        run_root=run_root,
        shard_index=0,
        config_path=Path(fixture["config"]),
        fallback_policy_path=Path(fixture["fallback_policy"]),
        transport_name="mock",
        workers=4,
        confirm_paid_run=False,
    )
    assert first["newly_completed"] == 20
    assert first["api_requests"] > plan["base_request_count"]
    second = run_physics_shard(
        run_root=run_root,
        shard_index=0,
        config_path=Path(fixture["config"]),
        fallback_policy_path=Path(fixture["fallback_policy"]),
        transport_name="mock",
        workers=4,
        confirm_paid_run=False,
    )
    assert second["resumed"] == 20
    assert second["api_requests"] == 0
    release = aggregate_physics_release(
        run_root=run_root,
        asset_release_dir=Path(fixture["asset_release"]),
        physical_release_id="fixture-physics-v1",
        strict=True,
        allow_nonpublishable=True,
    )
    assert release["bundle_count"] == 20
    assert release["publishable"] is False
    index = Path(fixture["asset_release"]) / "physical/fixture-physics-v1/physical_bundle_index.jsonl"
    assert len(list(iter_jsonl(index))) == 20
    bundle = read_json(
        Path(fixture["asset_release"])
        / "physical/fixture-physics-v1/bundles/fixture_physics_000000/physics.json"
    )
    assert bundle["status"] == "predicted"
    assert bundle["provenance"]["simulation_calibrated"] is False
    assert bundle["verification"]["heldout_probe_suite_hash"] is None
    assert {item["source_type"] for item in bundle["parameters"]} == {"synthetic_fixture"}


def test_mock_release_requires_explicit_nonpublishable_override(tmp_path: Path) -> None:
    fixture = create_ledger_fixture(tmp_path / "fixture", 1)
    run_root = tmp_path / "fixture/run"
    plan_physics(
        asset_ledger=Path(fixture["asset_ledger"]),
        output_root=run_root,
        config_path=Path(fixture["config"]),
        legacy_src=None,
        num_shards=1,
        expected_ready_count=1,
    )
    prepare_physics(run_root=run_root, shard_index=0, roots={}, workers=1, allow_metadata_only=True)
    run_physics_shard(
        run_root=run_root,
        shard_index=0,
        config_path=Path(fixture["config"]),
        fallback_policy_path=Path(fixture["fallback_policy"]),
        transport_name="mock",
        workers=1,
        confirm_paid_run=False,
    )
    with pytest.raises(ValueError, match="cannot_be_published"):
        aggregate_physics_release(
            run_root=run_root,
            asset_release_dir=Path(fixture["asset_release"]),
            physical_release_id="must-fail",
            strict=True,
            allow_nonpublishable=False,
        )


def test_paid_transport_requires_confirmation_before_api_key(tmp_path: Path) -> None:
    fixture = create_ledger_fixture(tmp_path / "fixture", 1)
    run_root = tmp_path / "fixture/run"
    plan_physics(
        asset_ledger=Path(fixture["asset_ledger"]),
        output_root=run_root,
        config_path=Path(fixture["config"]),
        legacy_src=None,
        num_shards=1,
        expected_ready_count=1,
    )
    with pytest.raises(ValueError, match="confirm-paid-run"):
        run_physics_shard(
            run_root=run_root,
            shard_index=0,
            config_path=Path(fixture["config"]),
            fallback_policy_path=Path(fixture["fallback_policy"]),
            transport_name="qwen",
            workers=1,
            confirm_paid_run=False,
        )


def test_static_shards_are_disjoint_and_complete(tmp_path: Path) -> None:
    fixture = create_ledger_fixture(tmp_path / "fixture", 101)
    run_root = tmp_path / "fixture/run"
    plan_physics(
        asset_ledger=Path(fixture["asset_ledger"]),
        output_root=run_root,
        config_path=Path(fixture["config"]),
        legacy_src=None,
        num_shards=7,
        expected_ready_count=101,
    )
    ids: list[str] = []
    for path in sorted((run_root / "plans").glob("*.jsonl")):
        ids.extend(str(row["sample_id"]) for row in iter_jsonl(path))
    assert len(ids) == 101
    assert len(set(ids)) == 101
