# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from softsim_asset_pipeline.contracts import validate_physical_bundle, with_record_hash
from softsim_asset_pipeline.io import atomic_write_jsonl, iter_jsonl, safe_join, validate_relative_path


def test_jsonl_streaming_and_atomic_write(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    rows = [{"sample_id": f"sample_{index:03d}", "value": index} for index in range(25)]
    assert atomic_write_jsonl(path, rows) == 25
    assert list(iter_jsonl(path)) == rows
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("value", ["/absolute/file.glb", "../escape.glb", "a/../escape.glb", "a\\b.glb"])
def test_relative_path_rejects_escape(value: str) -> None:
    with pytest.raises(ValueError):
        validate_relative_path(value)


def test_safe_join_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        safe_join(root, "link/value.glb")


def test_verified_status_requires_evidence() -> None:
    record = with_record_hash(
        {
            "schema_name": "softsim.physical_bundle",
            "schema_version": 1,
            "sample_id": "sample",
            "asset_revision_id": "sha256:a",
            "physical_bundle_id": "sha256:b",
            "status": "verified",
            "simulation_domain": "volumetric_soft_body",
            "representation": "watertight_volume",
            "model_family": "elastic",
            "parameters": [],
            "solver_adapter": {},
            "solver_numerics": {},
            "estimator": {},
            "agent": {},
            "verification": {},
            "limitations": [],
            "warnings": [],
            "provenance": {},
        }
    )
    with pytest.raises(ValueError, match="calibration evidence"):
        validate_physical_bundle(record)

