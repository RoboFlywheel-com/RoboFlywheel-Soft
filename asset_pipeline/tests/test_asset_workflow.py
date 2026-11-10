# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
from pathlib import Path

from softsim_asset_pipeline.asset import audit_assets, inventory_assets, plan_assets, publish_asset_release
from softsim_asset_pipeline.cloth import run_cloth_backfill
from softsim_asset_pipeline.fixtures import create_asset_fixture
from softsim_asset_pipeline.io import iter_jsonl, sha256_file


def test_asset_workflow_and_strict_resume(tmp_path: Path) -> None:
    fixture = create_asset_fixture(tmp_path / "fixture", 10)
    manifest = Path(fixture["manifest"])
    mesh_root = Path(fixture["mesh_root"])
    image_root = Path(fixture["image_root"])
    overlay = Path(fixture["overlay_root"])
    source_hashes = {path: sha256_file(path) for path in mesh_root.rglob("*.glb")}
    plan = plan_assets(
        manifest=manifest,
        output=tmp_path / "plan.json",
        mesh_roots={"mesh_primary": mesh_root},
        overlay_root=overlay,
        expected_count=10,
        shard_size=4,
        estimated_cloth_bytes_per_asset=1024,
    )
    assert plan["dynamic_cloth_count"] == 4
    assert not plan["promotion_gate_failures"]

    inventory_dir = tmp_path / "inventory"
    inventory = inventory_assets(
        manifest=manifest,
        mesh_roots={"mesh_primary": mesh_root},
        image_root=image_root,
        output_dir=inventory_dir,
    )
    assert inventory["by_status"] == {"cloth_pending": 4, "pass": 6}
    audit_dir = tmp_path / "audit"
    audit = audit_assets(
        inventory_path=inventory_dir / "inventory_records.jsonl",
        qa_config_path=Path(fixture["qa_config"]),
        output_dir=audit_dir,
        workers=2,
    )
    assert audit["by_status"] == {"cloth_pending": 4, "pass": 6}

    result_path = tmp_path / "cloth_results.jsonl"
    first = run_cloth_backfill(
        manifest=manifest,
        image_root=image_root,
        overlay_root=overlay,
        config_path=Path(fixture["cloth_config"]),
        result_path=result_path,
        workers=2,
        shard_index=0,
        num_shards=1,
    )
    assert first["generated"] == 4 and first["failed"] == 0
    first_mesh_hashes = {path: sha256_file(path) for path in overlay.rglob("*.glb")}
    second = run_cloth_backfill(
        manifest=manifest,
        image_root=image_root,
        overlay_root=overlay,
        config_path=Path(fixture["cloth_config"]),
        result_path=result_path,
        workers=2,
        shard_index=0,
        num_shards=1,
    )
    assert second["skipped"] == 4 and second["generated"] == 0
    assert first_mesh_hashes == {path: sha256_file(path) for path in overlay.rglob("*.glb")}
    assert source_hashes == {path: sha256_file(path) for path in mesh_root.rglob("*.glb")}

    release = publish_asset_release(
        manifest=manifest,
        inventory_path=inventory_dir / "inventory_records.jsonl",
        audit_path=audit_dir / "asset_audit_records.jsonl",
        backfill_paths=[result_path],
        releases_root=tmp_path / "releases",
        dataset_release_id="fixture-assets-v1",
        expected_count=10,
        strict=True,
    )
    assert release["asset_count"] == 10
    ledger = tmp_path / "releases/fixture-assets-v1/asset_ledger.jsonl"
    rows = list(iter_jsonl(ledger))
    assert len(rows) == 10
    assert {row["asset_status"] for row in rows} == {"ready_with_warnings"}
    assert all(row["physical_bundle_ref"] is None for row in rows)


def test_duplicate_and_orphan_are_explicit(tmp_path: Path) -> None:
    fixture = create_asset_fixture(tmp_path / "fixture", 3)
    primary = Path(fixture["mesh_root"])
    duplicate = tmp_path / "duplicate"
    source = next(primary.rglob("*.glb"))
    relative = source.relative_to(primary)
    (duplicate / relative).parent.mkdir(parents=True)
    shutil.copy2(source, duplicate / relative)
    orphan = primary / "orphan/unclaimed.glb"
    orphan.parent.mkdir(parents=True)
    shutil.copy2(source, orphan)
    output = tmp_path / "inventory"
    summary = inventory_assets(
        manifest=Path(fixture["manifest"]),
        mesh_roots={"primary": primary, "duplicate": duplicate},
        image_root=Path(fixture["image_root"]),
        output_dir=output,
        scan_orphans=True,
    )
    assert summary["by_status"]["duplicate"] == 1
    assert summary["orphan_count"] == 1


def test_indexed_inventory_can_defer_per_file_checks(tmp_path: Path) -> None:
    fixture = create_asset_fixture(tmp_path / "fixture", 6)
    mesh_root = Path(fixture["mesh_root"])
    index_path = tmp_path / "mesh_names.txt"
    index_path.write_text(
        "\n".join(path.relative_to(mesh_root).as_posix() for path in mesh_root.rglob("*.glb")) + "\n",
        encoding="utf-8",
    )
    summary = inventory_assets(
        manifest=Path(fixture["manifest"]),
        mesh_roots={"primary": mesh_root},
        image_root=Path(fixture["image_root"]),
        output_dir=tmp_path / "inventory",
        workers=2,
        mesh_indexes={"primary": set(index_path.read_text(encoding="utf-8").splitlines())},
        verify_glb_headers=False,
        verify_image_paths=False,
    )
    assert summary["by_status"] == {"cloth_pending": 2, "pass": 4}
    records = list(iter_jsonl(tmp_path / "inventory/inventory_records.jsonl"))
    assert all(record["image"]["check"] == "root_only" for record in records)
