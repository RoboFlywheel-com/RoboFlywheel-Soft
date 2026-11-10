# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from fragile_soft_benchmark.adapter import P0Adapter
from fragile_soft_benchmark.builder import build_instances
from fragile_soft_benchmark.physical_join import join_physical_bundles
from fragile_soft_benchmark.mock_backend import run_mock_episode
from fragile_soft_benchmark.qualification import make_default_qualification_instances, run_qualification


FIXTURE = Path(__file__).parent / "fixtures" / "p0_legacy"


def _adapter():
    return P0Adapter.from_files(
        FIXTURE / "manifest.json",
        FIXTURE / "material_profiles.json",
        summary_json_path=FIXTURE / "summary.json",
        summary_csv_path=FIXTURE / "summary.csv",
        source_root=FIXTURE,
    )


def test_builder_requires_explicit_verified_join_for_formal_scope():
    rows = build_instances(
        _adapter(),
        {
            "canonical_groups": [{"canonical_group_id": "soft_food_form", "source_asset_id": "tofu", "split": "calibration"}],
            "geometry_revisions_per_group": 1,
            "difficulty_ids": ["d00"],
        },
    )
    assert rows[0]["physical_bundle_id"] is None
    assert rows[0]["physical_join_status"] == "unjoined"
    assert rows[0]["formal_join_ready"] is False


def test_verified_index_join_replaces_provisional_identity():
    rows = build_instances(
        _adapter(),
        {
            "canonical_groups": [{"canonical_group_id": "soft_food_form", "source_asset_id": "tofu", "split": "calibration"}],
            "geometry_revisions_per_group": 1,
            "difficulty_ids": ["d00"],
        },
        physical_index=[{"sample_id": "sample-0", "asset_revision_id": "sample-0-r1", "physical_bundle_id": "pb-real", "status": "verified"}],
        physical_join_map=[{"benchmark_instance_id": "fsb-soft_food_form-r000-d00", "physical_bundle_id": "pb-real"}],
        physical_release_id="physical-v12",
        require_physical_join=True,
    )
    row = rows[0]
    assert row["physical_join_status"] == "joined_verified"
    assert row["formal_join_ready"] is True
    assert row["physical_bundle_id"] == "pb-real"
    assert row["sample_id"] == "sample-0"
    assert row["asset_revision_id"] == "sample-0-r1"
    assert row.get("instance_hash")


def test_formal_qualification_rejects_unjoined_instances():
    rows = make_default_qualification_instances(_adapter(), {"qualification_revisions_per_group": 1, "qualification_difficulty_ids": ["d00", "d02", "d05"]})

    def formal_runner(instance, **kwargs):
        record = run_mock_episode(instance, **kwargs)
        record["backend_id"] = "fixture.formal.v1"
        return record

    result = run_qualification(rows, backend_id="fixture.formal.v1", formal_scope=True, episode_runner=formal_runner)
    assert result["report"]["status"] == "rejected"
    assert result["report"]["e3_a_eligible"] is False


def test_nonverified_join_is_counted_and_never_formal_ready():
    rows, report = join_physical_bundles(
        [{"benchmark_instance_id": "i", "base_asset_id": "a", "geometry_revision_id": "g"}],
        [{"sample_id": "s", "asset_revision_id": "r", "physical_bundle_id": "pb", "status": "rejected"}],
        join_map=[{"benchmark_instance_id": "i", "physical_bundle_id": "pb"}],
        require_verified=False,
    )
    assert report["counts"]["joined_nonverified"] == 1
    assert rows[0]["formal_join_ready"] is False


def test_predicted_physical_rows_are_rejected_before_join():
    try:
        join_physical_bundles(
            [{"benchmark_instance_id": "i", "sample_id": "s", "asset_revision_id": "r"}],
            [{"sample_id": "s", "asset_revision_id": "r", "physical_bundle_id": "pb", "status": "predicted"}],
        )
    except ValueError as exc:
        assert "predicted" in str(exc)
    else:
        raise AssertionError("predicted row unexpectedly joined")


def test_join_rejects_symlinked_index(tmp_path):
    target = tmp_path / "index.jsonl"
    target.write_text('{"sample_id":"s","asset_revision_id":"r","physical_bundle_id":"p","status":"verified"}\n', encoding="utf-8")
    link = tmp_path / "index-link.jsonl"
    link.symlink_to(target)
    try:
        join_physical_bundles([], link)
    except ValueError as exc:
        assert "symlink" in str(exc)
    else:
        raise AssertionError("symlinked index unexpectedly accepted")
