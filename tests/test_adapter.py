# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from fragile_soft_benchmark.adapter import P0Adapter


FIXTURE = Path(__file__).parent / "fixtures" / "p0_legacy"


def test_adapter_merges_json_csv_and_evidence_without_absolute_paths():
    adapter = P0Adapter.from_files(
        FIXTURE / "manifest.json",
        FIXTURE / "material_profiles.json",
        summary_json_path=FIXTURE / "summary.json",
        summary_csv_path=FIXTURE / "summary.csv",
        video_manifest_path=FIXTURE / "videos.json",
        state_manifest_path=FIXTURE / "states.json",
        source_root=FIXTURE,
    )
    assert len(adapter.assets) == 4
    tofu = adapter.get_asset("tofu")
    assert tofu["material_profile"]["model"] == "damage"
    assert len(tofu["cases"]) == 3
    assert tofu["asset_kind"] == "p0_reference"
    assert tofu["unique_geometry_asset"] is False
    assert all(not str(ref["path"]).startswith("/") for ref in tofu["source_refs"])
    assert adapter.source_sha256.startswith("sha256:")
