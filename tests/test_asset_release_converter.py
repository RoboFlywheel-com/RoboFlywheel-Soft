# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from softsim_formal.asset_release import convert_v1_to_v2
from softsim_formal.canonical import fingerprint, sha256_file


class AssetReleaseConverterTest(unittest.TestCase):
    def test_ready_v1_row_converts_to_independent_v2_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "samples" / "s0" / "mesh.glb"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"fixture-mesh")
            source = root / "v1.jsonl"
            row = {
                "schema_name": "softsim.asset_ledger",
                "schema_version": 1,
                "sample_id": "s0",
                "asset_revision_id": "rev0",
                "asset_status": "ready",
                "canonical_id": "solid-s0",
                "geometry_family": "solid",
                "metric_scale": {"scale_to_meter": 0.001, "provenance": "derived_geometry_prior"},
                "artifacts": [{"role": "visual_mesh", "root_alias": "asset_artifacts", "path": "samples/s0/mesh.glb", "sha256": sha256_file(artifact)}],
                "physical_bundle_ref": None,
            }
            row["record_sha256"] = fingerprint(row)
            source.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            result = convert_v1_to_v2(source, output_dir=root / "v2", artifact_root=root, expected_count=1)
            self.assertTrue(result["manifest"]["publishable"])
            converted = json.loads((root / "v2" / "asset_ledger.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(converted["status"], "released")
            self.assertEqual(converted["geometry_ref"]["sha256"], sha256_file(artifact))
            self.assertTrue(converted["provenance"]["physical_bundle_join_required"])

    def test_missing_scale_is_nonpublishable_and_never_invented(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "mesh.glb"
            artifact.write_bytes(b"mesh")
            source = root / "v1.jsonl"
            row = {"sample_id": "s0", "asset_revision_id": "rev0", "asset_status": "ready", "artifacts": [{"role": "visual_mesh", "path": "mesh.glb", "sha256": sha256_file(artifact)}]}
            row["record_sha256"] = fingerprint(row)
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = convert_v1_to_v2(source, output_dir=root / "v2", artifact_root=root, expected_count=1)
            self.assertFalse(result["manifest"]["publishable"])
            self.assertIn("metric_scale_missing", result["report"]["errors"][0]["failure_codes"])
            self.assertIsNone(json.loads((root / "v2" / "asset_ledger.jsonl").read_text(encoding="utf-8").splitlines()[0])["metric_scale"])

    def test_missing_v1_record_hash_is_nonpublishable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "mesh.glb"
            artifact.write_bytes(b"mesh")
            source = root / "v1.jsonl"
            source.write_text(json.dumps({
                "sample_id": "s0",
                "asset_revision_id": "rev0",
                "asset_status": "ready",
                "metric_scale": {"scale_to_meter": 1.0},
                "artifacts": [{"role": "visual_mesh", "path": "mesh.glb", "sha256": sha256_file(artifact)}],
            }) + "\n", encoding="utf-8")
            result = convert_v1_to_v2(source, output_dir=root / "v2", artifact_root=root, expected_count=1)
            self.assertFalse(result["manifest"]["publishable"])
            self.assertIn("v1_record_hash_missing", result["report"]["errors"][0]["failure_codes"])

    def test_utf8_v1_record_hash_with_non_ascii_provenance_is_accepted(self) -> None:
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "mesh.glb"
            artifact.write_bytes(b"mesh")
            row = {
                "sample_id": "s0",
                "asset_revision_id": "rev0",
                "asset_status": "ready",
                "metric_scale": {"scale_to_meter": 1.0, "provenance": "人工审核"},
                "artifacts": [{"role": "visual_mesh", "path": "mesh.glb", "sha256": sha256_file(artifact)}],
            }
            row["record_sha256"] = "sha256:" + hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            source = root / "v1.jsonl"
            source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            result = convert_v1_to_v2(source, output_dir=root / "v2", artifact_root=root, expected_count=1)
            self.assertTrue(result["manifest"]["publishable"])


if __name__ == "__main__":
    unittest.main()
