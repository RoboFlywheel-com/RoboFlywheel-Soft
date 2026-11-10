# SPDX-License-Identifier: Apache-2.0
"""Parallel content-hash verification must be observationally equivalent."""

from __future__ import annotations

import unittest
from pathlib import Path

from softsim_formal.release import ReleaseReader, ReleaseError

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RELEASE = ROOT / "tests" / "fixtures" / "formal_release_v2" / "data"


class ParallelHashValidationTest(unittest.TestCase):
    def test_parallel_matches_serial_on_valid_release(self) -> None:
        serial = ReleaseReader(FIXTURE_RELEASE, allow_synthetic=True, hash_workers=1).read()
        parallel = ReleaseReader(FIXTURE_RELEASE, allow_synthetic=True, hash_workers=8).read()
        self.assertEqual(serial.as_dict()["source_hashes"], parallel.as_dict()["source_hashes"])
        self.assertEqual(len(serial.assets), len(parallel.assets))
        self.assertEqual(len(serial.physical), len(parallel.physical))

    def test_parallel_still_rejects_hash_mismatch(self) -> None:
        import json
        import tempfile

        from softsim_formal.canonical import fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            import shutil

            root = Path(tmp) / "release"
            shutil.copytree(FIXTURE_RELEASE, root)
            geometry = root / "artifacts" / "fixture-00000" / "geometry.proxy"
            geometry.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ReleaseError) as ctx:
                ReleaseReader(root, allow_synthetic=True, hash_workers=8).read()
            self.assertEqual(ctx.exception.code, "artifact_hash_mismatch")

    def test_serial_still_rejects_hash_mismatch(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            import shutil

            root = Path(tmp) / "release"
            shutil.copytree(FIXTURE_RELEASE, root)
            geometry = root / "artifacts" / "fixture-00000" / "geometry.proxy"
            geometry.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ReleaseError) as ctx:
                ReleaseReader(root, allow_synthetic=True, hash_workers=1).read()
            self.assertEqual(ctx.exception.code, "artifact_hash_mismatch")

    def test_content_hash_skip_keeps_structural_checks(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            import shutil

            root = Path(tmp) / "release"
            shutil.copytree(FIXTURE_RELEASE, root)
            # Content tamper is skipped at plan-time reads...
            geometry = root / "artifacts" / "fixture-00000" / "geometry.proxy"
            geometry.write_text("tampered", encoding="utf-8")
            bundle = ReleaseReader(root, allow_synthetic=True, verify_content_hash=False).read()
            self.assertEqual(len(bundle.assets), 12)
            # ...and so is existence (both are covered by the inventory gate
            # and the executor's per-episode check); undeclared aliases and
            # all schema/count/hash-of-ledger checks still apply.
            geometry.unlink()
            bundle = ReleaseReader(root, allow_synthetic=True, verify_content_hash=False).read()
            self.assertEqual(len(bundle.assets), 12)
            ledger = root / "asset_release_v2" / "asset_ledger.jsonl"
            rows = ledger.read_text(encoding="utf-8").splitlines()
            ledger.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
            with self.assertRaises(ReleaseError) as ctx:
                ReleaseReader(root, allow_synthetic=True, verify_content_hash=False).read()
            self.assertIn(ctx.exception.code, {"asset_release_incomplete", "hash_mismatch"})


if __name__ == "__main__":
    unittest.main()
