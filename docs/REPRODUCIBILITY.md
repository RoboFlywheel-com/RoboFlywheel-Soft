# Reproducing the analysis snapshot

The table generator consumes the CSVs checked into `paper/analysis_data/`. This reproduces the archived analysis and tables; rerunning the original simulator experiments additionally requires external production inputs and the pinned runtime.

```bash
python -m pip install --no-build-isolation './code[analysis]'
python paper/build_support/verify_analysis.py
python paper/build_support/build_results.py
```

Outputs are written to `paper/results/`. The generator records input hashes, bootstrap settings and population information in `analysis_manifest.json`. The baseline hashes are also listed in `paper/analysis_data/source_registry.json`. Verification exits with an error if any CSV content changes.

`T02_dataset.tex` reports the full-pool census. The remaining tables cover structure, independent verification, routing, recovery/filter budgets and the grasping application. CSVs can contain multiple episodes or methods for one asset; use the generator's defined paired units and denominators.

For code regression and the small end-to-end checks:

```bash
python -m pytest -q
bash code/scripts/run_formal_smoke.sh ./runs/formal-smoke
bash code/scripts/run_fragile_soft_smoke.sh ./runs/fragile-smoke
```

Tests intentionally retain failure paths, hash mismatches, revision joins and the hidden-target boundary. Fixtures remain synthetic and do not substitute for full-pool evidence. See [VALIDATION.md](VALIDATION.md) for checks performed for this release.

Private host paths in two provenance-only CSV columns were converted to `external-evidence/` references for portability. The registry records both original and distributed file hashes. Numeric values and generated tables are unchanged; production evidence files are not bundled.
