# Data availability and contracts

## Included in Git

- Synthetic contract fixtures under `tests/fixtures/` and fixture generators in both packages.
- Derived manuscript CSVs under `paper/analysis_data/`, their SHA-256 manifest, and generated tables under `paper/results/`.
- Versioned method configs in `configs/` and JSON schemas in `configs/schema/` and `code/contracts/v1/`.

The `v1` schema directory and `schema_version` values describe recorded interfaces; retain them when loading existing artifacts. The baseline `configs/method_contract.json` is accompanied by historical v12 variants. Choose the exact contract referenced by a release rather than silently replacing it.

## Distributed separately

The full 100,000-asset pool, images, meshes, physical bundles, videos, production execution evidence, and evaluator-only parameter sidecars are absent from this Git release. No confirmed public full-data URL is configured yet. Local fixtures and analysis CSVs are usable immediately; replaying full-scale simulation requires the corresponding external payload and hash manifests.

A formal release is expected to expose:

```text
release-root/
  asset_release_v2/release_manifest.json
  asset_release_v2/asset_ledger.jsonl
  physical_release_v12/physical_release_manifest.json
  physical_release_v12/physical_bundle_index.jsonl
  fragile_soft_benchmark/instance_manifest.jsonl
  method_contract.json
```

Artifact references use logical roots, relative paths, identities and content hashes. Bind local storage explicitly with repeated `--root ALIAS=PATH` arguments or the adapter's `SOFTSIM_ROOT_<ALIAS>` environment variables. Do not rename a revision or alter a hashed record in place.

```bash
softsim-exp inventory validate --release-root ./data/release \
  --root primary=./data/meshes --reject-synthetic
```

The physical agent emits immutable candidate revisions, calibration and independent-probe reports, and terminal states. Preserve rejected, uncertain, and infrastructure-failure records in denominators. Predicted priors and synthetic/mock bundles do not satisfy the verified release gate.

## Interpretation

The manuscript reports a 100,000-asset pool and 83,365 assets passing both physical and runtime checks; these populations differ. Categories, assets, configurations and episodes are distinct units. Calibration and independent tests evaluate effective solver behavior. Material parameters are not independently measured ground truth. The fragile-soft model uses a damage/stiffness-loss surrogate, and integrity is derived from deformation. Genesis 1.3.3 LegacyCoupler does not expose the required rigid-MPM contact force stream; unavailable force fields retain explicit missing-evidence status.
