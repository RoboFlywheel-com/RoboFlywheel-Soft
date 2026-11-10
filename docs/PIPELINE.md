# Pipeline walkthrough

Run commands from the repository root after installing both packages. All examples in the first three sections use local synthetic inputs; no API calls or production jobs are launched.

## 1. Asset inventory and procedural cloth

```bash
softsim-assets fixture-assets --output-root ./runs/assets-fixture --count 10
softsim-assets asset-inventory \
  --manifest ./runs/assets-fixture/primary.jsonl \
  --mesh-root mesh_primary=./runs/assets-fixture/meshes \
  --image-root ./runs/assets-fixture/images \
  --output-dir ./runs/assets-inventory
```

`fixture-assets` prints the exact manifest/root/config paths it created; use those values for subsequent commands. Real manifests can use multiple explicit root aliases. Inventory records duplicates, missing geometry and pending cloth explicitly. `asset-audit` validates the referenced GLBs in isolated workers. `cloth-backfill` writes content-addressed geometry into an overlay and resumes by matching content/config identities. `asset-publish` joins inventory, audit and backfill into an immutable ledger. See each command's `--help` for required inputs and the complete workflows in `asset_pipeline/tests/test_asset_workflow.py`.

## 2. Physical authoring

```bash
softsim-physical-agent smoke --output ./runs/physical-smoke
softsim-assets fixture-ledger --output-root ./runs/ledger-fixture --count 20
```

The first command exercises calibration, correction and independent-probe states using mock observations. The ledger fixture prints config paths for the prior workflow: `physics-plan` → `physics-prepare` → `physics-run --transport mock` → `physics-aggregate`. Mock priors retain `synthetic_fixture` provenance and cannot pass a verified publication gate. The complete example is in `asset_pipeline/tests/test_physics_workflow.py`.

For an external asset release, author a task plan with `softsim-physical-agent plan`, execute with `run`, and aggregate with `aggregate`. Use the config whose method/solver versions match the release. API credentials belong in process environment variables; paid prior generation requires `--confirm-paid-run`.

## 3. Experiments and fragile-object qualification

```bash
bash code/scripts/run_formal_smoke.sh ./runs/formal-smoke
bash code/scripts/run_fragile_soft_smoke.sh ./runs/fragile-smoke
softsim-exp status --run ./runs/formal-smoke/e1-10
softsim-exp validate-run --run ./runs/formal-smoke/e1-10
```

The formal smoke creates E1 fixtures at 10/150/200 inputs and E2/E3-A fixtures at 10 inputs. E2 can expand one input into multiple budget/method episodes. A successful smoke includes deliberately failed synthetic tasks: success means complete valid records, including failures, rather than every task being accepted.

## 4. Real simulator execution

Configure the separate Genesis environment as described in [INSTALL.md](INSTALL.md), obtain the external release payload, and bind its logical roots. Use `softsim-exp inventory validate --reject-synthetic` before planning. Production execution must select the real adapter explicitly:

```bash
softsim-exp plan e1-runtime --release-root ./data/release \
  --root primary=./data/meshes --output ./runs/e1-real \
  --config ./configs/e1_formal.json --reject-synthetic --shards 1
softsim-exp run-all --run ./runs/e1-real \
  --episode-executor softsim_genesis_runner:run_episode --process-isolation
softsim-exp aggregate --run ./runs/e1-real --strict
```

The shipped `e1_formal.json` keeps `production_full_run=false`; create and freeze a reviewed run config for your experiment. E2 additionally needs an evaluator-side target provider; hidden parameters must not enter proposer inputs or public task manifests. Keep failed/blocked attempts and hash evidence when resuming.

Reference-image generation, semantic filtering, and non-cloth image-to-mesh inference use external upstream tools. Their cloud launchers and model weights are not included in this code release.
