# SPDX-License-Identifier: Apache-2.0
"""Deterministic plan and shard generation for formal experiments."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .canonical import fingerprint, read_json, sha256_file
from .io import atomic_write_json, write_jsonl
from .records import make_run_manifest, make_task_spec, task_id, validate_task_spec
from .release import BlockedRelease, ReleaseBundle, ReleaseError, ReleaseReader


EXPERIMENTS = {"e1-contract", "e1-runtime", "e1-task", "e2-routing", "e2-recovery", "e2-filter", "e3a"}
EXPERIMENT_ALIASES = {"e1": "e1-runtime", "e2": "e2-recovery", "e3-a": "e3a", "e3_a": "e3a"}
PROMOTION_COUNTS = {"fixture": 10, "showcase": 150, "pilot": 200}
DEFAULT_DISABLED_METHOD_REASONS = {
    "gp_bo_disabled_optional": "optional dependency/checkpoint not installed",
    "external_policy_baseline": "checkpoint/data/source hash unavailable",
}


def package_code_hash() -> str:
    """Hash shipped Python source so code edits invalidate old terminal tasks."""
    entries = []
    package_dir = Path(__file__).resolve().parent
    source_root = package_dir.parent.parent
    candidates = [package_dir]
    sibling_root = source_root / "src"
    if (sibling_root / "softsim_physical_agent").is_dir():
        candidates.append(sibling_root / "softsim_physical_agent")
    if (source_root / "fragile_soft_benchmark").is_dir():
        candidates.append(source_root / "fragile_soft_benchmark")
    for base in candidates:
        for path in sorted(base.glob("*.py")):
            try:
                relative = path.relative_to(source_root.parent).as_posix()
            except ValueError:
                relative = path.name
            entries.append({"path": relative, "sha256": sha256_file(path)})
    for path in sorted((source_root / "contracts").rglob("*.json")) if (source_root / "contracts").is_dir() else ():
        entries.append({"path": path.relative_to(source_root.parent).as_posix(), "sha256": sha256_file(path)})
    for path in (source_root / "pyproject.toml", source_root / "setup.py"):
        if path.is_file():
            entries.append({"path": path.relative_to(source_root.parent).as_posix(), "sha256": sha256_file(path)})
    return fingerprint({"package": "softsim_formal", "entries": entries})


@dataclass(frozen=True)
class PlanResult:
    status: str
    run_id: str
    output_dir: str
    task_count: int
    shard_count: int
    blocked_code: str | None = None
    message: str | None = None
    manifest: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "output_dir": self.output_dir,
            "task_count": self.task_count,
            "shard_count": self.shard_count,
            "blocked_code": self.blocked_code,
            "message": self.message,
            "manifest": self.manifest,
        }


def _config_value(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config))
    return dict(read_json(Path(config)))


def _placeholder_hash(label: str) -> str:
    return fingerprint({"placeholder": label})


def _fixture_assets(count: int) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        sample = f"fixture-{index:05d}"
        family = ("solid", "cloth", "elastoplastic", "shell")[index % 4]
        rows.append({
            "sample_id": sample,
            "asset_revision_id": f"{sample}-r001",
            "dataset_release_id": "fixture-asset-release-v2",
            "physical_bundle_id": f"fixture-pb-{index:05d}",
            "status": "verified" if index % 13 else "task_failure",
            "geometry_family": family,
            "canonical_id": f"canonical-{index % 5:02d}",
            "canonical_strata": {"family": family, "canonical_id": f"canonical-{index % 5:02d}"},
            "method_id": "softsim.genesis_proxy.v12.2",
            "source_hashes": {"asset": _placeholder_hash(sample), "physical": _placeholder_hash(sample + "-physical")},
        })
    return rows


def _rows_for_experiment(experiment_id: str, count: int, release: ReleaseBundle | None, config: Mapping[str, Any], selection_keys: set[tuple[str, str, str]] | None = None) -> list[dict[str, Any]]:
    if release is None:
        rows = _fixture_assets(count)
    elif experiment_id == "e3a":
        rows = []
        physical_by_identity = {
            (str(row.get("sample_id")), str(row.get("asset_revision_id")), str(row.get("physical_bundle_id"))): row
            for row in release.physical
        }
        for item in release.benchmark_manifest:
            physical = physical_by_identity.get((str(item.get("sample_id")), str(item.get("asset_revision_id")), str(item.get("physical_bundle_id"))))
            if item.get("qualification_status") == "qualified" and item.get("formal_e3_a_manifest", False) and physical and physical.get("status") == "verified":
                rows.append(dict(item))
    else:
        # Asset geometry/scale join happens in plan_run (main-window contract:
        # geometry_ref/metric_scale via setdefault on the physical row).
        rows = [row for row in release.physical if experiment_id.startswith("e1") or row.get("status") == "verified"]
    if selection_keys is not None:
        # Frozen selection applies BEFORE the count cut.
        rows = [
            row for row in rows
            if (str(row.get("sample_id")), str(row.get("asset_revision_id")), str(row.get("physical_bundle_id"))) in selection_keys
        ]
    return rows[:count] if count else rows


def _task_population(experiment_id: str) -> str:
    return {
        "e1-contract": "e1_contract_population",
        "e1-runtime": "e1_runtime_population",
        "e1-task": "e1_task_probe_population",
        "e2-routing": "e2_routing_population",
        "e2-recovery": "e2_controlled_recovery_population",
        "e2-filter": "e2_filter_population",
        "e3a": "e3a_fragile_soft_episodes",
    }[experiment_id]


_HIDDEN_KEYS = {
    "hidden_target", "target", "ground_truth", "groundtruth", "oracle_target",
    "oracle_parameters", "true_parameters", "heldout_target", "secret_target",
}


def _is_hidden_key(key: Any) -> bool:
    normalized = str(key).lower()
    if normalized in {"target_handle", "hidden_target_handle", "target_access_policy", "target_provider", "target_provider_required"}:
        return False
    return normalized in _HIDDEN_KEYS or normalized.startswith("target_") or normalized.endswith("_target") or any(
        token in normalized for token in ("hidden_target", "ground_truth", "groundtruth", "oracle_", "oracle-", "true_parameters", "secret_target")
    )


def _redact_hidden(value: Any) -> Any:
    """Remove evaluator-only target payloads from public task specifications."""
    if isinstance(value, Mapping):
        return {str(key): _redact_hidden(item) for key, item in value.items() if not _is_hidden_key(key)}
    if isinstance(value, list):
        return [_redact_hidden(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_hidden(item) for item in value]
    return copy.deepcopy(value)


def _disabled_method_reasons(config: Mapping[str, Any]) -> dict[str, str]:
    """Return methods that must remain out of formal comparison tables."""
    disabled: dict[str, str] = dict(DEFAULT_DISABLED_METHOD_REASONS)
    raw = config.get("disabled_methods")
    if isinstance(raw, Mapping):
        disabled.update({str(key): str(value) for key, value in raw.items()})
    elif isinstance(raw, (list, tuple, set)):
        for item in raw:
            if isinstance(item, Mapping):
                method_id = item.get("method_id") or item.get("id")
                if method_id:
                    disabled[str(method_id)] = str(item.get("reason") or "disabled by frozen config")
            elif item:
                disabled[str(item)] = "disabled by frozen config"
    baselines = config.get("external_baselines")
    if isinstance(baselines, Mapping) and str(baselines.get("status", "")).lower() == "disabled":
        reason = str(baselines.get("reason") or "checkpoint/data/source hash unavailable")
        disabled.setdefault("gp_bo_disabled_optional", reason)
        disabled.setdefault("external_policy_baseline", reason)
    return disabled


def _enabled_methods(config: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
    requested = [str(value) for value in (config.get("methods") or ["contract_probe"])]
    disabled = _disabled_method_reasons(config)
    return [method for method in requested if method not in disabled], {method: disabled[method] for method in requested if method in disabled}


def _make_tasks(experiment_id: str, rows: list[dict[str, Any]], config: Mapping[str, Any], protocol_hash: str) -> list[dict[str, Any]]:
    population = _task_population(experiment_id)
    tasks: list[dict[str, Any]] = []
    default_methods = ["scripted_contact_regulation"] if experiment_id == "e3a" else ["contract_probe"]
    methods, _ = _enabled_methods({**config, "methods": config.get("methods") or default_methods})
    seeds = [int(value) for value in (config.get("seeds") or [0])]
    trials = max(1, int(config.get("trials_per_case", 1)))
    raw_budgets = config.get("candidate_budget", 8)
    if isinstance(raw_budgets, (list, tuple, set)):
        budgets = [int(value) for value in raw_budgets]
    else:
        budgets = [int(raw_budgets)]
    budgets = sorted(set(value for value in budgets if value > 0)) or [8]
    synthetic_fixture = bool(config.get("synthetic_fixture", False))
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or row.get("base_asset_id") or row.get("benchmark_instance_id") or f"fixture-{index:05d}")
        revision = str(row.get("asset_revision_id") or row.get("geometry_revision_id") or f"{sample_id}-r001")
        physical_id = row.get("physical_bundle_id")
        benchmark_id = row.get("benchmark_instance_id")
        family = str(row.get("geometry_family") or row.get("canonical_group_id") or row.get("material_family") or "unknown")
        canonical = str(row.get("canonical_id") or row.get("canonical_group_id") or f"canonical-{index % 5:02d}")
        row_status = str(row.get("status") or row.get("qualification_status") or "verified")
        for method in methods:
            for seed in seeds:
                for trial in range(trials):
                    for candidate_budget in budgets:
                        condition = str(row.get("condition_id") or ("default" if experiment_id != "e3a" else ("low" if index % 3 == 0 else ("medium" if index % 3 == 1 else "high"))))
                        paired = str(row.get("paired_scenario_key") or f"{benchmark_id or sample_id}::state-{trial}")
                        extra: dict[str, Any] = {
                            "source_status": row_status,
                            "source_row": _redact_hidden(row) if experiment_id.startswith("e2") else copy.deepcopy(row),
                            "promotion_stage": config.get("promotion_stage", "fixture"),
                            "config_hash": fingerprint(config),
                            "synthetic_fixture": synthetic_fixture,
                        }
                        if experiment_id.startswith("e2"):
                            prior_parameters = row.get("prior_parameters")
                            if prior_parameters is None and synthetic_fixture:
                                # Fixtures stay self-contained; production rows
                                # resolve frozen bundle parameters at run time.
                                prior_parameters = {"youngs_modulus_pa": 50000.0, "density_kg_m3": 600.0, "poisson_ratio": 0.3}
                            target_handle = fingerprint({
                                "sample_id": sample_id,
                                "asset_revision_id": revision,
                                "calibration_probe_id": "calibration.synthetic.compression.v1",
                                "heldout_probe_id": "heldout.synthetic.push.v1",
                            })
                            extra.update({
                                "hidden_target_handle": target_handle,
                                "target_access_policy": "inline_synthetic_fixture" if synthetic_fixture else "evaluator_sidecar_required",
                                "calibration_probe_id": "calibration.synthetic.compression.v1",
                                "heldout_probe_id": "heldout.synthetic.push.v1",
                                "candidate_budget": candidate_budget,
                            })
                            if prior_parameters is not None:
                                extra["prior_parameters"] = _redact_hidden(prior_parameters)
                            # Production rows carry no inline prior: the executor
                            # resolves the frozen bundle parameters (physics.json)
                            # at run time so records never inherit placeholder values.
                            if synthetic_fixture:
                                extra["hidden_target"] = row.get("hidden_target") or {"youngs_modulus_pa": 60000.0, "density_kg_m3": 650.0, "poisson_ratio": 0.31}
                        if experiment_id == "e1-task":
                            # Deterministic probe assignment, balanced by hash;
                            # cloth-like families swap compress for drape.
                            cloth_like = family.lower() in {"cloth", "sheet", "bag", "garment", "textile", "membrane", "shell"}
                            probe_set = ["drop", "drape", "push", "place", "pinch_lift"] if cloth_like else ["drop", "compress", "push", "place", "pinch_lift"]
                            probe_index = int(fingerprint({"task_probe_assign": sample_id, "revision": revision})[7:15], 16) % len(probe_set)
                            extra["task_probe"] = probe_set[probe_index]
                        if experiment_id == "e3a":
                            extra.update({
                                "fragility_regime": row.get("fragility_regime", "medium"),
                                "qualification_status": row.get("qualification_status", "candidate"),
                                "damage_model_id": row.get("damage_model_id", "TofuDamage.continuum_damage_stiffness_loss_surrogate"),
                                "force_n": float(row.get("force_n", {"low": 6.0, "medium": 15.0, "high": 30.0}.get(condition, 15.0))),
                            })
                        task = make_task_spec(
                            experiment_id=experiment_id,
                            protocol_hash=protocol_hash,
                            population=population,
                            sample_id=sample_id,
                            asset_revision_id=revision,
                            physical_bundle_id=str(physical_id) if physical_id is not None else None,
                            benchmark_instance_id=str(benchmark_id) if benchmark_id is not None else None,
                            method_id=str(method),
                            condition_id=condition,
                            paired_scenario_key=paired,
                            seed=seed,
                            trial_index=trial,
                            canonical_strata={"family": family, "canonical_id": canonical, "split": row.get("split", "test")},
                            data_role=str(row.get("data_role") or "test"),
                            extra=extra,
                        )
                        validate_task_spec(task)
                        tasks.append(task)
    tasks.sort(key=lambda item: str(item["task_id"]))
    return tasks


def _write_selection_manifest(output: Path, experiment_id: str, selected: list[dict[str, Any]], source_rows: list[dict[str, Any]], *, source_release_ids: Mapping[str, Any], rule: str) -> str:
    """Persist the exact population/exclusion decision before task execution."""
    selected_keys = {
        (str(row.get("sample_id")), str(row.get("asset_revision_id") or row.get("geometry_revision_id")), str(row.get("physical_bundle_id", "")))
        for row in selected
    }
    entries: list[dict[str, Any]] = []
    for row in source_rows:
        key = (str(row.get("sample_id")), str(row.get("asset_revision_id") or row.get("geometry_revision_id")), str(row.get("physical_bundle_id", "")))
        included = key in selected_keys
        entries.append({
            "sample_id": row.get("sample_id"),
            "asset_revision_id": row.get("asset_revision_id") or row.get("geometry_revision_id"),
            "physical_bundle_id": row.get("physical_bundle_id"),
            "benchmark_instance_id": row.get("benchmark_instance_id"),
            "source_status": row.get("status") or row.get("qualification_status"),
            "included": included,
            "exclusion_reason": None if included else ("non_verified_physical_bundle" if row.get("status") != "verified" else "not_qualified_or_not_selected"),
            "selection_rule": rule,
        })
    entries.sort(key=lambda item: (str(item.get("sample_id")), str(item.get("benchmark_instance_id")), str(item.get("physical_bundle_id"))))
    path = output / "selection_manifest.jsonl"
    selection_hash = write_jsonl(path, entries)
    metadata = {
        "schema_name": "softsim.selection_manifest",
        "schema_version": 1,
        "experiment_id": experiment_id,
        "population": _task_population(experiment_id),
        "source_release_ids": dict(source_release_ids),
        "selection_rule": rule,
        "source_population_count": len(source_rows),
        "selected_population_count": sum(bool(item["included"]) for item in entries),
        "excluded_population_count": sum(not bool(item["included"]) for item in entries),
        "selection_manifest_sha256": selection_hash,
        "entries_ref": "selection_manifest.jsonl",
    }
    atomic_write_json(output / "selection_manifest.json", metadata)
    return selection_hash


def _write_blocked(output: Path, experiment_id: str, config: Mapping[str, Any], code: str, message: str) -> PlanResult:
    output.mkdir(parents=True, exist_ok=True)
    protocol_hash = fingerprint({"experiment_id": experiment_id, "config": config})
    run_id = "run-" + fingerprint({"experiment_id": experiment_id, "config": config, "blocked": code})[7:19]
    empty_hash = fingerprint([])
    manifest = {
        "schema_name": "softsim.run_manifest",
        "schema_version": 1,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "protocol_hash": protocol_hash,
        "expected_task_count": 0,
        "task_manifest_sha256": empty_hash,
        "config_hash": fingerprint(config),
        "code_hash": package_code_hash(),
        "population": _task_population(experiment_id),
        "data_role": "test",
        "mode": "blocked",
        "scheduler": "none",
        "method_id": "none",
        "denominator_scope": "blocked_no_tasks",
        "source_hashes": {},
        "logical_roots": {},
        "status": "blocked",
        "blocked_code": code,
        "blocked_message": message,
    }
    atomic_write_json(output / "run_manifest.json", manifest)
    empty_task_hash = write_jsonl(output / "task_manifest.jsonl", [])
    manifest["task_manifest_sha256"] = empty_task_hash
    atomic_write_json(output / "run_manifest.json", manifest)
    atomic_write_json(output / "run_status.json", {"status": "blocked", "blocked_code": code, "message": message, "run_id": run_id})
    (output / "shards").mkdir(exist_ok=True)
    return PlanResult("blocked", run_id, output.as_posix(), 0, 0, code, message, manifest)


def plan_run(
    experiment_id: str,
    *,
    output_dir: str | Path,
    config: Mapping[str, Any] | str | Path | None = None,
    release_root: str | Path | None = None,
    fixture: bool = False,
    count: int | None = None,
    shard_count: int = 1,
    mode: str = "local",
    scheduler: str | None = None,
    allow_synthetic: bool = True,
    root_bindings: Mapping[str, str | Path] | None = None,
    benchmark_join_map: str | Path | None = None,
    hash_workers: int = 1,
    verify_artifact_content: bool = True,
    selection_file: str | Path | None = None,
) -> PlanResult:
    experiment_id = EXPERIMENT_ALIASES.get(experiment_id, experiment_id)
    if experiment_id not in EXPERIMENTS:
        raise ValueError(f"unknown experiment: {experiment_id}")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    cfg = _config_value(config)
    # Inline hidden targets are permitted only for explicitly synthetic
    # fixtures.  Production plans always require an evaluator-side provider.
    cfg["synthetic_fixture"] = bool(fixture)
    disabled_method_reasons = _disabled_method_reasons(cfg)
    cfg["disabled_method_reasons"] = disabled_method_reasons
    output = Path(output_dir).resolve()
    release: ReleaseBundle | None = None
    if not fixture:
        try:
            release = ReleaseReader(
                release_root,
                allow_synthetic=allow_synthetic,
                require_benchmark=(experiment_id == "e3a"),
                root_bindings=root_bindings,
                benchmark_join_map=benchmark_join_map,
                hash_workers=hash_workers,
                verify_content_hash=verify_artifact_content,
            ).read()
        except BlockedRelease as exc:
            return _write_blocked(output, experiment_id, cfg, exc.code, str(exc))
        except ReleaseError as exc:
            return _write_blocked(output, experiment_id, cfg, exc.code, str(exc))
    if count is None:
        count = int(cfg.get("count", PROMOTION_COUNTS.get(str(cfg.get("promotion_stage", "fixture")), 10) if fixture else 0))
    if fixture and count <= 0:
        count = 10
    selection_keys: set[tuple[str, str, str]] | None = None
    if selection_file is not None:
        # Explicit frozen selection (e.g. the E2.2 review-frozen problem
        # matrix): restrict rows to the listed identities, preserving the
        # selection manifest's audit trail.
        selection_keys = set()
        with open(selection_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                selection_keys.add((str(entry.get("sample_id")), str(entry.get("asset_revision_id")), str(entry.get("physical_bundle_id"))))
        cfg["selection_file"] = str(selection_file)
    rows = _rows_for_experiment(experiment_id, int(count), release, cfg, selection_keys=selection_keys)
    if not rows:
        return _write_blocked(output, experiment_id, cfg, "blocked_empty_input", "no qualified input rows")
    if release is not None:
        if experiment_id == "e3a":
            source_rows = [dict(item) for item in release.benchmark_manifest]
            rule = "qualification_status=qualified AND formal_e3_a_manifest=true AND physical.status=verified"
        else:
            source_rows = [dict(item) for item in release.physical]
            rule = "all physical terminal rows for E1; status=verified for E2"
            # Physical index rows carry bundle/evidence refs but not the asset
            # geometry/scale; the episode executor needs both.  Join the asset
            # ledger row on (sample_id, asset_revision_id) into the task rows
            # without overwriting physical-side fields.  source_rows (used by
            # the selection manifest) stay pristine physical rows.
            asset_by_key = {
                (str(a.get("sample_id")), str(a.get("asset_revision_id"))): a
                for a in release.assets
            }
            for row in rows:
                if row.get("benchmark_instance_id"):
                    continue
                asset_row = asset_by_key.get((str(row.get("sample_id")), str(row.get("asset_revision_id"))))
                if asset_row:
                    row.setdefault("geometry_ref", asset_row.get("geometry_ref"))
                    row.setdefault("metric_scale", asset_row.get("metric_scale"))
        source_release_ids = {"asset_release_id": release.asset_release_id, "physical_release_id": release.physical_release_id}
    else:
        source_rows = _fixture_assets(int(count))
        source_release_ids = {"asset_release_id": "fixture-asset-release-v2", "physical_release_id": "fixture-physical-release-v12"}
        rule = "synthetic fixture selection"
    selection_hash = _write_selection_manifest(output, experiment_id, rows, source_rows, source_release_ids=source_release_ids, rule=rule)
    protocol_payload = {"experiment_id": experiment_id, "config": cfg, "method_contract": release.method_contract if release else "fixture"}
    protocol_hash = fingerprint(protocol_payload)
    tasks = _make_tasks(experiment_id, rows, cfg, protocol_hash)
    if not tasks:
        return _write_blocked(
            output,
            experiment_id,
            cfg,
            "blocked_no_enabled_methods",
            "all requested methods are disabled by the frozen baseline provenance gate",
        )
    task_file = output / "task_manifest.jsonl"
    task_hash = write_jsonl(task_file, tasks)
    config_hash = fingerprint(cfg)
    code_hash = package_code_hash()
    run_spec = {"experiment_id": experiment_id, "protocol_hash": protocol_hash, "task_hash": task_hash, "config_hash": config_hash, "code_hash": code_hash, "mode": mode, "scheduler": scheduler or mode}
    run_id = "run-" + fingerprint(run_spec)[7:19]
    manifest = make_run_manifest(
        run_id=run_id,
        experiment_id=experiment_id,
        protocol_hash=protocol_hash,
        expected_task_count=len(tasks),
        task_manifest_sha256=task_hash,
        config_hash=config_hash,
        code_hash=code_hash,
        population=_task_population(experiment_id),
        mode=mode,
        scheduler=scheduler or mode,
        source_hashes=release.source_hashes if release else {"fixture": fingerprint({"fixture": True})},
        method_id=str(cfg.get("method_id") or (release.method_contract.get("method_id") if release else "softsim.mock.v1")),
        denominator_scope="planned_tasks_including_failures",
        logical_roots=release.roots if release else {"fixture": "tests/fixtures/formal"},
    )
    manifest["shard_count"] = int(shard_count)
    manifest["synthetic"] = bool(fixture or (release.synthetic if release else False))
    manifest["production_full_run"] = False
    manifest["real_executor_required"] = not fixture
    manifest["artifact_content_hash_check"] = "full" if verify_artifact_content else "deferred_to_inventory_validate; executor verifies per-episode mesh sha256"
    manifest["simulator_fingerprint"] = fingerprint({"backend": cfg.get("backend", "mock"), "simulator_config": cfg.get("simulator", {}), "method": manifest.get("method_id")})
    if experiment_id.startswith("e2"):
        manifest["target_access_policy"] = "inline_synthetic_fixture" if fixture else "evaluator_sidecar_required"
        manifest["hidden_target_public"] = bool(fixture)
        manifest["target_provider_required"] = not fixture
    manifest["source_release_ids"] = {
        "asset_release_id": release.asset_release_id if release else "fixture-asset-release-v2",
        "physical_release_id": release.physical_release_id if release else "fixture-physical-release-v12",
    }
    manifest["selection_manifest_sha256"] = selection_hash
    manifest["selection_manifest_ref"] = "selection_manifest.json"
    manifest["disabled_methods"] = dict(disabled_method_reasons)
    manifest["comparison_methods"] = sorted({str(task.get("method_id")) for task in tasks})
    if release:
        manifest["verified_physical_count"] = len(release.verified_physical)
        manifest["physical_execution_population"] = len(release.physical)
        if release.benchmark_join_report is not None:
            join_report_path = output / "benchmark_join_report.json"
            atomic_write_json(join_report_path, release.benchmark_join_report)
            manifest["benchmark_join_report_ref"] = join_report_path.name
            manifest["benchmark_join_report_sha256"] = sha256_file(join_report_path)
    atomic_write_json(output / "run_manifest.json", manifest)
    shards_dir = output / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_rows: dict[int, list[dict[str, Any]]] = {index: [] for index in range(shard_count)}
    for task in tasks:
        index = int(fingerprint(task["task_id"])[7:15], 16) % shard_count
        task_with_shard = dict(task)
        task_with_shard["shard_index"] = index
        shard_rows[index].append(task_with_shard)
    for index in shard_rows:
        shard_rows[index].sort(key=lambda row: str(row["task_id"]))
        write_jsonl(shards_dir / f"shard-{index:04d}.jsonl", shard_rows[index])
    atomic_write_json(output / "run_status.json", {"status": "planned", "run_id": run_id, "expected_task_count": len(tasks), "completed_task_count": 0, "shard_count": shard_count, "mode": mode, "scheduler": scheduler or mode})
    return PlanResult("planned", run_id, output.as_posix(), len(tasks), shard_count, manifest=manifest)


def read_plan(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(path)
    manifest = read_json(root / "run_manifest.json")
    tasks = []
    task_path = root / "task_manifest.jsonl"
    if task_path.is_file():
        from .io import read_jsonl
        tasks = list(read_jsonl(task_path))
    return manifest, tasks
