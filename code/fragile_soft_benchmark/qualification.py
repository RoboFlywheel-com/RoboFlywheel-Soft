# SPDX-License-Identifier: Apache-2.0
"""36-instance qualification runner and explicit failure disposition."""

from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .builder import build_instances
from .evaluator import evaluate_episode
from .mock_backend import run_mock_episode
from .hashing import sha256_json
from .runner import EpisodeRunnerContractError, validate_episode_record


QUALIFICATION_SCHEMA_VERSION = "fragile-soft.qualification.v1"
DEFAULT_QUALIFICATION_GROUPS = (
    {"canonical_group_id": "soft_food_form", "source_asset_id": "tofu", "split": "calibration"},
    {"canonical_group_id": "compressible_fruit", "source_asset_id": "banana", "split": "calibration"},
    {"canonical_group_id": "mashed_food", "source_asset_id": "dough", "split": "hidden_test"},
    {"canonical_group_id": "sponge_foam", "source_asset_id": "slow_rise_foam_toy", "split": "hidden_test"},
)
DEFAULT_QUALIFICATION_DIFFICULTIES = ("d00", "d02", "d05")
DEFAULT_THRESHOLDS = {
    "no_contact_damage_max": 1e-9,
    "monotonic_tolerance": 0.03,
    "repeat_variance_max": 0.01,
    "safe_window_min_width_n": 0.0,
    "max_penetration_m": 0.004,
}


def _variance(values: Sequence[float]) -> float:
    return statistics.pvariance(values) if len(values) > 1 else 0.0


def _monotonic(values: Sequence[float], tolerance: float) -> bool:
    return all(float(right) + tolerance >= float(left) for left, right in zip(values, values[1:]))


def _split_gate(instances: Sequence[Mapping[str, Any]]) -> tuple[bool, dict[str, Any]]:
    groups: dict[str, set[str]] = {}
    for instance in instances:
        split = str(instance.get("split", "unknown"))
        group = str(instance.get("canonical_group_id"))
        groups.setdefault(split, set()).add(group)
    calibration = groups.get("calibration", set())
    hidden = groups.get("hidden_test", set()) | groups.get("hidden", set())
    overlap = sorted(calibration & hidden)
    calibration_hashes = {
        (instance.get("source") or {}).get("p0_asset_source_sha256")
        for instance in instances if instance.get("split") == "calibration"
    }
    hidden_hashes = {
        (instance.get("source") or {}).get("p0_asset_source_sha256")
        for instance in instances if instance.get("split") in {"hidden_test", "hidden"}
    }
    hash_overlap = sorted(h for h in calibration_hashes & hidden_hashes if h)
    details = {
        "calibration_groups": sorted(calibration),
        "hidden_groups": sorted(hidden),
        "group_overlap": overlap,
        "source_hash_overlap": hash_overlap,
    }
    return not overlap and not hash_overlap and bool(calibration) and bool(hidden), details


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _episode_rows_for_instance(instance: Mapping[str, Any], *, seeds: Sequence[int], thresholds: Mapping[str, float], episode_runner: Callable[..., dict[str, Any]] = run_mock_episode) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    difficulty = instance.get("difficulty") or {}
    schedule = difficulty.get("force_schedule") or {}
    if isinstance(schedule, Mapping):
        forces = [float(schedule[key]) for key in ("low", "medium", "high") if key in schedule]
    else:
        forces = [float(value) for value in schedule]
    if not forces:
        forces = [4.0, 10.0, 20.0]
    episode_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for seed in seeds:
        control = episode_runner(instance, seed=seed, force_n=0.0, no_contact=True)
        control["episode_kind"] = "no_contact_control"
        _attach_instance_metadata(control, instance)
        _rehash(control, "episode_hash")
        episode_rows.append(control)
        metric = {**evaluate_episode(control, penetration_threshold_m=thresholds["max_penetration_m"]), "episode_kind": "no_contact_control"}
        _attach_instance_metadata(metric, instance)
        _rehash(metric, "metric_record_hash")
        metric_rows.append(metric)
        for level, force in zip(("low", "medium", "high"), forces):
            episode = episode_runner(instance, seed=seed, force_n=force, force_sweep=forces)
            episode["episode_kind"] = level
            _attach_instance_metadata(episode, instance)
            _rehash(episode, "episode_hash")
            episode_rows.append(episode)
            metric = {**evaluate_episode(episode, penetration_threshold_m=thresholds["max_penetration_m"]), "episode_kind": level, "force_n": force}
            _attach_instance_metadata(metric, instance)
            _rehash(metric, "metric_record_hash")
            metric_rows.append(metric)
    return episode_rows, metric_rows


def _attach_instance_metadata(record: dict[str, Any], instance: Mapping[str, Any]) -> None:
    record.update(
        {
            "benchmark_instance_id": instance.get("benchmark_instance_id"),
            "geometry_revision_id": instance.get("geometry_revision_id"),
            "physical_bundle_id": instance.get("physical_bundle_id"),
            "provisional_physical_bundle_id": instance.get("provisional_physical_bundle_id"),
            "physical_join_status": instance.get("physical_join_status"),
            "physical_release_id": instance.get("physical_release_id"),
            "sample_id": instance.get("sample_id"),
            "asset_revision_id": instance.get("asset_revision_id"),
            "difficulty_id": instance.get("difficulty_id"),
            "damage_model_id": instance.get("damage_model_id", "none"),
            "calibration_or_hidden_split": instance.get("calibration_or_hidden_split"),
            "source_hash": instance.get("source_hash") or (instance.get("source") or {}).get("p0_asset_source_sha256"),
        }
    )


def _rehash(record: dict[str, Any], key: str) -> None:
    record.pop(key, None)
    record[key] = sha256_json(record)


def _gate_for_instance(instance: Mapping[str, Any], metric_rows: Sequence[Mapping[str, Any]], *, thresholds: Mapping[str, float], seeds: Sequence[int], episode_runner: Callable[..., dict[str, Any]] = run_mock_episode, require_physical_join: bool = False) -> dict[str, Any]:
    controls = [row for row in metric_rows if row.get("episode_kind") == "no_contact_control"]
    force_rows = [row for row in metric_rows if row.get("episode_kind") in {"low", "medium", "high"}]
    damage_model = str(instance.get("material_model_id")) == "tofu_damage"
    finite_ok = all(
        float((row.get("metrics") or {}).get("finite_fraction", 0.0)) >= 1.0 - 1e-12
        and bool((row.get("metrics") or {}).get("converged", False)) for row in metric_rows
    )
    no_contact_damage = max([float((row.get("metrics") or {}).get("terminal_damage_fraction", 1.0)) for row in controls] or [1.0])
    first_seed = seeds[0] if seeds else 0
    ordered_damage = [
        float((next((row for row in force_rows if row.get("seed") == first_seed and row.get("episode_kind") == level), {}).get("metrics") or {}).get("terminal_damage_fraction", 1.0))
        for level in ("low", "medium", "high")
    ]
    damage_by_seed = {
        int(seed): [
            float((next((row for row in force_rows if row.get("seed") == seed and row.get("episode_kind") == level), {}).get("metrics") or {}).get("terminal_damage_fraction", 1.0))
            for level in ("low", "medium", "high")
        ]
        for seed in seeds
    }
    monotonic_ok = all(_monotonic(values, thresholds["monotonic_tolerance"]) for values in damage_by_seed.values())
    if damage_model and max(ordered_damage) <= thresholds["monotonic_tolerance"]:
        monotonic_ok = False
    repeat_values = [float((row.get("metrics") or {}).get("terminal_damage_fraction", 1.0)) for row in force_rows if row.get("episode_kind") == "high"]
    repeat_variance = _variance(repeat_values)
    windows = [(row.get("metrics") or {}).get("safe_force_window") or {} for row in force_rows]
    safe_ok = any(bool(window.get("non_empty")) and float(window.get("width_n", 0.0)) >= thresholds["safe_window_min_width_n"] for window in windows)
    onset_values = [
        float(((row.get("metrics") or {}).get("damage_onset") or {}).get("force_n", 0.0))
        for row in force_rows
    ]
    onset_ok = all(isinstance((row.get("metrics") or {}).get("damage_onset"), Mapping) for row in force_rows)
    penetration_ok = all(float((row.get("metrics") or {}).get("max_penetration_m", float("inf"))) <= thresholds["max_penetration_m"] for row in metric_rows)
    replay_source = next((row for row in force_rows if row.get("episode_kind") == "medium"), None)
    replay_ok = False
    if replay_source is not None:
        replay = episode_runner(instance, seed=int(replay_source.get("seed", 0)), force_n=float(replay_source.get("force_n", 0.0)))
        replay_ok = replay.get("action_replay_hash") == replay_source.get("action_replay_hash")
    gates = {
        "finite_convergence": {"passed": finite_ok},
        "no_contact_no_damage": {"passed": no_contact_damage <= thresholds["no_contact_damage_max"], "max_damage": no_contact_damage},
        "force_damage_monotonic": {"passed": monotonic_ok, "damage_by_force": ordered_damage, "damage_by_seed": damage_by_seed, "damage_model": damage_model},
        "safe_window_non_empty": {"passed": safe_ok, "windows": windows},
        "damage_onset_recorded": {"passed": onset_ok, "onset_force_n": onset_values},
        "contact_penetration": {"passed": penetration_ok},
        "repeat_seed_variance": {"passed": repeat_variance <= thresholds["repeat_variance_max"], "variance": repeat_variance},
        "action_replay": {"passed": replay_ok},
        "physical_bundle_join": {
            "passed": (not require_physical_join) or (
                instance.get("physical_join_status") == "joined_verified"
                and bool(instance.get("physical_bundle_id"))
                and bool(instance.get("sample_id"))
                and bool(instance.get("asset_revision_id"))
            ),
            "required": require_physical_join,
        },
    }
    return {"benchmark_instance_id": instance.get("benchmark_instance_id"), "passed": all(bool(value.get("passed")) for value in gates.values()), "gates": gates}


def run_qualification(
    instances: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path | None = None,
    seeds: Sequence[int] = (0, 1),
    thresholds: Mapping[str, float] | None = None,
    backend_id: str = "mock.deterministic.v1",
    formal_scope: bool = False,
    episode_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run qualification and emit every required manifest/record family."""

    limits = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    if formal_scope and episode_runner is None:
        raise ValueError("formal qualification requires an explicit simulator episode_runner")
    raw_runner = episode_runner or run_mock_episode

    def runner(instance: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        declared_backend = "mock.deterministic.v1" if not formal_scope else backend_id
        try:
            record = raw_runner(instance, **kwargs)
            validate_episode_record(record, backend_id=declared_backend)
            return record
        except (Exception, EpisodeRunnerContractError) as exc:
            # Preserve a metric-bearing failed episode when a production
            # executor crashes or returns a malformed record.  This fallback is
            # explicitly marked failed and can never satisfy formal gates.
            fallback = run_mock_episode(
                instance,
                seed=int(kwargs.get("seed", 0)),
                force_n=kwargs.get("force_n"),
                no_contact=bool(kwargs.get("no_contact", False)),
                force_sweep=kwargs.get("force_sweep"),
                backend_failure=True,
            )
            fallback["backend_id"] = declared_backend
            fallback["backend_status"] = "failed"
            # Do not serialize exception text: a production runner may include
            # evaluator-only values in its diagnostic message.
            fallback["runner_error"] = {"type": type(exc).__name__}
            for sample in fallback.get("samples", []) + fallback.get("force_sweep_samples", []):
                sample["finite_fraction"] = 0.0
                sample["converged"] = False
            fallback["episode_hash"] = sha256_json({key: value for key, value in fallback.items() if key != "episode_hash"})
            return fallback
    instance_rows = [copy.deepcopy(dict(row)) for row in instances]
    all_episodes: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    instance_reports: list[dict[str, Any]] = []
    for instance in instance_rows:
        if instance.get("status") != "candidate":
            failed_episode = runner(instance, seed=0, force_n=0.0, backend_failure=True)
            failed_episode["episode_kind"] = "construction_failure"
            _attach_instance_metadata(failed_episode, instance)
            _rehash(failed_episode, "episode_hash")
            failed_metric = {**evaluate_episode(failed_episode), "episode_kind": "construction_failure"}
            _attach_instance_metadata(failed_metric, instance)
            _rehash(failed_metric, "metric_record_hash")
            all_episodes.append(failed_episode)
            all_metrics.append(failed_metric)
            report = {"benchmark_instance_id": instance.get("benchmark_instance_id"), "passed": False, "gates": {"finite_convergence": {"passed": False, "reason": "instance_not_candidate"}}}
            instance_reports.append(report)
            continue
        episodes, metrics = _episode_rows_for_instance(instance, seeds=seeds, thresholds=limits, episode_runner=runner)
        all_episodes.extend(episodes)
        all_metrics.extend(metrics)
        instance_reports.append(_gate_for_instance(instance, metrics, thresholds=limits, seeds=seeds, episode_runner=runner, require_physical_join=formal_scope))
    split_passed, split_details = _split_gate(instance_rows)
    backend_mismatch = sorted(
        {
            str(row.get("backend_id"))
            for row in all_episodes
            if formal_scope and str(row.get("backend_id")) != str(backend_id)
        }
    )
    for report in instance_reports:
        report.setdefault("gates", {})["hidden_split_no_leakage"] = {"passed": split_passed, **split_details}
        report.setdefault("gates", {})["backend_contract"] = {"passed": not backend_mismatch, "declared_backend_id": backend_id, "mismatched_backend_ids": backend_mismatch}
        report["passed"] = bool(report.get("passed")) and split_passed
    rejected: list[dict[str, Any]] = []
    for instance, instance_report in zip(instance_rows, instance_reports):
        instance["qualification_status"] = "qualified" if instance_report["passed"] else "rejected"
        instance["e3_a_eligible"] = bool(instance_report["passed"] and formal_scope)
        instance["qualification_scope"] = "formal_server" if formal_scope else "local_mock_smoke"
        instance["formal_e3_a_manifest"] = False
        _rehash(instance, "instance_hash")
        if not instance_report["passed"]:
            failed_gates = [name for name, gate in instance_report.get("gates", {}).items() if not gate.get("passed")]
            disposition = "replace" if "unsupported_material_model" in instance.get("failure_codes", []) else ("repair" if "finite_convergence" in failed_gates else "rejected")
            rejected.append({"benchmark_instance_id": instance.get("benchmark_instance_id"), "disposition": disposition, "failed_gates": failed_gates, "reason": "qualification_gate_failed", "repair_action": "rerun_with_same_contract" if disposition == "repair" else None, "replacement_instance_id": None if disposition != "replace" else "reserve_required"})
    report = {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "qualification_id": "fragile-soft-qualification-v1",
        "backend_id": backend_id,
        "release_scope": "formal_qualification" if formal_scope else "local_mock_qualification_only",
        "formal_e3_a_manifest": False,
        "instance_count": len(instance_rows),
        "qualified_count": sum(bool(row["passed"]) for row in instance_reports),
        "rejected_count": sum(not bool(row["passed"]) for row in instance_reports),
        "status": "qualified" if instance_reports and all(row["passed"] for row in instance_reports) else "rejected",
        "e3_a_eligible": bool(formal_scope and instance_reports and all(row["passed"] for row in instance_reports)),
        "qualification_scope": "formal_server" if formal_scope else "local_mock_smoke",
        "thresholds": limits,
        "seeds": list(seeds),
        "instance_reports": instance_reports,
        "failure_disposition": {"rejected": sum(r["disposition"] == "rejected" for r in rejected), "repair": sum(r["disposition"] == "repair" for r in rejected), "replace": sum(r["disposition"] == "replace" for r in rejected)},
        "evidence_boundary": "TofuDamage is a continuum stiffness-loss surrogate, not explicit fracture; P0 video count is not a pass count.",
    }
    report["qualification_report_hash"] = sha256_json(report)
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        _write_jsonl(target / "instance_manifest.jsonl", instance_rows)
        _write_jsonl(target / "episode_records.jsonl", all_episodes)
        _write_jsonl(target / "metric_records.jsonl", all_metrics)
        _write_jsonl(target / "material_records.jsonl", [{"benchmark_instance_id": row.get("benchmark_instance_id"), "geometry_revision_id": row.get("geometry_revision_id"), "physical_bundle_id": row.get("physical_bundle_id"), "material_model_id": row.get("material_model_id"), "damage_model_id": row.get("damage_model_id"), "source_hash": row.get("source_hash"), "material": row.get("material")} for row in instance_rows])
        _write_jsonl(target / "material_config_records.jsonl", [{"benchmark_instance_id": row.get("benchmark_instance_id"), "geometry_revision_id": row.get("geometry_revision_id"), "physical_bundle_id": row.get("physical_bundle_id"), "material_model_id": row.get("material_model_id"), "damage_model_id": row.get("damage_model_id"), "source_hash": row.get("source_hash"), "material": row.get("material"), "difficulty_id": row.get("difficulty_id"), "difficulty": row.get("difficulty"), "scenario": row.get("scenario"), "split": row.get("split")} for row in instance_rows])
        _write_jsonl(target / "config_records.jsonl", [{"benchmark_instance_id": row.get("benchmark_instance_id"), "geometry_revision_id": row.get("geometry_revision_id"), "physical_bundle_id": row.get("physical_bundle_id"), "difficulty_id": row.get("difficulty_id"), "source_hash": row.get("source_hash"), "difficulty": row.get("difficulty"), "scenario": row.get("scenario"), "evaluator_id": row.get("evaluator_id"), "split": row.get("split")} for row in instance_rows])
        media_rows = [{"benchmark_instance_id": row.get("benchmark_instance_id"), "geometry_revision_id": row.get("geometry_revision_id"), "physical_bundle_id": row.get("physical_bundle_id"), "difficulty_id": row.get("difficulty_id"), "source_hash": row.get("source_hash"), "video_status": "not_generated_local_mock", "keyframe_status": "not_generated_local_mock", "p0_reference_paths": [ref.get("path") for ref in (row.get("source", {}).get("source_refs", []) if isinstance(row.get("source"), Mapping) else []) if isinstance(ref, Mapping)], "is_evidence": False} for row in instance_rows]
        _write_jsonl(target / "video_keyframe_manifest.jsonl", media_rows)
        _write_jsonl(target / "video_manifest.jsonl", [{"benchmark_instance_id": row["benchmark_instance_id"], "status": row["video_status"], "is_evidence": False} for row in media_rows])
        _write_jsonl(target / "keyframe_manifest.jsonl", [{"benchmark_instance_id": row["benchmark_instance_id"], "status": row["keyframe_status"], "is_evidence": False} for row in media_rows])
        (target / "qualification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_jsonl(target / "failed_repair_replace.jsonl", rejected)
        joined = sum(row.get("physical_join_status") == "joined_verified" for row in instance_rows)
        nonverified = sum(row.get("physical_join_status") == "joined_nonverified" for row in instance_rows)
        missing = sum(row.get("physical_join_status") in {None, "unjoined", "missing"} for row in instance_rows)
        ambiguous = sum(row.get("physical_join_status") == "ambiguous" for row in instance_rows)
        join_report = {
            "schema_version": "fragile-soft.physical-join.v1",
            "physical_release_id": next((row.get("physical_release_id") for row in instance_rows if row.get("physical_release_id")), None),
            "index_count": joined + nonverified,
            "instance_count": len(instance_rows),
            "counts": {"joined_verified": joined, "joined_nonverified": nonverified, "missing": missing, "ambiguous": ambiguous},
            "errors": [],
            "all_joined_verified": bool(instance_rows) and joined == len(instance_rows),
        }
        (target / "physical_join_report.json").write_text(json.dumps(join_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"report": report, "instances": instance_rows, "episodes": all_episodes, "metrics": all_metrics, "rejected": rejected}


def make_default_qualification_instances(
    adapter: Any,
    config: Mapping[str, Any] | None = None,
    *,
    physical_index: Any = None,
    physical_join_map: Any = None,
    physical_release_id: str | None = None,
    require_physical_join: bool = False,
) -> list[dict[str, Any]]:
    cfg = dict(config or {})
    cfg["canonical_groups"] = cfg.get("qualification_groups", DEFAULT_QUALIFICATION_GROUPS)
    cfg["geometry_revisions_per_group"] = int(cfg.get("qualification_revisions_per_group", 3))
    cfg["difficulty_ids"] = list(cfg.get("qualification_difficulty_ids", DEFAULT_QUALIFICATION_DIFFICULTIES))
    return build_instances(
        adapter,
        cfg,
        physical_index=physical_index,
        physical_join_map=physical_join_map,
        physical_release_id=physical_release_id,
        require_physical_join=require_physical_join,
    )
