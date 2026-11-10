# SPDX-License-Identifier: Apache-2.0
"""Join benchmark instances to an authoritative physical bundle index.

The constructor deliberately creates only provisional IDs.  Formal E3-A input
must call this module with an immutable physical index and an explicit mapping
when the P0 asset ID is not the physical release's sample ID.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from .hashing import sha256_json


class PhysicalJoinError(ValueError):
    """Physical index or mapping is malformed or ambiguous."""


def _rows(source: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file() or path.is_symlink():
            raise PhysicalJoinError(f"physical join input is missing or symlinked: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() != ".jsonl":
            value = json.loads(text)
            if isinstance(value, Mapping):
                value = value.get("rows") or value.get("bundles") or value.get("instances") or value.get("mapping") or []
            if not isinstance(value, list):
                raise PhysicalJoinError(f"expected list in {path}")
            return [dict(row) for row in value if isinstance(row, Mapping)]
        result = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise PhysicalJoinError(f"{path}:{line_number} is not an object")
                result.append(dict(value))
        return result
    return [dict(row) for row in source]


def _key(sample_id: Any, asset_revision_id: Any) -> tuple[str, str]:
    if sample_id is None or asset_revision_id is None:
        raise PhysicalJoinError("physical join key requires sample_id and asset_revision_id")
    return str(sample_id), str(asset_revision_id)


def _record_hash(row: Mapping[str, Any]) -> str | None:
    value = row.get("record_sha256")
    if value is None:
        return None
    payload = {key: value for key, value in row.items() if key != "record_sha256"}
    expected = sha256_json(payload)
    legacy = "sha256:" + hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return str(value) if str(value) in {expected, legacy} else None


def _mapping_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
    instance_id = row.get("benchmark_instance_id") or row.get("instance_id")
    if instance_id:
        return ("instance", str(instance_id))
    base = row.get("base_asset_id") or row.get("asset_id")
    revision = row.get("geometry_revision_id") or row.get("revision_id")
    if base is not None and revision is not None:
        return ("geometry", f"{base}::{revision}")
    return None


def join_physical_bundles(
    instances: Iterable[Mapping[str, Any]],
    physical_index: str | Path | Iterable[Mapping[str, Any]],
    *,
    join_map: str | Path | Iterable[Mapping[str, Any]] | None = None,
    physical_release_id: str | None = None,
    require_verified: bool = True,
    require_all: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach exact ``sample_id/asset_revision_id/physical_bundle_id`` joins.

    No positional or fuzzy matching is performed.  A mapping row may identify
    an instance directly or map a base asset plus geometry revision to a
    physical key.  Ambiguous/missing/non-verified rows remain explicit and can
    never be promoted to formal E3-A.
    """

    source_rows = _rows(physical_index)
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_bundle: dict[str, list[dict[str, Any]]] = {}
    for row in source_rows:
        key = _key(row.get("sample_id"), row.get("asset_revision_id"))
        status = str(row.get("status", "")).lower()
        if status in {"predicted", "predicted_prior", "calibrated", "calibrated_candidate", "prior"}:
            raise PhysicalJoinError(f"predicted or intermediate physical row cannot be joined: {key}")
        if status not in {"verified", "rejected", "task_failure"}:
            raise PhysicalJoinError(f"unknown physical status cannot be joined: {status or '<missing>'}")
        if row.get("record_sha256") is not None and _record_hash(row) is None:
            raise PhysicalJoinError(f"physical index record hash mismatch for {key}")
        by_key.setdefault(key, []).append(row)
        if row.get("physical_bundle_id") is not None:
            by_bundle.setdefault(str(row["physical_bundle_id"]), []).append(row)
    map_rows = _rows(join_map) if join_map is not None else []
    direct_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in map_rows:
        key = _mapping_key(row)
        if key is None:
            raise PhysicalJoinError("join mapping row needs benchmark_instance_id or base_asset_id+geometry_revision_id")
        direct_map.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    counts = {"joined_verified": 0, "joined_nonverified": 0, "missing": 0, "ambiguous": 0}
    errors: list[dict[str, Any]] = []
    for original in instances:
        instance = dict(original)
        instance_id = str(instance.get("benchmark_instance_id", ""))
        original_hash = instance.get("instance_hash")
        if original_hash is not None:
            payload = {key: value for key, value in instance.items() if key != "instance_hash"}
            if str(original_hash) not in {sha256_json(payload), "sha256:" + hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()}:
                raise PhysicalJoinError(f"benchmark instance hash mismatch: {instance_id}")
            instance["source_instance_hash"] = original_hash
        lookup_key = ("instance", instance_id)
        mappings = direct_map.get(lookup_key, [])
        if not mappings:
            geometry_id = instance.get("geometry_revision_id")
            mappings = direct_map.get(("geometry", f"{instance.get('base_asset_id')}::{geometry_id}"), [])
        candidates: list[dict[str, Any]] = []
        if len(mappings) > 1:
            counts["ambiguous"] += 1
            errors.append({"benchmark_instance_id": instance_id, "code": "ambiguous_join_mapping"})
        elif mappings:
            mapping = mappings[0]
            mapping_release = mapping.get("physical_release_id")
            if mapping_release is not None and physical_release_id is not None and str(mapping_release) != str(physical_release_id):
                counts["missing"] += 1
                errors.append({"benchmark_instance_id": instance_id, "code": "physical_release_mismatch"})
                mappings = []
                mapping = None
            if mapping is None:
                candidates = []
            elif mapping.get("physical_bundle_id"):
                candidates = list(by_bundle.get(str(mapping["physical_bundle_id"]), []))
                candidates = [
                    row for row in candidates
                    if (mapping.get("sample_id") is None or str(mapping.get("sample_id")) == str(row.get("sample_id")))
                    and (mapping.get("asset_revision_id") is None or str(mapping.get("asset_revision_id")) == str(row.get("asset_revision_id")))
                ]
            else:
                try:
                    candidates = list(by_key.get(_key(mapping.get("sample_id"), mapping.get("asset_revision_id")), []))
                except PhysicalJoinError:
                    candidates = []
        else:
            # Exact direct convention is safe only when it yields one row.
            sample = instance.get("sample_id") or instance.get("base_asset_id")
            revision = instance.get("asset_revision_id")
            if revision is not None:
                try:
                    candidates = list(by_key.get(_key(sample, revision), []))
                except PhysicalJoinError:
                    candidates = []
        if len(candidates) != 1:
            if candidates:
                counts["ambiguous"] += 1
            elif not mappings:
                counts["missing"] += 1
            instance["physical_join_status"] = "missing" if not candidates else "ambiguous"
            instance["physical_bundle_status"] = instance["physical_join_status"]
            instance["physical_bundle_id"] = None
            instance["physical_release_id"] = physical_release_id
            instance["formal_join_ready"] = False
            instance.setdefault("failure_codes", []).append("physical_bundle_join_missing" if not candidates else "physical_bundle_join_ambiguous")
            errors.append({"benchmark_instance_id": instance_id, "code": instance["physical_join_status"]})
            instance.pop("instance_hash", None)
            instance["instance_hash"] = sha256_json(instance)
            output.append(instance)
            continue
        physical = candidates[0]
        status = str(physical.get("status", "")).lower()
        verified = status == "verified"
        if verified:
            counts["joined_verified"] += 1
        else:
            counts["joined_nonverified"] += 1
        instance["sample_id"] = str(physical["sample_id"])
        instance["asset_revision_id"] = str(physical["asset_revision_id"])
        instance["physical_bundle_id"] = str(physical.get("physical_bundle_id")) if physical.get("physical_bundle_id") else None
        instance["physical_bundle_status"] = status
        instance["physical_join_status"] = "joined_verified" if verified else "joined_nonverified"
        instance["physical_bundle_record_sha256"] = physical.get("record_sha256")
        instance["physical_release_id"] = physical_release_id
        instance["formal_join_ready"] = bool(verified and instance.get("physical_bundle_id"))
        if not verified:
            instance.setdefault("failure_codes", []).append("physical_bundle_not_verified")
        instance.pop("instance_hash", None)
        instance["instance_hash"] = sha256_json(instance)
        output.append(instance)
    report = {
        "schema_name": "fragile-soft.physical-join-report",
        "schema_version": "fragile-soft.physical-join.v1",
        "physical_release_id": physical_release_id,
        "index_count": len(source_rows),
        "instance_count": len(output),
        "counts": counts,
        "errors": errors,
        "all_joined_verified": counts["joined_verified"] == len(output) and not errors,
        "verification_gate": "required" if require_verified else "advisory",
        "verified_required": bool(require_verified),
    }
    if require_all and not report["all_joined_verified"]:
        raise PhysicalJoinError(json.dumps(report, sort_keys=True))
    return output, report


def make_join_template(
    instances: Iterable[Mapping[str, Any]],
    *,
    physical_release_id: str | None = None,
) -> list[dict[str, Any]]:
    """Create a reviewable mapping with no guessed physical identity."""

    return [
        {
            "benchmark_instance_id": instance.get("benchmark_instance_id"),
            "base_asset_id": instance.get("base_asset_id"),
            "geometry_revision_id": instance.get("geometry_revision_id"),
            "physical_release_id": physical_release_id,
            "sample_id": None,
            "asset_revision_id": None,
            "physical_bundle_id": None,
            "mapping_status": "needs_verified_physical_mapping",
        }
        for instance in instances
    ]
