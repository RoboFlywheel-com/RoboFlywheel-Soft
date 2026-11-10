# SPDX-License-Identifier: Apache-2.0
"""Strict converter from the legacy v1 asset ledger to formal asset release v2.

The converter is intentionally a *release gate*, not a repair script.  It
copies no meshes and invents no metric scale.  Rows with missing scale,
missing visual/collision artifacts, invalid status, absolute paths, or null
physical references are retained in a conversion report and make the output
non-publishable until a server-side owner resolves them.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonical_dumps, fingerprint, read_json, sha256_file
from .io import atomic_write_json, write_jsonl, read_jsonl


class AssetReleaseConversionError(ValueError):
    pass


_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_FORBIDDEN = ("/tmp/", "/var/tmp/", "/node10_nvme/", "latest", "staging")


def _hash(value: Any, field: str) -> str:
    if isinstance(value, str) and value.startswith("sha256:") and len(value) == 71:
        try:
            int(value[7:], 16)
            return value.lower()
        except ValueError:
            pass
    if isinstance(value, str) and _HEX64.fullmatch(value):
        return "sha256:" + value.lower()
    raise AssetReleaseConversionError(f"{field} must be sha256:<64 hex>")


def _record_hashes(row: Mapping[str, Any]) -> set[str]:
    payload = {key: value for key, value in row.items() if key not in {"record_sha256", "record_hash"}}
    canonical = fingerprint(payload)
    utf8 = "sha256:" + hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return {canonical, utf8}


def _rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise AssetReleaseConversionError(f"asset v1 input is missing or symlinked: {path}")
    if path.suffix.lower() not in {".jsonl", ".ndjson"}:
        value = read_json(path)
        if isinstance(value, Mapping):
            value = value.get("rows") or value.get("assets") or value.get("ledger") or value.get("items")
        if not isinstance(value, list):
            raise AssetReleaseConversionError(f"expected rows list in {path}")
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return list(read_jsonl(path))


def _forbidden(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_forbidden(k) or _forbidden(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_forbidden(v) for v in value)
    if isinstance(value, str):
        text = value.lower().replace("\\", "/")
        return any(token in text for token in _FORBIDDEN)
    return False


def _relative_artifact(raw: Mapping[str, Any], root: Path, *, roots: Mapping[str, Path] | None = None, row_id: str, role: str) -> tuple[dict[str, Any] | None, str | None]:
    raw_path = raw.get("relative_path") or raw.get("path") or raw.get("relative_to_revision")
    if not isinstance(raw_path, str) or not raw_path:
        return None, f"{role}_path_missing"
    candidate = Path(raw_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None, f"{role}_path_not_relative"
    if _forbidden(raw_path):
        return None, f"{role}_path_forbidden"
    expected = raw.get("sha256") or raw.get("hash")
    try:
        expected_hash = _hash(expected, f"{row_id}.{role}.sha256")
    except AssetReleaseConversionError as exc:
        return None, str(exc)
    alias = str(raw.get("root_alias") or "asset_artifacts")
    path = (roots.get(alias, root) if roots else root) / candidate
    if not path.is_file() or path.is_symlink():
        return None, f"{role}_artifact_missing"
    actual = sha256_file(path)
    if actual != expected_hash:
        return None, f"{role}_artifact_hash_mismatch"
    return {"root_alias": str(raw.get("root_alias") or "asset_artifacts"), "relative_path": candidate.as_posix(), "sha256": actual}, None


def _metric_scale(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    value = row.get("metric_scale") or row.get("scale") or row.get("metric_scale_m")
    if not isinstance(value, Mapping):
        return None, "metric_scale_missing"
    try:
        scale_m = float(
            value.get("meters_per_unit")
            if value.get("meters_per_unit") is not None
            else (value.get("scale_m") if value.get("scale_m") is not None else value.get("scale_to_meter"))
        )
    except (TypeError, ValueError):
        return None, "metric_scale_invalid"
    if scale_m <= 0:
        return None, "metric_scale_nonpositive"
    source = str(value.get("source") or row.get("metric_scale_source") or "unspecified")
    if source.lower() in {"measured", "ground_truth", "ground-truth"}:
        # The converter does not certify measurements; preserve the label but
        # require an explicit evidence reference before release.
        evidence = value.get("evidence_ref") or row.get("metric_scale_evidence_ref")
        if not evidence:
            return None, "metric_scale_measurement_evidence_missing"
    result = {
        "meters_per_unit": scale_m,
        "unit": str(value.get("unit") or "m"),
        "source": source,
        "evidence_ref": value.get("evidence_ref") or row.get("metric_scale_evidence_ref"),
    }
    if value.get("provenance") is not None:
        result["provenance"] = value.get("provenance")
    if value.get("confidence") is not None:
        try:
            result["confidence"] = float(value["confidence"])
        except (TypeError, ValueError):
            return None, "metric_scale_confidence_invalid"
    return result, None


def convert_v1_to_v2(
    ledger_path: str | Path,
    *,
    output_dir: str | Path,
    artifact_root: str | Path,
    artifact_roots: Mapping[str, str | Path] | None = None,
    release_id: str = "softsim-100k-assets-v2",
    source_manifest_path: str | Path | None = None,
    expected_count: int | None = None,
    allow_ready_with_warnings: bool = False,
) -> dict[str, Any]:
    """Convert a v1 ledger into an independent v2 release workspace."""
    source = Path(ledger_path)
    if source.is_symlink():
        raise AssetReleaseConversionError(f"asset v1 ledger is symlinked: {source}")
    raw_root = Path(artifact_root).expanduser().absolute()
    if raw_root.is_symlink():
        raise AssetReleaseConversionError(f"asset v1 artifact root is symlinked: {artifact_root}")
    root = raw_root.resolve()
    roots: dict[str, Path] = {}
    for alias, path in (artifact_roots or {}).items():
        raw_path = Path(path).expanduser().absolute()
        if raw_path.is_symlink():
            raise AssetReleaseConversionError(f"asset v1 artifact root is symlinked: {alias}")
        roots[str(alias)] = raw_path.resolve()
    roots.setdefault("asset_artifacts", root)
    out = Path(output_dir).resolve()
    rows = _rows(source)
    source_hash = sha256_file(source)
    source_manifest_hash = sha256_file(source_manifest_path) if source_manifest_path else None
    converted: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(rows):
        sample_id = str(raw.get("sample_id") or "")
        revision = str(raw.get("asset_revision_id") or "")
        row_errors: list[str] = []
        if not sample_id:
            row_errors.append("sample_id_missing")
        if not revision:
            row_errors.append("asset_revision_id_missing")
        key = (sample_id, revision)
        if key in seen:
            row_errors.append("duplicate_sample_revision")
        seen.add(key)
        if _forbidden(raw):
            row_errors.append("forbidden_tmp_staging_latest_reference")
        source_record_hash = raw.get("record_sha256") or raw.get("record_hash")
        if source_record_hash is None:
            row_errors.append("v1_record_hash_missing")
        else:
            try:
                expected_record_hashes = _record_hashes(raw)
                if str(source_record_hash) not in expected_record_hashes:
                    row_errors.append("v1_record_hash_mismatch")
            except (TypeError, ValueError):
                row_errors.append("v1_record_hash_invalid")
        old_status = str(raw.get("asset_status") or raw.get("status") or "unknown")
        if old_status not in {"ready", "ready_with_warnings"}:
            row_errors.append("v1_asset_not_ready")
        scale, scale_error = _metric_scale(raw)
        if scale_error:
            row_errors.append(scale_error)
        artifacts = raw.get("artifacts")
        if not isinstance(artifacts, list):
            artifacts = []
        geometry_candidates = [item for item in artifacts if isinstance(item, Mapping) and str(item.get("role", "")).lower() in {"visual_mesh", "collision_mesh", "geometry", "mesh"}]
        geometry_ref = None
        artifact_error = "geometry_artifact_missing"
        for item in geometry_candidates:
            geometry_ref, artifact_error = _relative_artifact(item, root, roots=roots, row_id=sample_id or f"row-{index}", role="geometry")
            if geometry_ref is not None:
                break
        if geometry_ref is None:
            row_errors.append(artifact_error or "geometry_artifact_missing")
        physical_ref = raw.get("physical_bundle_ref")
        if physical_ref not in (None, ""):
            # A v1 physical reference is not trusted as a v2 physical release
            # identity; preserve it only as provenance.
            if _forbidden(physical_ref):
                row_errors.append("physical_bundle_ref_forbidden")
        status = "released" if not row_errors and (old_status == "ready" or allow_ready_with_warnings) else "rejected"
        converted_row: dict[str, Any] = {
            "schema_name": "softsim.asset_release_record",
            "schema_version": 2,
            "sample_id": sample_id,
            "asset_revision_id": revision,
            "dataset_release_id": release_id,
            "status": status,
            "canonical_id": raw.get("canonical_id"),
            "object_type": raw.get("object_type"),
            "geometry_family": raw.get("geometry_family"),
            "material_family": raw.get("material_family"),
            "geometry_ref": geometry_ref,
            "metric_scale": scale,
            "source_v1": {"record_sha256": raw.get("record_sha256"), "asset_status": old_status, "physical_bundle_ref": physical_ref},
            "conversion": {"rule": "v1-ledger-to-v2-release-gate", "source_ledger_sha256": source_hash, "source_manifest_sha256": source_manifest_hash},
            "failure_codes": sorted(set(row_errors)),
            "provenance": {"is_measured_ground_truth": False, "physical_bundle_join_required": True},
        }
        converted_row["record_sha256"] = fingerprint(converted_row)
        converted.append(converted_row)
        if row_errors:
            errors.append({"row": index + 1, "sample_id": sample_id, "asset_revision_id": revision, "failure_codes": sorted(set(row_errors))})
    converted.sort(key=lambda row: (str(row.get("sample_id")), str(row.get("asset_revision_id"))))
    out.mkdir(parents=True, exist_ok=True)
    ledger_out = out / "asset_ledger.jsonl"
    write_jsonl(ledger_out, converted)
    ledger_hash = sha256_file(ledger_out)
    (out / "asset_ledger.sha256").write_text(ledger_hash.split(":", 1)[-1] + "  asset_ledger.jsonl\n", encoding="utf-8")
    manifest = {
        "schema_name": "softsim.asset_release_manifest",
        "schema_version": 2,
        "release_id": release_id,
        "publishable": bool(converted) and not errors and (expected_count is None or len(converted) == expected_count),
        "population_count": len(converted),
        "released_count": sum(row["status"] == "released" for row in converted),
        "rejected_count": sum(row["status"] == "rejected" for row in converted),
        "ledger_sha256": ledger_hash,
        "source_v1_ledger_sha256": source_hash,
        "source_v1_manifest_sha256": source_manifest_hash,
        # Do not serialize workstation paths as artifact identity.  The server
        # binds these aliases to persistent roots before running the formal
        # reader; relative paths plus hashes remain the immutable identity.
        "logical_roots": {alias: f"<mount:{alias}>" for alias in sorted(roots)},
        "root_bindings_required": sorted(roots),
        "physical_release_required": True,
        "conversion_report_ref": "conversion_report.json",
        "conversion_rule": "v1-ledger-to-v2-release-gate; no scale/artifact invention",
    }
    atomic_write_json(out / "release_manifest.json", manifest)
    report = {
        "schema_name": "softsim.asset_release_conversion_report",
        "schema_version": 1,
        "source_ledger_sha256": source_hash,
        "output_release_id": release_id,
        "input_count": len(rows),
        "output_count": len(converted),
        "released_count": manifest["released_count"],
        "rejected_count": manifest["rejected_count"],
        "errors": errors,
        "publishable": manifest["publishable"],
        "claim_boundary": "v2 conversion does not provide physical verification or material ground truth",
    }
    atomic_write_json(out / "conversion_report.json", report)
    return {"manifest": manifest, "report": report, "ledger_path": ledger_out.as_posix()}
