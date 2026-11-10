# SPDX-License-Identifier: Apache-2.0
"""Readers and qualification gates for immutable SoftSim releases.

The reader is intentionally stricter than the legacy pilot importers.  A
directory merely containing plausible JSON is not a release: every authority
file and every publishable physical bundle must be hashed, traceable, and
compatible with the frozen Genesis proxy method.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .canonical import fingerprint, read_json, sha256_file
from .io import read_jsonl
from .schema import (
    SchemaError,
    require,
    require_hash,
    require_id,
    validate_artifact_ref,
    validate_asset_row,
    validate_physical_row,
)


class ReleaseError(ValueError):
    """An input exists but violates the formal release contract."""

    def __init__(self, message: str, code: str = "release_invalid") -> None:
        super().__init__(message)
        self.code = code


class BlockedRelease(ReleaseError):
    """A formal run cannot start because a required release is unavailable."""

    def __init__(self, message: str, code: str = "blocked_missing_release") -> None:
        super().__init__(message, code)


@dataclass(frozen=True)
class ReleaseBundle:
    asset_manifest: dict[str, Any]
    assets: tuple[dict[str, Any], ...]
    physical_manifest: dict[str, Any]
    physical: tuple[dict[str, Any], ...]
    benchmark_manifest: tuple[dict[str, Any], ...]
    method_contract: dict[str, Any]
    roots: dict[str, str]
    source_hashes: dict[str, str]
    synthetic: bool = False
    benchmark_join_report: dict[str, Any] | None = None

    @property
    def verified_physical(self) -> tuple[dict[str, Any], ...]:
        return tuple(row for row in self.physical if row.get("status") == "verified")

    @property
    def rejected_physical(self) -> tuple[dict[str, Any], ...]:
        return tuple(row for row in self.physical if row.get("status") == "rejected")

    @property
    def task_failures(self) -> tuple[dict[str, Any], ...]:
        return tuple(row for row in self.physical if row.get("status") == "task_failure")

    @property
    def asset_release_id(self) -> str:
        return str(self.asset_manifest["release_id"])

    @property
    def physical_release_id(self) -> str:
        return str(self.physical_manifest["release_id"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_manifest": copy.deepcopy(self.asset_manifest),
            "asset_count": len(self.assets),
            "physical_manifest": copy.deepcopy(self.physical_manifest),
            "physical_count": len(self.physical),
            "verified_count": len(self.verified_physical),
            "rejected_count": len(self.rejected_physical),
            "task_failure_count": len(self.task_failures),
            "benchmark_count": len(self.benchmark_manifest),
            "method_id": self.method_contract.get("method_id"),
            "source_hashes": dict(self.source_hashes),
            "synthetic": self.synthetic,
            "benchmark_join_report": copy.deepcopy(self.benchmark_join_report),
        }


def _row_hash(row: Mapping[str, Any], field: str = "record_sha256") -> str:
    payload = {key: value for key, value in row.items() if key != field}
    return fingerprint(payload)


def _row_hash_legacy_utf8(row: Mapping[str, Any], field: str = "record_sha256") -> str:
    """Accept the physical-agent v0.1 UTF-8 canonicalization during handoff."""
    payload = {key: value for key, value in row.items() if key != field}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _read_manifest(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise BlockedRelease(f"missing {name}: {path}", f"blocked_missing_{name}")
    if path.is_symlink():
        raise ReleaseError(f"symlink is not a formal input: {path}", "symlink_input")
    try:
        value = read_json(path)
    except (OSError, ValueError) as exc:
        raise ReleaseError(f"cannot read {name}: {path}", "manifest_unreadable") from exc
    if not isinstance(value, dict):
        raise ReleaseError(f"{name} must be a JSON object", "manifest_shape")
    return value


def _read_rows(path: Path, name: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BlockedRelease(f"missing {name}: {path}", f"blocked_missing_{name}")
    if path.is_symlink():
        raise ReleaseError(f"symlink is not a formal input: {path}", "symlink_input")
    try:
        # JSONL is authoritative, but accepting a wrapper object makes the
        # reader interoperable with release tooling that stores ``instances``
        # or ``rows`` in one JSON document.
        if path.suffix.lower() not in {".jsonl", ".ndjson"}:
            try:
                value = read_json(path)
                if isinstance(value, Mapping):
                    rows = value.get("instances") or value.get("rows") or value.get("assets") or value.get("bundles")
                    if isinstance(rows, list):
                        return [dict(row) for row in rows if isinstance(row, Mapping)]
            except (ValueError, OSError):
                pass
        return list(read_jsonl(path))
    except (OSError, ValueError) as exc:
        raise ReleaseError(f"cannot read {name}: {path}", "rows_unreadable") from exc


def _contains_forbidden_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_forbidden_path(k) or _contains_forbidden_path(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_path(v) for v in value)
    if isinstance(value, str):
        lower = value.lower().replace("\\", "/")
        if lower.startswith("/") or (len(lower) > 2 and lower[1] == ":" and lower[2:3] == "/"):
            return True
        if lower in {"/tmp", "/var/tmp"} or lower.startswith("/tmp/") or lower.startswith("/var/tmp/"):
            return True
        if any("latest" in part or "staging" in part for part in lower.split("/")):
            return True
        if lower.startswith("/node10_nvme/"):
            return True
    return False


def _verify_file_hash(path: Path, expected: Any, label: str) -> str:
    actual = sha256_file(path)
    if expected is not None:
        require_hash(expected, label)
        if actual != expected:
            raise ReleaseError(f"{label} mismatch for {path}: {actual} != {expected}", "hash_mismatch")
    return actual


def _validate_method(contract: Mapping[str, Any]) -> None:
    require(contract, "schema_name", "schema_version", "method_id", "genesis_version", "solver", "ipc_used", "uipc_used")
    if contract.get("schema_name") != "softsim.method_contract":
        raise ReleaseError("wrong method contract schema", "method_contract_schema")
    if str(contract.get("genesis_version")) != "1.3.3":
        raise ReleaseError("formal method requires Genesis 1.3.3", "method_version_mismatch")
    if contract.get("ipc_used") is not False or contract.get("uipc_used") is not False:
        raise ReleaseError("formal proxy must declare ipc_used=false and uipc_used=false", "method_solver_mismatch")
    solver = contract.get("solver")
    if not isinstance(solver, Mapping):
        raise ReleaseError("method solver contract missing", "method_solver_missing")
    if solver.get("cloth") != "PBD.Cloth":
        raise ReleaseError("cloth solver must be PBD.Cloth", "method_solver_mismatch")
    if solver.get("coupler") != "LegacyCoupler":
        raise ReleaseError("coupler must be LegacyCoupler", "method_solver_mismatch")
    volume = solver.get("volume") or solver.get("solid")
    if volume not in {"MPM.Elastic", "MPM.ElastoPlastic", "MPM.Elastic/MPM.ElastoPlastic"}:
        raise ReleaseError("volume solver must be MPM.Elastic/MPM.ElastoPlastic", "method_solver_mismatch")


class ReleaseReader:
    """Load and validate v2 asset, v12 physical, benchmark, and method inputs."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        asset_manifest: str | Path | None = None,
        asset_ledger: str | Path | None = None,
        physical_manifest: str | Path | None = None,
        physical_index: str | Path | None = None,
        benchmark_manifest: str | Path | None = None,
        method_contract: str | Path | None = None,
        benchmark_join_map: str | Path | None = None,
        root_bindings: Mapping[str, str | Path] | None = None,
        allow_synthetic: bool = True,
        require_benchmark: bool = True,
        hash_workers: int = 1,
        verify_content_hash: bool = True,
    ) -> None:
        if root is None:
            explicit = next((Path(value) for value in (asset_manifest, physical_manifest, benchmark_manifest, method_contract) if value), None)
            inferred = None
            if explicit is not None and explicit.suffix:
                inferred = explicit.parent.parent if explicit.parent.name in {"asset_release_v2", "physical_release_v12", "fragile_soft_benchmark"} else explicit.parent
            else:
                inferred = explicit
            self.root = (inferred or Path(".")).resolve()
        else:
            root_path = Path(root)
            if root_path.is_symlink():
                raise ReleaseError(f"symlink release root is not formal input: {root_path}", "symlink_input")
            self.root = root_path.resolve()
        self.paths = {
            # ``absolute()`` preserves a symlink in the final component so the
            # reader can reject it; ``resolve()`` here would silently follow
            # the forbidden staging/tmp link before validation.
            "asset_manifest": Path(asset_manifest).expanduser().absolute() if asset_manifest else None,
            "asset_ledger": Path(asset_ledger).expanduser().absolute() if asset_ledger else None,
            "physical_manifest": Path(physical_manifest).expanduser().absolute() if physical_manifest else None,
            "physical_index": Path(physical_index).expanduser().absolute() if physical_index else None,
            "benchmark_manifest": Path(benchmark_manifest).expanduser().absolute() if benchmark_manifest else None,
            "method_contract": Path(method_contract).expanduser().absolute() if method_contract else None,
            "benchmark_join_map": Path(benchmark_join_map).expanduser().absolute() if benchmark_join_map else None,
        }
        self.allow_synthetic = allow_synthetic
        self.require_benchmark = require_benchmark
        # >1 defers content-hash verification to an ordered thread-pool drain;
        # cheap checks (symlink/escape/existence) still raise serially in row
        # order.  Semantics are unchanged for valid releases; on invalid
        # releases a cheap error may surface before an earlier-row hash error.
        self.hash_workers = max(1, int(hash_workers))
        # False is allowed only for plan-time reads: structural checks still
        # run, per-file content hashing is skipped, and the run manifest must
        # record the deferral (per-episode mesh sha256 is still enforced by
        # the executor; full-population hashing remains the inventory gate).
        self.verify_content_hash = bool(verify_content_hash)
        self._pending_hashes: list[tuple[Path, Any, str]] = []
        self.root_bindings: dict[str, Path] = {}
        for alias, path in (root_bindings or {}).items():
            raw_binding = Path(path).expanduser().absolute()
            if raw_binding.is_symlink():
                raise ReleaseError(f"logical root binding is a symlink: {alias}", "symlink_input")
            self.root_bindings[str(alias)] = raw_binding.resolve()

    def _discover(self) -> dict[str, Path]:
        root = self.root
        asset_dir = root / "asset_release_v2"
        physical_dir = root / "physical_release_v12"
        benchmark_dir = root / "fragile_soft_benchmark"
        method_default = root / "method_contract.json"
        paths = {
            "asset_manifest": self.paths["asset_manifest"] or (asset_dir / "release_manifest.json"),
            "asset_ledger": self.paths["asset_ledger"] or (asset_dir / "asset_ledger.jsonl"),
            "physical_manifest": self.paths["physical_manifest"] or (physical_dir / "physical_release_manifest.json"),
            "physical_index": self.paths["physical_index"] or (physical_dir / "physical_bundle_index.jsonl"),
            "benchmark_manifest": self.paths["benchmark_manifest"] or (benchmark_dir / "instance_manifest.jsonl"),
            "method_contract": self.paths["method_contract"] or method_default,
            "benchmark_join_map": self.paths["benchmark_join_map"],
        }
        # Also accept a root containing the authority files directly.
        fallbacks = {
            "asset_manifest": root / "release_manifest.json",
            "asset_ledger": root / "asset_ledger.jsonl",
            "physical_manifest": root / "physical_release_manifest.json",
            "physical_index": root / "physical_bundle_index.jsonl",
            "benchmark_manifest": root / "instance_manifest.jsonl",
        }
        for key, fallback in fallbacks.items():
            if not paths[key].is_file() and fallback.is_file():
                paths[key] = fallback
        return paths

    def read(self) -> ReleaseBundle:
        paths = self._discover()
        # Missing physical is a distinct blocked state required by the run
        # planner; do this before parsing other inputs.
        if not paths["physical_manifest"].is_file() or not paths["physical_index"].is_file():
            raise BlockedRelease("physical release v12 is required before a formal run", "blocked_missing_physical_release")
        asset_manifest = _read_manifest(paths["asset_manifest"], "asset release manifest")
        physical_manifest = _read_manifest(paths["physical_manifest"], "physical release manifest")
        assets = _read_rows(paths["asset_ledger"], "asset ledger")
        physical = _read_rows(paths["physical_index"], "physical bundle index")
        benchmark: list[dict[str, Any]] = []
        benchmark_join_report: dict[str, Any] | None = None
        if self.require_benchmark:
            benchmark = _read_rows(paths["benchmark_manifest"], "fragile-soft benchmark manifest")
            benchmark_hash = physical_manifest.get("benchmark_manifest_sha256")
            if benchmark_hash is None:
                sidecar = paths["benchmark_manifest"].with_suffix(paths["benchmark_manifest"].suffix + ".sha256")
                if sidecar.is_file():
                    token = sidecar.read_text(encoding="utf-8").strip().split()[0]
                    benchmark_hash = token if token.startswith("sha256:") else "sha256:" + token
            if benchmark_hash is None:
                raise ReleaseError("fragile-soft benchmark manifest must carry a hash", "benchmark_manifest_hash_missing")
            _verify_file_hash(paths["benchmark_manifest"], benchmark_hash, "benchmark_manifest_sha256")
            join_path = paths["benchmark_join_map"]
            if join_path is not None and not join_path.is_file():
                raise BlockedRelease(
                    f"missing benchmark physical join map: {join_path}",
                    "blocked_missing_benchmark_join_map",
                )
            if join_path is None:
                for candidate in (paths["benchmark_manifest"].with_name("benchmark_physical_join.jsonl"), paths["benchmark_manifest"].with_name("physical_join_map.jsonl")):
                    if candidate.is_file():
                        join_path = candidate
                        break
            if join_path is not None and join_path.is_file():
                join_rows = _read_rows(join_path, "benchmark physical join map")
                if any(_contains_forbidden_path(row) for row in join_rows):
                    raise ReleaseError("join map contains staging/latest/tmp reference", "forbidden_path")
                benchmark, benchmark_join_report = self._join_benchmark_rows(
                    benchmark,
                    join_rows,
                    physical,
                    physical_release_id=str(physical_manifest.get("release_id") or ""),
                )
                if benchmark_join_report.get("errors"):
                    first_error = benchmark_join_report["errors"][0]
                    error_code = str(first_error.get("code") or "benchmark_physical_join_missing")
                    mapped_code = {
                        "source_instance_hash_mismatch": "benchmark_hash_mismatch",
                        "physical_release_mismatch": "benchmark_physical_join_mismatch",
                        "physical_identity_mismatch": "benchmark_physical_join_mismatch",
                        "ambiguous_join_mapping": "benchmark_physical_join_invalid",
                        "ambiguous": "benchmark_physical_join_invalid",
                        "missing": "benchmark_physical_join_missing",
                    }.get(error_code, "benchmark_physical_join_invalid")
                    raise ReleaseError(
                        f"benchmark physical join failed for {first_error.get('benchmark_instance_id')}: {error_code}",
                        mapped_code,
                    )
                source_join_hash = sha256_file(join_path)
            else:
                source_join_hash = None
        else:
            source_join_hash = None
        method = _read_manifest(paths["method_contract"], "method contract")
        # The local authority file may itself live below a temporary directory
        # during tests.  Only references *inside* release content are subject
        # to the tmp/staging/latest ban; the filesystem location is not an
        # artifact identity.
        for value in (asset_manifest, physical_manifest, method):
            if _contains_forbidden_path(value):
                raise ReleaseError("release contains staging/latest/tmp reference", "forbidden_path")
        if any(_contains_forbidden_path(row) for row in assets + physical + benchmark):
            raise ReleaseError("release row contains staging/latest/tmp reference", "forbidden_path")

        self._validate_asset_manifest(asset_manifest, paths["asset_ledger"])
        self._validate_physical_manifest(physical_manifest, paths["physical_index"], paths["benchmark_manifest"] if paths["benchmark_manifest"].is_file() else None, paths["method_contract"])
        raw_roots = asset_manifest.get("logical_roots") or physical_manifest.get("logical_roots") or {}
        # References are rooted at the explicit release root, not at the
        # directory containing one authority file.  This also handles a
        # release laid out directly as ``release_manifest.json`` plus
        # ``bundles/`` instead of nested v2/v12 directories.
        artifact_roots: dict[str, Path] = {"fixture": self.root, "release": self.root, "asset": self.root, "physical": self.root}
        artifact_roots.update(self.root_bindings)
        if isinstance(raw_roots, Mapping):
            for alias, value in raw_roots.items():
                if isinstance(alias, str) and isinstance(value, str):
                    if alias in self.root_bindings:
                        artifact_roots[alias] = self.root_bindings[alias]
                    else:
                        candidate = Path(value)
                        if candidate.is_absolute():
                            raise ReleaseError(
                                f"logical root {alias} must use an explicit root binding",
                                "absolute_root_reference",
                            )
                        artifact_roots[alias] = (self.root / candidate).resolve() if not candidate.is_absolute() else candidate
        self._validate_rows(assets, physical, benchmark, asset_manifest=asset_manifest, physical_manifest=physical_manifest, artifact_roots=artifact_roots, method_hash=sha256_file(paths["method_contract"]))
        self._drain_hashes()
        _validate_method(method)

        roots = {}
        if isinstance(raw_roots, Mapping):
            for alias, value in raw_roots.items():
                if isinstance(alias, str) and isinstance(value, str):
                    roots[alias] = value
        synthetic = bool(asset_manifest.get("synthetic") or physical_manifest.get("synthetic") or method.get("synthetic"))
        if synthetic and not self.allow_synthetic:
            raise ReleaseError("synthetic fixture release is disabled", "synthetic_release")
        source_hashes = {key: sha256_file(path) for key, path in paths.items() if path is not None and path.is_file()}
        if source_join_hash is not None:
            source_hashes["benchmark_join_map"] = source_join_hash
        return ReleaseBundle(
            asset_manifest=asset_manifest,
            assets=tuple(assets),
            physical_manifest=physical_manifest,
            physical=tuple(physical),
            benchmark_manifest=tuple(benchmark),
            method_contract=method,
            roots=roots,
            source_hashes=source_hashes,
            synthetic=synthetic,
            benchmark_join_report=benchmark_join_report,
        )

    load = read
    validate = read

    def _validate_asset_manifest(self, manifest: Mapping[str, Any], ledger_path: Path) -> None:
        require(manifest, "schema_name", "schema_version", "release_id", "publishable", "population_count")
        if manifest.get("schema_name") != "softsim.asset_release_manifest" or manifest.get("schema_version") != 2:
            raise ReleaseError("asset release must be schema v2", "asset_release_schema")
        require_id(manifest["release_id"], "asset release_id")
        try:
            if int(manifest["population_count"]) <= 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ReleaseError("asset population_count must be a positive integer", "asset_release_schema") from exc
        if not bool(manifest.get("publishable")):
            raise BlockedRelease("asset release is not publishable", "blocked_asset_release_not_publishable")
        ledger_hash = manifest.get("ledger_sha256") or manifest.get("asset_ledger_sha256")
        if ledger_hash is None and isinstance(manifest.get("ledger_ref"), Mapping):
            ledger_hash = manifest["ledger_ref"].get("sha256")
        if ledger_hash is None:
            raise ReleaseError("asset release manifest must carry ledger_sha256", "asset_manifest_hash_missing")
        _verify_file_hash(ledger_path, ledger_hash, "ledger_sha256")

    def _validate_physical_manifest(self, manifest: Mapping[str, Any], index_path: Path, benchmark_path: Path | None = None, method_path: Path | None = None) -> None:
        require(manifest, "schema_name", "schema_version", "release_id", "publishable", "population_count")
        if manifest.get("schema_name") != "softsim.physical_release_manifest" or manifest.get("schema_version") != 12:
            raise ReleaseError("physical release must be schema v12", "physical_release_schema")
        require_id(manifest["release_id"], "physical release_id")
        if manifest.get("physical_release_id") is not None and str(manifest.get("physical_release_id")) != str(manifest["release_id"]):
            raise ReleaseError("physical_release_id does not match release_id", "physical_release_id_mismatch")
        if not bool(manifest.get("publishable")):
            raise BlockedRelease("physical release is not publishable", "blocked_physical_release_not_publishable")
        index_hash = manifest.get("index_sha256") or manifest.get("physical_bundle_index_sha256")
        if index_hash is None and isinstance(manifest.get("index_ref"), Mapping):
            index_hash = manifest["index_ref"].get("sha256")
        if index_hash is None:
            raise ReleaseError("physical release manifest must carry index_sha256", "physical_manifest_hash_missing")
        _verify_file_hash(index_path, index_hash, "index_sha256")
        if benchmark_path is not None and benchmark_path.is_file() and manifest.get("benchmark_manifest_sha256") is not None:
            _verify_file_hash(benchmark_path, manifest["benchmark_manifest_sha256"], "benchmark_manifest_sha256")
        if method_path is not None and method_path.is_file() and manifest.get("method_contract_sha256") is not None:
            _verify_file_hash(method_path, manifest["method_contract_sha256"], "method_contract_sha256")

    def _validate_rows(self, assets: list[dict[str, Any]], physical: list[dict[str, Any]], benchmark: list[dict[str, Any]], *, asset_manifest: Mapping[str, Any] | None = None, physical_manifest: Mapping[str, Any] | None = None, artifact_roots: Mapping[str, Path] | None = None, method_hash: str | None = None) -> None:
        if not assets:
            raise ReleaseError("asset ledger is empty", "asset_release_incomplete")
        asset_ids: set[tuple[str, str]] = set()
        for row in assets:
            try:
                validate_asset_row(row)
            except SchemaError as exc:
                raise ReleaseError(str(exc), "asset_row_invalid") from exc
            key = (str(row["sample_id"]), str(row["asset_revision_id"]))
            if str(row.get("dataset_release_id")) != str(asset_manifest.get("release_id") if asset_manifest else row.get("dataset_release_id")):
                raise ReleaseError(f"asset row dataset release mismatch: {key}", "asset_release_id_mismatch")
            if key in asset_ids:
                raise ReleaseError(f"duplicate asset revision: {key}", "duplicate_asset")
            asset_ids.add(key)
            if row["record_sha256"] not in {_row_hash(row), _row_hash_legacy_utf8(row)}:
                raise ReleaseError(f"asset row hash mismatch: {key}", "hash_mismatch")
            try:
                self._verify_artifact_ref(row["geometry_ref"], artifact_roots or {"fixture": self.root}, "geometry_ref")
            except ReleaseError:
                raise
        if not physical:
            raise BlockedRelease("physical bundle index is empty", "blocked_missing_physical_release")
        if physical_manifest is not None and physical_manifest.get("population_count") is not None:
            try:
                expected_population = int(physical_manifest["population_count"])
            except (TypeError, ValueError) as exc:
                raise ReleaseError("physical population_count is invalid", "physical_release_schema") from exc
            if expected_population != len(physical):
                raise ReleaseError(f"physical index count {len(physical)} != population_count {expected_population}", "physical_release_incomplete")
            declared_counts = {
                "verified_count": sum(row.get("status") == "verified" for row in physical),
                "rejected_count": sum(row.get("status") == "rejected" for row in physical),
                "task_failure_count": sum(row.get("status") == "task_failure" for row in physical),
            }
            for field, actual in declared_counts.items():
                if field in physical_manifest and int(physical_manifest[field]) != actual:
                    if any(str(row.get("status", "")).lower() in {"predicted", "predicted_prior", "calibrated", "calibrated_candidate"} for row in physical):
                        raise ReleaseError("predicted or intermediate bundle cannot enter release", "predicted_bundle")
                    raise ReleaseError(f"physical manifest {field}={physical_manifest[field]} disagrees with index ({actual})", "physical_release_incomplete")
        if asset_manifest is not None and asset_manifest.get("population_count") is not None:
            try:
                expected_assets = int(asset_manifest["population_count"])
            except (TypeError, ValueError) as exc:
                raise ReleaseError("asset population_count is invalid", "asset_release_schema") from exc
            if expected_assets != len(assets):
                raise ReleaseError(f"asset ledger count {len(assets)} != population_count {expected_assets}", "asset_release_incomplete")
            if "released_count" in asset_manifest and int(asset_manifest["released_count"]) != len(assets):
                raise ReleaseError("asset release released_count disagrees with ledger", "asset_release_incomplete")
        physical_keys: set[tuple[str, str]] = set()
        physical_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        method_contract_path = self._discover()["method_contract"]
        canonical_method_hash = fingerprint(read_json(method_contract_path)) if method_contract_path.is_file() else None
        for row in physical:
            try:
                validate_physical_row(row, allow_nonverified=True)
            except SchemaError as exc:
                # Predicted/prior-only bundles get a dedicated rejection code.
                code = "predicted_bundle" if str(row.get("status", "")).lower() in {"predicted", "predicted_prior", "calibrated"} else "physical_row_invalid"
                raise ReleaseError(str(exc), code) from exc
            if str(row.get("status", "")).lower() in {"predicted", "predicted_prior", "calibrated", "calibrated_candidate"}:
                raise ReleaseError("predicted or intermediate bundle cannot enter release", "predicted_bundle")
            key = (str(row["sample_id"]), str(row["asset_revision_id"]))
            if key in physical_keys:
                raise ReleaseError(f"duplicate physical revision: {key}", "duplicate_physical")
            physical_keys.add(key)
            physical_by_key[key] = row
            if row["record_sha256"] not in {_row_hash(row), _row_hash_legacy_utf8(row)}:
                raise ReleaseError(f"physical row hash mismatch: {key}", "hash_mismatch")
            bundle = row.get("bundle")
            if bundle is not None:
                if not isinstance(bundle, Mapping):
                    raise ReleaseError(f"physical bundle reference is malformed: {key}", "physical_row_invalid")
                bundle_path = bundle.get("relative_path") or bundle.get("path")
                bundle_hash = bundle.get("sha256")
                bundle_ref = {"root_alias": bundle.get("root_alias", "release"), "relative_path": bundle_path, "sha256": bundle_hash}
                try:
                    validate_artifact_ref(bundle_ref, "bundle")
                except SchemaError as exc:
                    raise ReleaseError(str(exc), "physical_artifact_invalid") from exc
                self._verify_artifact_ref(bundle_ref, artifact_roots or {"release": self.root}, "bundle")
            if row.get("method_contract_hash") is None:
                raise ReleaseError(f"physical row missing method_contract_hash: {key}", "method_hash_missing")
            try:
                require_hash(row.get("method_contract_hash"), "method_contract_hash")
            except SchemaError as exc:
                raise ReleaseError(str(exc), "method_hash_missing") from exc
            trace_fields = (("agent_trace_ref", "trace_ref"),)
            if row.get("status") == "verified":
                trace_fields = (("agent_trace_ref", "trace_ref"), "calibration_evidence_ref", "heldout_evidence_ref")
            for field_spec in trace_fields:
                if isinstance(field_spec, tuple):
                    field = next((name for name in field_spec if row.get(name) is not None), field_spec[0])
                else:
                    field = field_spec
                ref = row.get(field)
                try:
                    validate_artifact_ref(ref, field)
                except SchemaError as exc:
                    raise ReleaseError(str(exc), "trace_missing") from exc
                self._verify_artifact_ref(ref, artifact_roots or {"fixture": self.root}, field)
            if method_hash is not None and row.get("method_contract_hash"):
                declared = str(row.get("method_contract_hash"))
                if declared not in {method_hash, canonical_method_hash}:
                    raise ReleaseError(f"method contract hash mismatch for {key}", "method_hash_mismatch")
        asset_set = {(str(row["sample_id"]), str(row["asset_revision_id"])) for row in assets}
        missing = asset_set - physical_keys
        if missing:
            raise ReleaseError(f"physical release incomplete; missing {len(missing)} asset revisions", "physical_release_incomplete")
        if benchmark:
            seen: set[str] = set()
            for row in benchmark:
                instance_id = row.get("benchmark_instance_id") or row.get("instance_id")
                if not isinstance(instance_id, str) or not instance_id:
                    raise ReleaseError("benchmark row missing benchmark_instance_id", "benchmark_row_invalid")
                if instance_id in seen:
                    raise ReleaseError(f"duplicate benchmark instance: {instance_id}", "duplicate_benchmark")
                seen.add(instance_id)
                if row.get("instance_hash") is None:
                    raise ReleaseError(f"benchmark instance hash missing: {instance_id}", "benchmark_hash_missing")
                if row.get("instance_hash") not in {
                    _row_hash(row, "instance_hash"),
                    _row_hash_legacy_utf8(row, "instance_hash"),
                }:
                    raise ReleaseError(f"benchmark instance hash mismatch: {instance_id}", "benchmark_hash_mismatch")
                if row.get("qualification_status") != "qualified" or row.get("formal_e3_a_manifest") is not True:
                    raise ReleaseError(f"benchmark instance not formally qualified/promoted: {instance_id}", "benchmark_not_qualified")
                if (
                    physical_manifest is not None
                    and row.get("physical_release_id") is not None
                    and str(row.get("physical_release_id")) != str(physical_manifest.get("release_id"))
                ):
                    raise ReleaseError(f"benchmark instance physical release mismatch: {instance_id}", "benchmark_physical_join_mismatch")
                if row.get("physical_join_status") != "joined_verified" or row.get("formal_join_ready") is not True:
                    raise ReleaseError(f"benchmark instance has no verified physical join: {instance_id}", "benchmark_physical_join_missing")
                sample_id = row.get("sample_id")
                asset_revision_id = row.get("asset_revision_id")
                physical_bundle_id = row.get("physical_bundle_id")
                if not sample_id or not asset_revision_id or not physical_bundle_id:
                    raise ReleaseError(f"benchmark instance physical identity incomplete: {instance_id}", "benchmark_physical_join_invalid")
                physical_row = physical_by_key.get((str(sample_id), str(asset_revision_id)))
                if physical_row is None or physical_row.get("status") != "verified" or str(physical_row.get("physical_bundle_id")) != str(physical_bundle_id):
                    raise ReleaseError(f"benchmark instance physical bundle does not match verified index: {instance_id}", "benchmark_physical_join_mismatch")
                if row.get("physical_bundle_record_sha256") is not None and row.get("physical_bundle_record_sha256") != physical_row.get("record_sha256"):
                    raise ReleaseError(f"benchmark instance physical record hash mismatch: {instance_id}", "benchmark_physical_join_mismatch")
                if (str(sample_id), str(asset_revision_id)) not in asset_set:
                    raise ReleaseError(f"benchmark instance references unknown asset revision: {instance_id}", "benchmark_physical_join_mismatch")

    def _verify_artifact_ref(self, ref: Mapping[str, Any], roots: Mapping[str, Path], field: str) -> None:
        alias = str(ref.get("root_alias"))
        relative = str(ref.get("relative_path"))
        base = roots.get(alias)
        if base is None:
            raise ReleaseError(f"{field} uses undeclared root alias: {alias}", "artifact_root_missing")
        if not getattr(self, "verify_content_hash", True):
            # Plan-time fast path: alias structure only.  Existence and content
            # hash are covered by the inventory-level full validation and by
            # the executor's per-episode mesh sha256 check.
            return
        if self._reader_workers() <= 1:
            self._verify_artifact_ref_serial(base, relative, ref.get("sha256"), field, verify_hash=getattr(self, "verify_content_hash", True))
        else:
            # Deferred: full per-ref verification (symlink walk, escape,
            # existence, content hash) runs in the ordered thread-pool drain.
            self._pending_hashes.append((base, relative, ref.get("sha256"), field))

    @staticmethod
    def _verify_artifact_ref_serial(base: Path, relative: str, expected: Any, field: str, *, verify_hash: bool = True) -> None:
        raw_path = base / relative
        current = base
        for component in Path(relative).parts:
            current = current / component
            if current.is_symlink():
                raise ReleaseError(f"{field} artifact is a symlink: {relative}", "symlink_input")
        path = raw_path.resolve()
        try:
            path.relative_to(base.resolve())
        except ValueError as exc:
            raise ReleaseError(f"{field} escapes logical root", "artifact_path_escape") from exc
        if not path.is_file() or path.is_symlink():
            raise ReleaseError(f"{field} artifact is missing: {relative}", "artifact_missing")
        if verify_hash and sha256_file(path) != expected:
            raise ReleaseError(f"{field} artifact hash mismatch: {relative}", "artifact_hash_mismatch")

    def _reader_workers(self) -> int:
        return getattr(self, "hash_workers", 1)

    def _drain_hashes(self) -> None:
        """Verify deferred artifact refs in deterministic append order."""
        jobs = self._pending_hashes
        self._pending_hashes = []
        if not jobs:
            return
        workers = self._reader_workers()
        verify_hash = getattr(self, "verify_content_hash", True)
        if workers <= 1:
            for base, relative, expected, field in jobs:
                self._verify_artifact_ref_serial(base, relative, expected, field, verify_hash=verify_hash)
            return
        from concurrent.futures import ThreadPoolExecutor

        def check(job: tuple[Path, str, Any, str]) -> ReleaseError | None:
            base, relative, expected, field = job
            try:
                self._verify_artifact_ref_serial(base, relative, expected, field, verify_hash=verify_hash)
            except ReleaseError as exc:
                return exc
            return None

        with ThreadPoolExecutor(max_workers=workers) as pool:
            errors = list(pool.map(check, jobs))
        for error in errors:
            if error is not None:
                raise error

    @staticmethod
    def _join_benchmark_rows(
        benchmark: list[dict[str, Any]],
        join_rows: list[dict[str, Any]],
        physical: list[dict[str, Any]],
        *,
        physical_release_id: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        by_bundle: dict[str, list[dict[str, Any]]] = {}
        by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in physical:
            by_key.setdefault((str(row.get("sample_id")), str(row.get("asset_revision_id"))), []).append(row)
            if row.get("physical_bundle_id") is not None:
                by_bundle.setdefault(str(row["physical_bundle_id"]), []).append(row)
        output: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        joined = 0
        counts = {"joined_verified": 0, "joined_nonverified": 0, "missing": 0, "ambiguous": 0}
        direct: dict[str, list[dict[str, Any]]] = {}
        for row in join_rows:
            instance_id = row.get("benchmark_instance_id") or row.get("instance_id")
            if instance_id:
                direct.setdefault(str(instance_id), []).append(row)
            else:
                errors.append({"benchmark_instance_id": "", "code": "mapping_key_missing"})
        benchmark_ids = {
            str(row.get("benchmark_instance_id") or row.get("instance_id") or "")
            for row in benchmark
        }
        for mapped_id in sorted(set(direct) - benchmark_ids):
            errors.append({"benchmark_instance_id": mapped_id, "code": "mapping_unknown_instance"})
        for original in benchmark:
            row = dict(original)
            instance_id = str(row.get("benchmark_instance_id") or row.get("instance_id") or "")
            # A join produces a derived benchmark manifest.  Keep the immutable
            # pre-join digest for provenance, then rehash the derived row after
            # its exact physical identity has been attached.
            source_instance_hash = row.get("instance_hash")
            if source_instance_hash is not None:
                expected_source_hashes = {
                    _row_hash(row, "instance_hash"),
                    _row_hash_legacy_utf8(row, "instance_hash"),
                }
                if source_instance_hash not in expected_source_hashes:
                    errors.append({"benchmark_instance_id": instance_id, "code": "source_instance_hash_mismatch"})
                row["source_instance_hash"] = source_instance_hash
            mapping = direct.get(instance_id, [])
            candidates: list[dict[str, Any]] = []
            mapping_invalid = False
            if len(mapping) == 1:
                item = mapping[0]
                mapping_release = item.get("physical_release_id")
                if (
                    mapping_release is not None
                    and physical_release_id
                    and str(mapping_release) != str(physical_release_id)
                ):
                    errors.append({"benchmark_instance_id": instance_id, "code": "physical_release_mismatch"})
                    item = None
                    mapping_invalid = True
                if item is not None and item.get("physical_bundle_id") is not None:
                    candidates = by_bundle.get(str(item["physical_bundle_id"]), [])
                elif item is not None and item.get("sample_id") is not None and item.get("asset_revision_id") is not None:
                    candidates = by_key.get((str(item["sample_id"]), str(item["asset_revision_id"])), [])
            elif len(mapping) > 1:
                errors.append({"benchmark_instance_id": instance_id, "code": "ambiguous_join_mapping"})
            if not mapping_invalid and not candidates and row.get("physical_bundle_id") is not None:
                candidates = by_bundle.get(str(row["physical_bundle_id"]), [])
            if len(candidates) == 1:
                physical_row = candidates[0]
                if len(mapping) == 1 and not mapping_invalid:
                    item = mapping[0]
                    identity_mismatch = any(
                        item.get(field) is not None
                        and str(item.get(field)) != str(physical_row.get(field))
                        for field in ("sample_id", "asset_revision_id", "physical_bundle_id")
                    )
                    if identity_mismatch:
                        errors.append({"benchmark_instance_id": instance_id, "code": "physical_identity_mismatch"})
                        candidates = []
                        physical_row = None
                if physical_row is not None:
                    row["physical_bundle_record_sha256"] = physical_row.get("record_sha256")
            if len(candidates) == 1:
                physical_row = candidates[0]
                row_release = row.get("physical_release_id")
                if row_release is not None and physical_release_id and str(row_release) != str(physical_release_id):
                    errors.append({"benchmark_instance_id": instance_id, "code": "physical_release_mismatch"})
                    candidates = []
                    physical_row = None
            if len(candidates) == 1:
                physical_row = candidates[0]
                row["sample_id"] = str(physical_row.get("sample_id"))
                row["asset_revision_id"] = str(physical_row.get("asset_revision_id"))
                row["physical_bundle_id"] = str(physical_row.get("physical_bundle_id"))
                row["physical_bundle_status"] = physical_row.get("status")
                row["physical_join_status"] = "joined_verified" if physical_row.get("status") == "verified" else "joined_nonverified"
                row["formal_join_ready"] = physical_row.get("status") == "verified"
                joined += 1
                counts[row["physical_join_status"]] += 1
            else:
                row["physical_join_status"] = "missing" if not candidates else "ambiguous"
                row["physical_bundle_status"] = row["physical_join_status"]
                row["formal_join_ready"] = False
                counts[row["physical_join_status"]] += 1
                errors.append({"benchmark_instance_id": instance_id, "code": row["physical_join_status"]})
            row["instance_hash"] = _row_hash(row, "instance_hash")
            output.append(row)
        all_joined_verified = counts["joined_verified"] == len(output) and not errors
        report = {
            "schema_name": "fragile-soft.physical-join-report",
            "schema_version": "fragile-soft.physical-join.v1",
            "physical_release_id": physical_release_id,
            "index_count": len(physical),
            "instance_count": len(output),
            "counts": counts,
            "errors": errors,
            "all_joined_verified": all_joined_verified,
            # Compatibility aliases for early Phase-I readers.
            "joined_count": joined,
            "all_joined": all_joined_verified,
        }
        return output, report


def read_release(*args: Any, **kwargs: Any) -> ReleaseBundle:
    return ReleaseReader(*args, **kwargs).read()
