# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .glb import audit_glb, prune_glb_components
from .io import atomic_write_json, content_id, sha256_file


def _family_limits(record: dict[str, Any], qa_config: dict[str, Any]) -> tuple[int, int]:
    family = str(record.get("geometry_family") or "unknown")
    profile = (qa_config.get("families") or {}).get(family, {})
    return (
        int(profile.get("max_faces", qa_config.get("max_faces", 2_000_000))),
        int(profile.get("max_components", qa_config.get("max_components", 128))),
    )


def repair_topology_one(
    inventory_record: dict[str, Any],
    audit_record: dict[str, Any],
    *,
    overlay_root: Path,
    qa_config: dict[str, Any],
    min_face_retention: float,
) -> dict[str, Any]:
    sample_id = str(inventory_record["sample_id"])
    if str(audit_record.get("sample_id")) != sample_id:
        raise ValueError("inventory_audit_sample_mismatch")
    if audit_record.get("status") != "qa_failed" or audit_record.get("failure_codes") != [
        "component_budget_exceeded"
    ]:
        raise ValueError("topology_repair_requires_component_only_failure")
    placements = inventory_record.get("placements") or []
    if len(placements) != 1 or placements[0].get("root_alias") != "primary":
        raise ValueError("topology_repair_requires_one_primary_mesh")
    source = Path(str(placements[0]["absolute_path"]))
    source_sha = sha256_file(source)
    expected_sha = str((audit_record.get("artifact") or {}).get("sha256") or "")
    if expected_sha and source_sha != expected_sha:
        raise ValueError("source_mesh_hash_mismatch")

    max_faces, max_components = _family_limits(inventory_record, qa_config)
    fingerprint = content_id(
        {
            "schema": "softsim.topology_repair.v1",
            "sample_id": sample_id,
            "source_sha256": source_sha,
            "max_faces": max_faces,
            "max_components": max_components,
            "min_face_retention": min_face_retention,
        }
    )
    revision = fingerprint.split(":", 1)[1]
    revision_dir = overlay_root / "samples" / sample_id / "revisions" / revision
    staging_root = overlay_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{sample_id}.{revision[:12]}.", dir=staging_root))
    try:
        relative_mesh = f"mesh/{sample_id}.glb"
        mesh_path = temporary / relative_mesh
        repair = prune_glb_components(
            source,
            mesh_path,
            max_components=max_components,
            min_face_retention=min_face_retention,
        )
        if repair["status"] == "retention_below_threshold":
            return {
                "sample_id": sample_id,
                "status": "deferred",
                "repair_action": "trellis_regenerate",
                "source_sha256": source_sha,
                "repair": repair,
            }
        if repair["status"] != "repaired":
            raise ValueError(f"unexpected_topology_repair_status:{repair['status']}")

        audit = audit_glb(mesh_path)
        if audit["geometry"]["faces"] > max_faces:
            audit["failure_codes"].append("face_budget_exceeded")
        if audit["geometry"]["components"] > max_components:
            audit["failure_codes"].append("component_budget_exceeded")
        audit["failure_codes"] = sorted(set(audit["failure_codes"]))
        audit["status"] = "pass" if not audit["failure_codes"] else "fail"
        if audit["status"] != "pass":
            raise ValueError("repaired_mesh_qa_failed:" + ",".join(audit["failure_codes"]))
        atomic_write_json(temporary / "qa.json", {**audit, "repair": repair})
        artifacts = [
            {
                "role": "visual_mesh",
                "relative_to_revision": relative_mesh,
                "sha256": sha256_file(mesh_path),
                "size_bytes": mesh_path.stat().st_size,
                "media_type": "model/gltf-binary",
            },
            {
                "role": "topology_qa",
                "relative_to_revision": "qa.json",
                "sha256": sha256_file(temporary / "qa.json"),
                "size_bytes": (temporary / "qa.json").stat().st_size,
                "media_type": "application/json",
            },
        ]
        marker = {
            "schema_name": "softsim.topology_repair_success",
            "schema_version": 1,
            "sample_id": sample_id,
            "task_fingerprint": fingerprint,
            "revision": revision,
            "source": {
                "root_alias": "primary",
                "path": placements[0]["path"],
                "sha256": source_sha,
            },
            "artifacts": artifacts,
            "repair": repair,
            "geometry": audit["geometry"],
            "producer_version": "1.0.0",
        }
        atomic_write_json(temporary / "SUCCESS.json", marker)
        revision_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, revision_dir)
        return {"sample_id": sample_id, "status": "repaired", **marker}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
