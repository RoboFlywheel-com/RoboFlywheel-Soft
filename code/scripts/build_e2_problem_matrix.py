#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the E2.2 controlled-recovery problem matrix candidate.

Scans verified v12.6 bundles, groups by canonical solver family x material
family, and selects (deterministically by hash) up to 6 assets per family for
the 90-problem matrix (5 families x 6 assets x 3 hidden parameter sets).
The output is a REVIEW CANDIDATE: the formal freeze (parameter-response gate,
d<=4 open parameters per problem) is a release decision; hidden targets
are generated evaluator-side afterwards via ``softsim_targets build``.

Usage:
    python3 code/scripts/build_e2_problem_matrix.py \
        --physical-root ./data/physical_release_v12 \
        --output ./runs/e2-matrix/matrix_candidate.jsonl \
        --report ./runs/e2-matrix/matrix_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

CANONICAL_SOLVER = {
    "genesis_mpm_elastoplastic": "MPM.ElastoPlastic",
    "genesis_mpm_elastic": "MPM.Elastic",
    "genesis_fem_cloth_ipc": "PBD.Cloth",
    "genesis_pbd_cloth": "PBD.Cloth",
    "genesis_fem_elastic": "MPM.Elastic",
    "genesis_pbd_elastic": "PBD.Elastic",
}
MUTABLE_DEFAULT = ["youngs_modulus_pa", "density_kg_m3", "poisson_ratio", "yield_stress_pa"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--families", type=int, default=5)
    parser.add_argument("--per-family", type=int, default=6)
    parser.add_argument("--include-groups", nargs="*", default=None,
                        help="explicit solver|material groups to select (release freeze); overrides size-based picking")
    args = parser.parse_args()
    root = Path(args.physical_root)
    bundles = root / "bundles"
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    scanned = skipped = 0
    for entry in sorted(bundles.iterdir()):
        physics_path = entry / "physics.json"
        if not physics_path.is_file():
            continue
        scanned += 1
        try:
            physics = json.loads(physics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            skipped += 1
            continue
        if physics.get("status") not in {"accepted_first_pass", "corrected_then_accepted"}:
            continue
        route = physics.get("route") or {}
        solver = CANONICAL_SOLVER.get(str(route.get("model_family")), "other")
        family = str(route.get("material_family") or "unknown")
        params = (physics.get("solver_adapter") or {}).get("parameters") or {}
        open_params = [name for name in MUTABLE_DEFAULT if isinstance(params.get(name), (int, float))]
        if len(open_params) < 2:
            continue
        groups[(solver, family)].append({
            "sample_id": physics.get("sample_id") or entry.name,
            "asset_revision_id": physics.get("asset_revision_id"),
            "physical_bundle_id": physics.get("physical_bundle_id"),
            "solver_family": solver,
            "material_family": family,
            "open_parameters": open_params[:4],
            "parameters": {k: params[k] for k in open_params[:4]},
        })
    report = {"scanned": scanned, "skipped_unreadable": skipped, "family_sizes": {f"{k[0]}|{k[1]}": len(v) for k, v in sorted(groups.items())}}
    matrix: list[dict] = []
    if args.include_groups:
        wanted = [tuple(g.split("|", 1)) for g in args.include_groups]
        selected_groups = [g for g in wanted if g in groups]
        missing = [g for g in wanted if g not in groups]
        if missing:
            raise SystemExit(f"requested groups absent from release: {missing}")
    else:
        selected_groups = [k for k, _ in sorted(groups.items(), key=lambda kv: -len(kv[1]))[: args.families]]
    for key in selected_groups:
        rows = groups[key]
        ranked = sorted(rows, key=lambda r: hashlib.sha256(str(r["sample_id"]).encode()).hexdigest())
        matrix.extend(ranked[: args.per_family])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in matrix:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    report["matrix_rows"] = len(matrix)
    report["status"] = "review_candidate_not_frozen"
    Path(args.report).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"matrix_rows": len(matrix), "families": len(groups), "report": args.report}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
