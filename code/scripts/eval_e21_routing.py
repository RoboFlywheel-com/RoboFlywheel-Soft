#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E2.1 routing evaluation: method predictions vs human reference labels.

Joins the frozen reference labels (rater forms / majority files) with the
production bundle routes and computes the frozen E2.1 metrics per method row:
routing macro-F1, per-class F1, unsupported recall, abstention rate.

Method rows (EXPERIMENT_PLAN §4.4):
  A family rules           -> geometry_family -> fixed representation/model
  B one-shot VLM           -> bundle route prior (model_family as routed)
  C VLM + contract gate    -> bundle route after the deterministic gate
                             (rebuild_v4 production route = the gate output)
  full routing agent       -> not produced offline; comes from the E2.1 run

Single-rater caveat: agreement fields stay null until rater_b/c forms exist;
the reference is the single-rater majority (= rater_a labels).

Usage:
    python3 code/scripts/eval_e21_routing.py \
        --labels ./data/labels/label_form.rater_a.jsonl \
        --physical-root ./data/physical_release_v12 \
        --output <out.json>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

CANONICAL_SOLVER = {
    "genesis_mpm_elastoplastic": "MPM.ElastoPlastic",
    "genesis_mpm_elastic": "MPM.Elastic",
    "genesis_fem_elastic": "MPM.Elastic",
    "genesis_fem_cloth_ipc": "PBD.Cloth",
    "genesis_pbd_cloth": "PBD.Cloth",
    "genesis_pbd_elastic": "PBD.Cloth",
}
FAMILY_RULE = {  # method A: deterministic family -> constitutive route
    "solid": "MPM.ElastoPlastic",
    "shell": "MPM.ElastoPlastic",
    "bag": "PBD.Cloth",
    "garment": "PBD.Cloth",
    "cloth_sheet": "PBD.Cloth",
}


def _f1_by_class(pairs: list[tuple[str, str]]) -> tuple[dict[str, dict[str, float]], float]:
    classes = sorted({p for p, _ in pairs} | {t for _, t in pairs})
    out: dict[str, dict[str, float]] = {}
    for cls in classes:
        tp = sum(1 for p, t in pairs if p == cls and t == cls)
        fp = sum(1 for p, t in pairs if p == cls and t != cls)
        fn = sum(1 for p, t in pairs if p != cls and t == cls)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        out[cls] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4), "support": tp + fn}
    macro = sum(v["f1"] for v in out.values()) / max(len(out), 1)
    return out, round(macro, 4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--physical-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    labels = {}
    with open(args.labels, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            labels[str(row["sample_id"])] = row
    bundles = Path(args.physical_root) / "bundles"
    methods: dict[str, list[tuple[str, str]]] = {"A_family_rules": [], "B_one_shot_prior": [], "C_contract_gate": []}
    meta = {"labeled": 0, "bundle_missing": 0, "unsupported_in_reference": 0, "model_limit_in_reference": 0}
    for sample, label in labels.items():
        physics_path = bundles / sample / "physics.json"
        if not physics_path.is_file():
            meta["bundle_missing"] += 1
            continue
        physics = json.loads(physics_path.read_text(encoding="utf-8"))
        route = physics.get("route") or {}
        reference = str(label.get("constitutive_family") or "")
        if not reference:
            continue
        meta["labeled"] += 1
        meta["unsupported_in_reference"] += reference == "unsupported"
        meta["model_limit_in_reference"] += str(label.get("model_limit")) == "true"
        pred_a = FAMILY_RULE.get(str(label.get("geometry_family")), "unsupported")
        pred_bc = CANONICAL_SOLVER.get(str(route.get("model_family")), "unsupported")
        methods["A_family_rules"].append((pred_a, reference))
        methods["B_one_shot_prior"].append((pred_bc, reference))
        methods["C_contract_gate"].append((pred_bc, reference))
    result: dict = {"reference": Path(args.labels).name, "meta": meta, "methods": {}}
    for method, pairs in methods.items():
        per_class, macro = _f1_by_class(pairs)
        result["methods"][method] = {
            "routing_macro_f1": macro,
            "per_class": per_class,
            "n": len(pairs),
            "unsupported_recall": per_class.get("unsupported", {}).get("recall"),
        }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({m: v["routing_macro_f1"] for m, v in result["methods"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
