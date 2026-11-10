#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the E2.1 routing reference-label package (300 stratified assets).

The reference labels are a human input, but everything around them is
mechanical: this script selects 300 assets stratified by geometry family
(deterministic by hash), copies their input images, and emits per-rater label
forms with the frozen label schema, a labeling guide, and an agreement
calculator for the 3-rater reconciliation.

Label schema (EXPERIMENT_PLAN §4.3):  correct_representation /
constitutive_family / adapter / repairability / final_acceptability /
model_limit.  Three raters fill independent copies; ``--mode agree`` computes
per-field agreement and majority labels.

Usage:
    python3 code/scripts/build_e21_labeling_package.py --mode build \
        --release-root <frozen root> --image-root <image_bank/primary> \
        --output <pkg dir> --count 300
    python3 code/scripts/build_e21_labeling_package.py --mode agree --package <pkg dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

LABEL_FIELDS = {
    "correct_representation": ["watertight_volume", "surface_mesh", "sheet_mid_surface", "closed_shell", "unsupported"],
    "constitutive_family": ["MPM.ElastoPlastic", "MPM.Elastic", "PBD.Cloth", "unsupported"],
    "adapter": ["genesis_mpm", "genesis_pbd", "unsupported"],
    "repairability": ["parameter_correctable", "model_limit", "numerical_failure", "uncertain"],
    "final_acceptability": ["accept", "reject", "uncertain"],
    "model_limit": ["true", "false"],
}

GUIDE = """# E2.1 Routing Reference Label 标注指南

每个资产一行（`label_form.jsonl`），三张表分别由三名专家独立填写，禁止讨论。

逐字段定义：
- `correct_representation`：该资产物理上正确的几何表示（watertight_volume=实心体 /
  surface_mesh=厚壳 / sheet_mid_surface=布料中面 / closed_shell=闭合薄壳 / unsupported=当前
  solver 集合不支持）。
- `constitutive_family`：正确本构族（MPM.ElastoPlastic 可永久塑形 / MPM.Elastic 纯弹性 /
  PBD.Cloth 薄膜布料 / unsupported）。
- `adapter`：正确 solver 路由。
- `repairability`：若初始参数不对，参数修正能否救回（parameter_correctable），还是模型
  能力上限（model_limit）、数值问题（numerical_failure）、无法判断（uncertain）。
- `final_acceptability`：作为 release 资产你最终的接受决定。
- `model_limit`：该资产是否超出当前 proxy 方法（MPM/PBD 无 fracture/无充气）的表达上限。

一致性计算：`--mode agree` 输出逐字段 majority 标签、完全一致比例与 pairwise agreement。
"""


def _build(args: argparse.Namespace) -> int:
    root = Path(args.release_root)
    ledger = root / "asset_release_v2" / "asset_ledger.jsonl"
    physical_index = root / "physical_release_v12" / "physical_bundle_index.jsonl"
    if not ledger.is_file():
        ledger = Path(args.asset_ledger)
    status_by_sample: dict[str, str] = {}
    if physical_index.is_file():
        with open(physical_index, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                status_by_sample[str(row["sample_id"])] = str(row.get("status"))
    by_family: dict[str, list[dict]] = defaultdict(list)
    with open(ledger, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_family[str(row.get("geometry_family") or "unknown")].append(row)
    per_family = max(1, args.count // max(1, len(by_family)))
    selected: list[dict] = []
    for family, rows in sorted(by_family.items()):
        ranked = sorted(rows, key=lambda r: hashlib.sha256(f"e21:{r['sample_id']}".encode()).hexdigest())
        selected.extend(ranked[:per_family])
    selected = selected[: args.count]
    out = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root)
    form_rows = []
    missing_images = 0
    for row in selected:
        sample = str(row["sample_id"])
        image_src = image_root / "samples" / sample / "input_image.png"
        image_rel = f"images/{sample}.png"
        if image_src.is_file():
            shutil.copyfile(image_src, out / image_rel)
        else:
            missing_images += 1
            image_rel = None
        form_rows.append({
            "sample_id": sample,
            "asset_revision_id": row.get("asset_revision_id"),
            "geometry_family": row.get("geometry_family"),
            "object_type": row.get("object_type"),
            "physical_status": status_by_sample.get(sample),
            "image": image_rel,
            **{field: "" for field in LABEL_FIELDS},
            "rater_notes": "",
        })
    for rater in ("rater_a", "rater_b", "rater_c"):
        with open(out / f"label_form.{rater}.jsonl", "w", encoding="utf-8") as handle:
            for row in form_rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    (out / "LABELING_GUIDE_ZH.md").write_text(GUIDE, encoding="utf-8")
    (out / "label_schema.json").write_text(json.dumps(LABEL_FIELDS, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"package": out.as_posix(), "assets": len(form_rows), "missing_images": missing_images,
                      "families": {k: min(per_family, len(v)) for k, v in sorted(by_family.items())}}, indent=1, ensure_ascii=False))
    return 0


def _agree(args: argparse.Namespace) -> int:
    pkg = Path(args.package)
    forms = {}
    for rater in ("rater_a", "rater_b", "rater_c"):
        path = pkg / f"label_form.{rater}.jsonl"
        if path.is_file():
            with open(path, encoding="utf-8") as handle:
                forms[rater] = {json.loads(line)["sample_id"]: json.loads(line) for line in handle if line.strip()}
    raters = sorted(forms)
    if len(raters) < 2:
        print(json.dumps({"error": "need >=2 rater forms"}))
        return 2
    report: dict = {"raters": raters, "fields": {}}
    for field in LABEL_FIELDS:
        total = agree_all = agree_pair = 0
        pairs = 0
        majority: dict[str, str] = {}
        samples = set.intersection(*[set(forms[r]) for r in raters])
        for sample in sorted(samples):
            votes = [forms[r][sample].get(field, "") for r in raters]
            votes = [v for v in votes if v != ""]
            if not votes:
                continue
            total += 1
            top = max(set(votes), key=votes.count)
            majority[sample] = top
            if len(set(votes)) == 1:
                agree_all += 1
            for i in range(len(votes)):
                for j in range(i + 1, len(votes)):
                    pairs += 1
                    agree_pair += votes[i] == votes[j]
        report["fields"][field] = {
            "labeled": total,
            "full_agreement_rate": round(agree_all / max(total, 1), 4),
            "pairwise_agreement": round(agree_pair / max(pairs, 1), 4),
        }
        with open(pkg / f"majority.{field}.jsonl", "w", encoding="utf-8") as handle:
            for sample, value in sorted(majority.items()):
                handle.write(json.dumps({"sample_id": sample, field: value}, ensure_ascii=True) + "\n")
    out = pkg / "agreement_report.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": out.as_posix(), "fields": list(report["fields"])}, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("build", "agree"), default="build")
    parser.add_argument("--release-root")
    parser.add_argument("--asset-ledger")
    parser.add_argument("--image-root")
    parser.add_argument("--package")
    parser.add_argument("--output")
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args()
    return _build(args) if args.mode == "build" else _agree(args)


if __name__ == "__main__":
    raise SystemExit(main())
