#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the DeliGrasp-style feed-forward commanded-force prior cache.

One small paid Qwen call per object type; the frozen cache
(``deligrasp_prior_cache.jsonl``, one row per sample/object type) is what the
``deligrasp_prior`` E3-A baseline reads at run time.  This is a
DeliGrasp-*style* adaptation: LLM physical prior -> feed-forward commanded
force.  No closed-loop force control (the rigid-MPM force stream does not
exist in the frozen Genesis 1.3.3 coupler).

Usage:
    DASHSCOPE_API_KEY=... python3 code/scripts/gen_deligrasp_prior_cache.py \
        --objects tofu=soft_food tomato=soft_food banana=soft_food porous_cake=soft_food \
        --output <path>/deligrasp_prior_cache.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

SYSTEM = (
    "You are estimating gentle-grasp force priors for a fragile-object benchmark "
    "(parallel-jaw gripper, lift 30 mm and hold). Given an object description, "
    "respond with strict JSON only: "
    '{"force_n": <number>, "rationale": "<one sentence>"}. The force is the '
    "commanded closure effort in newtons, calibrated so that tofu survives at "
    "6-8 N, a tomato at 5-8 N, a banana at 8-12 N, porous cake at 4-6 N, and a "
    "stiff foam block needs 20-30 N."
)


def _api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY") or os.environ.get("QWEN_VL_API_KEY")
    if not key:
        raise SystemExit("set DASHSCOPE_API_KEY before generating API-backed priors")
    return key


def _chat(model: str, prompt: str, timeout: float = 120.0, max_retries: int = 5) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }).encode("utf-8")
    attempt = 0
    while True:
        attempt += 1
        req = urllib.request.Request(
            BASE_URL.rstrip("/") + "/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {_api_key()}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return str(payload["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt <= max_retries:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                time.sleep(float(retry_after) if retry_after else min(60.0, 2.0 ** attempt))
                continue
            raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", nargs="+", required=True, help="NAME=material_family pairs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen3.8-max")
    args = parser.parse_args()
    rows = []
    for spec in args.objects:
        name, _, family = spec.partition("=")
        prompt = f"object: {name}; material family: {family or 'unknown'}; benchmark: fragile grasp-lift-hold"
        text = _chat(args.model, prompt)
        try:
            payload = json.loads(text[text.index("{"): text.rindex("}") + 1])
            force = float(payload["force_n"])
            rationale = str(payload.get("rationale", ""))
        except Exception as exc:
            print(f"skip {name}: unparsable response ({exc})", file=sys.stderr)
            continue
        rows.append({
            "sample_id": name,
            "material_family": family or "unknown",
            "force_n": force,
            "rationale": rationale,
            "prior_source": f"qwen:{args.model}",
            "style": "deligrasp_style_feedforward_prior",
        })
        print(f"{name}: {force} N")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    print(json.dumps({"output": out.as_posix(), "rows": len(rows)}, indent=1))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
