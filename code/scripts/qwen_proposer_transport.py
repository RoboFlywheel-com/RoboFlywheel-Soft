#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qwen (Bailian / DashScope compatible-mode) transport for softsim_physical_agent.QwenProposer.

QwenProposer invokes this as an external command with {request_json} / {response_json}
placeholders (see code/src/softsim_physical_agent/proposers.py).  Input request JSON:

    {"prior": {...}, "candidate": {name: value, ...},
     "calibration": {"metrics": {...}, "diagnosis": {...}}}

Output response JSON (written to --response):

    {"patch": {param_name: new_value, ...}, "rationale": "..."}

Only physical parameters inside ALLOWED_PHYSICAL_PARAMETERS may appear in the patch;
solver numerics (dt, substeps, grid, seed, ...) are never touched.  stdlib-only so any
python3 can run it. Concurrency is controlled by the caller (workers x shards)
and should respect the API account's own rate limits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.8-max"

ALLOWED_PHYSICAL_PARAMETERS = [
    "density_kg_m3",
    "youngs_modulus_pa",
    "poisson_ratio",
    "yield_stress_pa",
    "yield_lower",
    "yield_higher",
    "plastic_viscosity_pa_s",
    "tensile_cohesion_pa",
    "friction_static",
    "friction_dynamic",
]

SYSTEM_PROMPT = (
    "You are a physical-parameter correction proposer for soft-body simulation "
    "(Genesis MPM/PBD). You receive the prior physical bundle, the current candidate "
    "parameters, and calibration probe metrics/diagnosis. Propose ONE bounded update "
    "to the physical parameters so the next calibration probe better matches the "
    "expected behavior contract. Rules: only adjust keys listed in "
    "allowed_physical_parameters; every patch value must be a finite positive number; "
    "keep changes within one mechanism (fewest keys possible); never touch solver "
    "numerics (dt, substeps, grid, particle size, seed). Respond with strict JSON only: "
    '{"patch": {"<param>": <new_value>}, "rationale": "<one paragraph>"}.'
)


def _resolve_api_key(cli_key: str | None) -> str:
    for candidate in (
        cli_key,
        os.environ.get("DASHSCOPE_API_KEY"),
        os.environ.get("QWEN_API_KEY"),
        os.environ.get("QWEN_VL_API_KEY"),
    ):
        if candidate:
            return candidate
    raise SystemExit("no API key available (env DASHSCOPE_API_KEY or --api-key)")


def _build_user_prompt(request: dict) -> str:
    payload = {
        "prior": request.get("prior"),
        "candidate": request.get("candidate"),
        "calibration": request.get("calibration") or {},
        "allowed_physical_parameters": ALLOWED_PHYSICAL_PARAMETERS,
    }
    return (
        "Decide the next bounded physical-parameter update for this task.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    )


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response contains no JSON object")
    return json.loads(text[start : end + 1])


def _chat_completion(*, base_url: str, api_key: str, model: str, user_prompt: str, timeout: float, max_retries: int) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
    ).encode("utf-8")
    attempt = 0
    while True:
        attempt += 1
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"] and _extract_json_object(
                data["choices"][0]["message"]["content"]
            )
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            detail = exc.read().decode("utf-8", "replace")[:500]
            if not retryable or attempt > max_retries:
                raise SystemExit(f"qwen transport HTTP {exc.code}: {detail}")
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else min(2.0**attempt, 30.0)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
            if attempt > max_retries:
                raise SystemExit(f"qwen transport failed after {attempt} attempts: {type(exc).__name__}: {exc}")
            time.sleep(min(2.0**attempt, 30.0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="path to request JSON from QwenProposer")
    parser.add_argument("--response", required=True, help="path to write response JSON")
    parser.add_argument("--model", default=os.environ.get("SOFTSIM_QWEN_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("SOFTSIM_QWEN_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args(argv)

    request = json.loads(open(args.request, encoding="utf-8").read())
    api_key = _resolve_api_key(args.api_key)
    result = _chat_completion(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        user_prompt=_build_user_prompt(request),
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    patch = result.get("patch")
    if not isinstance(patch, dict) or not patch:
        raise SystemExit("qwen response has no non-empty 'patch' object")
    cleaned = {}
    for name, value in patch.items():
        if name not in ALLOWED_PHYSICAL_PARAMETERS:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number != number or number in (float("inf"), float("-inf")) or number <= 0:
            continue
        cleaned[name] = number
    if not cleaned:
        raise SystemExit("qwen patch has no usable physical parameters after validation")
    output = {"patch": cleaned, "rationale": str(result.get("rationale") or "qwen bounded update")}
    with open(args.response, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"status": "ok", "patched": sorted(cleaned)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
