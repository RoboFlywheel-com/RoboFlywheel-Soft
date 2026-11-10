# SPDX-License-Identifier: Apache-2.0
"""Evaluator-side hidden-target sidecar for formal E2 recovery runs.

The public ``task_manifest.jsonl`` carries only ``hidden_target_handle``; the
actual hidden parameters live in a sidecar JSONL that is injected at run time
via ``--target-provider softsim_targets:get_target`` and resolved through the
``SOFTSIM_E2_TARGET_SIDECAR`` environment variable.  The sidecar must never be
copied into run manifests, archives, or publication bundles.

Production E2.2 note: the hidden target is the *simulated behavior of hidden
parameters*; this module provides the evaluator-side parameter lookup.  For
the controlled recovery matrix, hidden parameter sets are derived from the
release prior by a seeded log-space perturbation (`build_sidecar`), so the
reference rollout is executable by the frozen Genesis probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Mapping

SIDECAR_ENV = "SOFTSIM_E2_TARGET_SIDECAR"

_cache: dict[str, dict[str, dict[str, float]]] = {}


class TargetSidecarError(RuntimeError):
    pass


def _load(path: str) -> dict[str, dict[str, float]]:
    if path not in _cache:
        rows: dict[str, dict[str, float]] = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                handle_id = row.get("hidden_target_handle")
                params = row.get("parameters")
                if not handle_id or not isinstance(params, dict):
                    raise TargetSidecarError("sidecar row missing handle/parameters")
                rows[str(handle_id)] = {str(k): float(v) for k, v in params.items()}
        _cache[path] = rows
    return _cache[path]


def get_target(task: Mapping[str, Any]) -> dict[str, float]:
    """Evaluator-side provider (``softsim_targets:get_target``)."""
    sidecar = os.environ.get(SIDECAR_ENV)
    if not sidecar:
        raise TargetSidecarError(f"{SIDECAR_ENV} is not set; the evaluator sidecar is required in production E2")
    handle = task.get("hidden_target_handle")
    rows = _load(sidecar)
    if handle not in rows:
        raise TargetSidecarError("hidden_target_handle not present in sidecar")
    return dict(rows[str(handle)])


def _perturb(prior: Mapping[str, float], seed: int, salt: str) -> dict[str, float]:
    """Deterministic log-space perturbation -> hidden but executable target."""
    out: dict[str, float] = {}
    for name, value in prior.items():
        digest = hashlib.sha256(f"{seed}:{salt}:{name}".encode()).hexdigest()
        u = int(digest[:16], 16) / float(16**16)
        # factor in [1.10, 1.60] for positive scale parameters; additive small
        # jitter for ratio-like parameters in (0, 0.5)
        if 0.0 < float(value) < 0.5:
            out[name] = round(min(0.49, float(value) + 0.05 + 0.1 * u), 8)
        else:
            out[name] = round(float(value) * math.exp(math.log(1.1) + u * math.log(1.6 / 1.1)), 8)
    return out


def build_sidecar(plan_dir: str | Path, *, seed: int, output: str | Path, physical_root: str | Path | None = None) -> dict[str, Any]:
    """Derive hidden targets for every task of a frozen E2 plan.

    Production plans carry no inline prior; the hidden target is then derived
    from the frozen bundle parameters (``source_row.bundle`` under
    ``physical_root``) so the recovery problem is anchored to real bundles.
    """
    plan_dir = Path(plan_dir)
    rows = []
    with open(plan_dir / "task_manifest.jsonl", "r", encoding="utf-8") as handle:
        for line in handle:
            task = json.loads(line)
            handle_id = task.get("hidden_target_handle")
            prior = task.get("prior_parameters") or {}
            if not prior and physical_root is not None:
                bundle = (task.get("source_row") or {}).get("bundle") or {}
                rel = bundle.get("path") or bundle.get("relative_path")
                if rel:
                    physics_path = Path(physical_root) / str(rel)
                    if physics_path.is_file():
                        physics = json.loads(physics_path.read_text(encoding="utf-8"))
                        params = (physics.get("solver_adapter") or {}).get("parameters") or {}
                        prior = {str(k): float(v) for k, v in params.items() if isinstance(v, (int, float))}
            if not handle_id or not prior:
                raise TargetSidecarError(f"task {task.get('task_id')} lacks handle/prior; not an E2 production plan?")
            rows.append({"hidden_target_handle": str(handle_id), "parameters": _perturb(prior, seed, str(task.get("sample_id"))),
                         "sample_id": task.get("sample_id"), "seed": int(seed)})
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
    os.chmod(out_path, stat.S_IRUSR | stat.S_IWUSR)
    return {"sidecar": out_path.as_posix(), "rows": len(rows), "seed": int(seed), "note": "evaluator-side only; never copy into run manifests or archives"}


def main() -> int:
    parser = argparse.ArgumentParser(prog="softsim_targets")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="derive hidden-target sidecar for a frozen E2 plan")
    build.add_argument("--plan", required=True)
    build.add_argument("--seed", type=int, required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--physical-root", help="resolve bundle physics.json priors when tasks carry no inline prior")
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_sidecar(args.plan, seed=args.seed, output=args.output, physical_root=args.physical_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
