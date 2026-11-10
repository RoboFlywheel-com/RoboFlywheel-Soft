# SPDX-License-Identifier: Apache-2.0
"""Worker-isolation helpers used by scheduler adapters.

The local mock runner can execute in-process for speed, but production MPM
episodes must be isolated.  This module provides a small JSON request/response
boundary that a Slurm/local process adapter can invoke without importing a
simulator into the coordinator.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


def run_isolated(run_dir: str | Path, task_id: str, *, python: str | None = None, timeout_s: float | None = None, episode_executor: str | None = None, target_provider: str | None = None, force: bool = False) -> dict[str, Any]:
    command = [python or sys.executable, "-m", "softsim_formal.worker", "--run", str(run_dir), "--task-id", str(task_id)]
    if episode_executor:
        command.extend(["--episode-executor", episode_executor])
    if target_provider:
        command.extend(["--target-provider", target_provider])
    if force:
        command.append("--force")
    environment = os.environ.copy()
    # Source-tree callers (including local smoke) may not have installed the
    # package yet.  Production clean checkouts use site-packages, while this
    # narrow fallback keeps the subprocess boundary usable during development.
    source_root = Path(__file__).resolve().parents[2]
    source_entries = [source_root / "src", source_root] if (source_root / "src" / "softsim_formal").is_dir() else []
    existing = [entry for entry in environment.get("PYTHONPATH", "").split(os.pathsep) if entry]
    for entry in reversed(source_entries):
        if entry.is_dir() and str(entry) not in existing:
            existing.insert(0, str(entry))
    if existing:
        environment["PYTHONPATH"] = os.pathsep.join(existing)
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_s, env=environment)
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -9,
            "stdout": str(exc.stdout or ""),
            "stderr": str(exc.stderr or ""),
            "task_id": task_id,
            "isolated": True,
            "timeout": True,
        }
    except OSError as exc:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "",
            "task_id": task_id,
            "isolated": True,
            "process_error": type(exc).__name__,
        }
    payload: dict[str, Any] = {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "task_id": task_id, "isolated": True}
    if completed.stdout.strip():
        try:
            value = json.loads(completed.stdout.strip().splitlines()[-1])
            if isinstance(value, Mapping):
                payload["result"] = dict(value)
        except json.JSONDecodeError:
            pass
    return payload
