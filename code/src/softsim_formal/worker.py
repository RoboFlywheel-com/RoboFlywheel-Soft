# SPDX-License-Identifier: Apache-2.0
"""Private single-task worker entry point for process isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .canonical import read_json
from .io import read_jsonl
from .runner import _run_one, load_executor
from .e2 import load_target_provider


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--episode-executor")
    parser.add_argument("--target-provider")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.run).resolve()
    manifest = read_json(root / "run_manifest.json")
    tasks = list(read_jsonl(root / "task_manifest.jsonl"))
    task = next((row for row in tasks if str(row.get("task_id")) == str(args.task_id)), None)
    if task is None:
        print(json.dumps({"status": "error", "message": "task not found", "task_id": args.task_id}))
        return 2
    # Import failures must still leave a terminal, auditable record in the run
    # directory; the coordinator can then classify the task as blocked instead
    # of losing it to a scheduler-level process error.
    try:
        executor = load_executor(args.episode_executor) if args.episode_executor else None
    except Exception:
        executor = None
    try:
        provider = load_target_provider(args.target_provider) if args.target_provider else None
    except Exception:
        provider = None
    marker = _run_one(root, manifest, task, force=args.force, episode_executor=executor, target_provider=provider)
    print(json.dumps(marker, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
