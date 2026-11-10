# SPDX-License-Identifier: Apache-2.0
"""External episode-runner loading contract for formal qualification."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping


class EpisodeRunnerContractError(ValueError):
    """A configured external runner does not satisfy the Phase-I contract."""


def load_episode_runner(reference: str) -> Callable[..., dict[str, Any]]:
    """Load ``module:function`` without importing a simulator in Phase I."""

    if ":" not in reference:
        raise EpisodeRunnerContractError("episode runner must use module:function syntax")
    module_name, function_name = reference.split(":", 1)
    module = importlib.import_module(module_name)
    runner = getattr(module, function_name, None)
    if not callable(runner):
        raise EpisodeRunnerContractError(f"episode runner is not callable: {reference}")
    return runner


def validate_episode_record(record: Mapping[str, Any], *, backend_id: str) -> None:
    required = ("samples", "actions", "action_replay_hash", "backend_id", "backend_status")
    missing = [key for key in required if key not in record]
    if missing:
        raise EpisodeRunnerContractError("episode runner missing fields: " + ", ".join(missing))
    if str(record.get("backend_id")) != backend_id:
        raise EpisodeRunnerContractError(
            f"episode backend_id {record.get('backend_id')!r} != declared {backend_id!r}"
        )
