# SPDX-License-Identifier: Apache-2.0
"""Small explicit task state machine used by local and scheduler runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


STATES = {"planned", "running", "completed", "failed", "invalid", "blocked"}
TERMINAL = {"completed", "failed", "invalid", "blocked"}
TRANSITIONS = {
    "planned": {"running", "blocked", "invalid"},
    "running": {"completed", "failed", "invalid"},
    "completed": set(),
    "failed": set(),
    "invalid": set(),
    "blocked": set(),
}


@dataclass
class TaskState:
    task_id: str
    state: str = "planned"
    attempt: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def transition(self, state: str, **metadata: Any) -> None:
        if state not in STATES:
            raise ValueError(f"unknown task state: {state}")
        if state not in TRANSITIONS[self.state]:
            raise ValueError(f"invalid task transition {self.state}->{state}")
        self.history.append({"from": self.state, "to": state, **metadata})
        self.state = state

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL

    def can_resume(self) -> bool:
        # A valid task failure is data, not an implicit retry request.
        return self.state in {"planned"}
