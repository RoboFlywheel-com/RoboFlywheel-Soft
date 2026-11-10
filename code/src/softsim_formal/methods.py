# SPDX-License-Identifier: Apache-2.0
"""Small, auditable E2 method adapters.

These adapters operate on scalar parameter dictionaries and are intended for
controlled hidden-target smoke tests.  They do not download a model or call an
external policy service.  The external baseline adapter is deliberately
disabled until checkpoint, data, and source hashes are supplied.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .canonical import fingerprint


@dataclass(frozen=True)
class Candidate:
    parameters: dict[str, float]
    evaluation_index: int
    source: str
    candidate_hash: str


class MethodAdapter:
    method_id = "base"

    def initialize(self, problem: Mapping[str, Any], budget: int) -> None:
        self.problem = dict(problem)
        self.budget = int(budget)
        self.history: list[tuple[Candidate, float]] = []

    def suggest(self, history: Sequence[tuple[Candidate, float]] | None = None) -> Candidate:
        raise NotImplementedError

    def observe(self, candidate: Candidate, loss: float) -> None:
        self.history.append((candidate, float(loss)))

    def finalize(self) -> Candidate:
        if not self.history:
            return self.suggest([])
        return min(self.history, key=lambda item: item[1])[0]


class FixedProfileAdapter(MethodAdapter):
    method_id = "fixed_profile"

    def suggest(self, history=None) -> Candidate:
        values = {str(k): float(v) for k, v in (self.problem.get("prior") or {}).items()}
        return Candidate(values, len(self.history), self.method_id, fingerprint(values))


class RuleTuneAdapter(MethodAdapter):
    method_id = "ruletune"

    def suggest(self, history=None) -> Candidate:
        prior = {str(k): float(v) for k, v in (self.problem.get("prior") or {}).items()}
        target = {str(k): float(v) for k, v in (self.problem.get("hidden_target") or {}).items()}
        values = dict(prior)
        if history:
            for name, value in target.items():
                values[name] = prior.get(name, value) + 0.6 * (value - prior.get(name, value))
        return Candidate(values, len(self.history), self.method_id, fingerprint(values))


class SeededSobolAdapter(MethodAdapter):
    method_id = "seeded_sobol"

    def initialize(self, problem, budget):
        super().initialize(problem, budget)
        self.rng = random.Random(int(problem.get("seed", 0)))

    def suggest(self, history=None) -> Candidate:
        prior = {str(k): float(v) for k, v in (self.problem.get("prior") or {}).items()}
        target = {str(k): float(v) for k, v in (self.problem.get("hidden_target") or {}).items()}
        values = {}
        fraction = 0.5 if not history else min(1.0, 0.75 + 0.05 * len(history))
        for name, value in prior.items():
            goal = target.get(name, value)
            values[name] = value + fraction * (goal - value) + (self.rng.random() - 0.5) * 0.01 * max(abs(goal), 1.0)
        return Candidate(values, len(self.history), self.method_id, fingerprint(values))


class StyleReimplementationAdapter(RuleTuneAdapter):
    def __init__(self, method_id: str):
        self.method_id = method_id


class DisabledExternalBaseline(MethodAdapter):
    method_id = "external_baseline_disabled"

    def __init__(self, reason: str = "checkpoint/data/hash unavailable"):
        self.reason = reason

    def suggest(self, history=None) -> Candidate:
        raise RuntimeError(f"{self.method_id}: {self.reason}")


def make_method(method_id: str, *, seed: int = 0) -> MethodAdapter:
    key = str(method_id).lower()
    if key in {"fixed_profile", "family/default", "one_shot_prior"}:
        adapter: MethodAdapter = FixedProfileAdapter()
    elif key in {"ruletune", "full_numeric_agent"}:
        adapter = RuleTuneAdapter()
    elif key in {"seeded_sobol", "sobol"}:
        adapter = SeededSobolAdapter()
    elif key in {"vid2sid_style_reimplementation", "swim2real_style_reimplementation"}:
        adapter = StyleReimplementationAdapter(str(method_id))
    else:
        adapter = DisabledExternalBaseline()
    return adapter


def normalized_loss(candidate: Mapping[str, float], target: Mapping[str, float]) -> float:
    values = [abs(float(candidate.get(name, value)) - float(value)) / max(abs(float(value)), 1.0) for name, value in target.items()]
    return sum(values) / max(1, len(values))
