# SPDX-License-Identifier: Apache-2.0
"""Auditable constructor and qualification tools for the fragile-soft benchmark.

The package deliberately has a standard-library-only core.  A simulator backend
can consume the emitted material/scenario contracts, while the local backend is
deterministic and is intended only for fixture qualification and smoke tests.
"""

from .adapter import P0Adapter, adapt_p0
from .builder import build_instances, make_instance
from .difficulty import construct_difficulty
from .evaluator import evaluate_episode, evaluate_force_sweep
from .materials import TofuDamage, UnsupportedMaterialModel, construct_material
from .qualification import run_qualification
from .scenario import make_scenario
from .validation import validate_instance, validate_output_dir
from .sharding import partition_instances, shard_for
from .physical_join import PhysicalJoinError, join_physical_bundles, make_join_template
from .runner import EpisodeRunnerContractError, load_episode_runner
from .release import promote_e3_a_manifest

__all__ = [
    "P0Adapter",
    "UnsupportedMaterialModel",
    "TofuDamage",
    "adapt_p0",
    "build_instances",
    "construct_difficulty",
    "evaluate_episode",
    "evaluate_force_sweep",
    "construct_material",
    "make_instance",
    "make_scenario",
    "run_qualification",
    "validate_instance",
    "validate_output_dir",
    "partition_instances",
    "shard_for",
    "PhysicalJoinError",
    "join_physical_bundles",
    "make_join_template",
    "EpisodeRunnerContractError",
    "load_episode_runner",
    "promote_e3_a_manifest",
]

__version__ = "0.1.0"
