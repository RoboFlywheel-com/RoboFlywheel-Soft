# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim for the repository's existing ``code/src`` layout."""

from pathlib import Path

# Keep one implementation in ``code/fragile_soft_benchmark`` while allowing
# projects that install the sibling src-layout packages to import this one.
_implementation = Path(__file__).resolve().parents[2] / "fragile_soft_benchmark"
if str(_implementation) not in __path__:
    __path__.append(str(_implementation))

from .adapter import P0Adapter, adapt_p0
from .builder import build_instances, make_instance
from .difficulty import construct_difficulty
from .evaluator import evaluate_episode, evaluate_force_sweep
from .materials import TofuDamage, UnsupportedMaterialModel, construct_material
from .qualification import run_qualification
from .release import promote_e3_a_manifest
from .scenario import make_scenario
from .sharding import partition_instances, shard_for
from .validation import validate_instance, validate_output_dir

__version__ = "0.1.0"
