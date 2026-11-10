# SPDX-License-Identifier: Apache-2.0
"""SoftSim formal experiment package.

The package is deliberately dependency-light.  Production Genesis execution is
represented by an adapter contract; the local backend is deterministic and
synthetic, so it is suitable for qualification and schema tests only.
"""

__version__ = "0.1.0"
METHOD_ID = "softsim.genesis_proxy.v12.2"
SCHEMA_VERSION = 1

from .canonical import canonical_bytes, canonical_dumps, sha256_bytes, sha256_file
from .release import ReleaseReader, ReleaseError, BlockedRelease, read_release
from .planner import plan_run
from .runner import run_shard, run_all_shards, status_run
from .aggregate import aggregate_run
from .render import render_run
from .asset_release import AssetReleaseConversionError, convert_v1_to_v2

__all__ = [
    "METHOD_ID",
    "SCHEMA_VERSION",
    "ReleaseReader",
    "ReleaseError",
    "BlockedRelease",
    "read_release",
    "canonical_bytes",
    "canonical_dumps",
    "sha256_bytes",
    "sha256_file",
    "plan_run",
    "run_shard",
    "run_all_shards",
    "status_run",
    "aggregate_run",
    "render_run",
    "convert_v1_to_v2",
    "AssetReleaseConversionError",
]
