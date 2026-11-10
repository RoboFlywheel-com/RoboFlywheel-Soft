# SPDX-License-Identifier: Apache-2.0
"""Auditable simulator-in-the-loop physical parameter agent.

This package is deliberately self contained.  It treats a physical estimate as
a versioned behavior contract, rather than as a successful process exit or a
static prior.
"""

__version__ = "0.1.0"
METHOD_ID = "softsim.genesis_proxy.v12.2"

from .models import (  # noqa: F401
    ACCEPTED_FIRST_PASS,
    CORRECTED_THEN_ACCEPTED,
    REJECTED_MODEL_LIMIT,
    REJECTED_NUMERICAL_FAILURE,
    UNCERTAIN_HUMAN_ESCALATION,
)

from .agent import build_tasks, run_task  # noqa: F401,E402
from .input_adapter import load_asset_release, load_legacy_assets, load_previous_staging, load_release_assets, load_stage3_manifest  # noqa: F401,E402

__all__ = [
    "METHOD_ID",
    "__version__",
    "build_tasks",
    "run_task",
    "load_asset_release",
    "load_release_assets",
    "load_legacy_assets",
    "load_previous_staging",
    "load_stage3_manifest",
    "ACCEPTED_FIRST_PASS",
    "CORRECTED_THEN_ACCEPTED",
    "REJECTED_MODEL_LIMIT",
    "REJECTED_NUMERICAL_FAILURE",
    "UNCERTAIN_HUMAN_ESCALATION",
]
