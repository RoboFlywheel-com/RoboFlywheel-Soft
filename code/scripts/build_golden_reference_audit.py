#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CLI wrapper for the stage3 reference-only audit."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))

from softsim_physical_agent.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["audit-golden", *sys.argv[1:]]))

