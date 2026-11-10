#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compatibility wrapper for the formal v1 -> v2 asset release converter."""

from softsim_formal.cli import main
import sys


if __name__ == "__main__":
    raise SystemExit(main(["release", "convert-v1-to-v2"] + sys.argv[1:]))
