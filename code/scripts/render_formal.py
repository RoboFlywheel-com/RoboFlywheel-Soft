#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Thin installed-package wrapper for deterministic analysis rendering."""

from softsim_formal.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["render"] + __import__("sys").argv[1:]))
