#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CLI wrapper for canonical formal analysis exports."""

import argparse

from softsim_formal.render import check_render, render_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Render canonical SoftSim formal records")
    parser.add_argument("--run", required=True)
    parser.add_argument("--spec")
    parser.add_argument("--output")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        result = check_render(args.output or args.run)
        print(result)
        return 0 if result.get("valid") else 2
    print(render_run(args.run, spec_path=args.spec, output_dir=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
