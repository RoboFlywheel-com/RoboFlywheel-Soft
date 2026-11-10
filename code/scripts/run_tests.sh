#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! python3 -c 'import softsim_formal, fragile_soft_benchmark, softsim_physical_agent' >/dev/null 2>&1; then
  # Source-tree fallback for minimal system Python; installed users do not
  # need this path (the package entry points are preferred).
  export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi
if python3 -c 'import pytest' >/dev/null 2>&1; then
  exec python3 -m pytest -q "$ROOT/../tests"
fi
exec python3 -m unittest discover -s "$ROOT/../tests" -p 'test_*.py' -v
