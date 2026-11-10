#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/../tests/_smoke_output}"
if ! command -v softsim-physical-agent >/dev/null 2>&1; then
  python3 -m pip install --no-build-isolation --no-deps -e "$ROOT" >/dev/null
fi
softsim-physical-agent smoke --output "$OUT"
