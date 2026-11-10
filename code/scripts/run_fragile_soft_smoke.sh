#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/outputs/local_smoke_6}"
if ! command -v fragile-soft-benchmark >/dev/null 2>&1; then
  python3 -m pip install --no-build-isolation --no-deps -e "$ROOT/code" >/dev/null
fi
fragile-soft-benchmark qualify \
  --manifest "$ROOT/tests/fixtures/p0_legacy/manifest.json" \
  --profiles "$ROOT/tests/fixtures/p0_legacy/material_profiles.json" \
  --summary-json "$ROOT/tests/fixtures/p0_legacy/summary.json" \
  --summary-csv "$ROOT/tests/fixtures/p0_legacy/summary.csv" \
  --video-manifest "$ROOT/tests/fixtures/p0_legacy/videos.json" \
  --state-manifest "$ROOT/tests/fixtures/p0_legacy/states.json" \
  --config "$ROOT/configs/qualification_config.json" \
  --output "$OUT" --limit 6 >/dev/null
fragile-soft-benchmark validate --output "$OUT" --expected-count 6
