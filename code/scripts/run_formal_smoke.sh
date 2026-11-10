#!/usr/bin/env bash
set -euo pipefail

# Run only local synthetic promotion stages.  This script intentionally never
# discovers cloud paths or starts the production 100k/E3-A jobs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/runs/formal-smoke}"
mkdir -p "$OUT"
if command -v softsim-exp >/dev/null 2>&1; then
  FORMAL_CMD=(softsim-exp)
else
  export PYTHONPATH="$ROOT/code/src:$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
  FORMAL_CMD=(python3 -m softsim_formal)
fi
for COUNT in 10 150 200; do
  "${FORMAL_CMD[@]}" fixture e1 --output "$OUT/e1-${COUNT}" --count "$COUNT"
done
"${FORMAL_CMD[@]}" fixture e2 --output "$OUT/e2-10" --count 10
"${FORMAL_CMD[@]}" fixture e3a --output "$OUT/e3a-10" --count 10
