#!/usr/bin/env bash
set -euo pipefail

# Server-only wrapper.  It refuses to fall back to the local mock backend.
: "${SOFTSIM_EPISODE_RUNNER:?Set SOFTSIM_EPISODE_RUNNER=module:function for the audited Genesis runner}"
: "${SOFTSIM_PHYSICAL_INDEX:?Set SOFTSIM_PHYSICAL_INDEX to physical_bundle_index.jsonl}"
: "${SOFTSIM_JOIN_MAP:?Set SOFTSIM_JOIN_MAP to the immutable benchmark join map}"
: "${SOFTSIM_P0_ROOT:?Set SOFTSIM_P0_ROOT to the logical P0 root}"

OUT="${1:?usage: run_formal_qualification.sh OUTPUT_DIR}"
fragile-soft-benchmark qualify \
  --p0-root "$SOFTSIM_P0_ROOT" \
  --physical-index "$SOFTSIM_PHYSICAL_INDEX" \
  --physical-join-map "$SOFTSIM_JOIN_MAP" \
  --physical-release-id "${SOFTSIM_PHYSICAL_RELEASE_ID:?Set SOFTSIM_PHYSICAL_RELEASE_ID}" \
  --episode-runner "$SOFTSIM_EPISODE_RUNNER" \
  --backend-id "${SOFTSIM_BACKEND_ID:?Set SOFTSIM_BACKEND_ID}" \
  --formal-scope --server \
  --config "${SOFTSIM_QUALIFICATION_CONFIG:-configs/qualification_config.json}" \
  --output "$OUT"
