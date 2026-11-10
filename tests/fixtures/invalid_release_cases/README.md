# Release-reader negative fixtures

These small snippets document the two mandatory rejection paths. They are not
complete releases and are never used as experiment inputs.

- `predicted_bundle.json`: a prior-only/predicted status must be rejected;
- `no_trace_bundle.json`: a terminal physical row without an agent trace must
  be rejected even when its process reported success.
