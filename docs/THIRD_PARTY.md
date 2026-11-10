# Third-party components

The original code in this repository is licensed under Apache-2.0. **The Apache-2.0 grant does not extend to third-party model weights, SDKs, or separately distributed datasets; each remains under its own upstream license.** This repository does not vendor simulator/model source or checkpoints.

| Component | Role | Upstream license | Bundled? |
| --- | --- | --- | --- |
| Genesis | Reference MPM/PBD simulation runtime | Apache-2.0 | No |
| Qwen / DashScope-compatible service | Optional physical-parameter proposer | Provider service terms | No |
| TRELLIS.2 | External image-to-geometry stage | MIT | No |
| NumPy, SciPy | Tests and analysis | BSD-3-Clause | No |
| trimesh, rtree | Optional geometry and tests | MIT | No |
| Pillow | Image helpers in tests | HPND | No |
| pytest, setuptools | Development and packaging | MIT | No |

Reference appearance generation, semantic filtering and TRELLIS execution require their upstream tools; this release starts with manifests/geometry for those stages and includes procedural cloth generation. Internal cloud deployment wrappers are excluded. Model availability and account permissions are configured by the user.

The root code license does not assert redistribution rights for third-party model weights or for the separately hosted full dataset. Add the full dataset's own distribution terms with its release.
