# Release validation

Validated on 2026-10-26 using Python 3.14.4, NumPy 2.5.4, trimesh 4.7.4, SciPy 1.18.0, scikit-image 0.26.0, rtree 1.4.1, and pytest 9.1.1. Optional dependencies were installed into temporary directories; the system environment was not modified.

| Check | Result |
| --- | --- |
| Full regression suite, including mesh/representation and asset pipeline tests | 189 passed, 4 skipped |
| Non-editable installation of both packages into an isolated target | Passed |
| Installed probe tools and Genesis adapter script resolution | Passed |
| Installed E1 / E2 / E3-A fixtures outside the source tree | Completed with valid synthetic records |
| E1 promotion fixtures | 10, 150, 200 planned/observed records |
| E2 10-input fixture | 40 planned/observed budget episodes |
| E3-A fixture | 10 planned/observed records |
| Physical-agent smoke | First-pass acceptance, corrected acceptance and rejection recorded |
| Asset fixture inventory | 10 inputs: 6 geometry passes, 4 pending cloth |
| Fragile-object qualification smoke | 6 instances, validation passed |
| Analysis CSV hash verification | 21 files verified |
| Paper table regeneration | Generated tables are byte-identical; the analysis manifest records portable CSV hashes |
| Formal deployment archive creation and verification | Passed |
| Credential literal scan and local documentation links | Passed |

The four skips cover two explicit GPU/EGL render cases, one unavailable production-reference manifest, and one internal cloud-launcher integration excluded from this release. Remaining public contract tests retain failure-record coverage. Real GPU simulation, full-pool replay, and paid API execution were not run during this packaging check.

Compatibility findings: Python 3.14 with the pre-existing NumPy 2.2.6 build produced inconsistent voxel subdivision results; the dependency metadata requires NumPy 2.3+ for Python 3.14. The geometry test extra pins the validated trimesh version and includes marching-cubes dependencies. Deprecation warnings from packaging and geometry libraries remain; all collected non-skipped tests pass.
