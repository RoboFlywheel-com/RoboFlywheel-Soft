# Installation

Use Python 3.10+ for the orchestration packages. Create a virtual environment and install from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install --no-build-isolation ./code ./asset_pipeline
```

The entry points are `softsim-exp`, `softsim-physical-agent`, `fragile-soft-benchmark`, and `softsim-assets`. Existing `softsim_*` package names remain stable for compatibility with recorded configs.

## Tests and analysis

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The test extra installs pytest, NumPy, Pillow, trimesh 4.7.4, SciPy, rtree, scikit-image and networkx. Python 3.14 requires NumPy 2.3 or newer; older NumPy builds produced incorrect results in voxel subdivision during release validation. Mesh tests skip when optional geometry dependencies are absent. The GPU render test runs only with `SOFTSIM_RUN_GPU_RENDER_TESTS=1`. One historical cloud-launcher test skips because that internal launcher is excluded; a production-reference comparison skips when its external manifest is unavailable. The asset audit uses process workers and requires local process/socket support.

For analysis alone, install `./code[analysis]`, verify the CSV hashes, then run `python paper/build_support/build_results.py`. A TeX installation is unnecessary for generating `.tex` tables.

## Genesis simulation

Keep Genesis in a separate compatible environment; the orchestration interpreter can invoke it through `GENESIS_PYTHON`. The recorded reference runtime is **Genesis 1.3.3**. Install that exact upstream runtime and its GPU dependencies following its own instructions. No simulator or model checkpoint is bundled here.

```bash
export GENESIS_PYTHON=/path/to/genesis-env/bin/python
export SOFTSIM_GENESIS_BEHAVIOR_RUNNER="$PWD/code/scripts/genesis_behavior_runner.py"
export PYOPENGL_PLATFORM=egl
"$GENESIS_PYTHON" -c 'import genesis; print(genesis.__version__)'
```

If your GPU driver requires an EGL vendor JSON, configure `__EGL_VENDOR_LIBRARY_FILENAMES` for that machine. The repository does not assume a private driver path.

## Optional API proposer

The default mock and rule-based examples need no API credentials. For the Qwen transport, export `DASHSCOPE_API_KEY` and select the model available to your account with `SOFTSIM_QWEN_MODEL` or the config's `--model` argument. `QWEN_API_KEY` and `QWEN_VL_API_KEY` are accepted aliases. `.env.example` lists variables; programs read the process environment and do not automatically load `.env` files. API-backed jobs incur provider charges; local smoke commands do not call the provider.

## Troubleshooting

- Missing build requirements: upgrade pip/setuptools before using `--no-build-isolation`.
- Missing geometry modules: install the test extra in the active interpreter.
- Missing Genesis runner: set the absolute `SOFTSIM_GENESIS_BEHAVIOR_RUNNER` path above.
- Missing assets or logical roots: supply the explicit `--root ALIAS=PATH` bindings described in [DATA.md](DATA.md).
- Reusing runs after changing code/configs: create a new output directory; resume checks fingerprints and preserves immutable input identities.
