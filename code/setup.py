# SPDX-License-Identifier: Apache-2.0
from setuptools import setup

setup(
    name="softsim-formal-experiments",
    version="0.1.0",
    packages=["fragile_soft_benchmark", "softsim_formal", "softsim_genesis_runner", "softsim_physical_agent", "softsim_tools"],
    py_modules=["softsim_targets"],
    package_dir={
        "": "src",
        "fragile_soft_benchmark": "fragile_soft_benchmark",
        "softsim_formal": "src/softsim_formal",
        "softsim_genesis_runner": "src/softsim_genesis_runner",
        "softsim_physical_agent": "src/softsim_physical_agent",
        "softsim_tools": "scripts",
    },
    exclude_package_data={"": ["__pycache__/*", "*.py[cod]"]},
    entry_points={"console_scripts": [
        "softsim-exp=softsim_formal.cli:main",
        "fragile-soft-benchmark=fragile_soft_benchmark.cli:main",
        "softsim-physical-agent=softsim_physical_agent.cli:main",
    ]},
)
