# SPDX-License-Identifier: Apache-2.0
from setuptools import setup

setup(
    name="softsim-fragile-soft-benchmark",
    version="0.1.0",
    packages=["fragile_soft_benchmark"],
    package_dir={"fragile_soft_benchmark": "."},
    entry_points={"console_scripts": ["fragile-soft-benchmark=fragile_soft_benchmark.cli:main"]},
)
