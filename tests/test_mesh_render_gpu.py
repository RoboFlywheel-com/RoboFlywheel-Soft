# SPDX-License-Identifier: Apache-2.0
"""Opt-in actual CUDA + EGL tests (SOFTSIM_RUN_GPU_RENDER_TESTS=1)."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('SOFTSIM_RUN_GPU_RENDER_TESTS') != '1',
                                reason='requires explicit CUDA + EGL integration environment')


@pytest.mark.parametrize('mode', ['solid', 'cloth_sheet'])
def test_bound_soft_body_is_present_in_rendered_pixels(tmp_path, mode):
    script = Path(__file__).resolve().parents[1]/'deployment/experiments/mesh_adapt/gpu_render_smoke.py'
    output = tmp_path/mode
    result = subprocess.run([sys.executable, str(script), '--output', str(output), '--mode', mode],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output/'report.json').read_text())
    assert report['metrics']['simulation_executed']
    assert report['metrics']['render_visibility']['max_visible_pixels'] > 0
    assert (output/'render/clip.mp4').stat().st_size > 0
