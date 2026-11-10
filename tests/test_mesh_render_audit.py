# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code/scripts'))
from mesh_render_audit import visibility_sample, summarize_visibility


def test_floor_and_plates_cannot_pass_visibility_check():
    pixels = np.array([[0, 0, 2], [2, 3, 3]])
    sample = visibility_sample(pixels, {0: 0, 2: 2, 3: 3}, 1, 0, np)
    assert not summarize_visibility([sample])['valid']
    assert sample['visible_pixels'] == 0
    # Registered but off-screen/occluded is a different failure from no node.
    sample = visibility_sample(pixels, {0: 0, 1: 1, 2: 2, 3: 3}, 1, 0, np)
    assert summarize_visibility([sample])['failure_reason'] == 'soft_body_has_no_visible_pixels'


def test_renderer_label_is_resolved_through_entity_mapping():
    sample = visibility_sample(np.array([[0, 7], [7, 2]]), {0: 0, 7: 1, 2: 2}, 1, 30, np)
    result = summarize_visibility([sample])
    assert result['valid']
    assert result['max_visible_pixels'] == 2
    assert result['visible_sampled_frames'] == 1
    assert not summarize_visibility([])['valid']
