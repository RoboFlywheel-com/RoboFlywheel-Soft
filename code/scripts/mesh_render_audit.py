# SPDX-License-Identifier: Apache-2.0
"""Entity-specific visibility evidence, independent of physical acceptance gates."""
from __future__ import annotations


def visibility_sample(segmentation, index_map, entity_index, frame_index, np):
    # Genesis segmentation pixels contain renderer indices, not entity indices.
    # In particular, floor/plate pixels must never count as a visible soft body.
    labels = [label for label, key in index_map.items() if key == entity_index]
    pixels = int(np.count_nonzero(np.isin(segmentation, labels))) if labels else 0
    return {"frame_index": int(frame_index), "entity_index": int(entity_index),
            "registered": bool(labels), "visible_pixels": pixels}


def summarize_visibility(samples):
    registered = bool(samples) and all(s["registered"] for s in samples)
    visible = sum(s["visible_pixels"] > 0 for s in samples)
    valid = registered and visible > 0
    reason = None if valid else ("soft_body_not_registered" if not registered
                                 else "soft_body_has_no_visible_pixels")
    return {"valid": valid, "failure_reason": reason,
            "sampled_frames": len(samples), "visible_sampled_frames": visible,
            "max_visible_pixels": max((s["visible_pixels"] for s in samples), default=0),
            "samples": samples, "affects_physics": False,
            "scope": "sampled entity segmentation; does not certify every frame or physical behavior"}
