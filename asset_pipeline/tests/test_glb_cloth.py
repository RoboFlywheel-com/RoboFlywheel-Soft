# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import struct
from copy import deepcopy
from pathlib import Path

import pytest

from softsim_asset_pipeline.cloth import build_cloth_glb
from softsim_asset_pipeline.fixtures import create_asset_fixture
from softsim_asset_pipeline.glb import _serialize_glb, audit_glb, parse_glb, prune_glb_components


def test_generated_cloth_is_deterministic_and_embedded(tmp_path: Path) -> None:
    fixture = create_asset_fixture(tmp_path / "fixture", 2)
    image = Path(fixture["image_root"]) / "samples/fixture_000000/input_image.png"
    recipe = {
        "width_m": 1.0,
        "height_m": 0.6,
        "cols": 8,
        "rows": 4,
        "broad_fold_amplitude_m": 0.01,
    }
    first = build_cloth_glb(recipe, image.read_bytes())
    second = build_cloth_glb(recipe, image.read_bytes())
    assert first == second
    path = tmp_path / "cloth.glb"
    path.write_bytes(first)
    result = audit_glb(path)
    assert result["status"] == "pass"
    assert result["geometry"]["components"] == 1
    assert result["geometry"]["non_manifold_edges"] == 0
    assert result["materials"]["embedded_images"] == 1


def test_glb_truncation_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.glb"
    path.write_bytes(b"glTF")
    with pytest.raises(ValueError, match="truncated"):
        parse_glb(path)


def test_component_prune_can_remove_an_entire_tiny_primitive(tmp_path: Path) -> None:
    fixture = create_asset_fixture(tmp_path / "fixture", 1)
    image = Path(fixture["image_root"]) / "samples/fixture_000000/input_image.png"
    source = tmp_path / "source.glb"
    source.write_bytes(
        build_cloth_glb(
            {"width_m": 1.0, "height_m": 1.0, "cols": 1, "rows": 1},
            image.read_bytes(),
        )
    )
    parsed = parse_glb(source)
    document = deepcopy(parsed.document)
    binary = bytearray(parsed.binary)
    while len(binary) % 4:
        binary.append(0)
    position_offset = len(binary)
    binary.extend(
        b"".join(
            struct.pack("<fff", *position)
            for position in ((10.0, 10.0, 0.0), (11.0, 10.0, 0.0), (10.0, 11.0, 0.0))
        )
    )
    index_offset = len(binary)
    binary.extend(struct.pack("<HHH", 0, 1, 2))
    position_view = len(document["bufferViews"])
    document["bufferViews"].append(
        {"buffer": 0, "byteOffset": position_offset, "byteLength": 36, "target": 34962}
    )
    index_view = len(document["bufferViews"])
    document["bufferViews"].append(
        {"buffer": 0, "byteOffset": index_offset, "byteLength": 6, "target": 34963}
    )
    position_accessor = len(document["accessors"])
    document["accessors"].append(
        {"bufferView": position_view, "componentType": 5126, "count": 3, "type": "VEC3"}
    )
    index_accessor = len(document["accessors"])
    document["accessors"].append(
        {"bufferView": index_view, "componentType": 5123, "count": 3, "type": "SCALAR"}
    )
    document["meshes"][0]["primitives"].append(
        {
            "attributes": {"POSITION": position_accessor},
            "indices": index_accessor,
            "material": 0,
            "mode": 4,
        }
    )
    document["buffers"][0]["byteLength"] = len(binary)
    source.write_bytes(_serialize_glb(document, bytes(binary)))
    assert audit_glb(source)["geometry"]["components"] == 2

    repaired = tmp_path / "repaired.glb"
    result = prune_glb_components(
        source,
        repaired,
        max_components=1,
        min_face_retention=0.5,
    )
    audit = audit_glb(repaired)
    assert result["status"] == "repaired"
    assert result["face_retention"] == pytest.approx(2 / 3)
    assert audit["status"] == "pass"
    assert audit["geometry"]["components"] == 1
    assert audit["geometry"]["faces"] == 2
