#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for code/scripts/mesh_adaptation.py (v12.5-candidate "v3").

Heavy voxel paths need scipy (present in the Genesis venv); those tests are
skipped where scipy is unavailable.  Everything else is pure trimesh/numpy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "scripts"))

import mesh_adaptation as ma  # noqa: E402


def _box(extent=(1.0, 1.0, 1.0), subdivisions=None):
    mesh = trimesh.creation.box(extents=extent)
    if subdivisions:
        mesh = mesh.subdivide(subdivisions)
    return mesh


def _cylinder(radius=0.4, height=1.0, sections=64):
    return trimesh.creation.cylinder(radius=radius, height=height, sections=sections)


def _fragmented(mesh, patch=8, shrink=0.97):
    """Realistic Trellis-style fragmentation: patches with micro-gaps that
    defeat exact welding (the dominant 100k asset defect)."""
    faces = np.asarray(mesh.faces)
    vertices = np.asarray(mesh.vertices, dtype=float)
    new_faces = []
    new_vertices = []
    for start in range(0, len(faces), patch):
        group = faces[start : start + patch]
        uniq = np.unique(group)
        local_v = vertices[uniq].copy()
        centroid = local_v.mean(axis=0)
        local_v = centroid + shrink * (local_v - centroid)
        base = len(new_vertices)
        new_vertices.append(local_v)
        remap = {int(old): base + i for i, old in enumerate(uniq)}
        new_faces.extend([[remap[int(v)] for v in face] for face in group])
    return trimesh.Trimesh(vertices=np.vstack(new_vertices), faces=np.asarray(new_faces), process=False)


# --------------------------------------------------------------------------- #
# hygiene / quality
# --------------------------------------------------------------------------- #


def test_weld_and_clean_removes_degenerate_faces():
    mesh = _box()
    # add a degenerate (zero-area) face on an existing vertex triple
    faces = np.vstack([mesh.faces, [[0, 0, 1]]])
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=faces, process=False)
    cleaned, info = ma.weld_and_clean(mesh, np)
    assert len(cleaned.faces) == len(mesh.faces) - 1
    assert info["degenerate_faces_removed"] == 1
    assert 3 <= info["weld_digits_vertex"] <= 8


def test_mesh_quality_watertight_cube():
    quality = ma.mesh_quality(_box(), np)
    assert quality["watertight"] is True
    assert quality["component_count"] == 1
    assert quality["boundary_edge_count"] == 0
    assert quality["nonmanifold_edge_count"] == 0
    assert quality["volume"] == pytest.approx(1.0)
    assert quality["triangle_aspect_p95"] < 4.0  # right-isoceles faces of a box ~= 3.46
    assert quality["thinness_ratio"] == pytest.approx(1.0)


def test_mesh_quality_fragmented_open_surface():
    # three disjoint open sheets (each has a boundary; not watertight)
    patches = []
    for i in range(3):
        sheet = ma.checkerboard_grid_sheet(
            np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0]), [8, 8], [0, 1], np
        )
        sheet.apply_translation([2.0 * i, 0.0, 0.0])
        patches.append(sheet)
    mesh = trimesh.util.concatenate(patches)
    quality = ma.mesh_quality(mesh, np)
    assert quality["component_count"] == 3
    assert quality["watertight"] is False
    assert quality["components_above_1pct_area"] == 3
    assert quality["boundary_edge_count"] == 3 * 32


def test_face_component_labels_cover_all_faces():
    mesh = trimesh.util.concatenate([_box(), _box().apply_translation([5, 0, 0])])
    labels = ma.face_component_labels(mesh, np)
    assert len(labels) == len(mesh.faces)
    assert len(set(labels.tolist())) == 2


# --------------------------------------------------------------------------- #
# sheet / projection
# --------------------------------------------------------------------------- #


def test_flat_sheet_mid_surface_grid_dimensions():
    sheet, details = ma.flat_sheet_mid_surface(_box(extent=(2.0, 1.0, 0.02)), np)
    assert details["thin_axis"] == 2
    assert details["segments_by_axis"]["0"] == ma.FLAT_SHEET_LONGEST_SEGMENTS
    assert details["segments_by_axis"]["1"] == 32
    # single connected open sheet with boundary edges
    quality = ma.mesh_quality(sheet, np)
    assert quality["component_count"] == 1
    assert quality["boundary_edge_count"] > 0
    assert quality["watertight"] is False


def test_projected_mid_surface_preserves_semantic_hole():
    # annulus-like silhouette: a square frame made of 4 bars
    bars = []
    for dx, dy, ex, ey in ((-0.5, -0.5, 1.0, 0.2), (-0.5, 0.3, 1.0, 0.2), (-0.5, -0.5, 0.2, 1.0), (0.3, -0.5, 0.2, 1.0)):
        bar = trimesh.creation.box(extents=(ex, ey, 0.05))
        bar.apply_translation((dx + ex / 2, dy + ey / 2, 0.0))
        bars.append(bar)
    frame = trimesh.util.concatenate(bars)
    sheet, details = ma.projected_mid_surface(frame, np)
    assert details["occupied_cell_fraction"] < 0.9  # hole preserved
    assert len(sheet.faces) >= ma.MIN_CLOTH_PROXY_FACES
    quality = ma.mesh_quality(sheet, np)
    assert quality["component_count"] == 1
    assert quality["boundary_edge_count"] > 4 * 4  # outer + inner boundary


def test_projected_mid_surface_rejects_sparse_input():
    # two tiny triangles at opposite bbox corners: the projection occupancy is
    # far below the v11 minimum face count
    tiny = trimesh.Trimesh(
        vertices=np.array(
            [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [1, 1, 0], [1, 0.99, 0], [0.99, 1, 0]],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]]),
        process=False,
    )
    with pytest.raises(ValueError):
        ma.projected_mid_surface(tiny, np, longest_segments=64)


def test_rasterize_triangles_marks_covered_cells():
    projected = np.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]])
    faces = np.array([[0, 1, 2]])
    occupancy = ma._rasterize_triangles(projected, faces, 16, 16, np)
    assert occupancy.sum() > 20
    assert occupancy[0, 0] == 0  # far corner untouched


def test_binary_close_2d_fills_gaps_preserves_holes():
    grid = np.zeros((32, 32), dtype=bool)
    grid[4:28, 4:28] = True
    grid[12:20, 12:20] = False  # semantic hole (8 cells)
    grid[10, 10] = False  # 1-cell rasterization gap
    closed = ma._binary_close_2d(grid, np)
    assert closed[10, 10]  # gap filled
    assert not closed[12:20, 12:20].any()  # semantic hole preserved


# --------------------------------------------------------------------------- #
# opening surgery
# --------------------------------------------------------------------------- #


def test_sealed_shell_opening_garment_uses_longest_axis():
    tube = _cylinder(radius=0.3, height=1.6)  # z is the longest axis
    opened, details = ma.sealed_shell_opening(tube, "garment", np)
    assert details["axis"] == 2
    assert details["removed_face_count"] >= 2
    assert not opened.is_watertight
    assert ma.mesh_quality(opened, np)["boundary_edge_count"] > 0


def test_sealed_shell_opening_shell_uses_thinnest_axis():
    shell = _box(extent=(1.0, 1.0, 0.2))
    opened, details = ma.sealed_shell_opening(shell, "shell", np)
    assert details["axis"] == 2


def test_sealed_shell_opening_bag_uses_semantic_top():
    bag = _cylinder(radius=0.3, height=1.2)  # z longest; bags ignore it
    opened, details = ma.sealed_shell_opening(bag, "bag", np)
    assert details["axis"] == 1
    assert details["side"] == "max"


def test_sealed_shell_opening_rejects_when_no_bounded_cut():
    # a tetrahedron: no 2.5-8% axial slab captures a bounded (>= 2) face set
    tetra = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        faces=np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]),
        process=False,
    )
    with pytest.raises(ValueError):
        ma.sealed_shell_opening(tetra, "garment", np)


# --------------------------------------------------------------------------- #
# verdict gate
# --------------------------------------------------------------------------- #


def test_verdict_unfit_when_no_faces():
    verdict, defects = ma.assess_proxy_verdict({"face_count": 0}, cloth=True)
    assert verdict == "unfit"
    assert "proxy_has_no_faces" in defects


def test_verdict_degraded_fragmented_cloth_proxy():
    # fragmentation alone must not gate (v12.2 simulated these); it degrades
    verdict, issues = ma.assess_proxy_verdict(
        {"face_count": 5000, "component_count": 500, "watertight": False, "boundary_edge_count": 100},
        cloth=True,
    )
    assert verdict == "degraded"
    assert any(d.startswith("cloth_proxy_fragmented") for d in issues)


def test_verdict_degraded_sealed_cloth_volume():
    verdict, issues = ma.assess_proxy_verdict(
        {"face_count": 5000, "component_count": 1, "watertight": True, "boundary_edge_count": 0},
        cloth=True,
    )
    assert verdict == "degraded"
    assert "cloth_proxy_is_sealed_volume_deflates_without_pressure_model" in issues


def test_verdict_unfit_mpm_proxy_open():
    verdict, defects = ma.assess_proxy_verdict({"face_count": 1000, "watertight": False}, cloth=False)
    assert verdict == "unfit"
    assert "mpm_proxy_not_watertight" in defects


def test_verdict_ok_paths():
    assert ma.assess_proxy_verdict(
        {"face_count": 4000, "component_count": 1, "watertight": False, "boundary_edge_count": 60},
        cloth=True,
    )[0] == "ok"
    assert ma.assess_proxy_verdict(
        {"face_count": 4000, "watertight": True, "volume": 0.01}, cloth=False
    )[0] == "ok"


# --------------------------------------------------------------------------- #
# voxel paths (need scipy)
# --------------------------------------------------------------------------- #


needs_scipy = pytest.mark.skipif(
    __import__("importlib").util.find_spec("scipy") is None,
    reason="voxel morphology paths need scipy (Genesis venv)",
)


@needs_scipy
def test_filled_volume_proxy_watertight_result():
    sphere = trimesh.creation.icosphere(subdivisions=3)
    proxy, details = ma.filled_volume_proxy(sphere, np)
    assert details["method"] == "voxel_fill_preserving_exterior_connected_voids"
    assert proxy.is_watertight and proxy.is_volume


@needs_scipy
def test_filled_volume_proxy_preserves_watertight_bag_boundary():
    sphere = trimesh.creation.icosphere(subdivisions=2)
    proxy, details = ma.filled_volume_proxy(sphere, np, preserve_watertight_boundary=True)
    assert details["method"] == "source_watertight_volume_boundary"
    assert len(proxy.faces) == len(sphere.faces)


@needs_scipy
def test_voxel_sealed_shell_seals_fragmented_surface():
    fragmented = _fragmented(_box(extent=(1.0, 1.0, 1.0)).subdivide().subdivide())
    assert not fragmented.is_watertight
    assert ma.mesh_quality(fragmented, np)["component_count"] > 20
    sealed, details = ma.voxel_sealed_shell(fragmented, np, resolution=40)
    assert sealed.is_watertight and sealed.is_volume
    assert details["voxel_resolution"] == 40


@needs_scipy
def test_adapt_solver_mesh_solid_end_to_end(tmp_path):
    source = trimesh.creation.icosphere(subdivisions=3)
    source_path = tmp_path / "source.obj"
    source.export(source_path)
    path, extent, median_edge, details = ma.adapt_solver_mesh(
        source_path, 0.32, "corner_first", tmp_path, np, cloth=False, geometry_family="solid"
    )
    assert path.is_file()
    assert details["verdict"] == "ok"
    assert details["proxy_quality"]["watertight"] is True
    assert max(extent) == pytest.approx(0.32, rel=0.05)
    # grounded: min z == 0
    loaded = trimesh.load(path)
    assert loaded.bounds[0, 2] == pytest.approx(0.0, abs=1e-6)


@needs_scipy
def test_adapt_solver_mesh_cloth_sheet_end_to_end(tmp_path):
    sheet = trimesh.creation.box(extents=(1.6, 1.1, 0.01))
    source_path = tmp_path / "sheet.obj"
    sheet.export(source_path)
    path, extent, median_edge, details = ma.adapt_solver_mesh(
        source_path, 1.4, "flat", tmp_path, np, cloth=True, geometry_family="cloth_sheet"
    )
    assert details["proxy"]["method"] == "flat_mid_surface"
    assert details["verdict"] == "ok"
    assert max(extent) == pytest.approx(1.4, rel=0.05)


@needs_scipy
def test_adapt_solver_mesh_fragmented_garment_sealed_and_opened(tmp_path):
    # fragmented garment-like tube -> voxel seal -> opening surgery
    tube = _fragmented(_cylinder(radius=0.35, height=1.3, sections=48).subdivide())
    source_path = tmp_path / "garment.obj"
    tube.export(source_path)
    path, extent, median_edge, details = ma.adapt_solver_mesh(
        source_path, 0.75, "corner_first", tmp_path, np, cloth=True, geometry_family="garment"
    )
    assert "voxel_sealed" in str(details["proxy"]["method"])
    assert details["verdict"] == "ok"
    quality = details["proxy_quality"]
    assert quality["component_count"] == 1
    assert quality["boundary_edge_count"] > 0


# --------------------------------------------------------------------------- #
# runner switch plumbing (no Genesis needed)
# --------------------------------------------------------------------------- #


def test_runner_v3_switch_detection():
    import genesis_behavior_runner as runner

    assert runner._mesh_adaptation_v3_enabled({"config": {"mesh_adaptation": "v3"}}, {}) is True
    assert runner._mesh_adaptation_v3_enabled({"config": {"mesh_adaptation": "v2"}}, {}) is False
    assert runner._mesh_adaptation_v3_enabled({}, {"mesh_adaptation": "v3"}) is True
    assert runner._mesh_adaptation_v3_enabled({}, {}) is False
    # v2 detection unchanged
    assert runner._mesh_adaptation_v2_enabled({"config": {"mesh_adaptation": "v2"}}, {}) is True
    assert runner._mesh_adaptation_v2_enabled({"config": {"mesh_adaptation": "v3"}}, {}) is False


def test_runner_module_import_does_not_import_v3_module():
    """Importing the runner must not import the v3 module (lazy import keeps
    the v12.2 default path byte-for-byte and dependency-free)."""
    import subprocess

    code = (
        "import sys; sys.path.insert(0, r'%s'); "
        "import genesis_behavior_runner; "
        "print('mesh_adaptation' in sys.modules)"
    ) % (Path(__file__).resolve().parents[1] / "code" / "scripts")
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert out.strip() == "False"
