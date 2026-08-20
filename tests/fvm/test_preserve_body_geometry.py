"""Geometry-preserving Cartesian meshing regression tests (AGENT_PLAN M3).

Certifies that ``AdaptiveCartesianMesher`` treats the STL body as the geometry
authority (``preserve_body_geometry=True``):

* A  body exactly commensurable with the requested dyadic lattice (the cube_flow
   cube) keeps the requested background spacing, zero padding and an exact wall.
* B  an incompatible body width is never modified: the finest spacing is
   refined to divide the body exactly and the outer domain is padded outward.
* C  a shifted body (float32 STL coordinates) stays exact on the resolved
   lattice with positive padding.
* D  a body with no compatible isotropic spacing raises a clear error and the
   body is never modified.
* G  the legacy body-snapping mode is rejected (``preserve_body_geometry=False``).

Every built mesh must satisfy the same invariants: zero positive-volume overlap
between any fluid cell AABB and the body AABB, and a wall patch whose vertex
bounds coincide with the STL bounds.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from source.solvers.FVM.mesh.adaptive_cartesian import (
    AdaptiveCartesianMesher,
)
from source.solvers.FVM.mesh.validation import MeshValidationError

DOMAIN = (-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
BACKGROUND = 0.0625
SURFACE_SIZE = 0.015625  # level 2 from the background size
WALL_PATCH = "cube"


def write_box_stl(path, lo, hi) -> None:
    """Write an axis-aligned box STL with outward normals and float32 vertices."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    corners = np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )
    centre = (lo + hi) / 2.0
    triangles = []
    for axis in range(3):
        tangents = [a for a in range(3) if a != axis]
        for side in (0, 1):
            value = lo[axis] if side == 0 else hi[axis]
            face = np.array([c for c in corners if abs(c[axis] - value) < 1e-9])
            face_centre = face.mean(axis=0)
            angles = np.arctan2(
                face[:, tangents[1]] - face_centre[tangents[1]],
                face[:, tangents[0]] - face_centre[tangents[0]],
            )
            face = face[np.argsort(angles)]
            for a, b, c in ((0, 1, 2), (0, 2, 3)):
                normal = np.cross(face[b] - face[a], face[c] - face[a])
                if np.dot(normal, face_centre - centre) < 0:
                    a, b = b, a
                triangles.append((face[a], face[b], face[c]))
    with open(path, "wb") as stream:
        stream.write(b" " * 80)
        stream.write(struct.pack("<I", len(triangles)))
        for a, b, c in triangles:
            normal = np.cross(b - a, c - a)
            normal = normal / np.linalg.norm(normal)
            stream.write(struct.pack("<3f", *normal))
            for point in (a, b, c):
                stream.write(struct.pack("<3f", *point))
            stream.write(struct.pack("<H", 0))


def cell_overlap_vs_body(mesh, body_bounds) -> np.ndarray:
    points = np.asarray(mesh["points"], dtype=np.float64)
    vertices = points[np.asarray(mesh["cell_vertices"], dtype=np.int64)]
    lo = vertices.min(axis=1)
    hi = vertices.max(axis=1)
    body_min = np.asarray(body_bounds[::2], dtype=np.float64)
    body_max = np.asarray(body_bounds[1::2], dtype=np.float64)
    overlap = np.maximum(0.0, np.minimum(hi, body_max) - np.maximum(lo, body_min))
    return overlap[:, 0] * overlap[:, 1] * overlap[:, 2]


def wall_bounds(mesh, wall_patch_name=WALL_PATCH):
    faces = np.asarray(mesh["faces"])
    points = np.asarray(mesh["points"], dtype=np.float64)
    patch = next(p for p in mesh["boundary"] if p["name"] == wall_patch_name)
    vertices = points[
        np.unique(faces[patch["start_face"] : patch["start_face"] + patch["n_faces"]])
    ]
    return vertices.min(axis=0), vertices.max(axis=0)


def build_mesher(tmp_path, lo, hi, extra=None, **kwargs):
    stl = tmp_path / "body.stl"
    write_box_stl(str(stl), lo, hi)
    return AdaptiveCartesianMesher(
        domain=DOMAIN,
        max_cell_size=BACKGROUND,
        surface_file=str(stl),
        wall_patch_name=WALL_PATCH,
        surface_cell_size=SURFACE_SIZE,
        merge_outer_patch="numericalBoundary",
        **(extra or {}),
        **kwargs,
    )


def assert_conformal(mesh, body_bounds):
    overlap = cell_overlap_vs_body(mesh, body_bounds)
    assert np.count_nonzero(overlap > 1e-12) == 0, (
        f"{np.count_nonzero(overlap > 1e-12)} fluid cells overlap the body "
        f"(max {overlap.max():.6g})"
    )
    lo, hi = wall_bounds(mesh)
    tolerance = 1e-6 * max(1.0, float(np.abs(np.asarray(body_bounds)).max()))
    np.testing.assert_allclose(lo, np.asarray(body_bounds[::2]), atol=tolerance, rtol=0.0)
    np.testing.assert_allclose(hi, np.asarray(body_bounds[1::2]), atol=tolerance, rtol=0.0)


def test_a_cubeflow_is_exact_without_padding(tmp_path):
    mesher = build_mesher(tmp_path, (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    mesh = mesher.build()
    assert_conformal(mesh, mesher.surface_bounds)
    assert mesher.padding == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert mesher.effective_domain == mesher.requested_domain
    assert mesher.max_cell_size == pytest.approx(BACKGROUND)
    assert mesher._resolved_h_min == pytest.approx(SURFACE_SIZE)
    assert mesher._body_lattice_indices == (64, 128, 64, 128, 64, 128)
    assert mesh["mesh_generation"]["preserve_body_geometry"] is True


def test_b_incompatible_width_preserves_body_and_pads_outer_domain(tmp_path):
    # Width 1.03 is not divisible by the requested finest spacing; the body is
    # preserved by refining the spacing and padding the outer domain outward.
    mesher = build_mesher(tmp_path, (-0.5, -0.5, -0.5), (0.53, 0.53, 0.53))
    mesh = mesher.build()
    assert_conformal(mesh, mesher.surface_bounds)
    assert all(pad > 0.0 for pad in mesher.padding)
    assert mesher.effective_domain != mesher.requested_domain
    # The resolved spacing is finer than requested and divides the body exactly.
    assert mesher._resolved_h_min < SURFACE_SIZE
    width = mesher.surface_bounds[1] - mesher.surface_bounds[0]
    assert np.isclose(width / mesher._resolved_h_min, round(width / mesher._resolved_h_min))
    # Body lattice indices reflect the 1.03 width at the resolved spacing.
    lo, hi = mesher._body_lattice_indices[0:2]
    assert hi - lo == round(width / mesher._resolved_h_min)
    expected_lo = (
        mesher.surface_bounds[0] + mesher.padding[0] - mesher.requested_domain[0]
    ) / mesher._resolved_h_min
    assert lo == pytest.approx(expected_lo, abs=1e-9)


def test_c_shifted_float32_body_stays_exact_on_lattice(tmp_path):
    # -0.45/0.55 are not exactly representable in float32; the STL stores
    # -0.449999988 and 0.550000012.  The resolved lattice must place the wall
    # exactly on those coordinates, never on the rounded -0.45/0.55.
    mesher = build_mesher(tmp_path, (-0.45, -0.45, -0.45), (0.55, 0.55, 0.55))
    mesh = mesher.build()
    stl_min = mesher.surface_bounds[0]
    assert stl_min != pytest.approx(-0.45, abs=1e-9)  # float32 rounding is real
    assert_conformal(mesh, mesher.surface_bounds)
    assert all(pad > 0.0 for pad in mesher.padding)
    # The wall plane is the exact STL coordinate, not the rounded decimal.
    lo, _hi = wall_bounds(mesh)
    assert lo[0] == pytest.approx(stl_min, abs=1e-12)


def test_d_incommeasurable_body_raises_without_modifying_body(tmp_path):
    # A body whose axis ratios have no small-denominator form cannot be
    # represented by a single isotropic spacing; the resolver must refuse at
    # construction time, before any mesh is built.
    with pytest.raises(ValueError, match="body"):
        build_mesher(tmp_path, (-0.5, -0.5, -0.5), (0.5, 0.4999999, 0.5))


def test_g_legacy_snapping_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="preserve_body_geometry"):
        build_mesher(tmp_path, (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), preserve_body_geometry=False)


def test_wall_on_surface_postcondition_fires_on_shifted_solid():
    """Sanity check that the wall-on-surface postcondition is not vacuous: a
    mesh whose wall is one fine cell off the STL face must be rejected."""
    import numpy as np

    from source.solvers.FVM.mesh.adaptive_cartesian import _validate_wall_on_surface

    fake_mesh = {
        "points": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float
        ),
        "faces": np.array([[0, 1, 2, 3]], dtype=np.int32),
        "boundary": [{"name": "cube", "start_face": 0, "n_faces": 1}],
    }
    with pytest.raises(ValueError, match="Wall patch"):
        _validate_wall_on_surface(fake_mesh, (0.0, 1.0, 0.0, 1.0, 0.0, 1.0), "cube")


def test_overlap_validator_catches_centroid_outside_volume_penetrating_cell():
    """A cell whose centroid sits outside the body but whose AABB crosses a
    body face must still be rejected: the check is on cell volume, not the
    centroid."""
    from source.solvers.FVM.mesh.validation import validate_no_fluid_solid_overlap

    points = np.array(
        [
            [0.3, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.6, 1.0, 0.0],
            [0.3, 1.0, 0.0],
            [0.3, 0.0, 1.0],
            [0.6, 0.0, 1.0],
            [0.6, 1.0, 1.0],
            [0.3, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    mesh_data = {
        "points": points,
        "cell_vertices": np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64),
        "n_cells": 1,
    }
    # Cell centroid is (0.45, 0.5, 0.5), outside body_bounds on x alone; the
    # cell's AABB (x in [0.3, 0.6]) still crosses the body face at x=0.5 with
    # positive volume in y and z fully contained.
    body_bounds = (0.5, 1.5, -1.0, 2.0, -1.0, 2.0)
    centroid = points.mean(axis=0)
    assert not (
        body_bounds[0] <= centroid[0] <= body_bounds[1]
        and body_bounds[2] <= centroid[1] <= body_bounds[3]
        and body_bounds[4] <= centroid[2] <= body_bounds[5]
    )
    with pytest.raises(MeshValidationError, match="overlap the solid"):
        validate_no_fluid_solid_overlap(mesh_data, body_bounds)


def test_resolver_rejects_non_integer_cell_count():
    from source.solvers.FVM.mesh.adaptive_cartesian import _resolve_preserved_lattice

    with pytest.raises(ValueError, match="cell counts"):
        _resolve_preserved_lattice(
            requested_domain=DOMAIN,
            background=BACKGROUND,
            requested_level=2,
            body_bounds=(-0.5, 0.5, -0.5, 0.4999999, -0.5, 0.5),
        )
