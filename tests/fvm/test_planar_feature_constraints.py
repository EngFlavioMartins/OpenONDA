"""Feature planes keep curved-body cap seams on their input STL."""

import numpy as np
import pytest

from source.solvers.fvm.mesh.cartesian.mesher import (
    _box_patch_point_constraints,
    _planar_feature_constraints,
    _project_to_feature_planes,
)
from source.solvers.fvm.mesh.surface_classification import SurfaceIndex
from tests.fvm.cartesian_acceptance_fixtures import box_triangles


@pytest.mark.parametrize("triangulation", [2, 4])
@pytest.mark.parametrize("rotated", [False, True])
def test_diagonal_wall_face_inherits_large_planar_cap(triangulation, rotated):
    z = -0.4
    # The same planar facet must be recognized under a different tessellation
    # or coordinate frame. The optimized wall face straddles its sharp rim.
    cap = np.array(
        [
            [[-0.08, -0.09, z], [-0.05, -0.09, z], [-0.05, -0.01, z]],
            [[-0.08, -0.09, z], [-0.05, -0.01, z], [-0.08, -0.01, z]],
            [[-0.05, -0.09, z], [-0.02, -0.09, z], [-0.02, -0.01, z]],
            [[-0.05, -0.09, z], [-0.02, -0.01, z], [-0.05, -0.01, z]],
        ]
    )
    if triangulation == 2:
        cap = np.array(
            [
                [[-0.08, -0.09, z], [-0.02, -0.09, z], [-0.02, -0.01, z]],
                [[-0.08, -0.09, z], [-0.02, -0.01, z], [-0.08, -0.01, z]],
            ]
        )
    rim_end_x = -0.02 if triangulation == 2 else -0.05
    side = np.array([[-0.08, -0.09, z], [rim_end_x, -0.09, z], [-0.08, -0.09, z + 0.05]])
    surface = np.concatenate((cap, side[None]))
    points = np.array(
        [
            [-0.055, -0.054, -0.384],
            [-0.040, -0.055, -0.390],
            [-0.040, -0.040, -0.399],
            [-0.055, -0.040, -0.399],
        ]
    )
    expected = points.copy()
    expected[:, 2] = z
    if rotated:
        angle = np.deg2rad(37.0)
        rotation = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        surface = surface @ rotation.T
        points = points @ rotation.T
        expected = expected @ rotation.T
    constraints = _planar_feature_constraints(points, [np.arange(4)], surface)
    assert set(constraints) == set(range(4))
    projected = np.array(
        [
            _project_to_feature_planes(point, constraints[i], scale=1.0)
            for i, point in enumerate(points)
        ]
    )
    np.testing.assert_allclose(projected, expected, atol=1e-14)


def test_narrow_coplanar_strip_does_not_capture_a_wider_wall_face():
    z = -0.4
    strip = np.array(
        [
            [[-0.05, -0.07, z], [-0.045, -0.07, z], [-0.045, 0.03, z]],
            [[-0.05, -0.07, z], [-0.045, 0.03, z], [-0.05, 0.03, z]],
            [[-0.05, -0.07, z], [-0.045, -0.07, z], [-0.05, -0.07, z + 0.05]],
        ]
    )
    face = np.array(
        [
            [-0.055, -0.054, -0.384],
            [-0.040, -0.055, -0.390],
            [-0.040, -0.040, -0.399],
            [-0.055, -0.040, -0.399],
        ]
    )
    assert _planar_feature_constraints(face, [np.arange(4)], strip) == {}


def test_smooth_facet_transition_does_not_create_a_feature_plane():
    z = -0.4
    surface = np.array(
        [
            [[-0.08, -0.09, z], [-0.02, -0.09, z], [-0.02, -0.01, z]],
            [[-0.08, -0.09, z], [-0.02, -0.01, z], [-0.08, -0.01, z]],
            [[-0.02, -0.09, z], [-0.08, -0.09, z], [-0.08, -0.10, z + 0.001]],
        ]
    )
    face = np.array(
        [
            [-0.055, -0.054, -0.384],
            [-0.040, -0.055, -0.390],
            [-0.040, -0.040, -0.399],
            [-0.055, -0.040, -0.399],
        ]
    )
    assert _planar_feature_constraints(face, [np.arange(4)], surface) == {}


def test_diagonal_box_rim_face_recovers_a_real_input_plane():
    surface = box_triangles((0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    face = np.array(
        [
            [0.02, 0.0, 0.2],
            [0.02, 0.0, 0.3],
            [0.0, 0.02, 0.3],
            [0.0, 0.02, 0.2],
        ]
    )
    ids = np.arange(4)
    features = _planar_feature_constraints(face, [ids], surface)
    constraints = _box_patch_point_constraints(
        face, [ids], (0.0, 1.0, 0.0, 1.0, 0.0, 1.0), features
    )
    assert len(constraints) == 4
    assert len({tuple(value.items()) for value in constraints.values()}) == 1
    assert next(iter(constraints.values())) in ({0: 0.0}, {1: 0.0})


def test_inconsistent_feature_planes_are_rejected():
    point = np.array([0.0, 0.0, 0.0])
    facet = SurfaceIndex.build(np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]))
    with pytest.raises(ValueError, match="inconsistent"):
        _project_to_feature_planes(
            point,
            [(np.array([0.0, 0.0, 1.0]), 0.0, facet), (np.array([0.0, 0.0, 1.0]), 1.0, facet)],
            scale=1.0,
        )


def test_projection_stays_on_finite_feature_near_adjacent_facet():
    facet = SurfaceIndex.build(np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]))
    # The infinite supporting plane contains (0.8, 0.8, 0), but the actual
    # facet ends at x+y=1. A wall vertex must land on that finite edge.
    point = np.array([0.8, 0.8, 0.2])
    mapped = _project_to_feature_planes(point, [(np.array([0.0, 0.0, 1.0]), 0.0, facet)], scale=1.0)
    np.testing.assert_allclose(mapped, [0.5, 0.5, 0.0], atol=1e-14)
    assert facet.nearest_point(mapped)[1] <= 1e-14


def test_two_feature_projection_clamps_to_shared_finite_edge():
    horizontal = SurfaceIndex.build(np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]))
    vertical = SurfaceIndex.build(np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]))
    mapped = _project_to_feature_planes(
        np.array([1.2, 0.2, 0.2]),
        [
            (np.array([0.0, 0.0, 1.0]), 0.0, horizontal),
            (np.array([0.0, 1.0, 0.0]), 0.0, vertical),
        ],
        scale=1.0,
    )
    np.testing.assert_allclose(mapped, [1.0, 0.0, 0.0], atol=1e-14)
