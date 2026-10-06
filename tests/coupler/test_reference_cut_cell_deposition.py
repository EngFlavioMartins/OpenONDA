"""Analytic moment and visibility guards for a frozen cut-cell control."""

import numpy as np
import pytest

from source.coupler.geometry import SolidBoundary, TriangulatedWall
from tests.support.cylinder.reference_cut_cell_deposition import deposit_cut_cell_circulation


def lattice():
    x, y = np.meshgrid(.04 * np.arange(-5, 6), .04 * np.arange(-5, 6), indexing="ij")
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))


def flat_wall(xmax=.017):
    return TriangulatedWall.from_box((-1., xmax, -1., 1., -.5, .5),
                                    (-2., 2., -2., 2., -1., 1.))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inside_control_cells_preserve_signed_circulation_and_xy_moments(dtype):
    boundary = SolidBoundary((flat_wall(),))
    nodes = lattice()
    centres = np.array([[0., -.04, 0.], [0., .04, 0.]])
    gamma = np.array([1., -.7])
    indices, strength, report = deposit_cut_cell_circulation(
        centres, gamma, nodes, .04, boundary, dtype)
    assert np.isclose(strength.sum(), gamma.sum(), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.sum(nodes[indices, :2] * strength[:, None], axis=0),
                               np.sum(centres[:, :2] * gamma[:, None], axis=0), rtol=0, atol=1e-13)
    assert not boundary.contains(nodes[indices].astype(dtype)).any()
    assert report["donor_count"] == 2
    for donor in report["donors"]:
        chosen = nodes[donor["node_indices"]].astype(dtype)
        starts = np.broadcast_to(donor["visibility_origin"], chosen.shape)
        assert not boundary.blocks_segments(starts, chosen).any()
        assert donor["weight_l1"] <= 2 and donor["radius"] <= 4
        assert donor["maximum_constraint_residual"] <= 1e-10


def test_secondary_wall_blocks_otherwise_exterior_deposition_support():
    blocker = TriangulatedWall.from_box((.08, .10, -.035, .035, -.5, .5),
                                       (-2., 2., -2., 2., -1., 1.))
    boundary = SolidBoundary((flat_wall(), blocker))
    nodes = lattice()
    indices, _, report = deposit_cut_cell_circulation(
        np.zeros((1, 3)), np.ones(1), nodes, .04, boundary)
    assert len(indices)
    selected = report["donors"][0]
    start = np.broadcast_to(selected["visibility_origin"], nodes.shape)
    hidden = (~boundary.contains(nodes.astype(np.float32))) & boundary.blocks_segments(
        start, nodes.astype(np.float32))
    assert hidden.any()
    assert not np.intersect1d(indices, np.flatnonzero(hidden)).size


def test_storage_geometry_and_unresolved_support_fail_without_relaxing_guards():
    nodes = lattice()
    boundary = SolidBoundary((TriangulatedWall.from_box(
        (.3, 1., -1., 1., -.5, .5), (-2., 2., -2., 2., -1., 1.)),))
    # The nominal wall node x=.3 rounds into the solid at float32 storage.
    nodes[:, 0] += .3
    at_wall = nodes[np.isclose(nodes[:, 0], .3)]
    assert boundary.contains(at_wall.astype(np.float32)).all()
    indices, _, _ = deposit_cut_cell_circulation(
        np.array([[.28, 0., 0.]]), np.ones(1), nodes, .04, boundary, np.float32)
    assert not boundary.contains(nodes[indices].astype(np.float32)).any()
    thin_support = nodes[np.isclose(nodes[:, 0], .26)]
    with pytest.raises(ValueError, match="insufficient wall-visible"):
        deposit_cut_cell_circulation(np.array([[.28, 0., 0.]]), np.ones(1), thin_support,
                                    .04, boundary, np.float32)
    with pytest.raises(ValueError, match="deeper than"):
        deposit_cut_cell_circulation(np.array([[.36, 0., 0.]]), np.ones(1), nodes,
                                    .04, boundary, np.float32)
    with pytest.raises(ValueError, match="cannot relax"):
        deposit_cut_cell_circulation(np.array([[.3, 0., 0.]]), np.ones(1), nodes,
                                    .04, boundary, maximum_weight_l1=2.1)
