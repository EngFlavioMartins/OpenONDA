"""Reject coordinate arithmetic that would silently wrap NumPy grid indices."""

import numpy as np
import pytest

from tests.vpm._finite_image_field_mesh_reference import _gather_fields
from tests.vpm._finite_image_mesh_reference import _assign, _gather_derivatives
from tests.vpm._finite_slab_field_mesh_reference import finite_slab_field_mesh


@pytest.mark.parametrize("first", [(-1, 0, 0), (0, 5, 0)])
def test_assignment_and_both_gather_paths_reject_out_of_domain_stencils(first):
    weights = np.ones((1, 3, 3, 4))
    indices = np.array([first], dtype=np.int64)
    for operation in (
        lambda: _assign((8, 8, 8), indices, weights, np.ones((1, 3))),
        lambda: _gather_derivatives(np.zeros((8, 8, 8, 3)), indices, weights),
        lambda: _gather_fields(np.zeros((8, 8, 8, 12)), indices, weights),
    ):
        with pytest.raises(ValueError, match="outside.*linear-convolution"):
            operation()


def test_finite_unresolved_translated_geometry_is_rejected_without_mutation():
    x, gamma, sigma = np.array([[1e20, 0., .1]]), np.ones((1, 3)), np.array([.04])
    before = x.copy()
    with pytest.raises((ValueError, FloatingPointError), match="stencil|grid"):
        finite_slab_field_mesh(x, gamma, sigma, x, [(0, True)], zmin=0., zmax=.193,
                               tau=.12, spacing=.04, order=10)
    np.testing.assert_array_equal(x, before)
