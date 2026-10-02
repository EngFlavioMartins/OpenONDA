import numpy as np
import pytest

from tests.vpm._finite_image_mesh_reference import direct_finite_images
from tests.vpm._finite_slab_native_qualification import (
    compare_fields,
    direct_small_block,
    finite_blocks,
    native_tail_maxima,
)


def test_native_blocks_cover_exact_finite_family_once():
    blocks = finite_blocks()
    assert [(b["start"], b["end"]) for b in blocks] == [
        (0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128)]
    images = [image for block in blocks for image in block["images"]]
    assert len(images) == len(set(images)) == 513
    assert (0, False) not in images
    assert all(block["complete"] for block in blocks)
    assert not finite_blocks(5)[-1]["complete"]


def test_direct_chunking_reflection_and_source_core():
    x = np.array([[.03, .01, .02], [-.02, .04, .13]])
    gamma = np.array([[.02, -.03, .05], [.04, .07, -.01]])
    sigma = np.array([.04, .07])
    q = np.array([[.03, .01, .02], [.02, .03, .1]])
    descriptors = [(0, True), (1, False)]
    actual = direct_small_block(x, gamma, sigma, q, descriptors, zmin=.02, zmax=.23, chunk=1)
    expected = direct_finite_images(x, gamma, sigma, q, [(.04, True), (.42, False)])
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-14, atol=2e-14)
    with pytest.raises(ValueError, match="budget"):
        direct_small_block(x, gamma, sigma, q, descriptors, zmin=.02, zmax=.23, max_pairs=1)


def test_error_report_keeps_zero_truth_and_rejects_nonfinite():
    u, j = np.zeros((2, 3)), np.zeros((2, 3, 3))
    report = compare_fields(np.ones_like(u), j, u, j)
    assert report["velocity"]["relative_l2"] is None
    assert report["velocity"]["point_relative_norms"] == [None, None]
    assert report["velocity"]["absolute_l2"] > 0
    with pytest.raises(ValueError, match="finiteness"):
        compare_fields(u * np.nan, j, u, j)


def test_tail_maxima_and_overflow_are_not_hidden():
    u = np.array([[3., 4., 0.], [0., 0., 1.]])
    j = np.zeros((2, 3, 3))
    j[1, 0, 0] = 12
    j[1, 1, 1] = 5
    assert native_tail_maxima(u, j) == (5., 13.)
    with np.errstate(over="ignore"), pytest.raises(FloatingPointError, match="nonfinite"):
        native_tail_maxima(u * 1e38, j)
