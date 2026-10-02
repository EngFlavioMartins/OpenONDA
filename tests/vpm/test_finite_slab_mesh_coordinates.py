"""Pure coordinate properties; no induction/GPU/runtime-accuracy claims."""

import math

import numpy as np
import pytest

from tests.vpm._finite_slab_field_mesh_reference import (
    image_descriptor_shift,
    slab_coordinates,
    slab_world_images,
)


@pytest.mark.parametrize(
    ("minimum", "maximum", "spacing"),
    [(-0.48, 0.48, 0.04), (0.081, 0.794, 0.04), (1024.125, 1025.25, 0.037),
     (1e12, 1e12 + 0.001, 0.0001)],
)
def test_planes_are_exact_half_integer_indices_without_particle_snapping(minimum, maximum, spacing):
    x = np.array([[0.013, -0.017, minimum], [0.071, 0.027, maximum]])
    original = x.copy()
    lattice, steps, count = slab_coordinates(x, minimum, maximum, spacing)
    assert count == math.ceil((maximum-minimum)/spacing)
    np.testing.assert_array_equal(lattice[:, 2], [-count/2, count/2])
    assert steps[2] == (maximum-minimum)/count
    assert steps[2] <= spacing
    np.testing.assert_array_equal(steps[:2], [spacing, spacing])
    np.testing.assert_array_equal(x, original)
    for plane, k in ((0, 0), (1, 1)):
        # Odd-image coincidences at the two physical slab planes align in
        # integer arithmetic even when N is odd (plane coordinates are halves).
        assert -lattice[plane, 2] + image_descriptor_shift(k, True, count) == lattice[plane, 2]


@pytest.mark.parametrize("offset", [0., 1024., -65536.])
def test_random_interior_coordinates_and_integer_image_transforms_covary(offset):
    rng = np.random.default_rng(8306)
    minimum, maximum, spacing = offset + 0.125, offset + 1.25, 0.037
    x = rng.uniform([-0.7, -0.9, minimum], [1.1, 0.4, maximum], (17, 3))
    original = x.copy()
    lattice, steps, count = slab_coordinates(x, minimum, maximum, spacing)
    reconstructed = lattice * steps
    reconstructed[:, 2] += (minimum+maximum)/2
    np.testing.assert_allclose(reconstructed, x, rtol=0, atol=4*np.spacing(max(abs(minimum), abs(maximum))))
    descriptors = [(k, odd) for k in (-128, -1, 0, 1, 128) for odd in (False, True)]
    world, indices = slab_world_images(descriptors, minimum, maximum, count)
    for (world_shift, odd), (integer_shift, reflected) in zip(world, indices, strict=True):
        transformed = x.copy()
        expected = lattice.copy()
        if odd:
            transformed[:, 2] *= -1
            expected[:, 2] *= -1
        transformed[:, 2] += world_shift
        expected[:, 2] += integer_shift
        actual, other_steps, other_count = slab_coordinates(transformed, minimum, maximum, spacing)
        # This is a covariance test with a floating arithmetic allowance, not
        # a claim that world-space add/subtract commutes bitwise with scaling.
        allowance = 64*np.finfo(float).eps*(abs(minimum)+abs(maximum)+abs(world_shift)+1)/steps[2]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=allowance)
        np.testing.assert_array_equal(other_steps, steps)
        assert other_count == count and reflected == odd
    np.testing.assert_array_equal(x, original)


def test_common_slab_translation_and_nonintegral_width_leave_auxiliary_geometry_covariant():
    x = np.array([[0.023, -0.011, 0.187], [0.069, 0.073, 0.611]])
    original = x.copy()
    first, steps, count = slab_coordinates(x, 0.125, 1.25, 0.037)
    x[:, 2] += 1024
    second, moved_steps, moved_count = slab_coordinates(x, 1024.125, 1025.25, 0.037)
    np.testing.assert_array_equal(moved_steps, steps)
    assert count == moved_count and steps[2] != 0.037
    np.testing.assert_allclose(first, second, rtol=0, atol=2e-11)
    assert not np.array_equal(first[:, 2], np.rint(first[:, 2]))
    np.testing.assert_array_equal(x[:, :2], original[:, :2])


@pytest.mark.parametrize("descriptor", [(True, False), (1.5, False), (0, 1), (0, "odd")])
def test_invalid_or_inexact_image_descriptors_are_rejected(descriptor):
    with pytest.raises(ValueError):
        image_descriptor_shift(*descriptor, 24)


def test_image_descriptor_integer_range_and_finite_image_cap_are_explicit():
    assert image_descriptor_shift(np.int64(-2), np.bool_(True), 31) == -155
    with pytest.raises(ValueError, match="exact f64"):
        image_descriptor_shift(2**52, False, 1)
    with pytest.raises(ValueError, match="image count"):
        slab_world_images([(0, True), (1, False)], -0.48, 0.48, 24, max_images=1)
    with pytest.raises(ValueError, match="slab-to-grid"):
        slab_coordinates([[0, 0, 0]], 0, 1, 1e-20)


def test_coordinate_failures_do_not_mutate_input():
    x = np.array([[0.03, -0.07, 0.081]])
    original = x.copy()
    for args in ((1, 1, 0.04), (0, 1, 0), (0, np.inf, 0.04)):
        with pytest.raises(ValueError):
            slab_coordinates(x, *args)
    np.testing.assert_array_equal(x, original)
