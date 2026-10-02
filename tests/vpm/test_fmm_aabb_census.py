"""Conservatism checks for tighter bounds, not a changed source partition."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from tests.vpm._fmm_aabb_census_prototype import (
    SourceMetadata,
    aabb_distance_bounds,
    classify_aabb,
    legacy_point_accept,
    transform_query,
)


def _assert_conservative(source, points, shift=0.0, odd=False):
    low, high = points.min(axis=0), points.max(axis=0)
    classification = classify_aabb(source, low, high, shift, odd)
    decisions = [legacy_point_accept(source, point, shift, odd) for point in points]
    if classification == "all":
        assert all(decisions)
    elif classification == "none":
        assert not any(decisions)
    bounds = aabb_distance_bounds(low, high, source.com, shift, odd)
    if bounds is not None:
        displacement = transform_query(points, shift, odd) - np.asarray(source.com, np.float32)
        squares = np.sum(displacement * displacement, axis=1, dtype=np.float32)
        distances = np.sqrt(squares, dtype=np.float32)
        assert np.all(bounds[0] <= squares) and np.all(squares <= bounds[1])
        assert np.all(bounds[2] <= distances) and np.all(distances <= bounds[3])
    return classification


@pytest.mark.parametrize("odd", [False, True])
@pytest.mark.parametrize("shift", [0.0, 1e8, -1e8])
def test_extreme_midpoint_rounding_and_image_subtraction_are_enclosed(odd, shift):
    points = np.array(list(product([1e8, 1e8 + 8], [4.0], [1e8 + 8, 1e8 + 16])), np.float32)
    query = transform_query(points, shift, odd)
    for offset in (-80, 80, 88):
        centre = query[0].copy()
        centre[0] += np.float32(offset)
        source = SourceMetadata(centre, centre, 4, 1, 1, 1, np.array([0, 1, 0]))
        _assert_conservative(source, points, shift, odd)


@pytest.mark.parametrize("scale", [1e-20, 1e-5, 1.0, 1e10])
@pytest.mark.parametrize("cores", ["common", "mixed", "cancelled"])
def test_random_anisotropic_boxes_preserve_exact_pointwise_admission(scale, cores):
    rng = np.random.default_rng(59153)
    points = (rng.uniform(-1, 1, (257, 3)) * [2, 0.001, 0.1] * scale).astype(np.float32)
    for z in (0.01, 0.5, 3, 30):
        centre = np.array([0.2, -0.3, z], np.float32) * np.float32(scale)
        source = SourceMetadata(centre, centre, 0.02 * scale, 0.01 * scale, 0.01 * scale, 0.01 * scale, np.array([0, 1, 0]))
        if cores == "mixed":
            source = replace(source, min_core=0.002 * scale, max_core=0.05 * scale)
        elif cores == "cancelled":
            source = replace(source, strength=np.zeros(3))
        _assert_conservative(source, points)


def test_strict_mac_mean_core_and_tail_thresholds_are_not_crossed():
    centre = np.zeros(3, np.float32)
    source = SourceMetadata(centre, centre, 0.1, 0.02, 0.02, 0.02, np.array([0, 1, 0]))
    for threshold, candidate in (
        (2.0, source),
        (0.02, replace(source, half_size=0)),
        (0.45, replace(source, half_size=0.1, min_core=0.002, max_core=0.05)),
    ):
        radius = np.float32(threshold)
        nearby = [np.nextafter(radius, -np.inf, dtype=np.float32), radius, np.nextafter(radius, np.inf, dtype=np.float32)]
        points = np.array([[0.0, 0.0, r] for r in nearby], np.float32)
        _assert_conservative(candidate, points)


def test_tight_box_bound_can_classify_an_elongated_packet_without_splitting():
    source = SourceMetadata(np.zeros(3), np.zeros(3), 0.05, 0.001, 0.001, 0.001, np.array([0, 1, 0]))
    points = np.array(list(product([-10, 10], [2, 2.01], [0, 0.001])), np.float32)
    # Its radius is about10, so a centre-distance2 bounding sphere overlaps
    # the source. The actual box stays2 away and all points pass diameter/r<.1.
    assert _assert_conservative(source, points) == "all"


def test_overflowed_interval_declines_instead_of_claiming_acceptance():
    assert aabb_distance_bounds(np.ones(3) * 1e30, np.ones(3) * 2e30, np.zeros(3)) is None
