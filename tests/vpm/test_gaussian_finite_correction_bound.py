"""Finite correction AABB proof checks; no CuPy/solver evaluation required."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
import itertools

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh.error_bounds import (
    finite_image_correction_bound,
    local_correction_bound,
)
from source.solvers.vpm.physics.induction.gaussian_tail.error_bounds import (
    prepare_tail_source,
    validate_source_values,
)
from tests.vpm._gaussian_broadening_reference import correction_fields


def _source(offset=0.0, dtype=np.float64):
    zmin, zmax = offset - 0.48, offset + 0.48
    x = np.array([[-0.2, 0.11, -0.46], [0.3, -0.2, 0.45], [0.04, 0.03, 0.07]], dtype=dtype)
    x[:, 2] += offset
    g = np.array([[0.2, -0.3, 0.4], [-0.2, 0.3, -0.4], [0.0, 0.0, 0.01]], dtype=dtype)
    core = np.array([0.025, 0.04, 0.055], dtype=dtype)
    snapshot = prepare_tail_source(x, g, core, z_min=zmin, z_max=zmax)
    return snapshot, x, g, core, zmin, zmax


def _bound(snapshot, lower, upper, images, **kwargs):
    return finite_image_correction_bound(
        snapshot, lower, upper, images=images, tau=0.12, omitted_distance_lower=0.6, **kwargs
    )


@pytest.mark.parametrize("offset", [0.0, 2.0**20, -(2.0**30)])
def test_outward_distance_never_exceeds_exact_rational_pair_distance(offset):
    snapshot, x, _, _, zmin, zmax = _source(offset)
    lower = np.array([-0.13, -0.15, zmin])
    upper = np.array([0.27, 0.11, zmax])
    images = [(k, odd) for k in (-128, -2, -1, 0, 1, 2, 128) for odd in (False, True)]
    result = finite_image_correction_bound(
        snapshot, lower, upper, images=images, tau=0.12, omitted_distance_lower=0.6
    )
    # Only the geometrically separated images claim a universal pair distance;
    # overlapping images rely on the separate omitted-distance prerequisite.
    period = 2 * (Fraction(zmax) - Fraction(zmin))
    corners = tuple(itertools.product(*zip(lower, upper, strict=True)))
    for (k, odd), lower_distance in zip(images, result.distance_lower_bounds, strict=True):
        if lower_distance <= 0.6:
            continue
        for source in x:
            transformed = list(map(Fraction, source))
            transformed[2] = period * k + (
                2 * Fraction(zmin) - transformed[2] if odd else transformed[2]
            )
            closest = tuple(
                min(max(transformed[i], Fraction(lower[i])), Fraction(upper[i])) for i in range(3)
            )
            for target in (*corners, closest):
                squared = sum((Fraction(target[i]) - transformed[i]) ** 2 for i in range(3))
                assert Fraction(lower_distance) ** 2 <= squared


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_all_descriptors_preserved_and_only_nearest_two_images_cutoff_limited(dtype):
    snapshot, x, g, _, _, _ = _source(dtype=dtype)
    images = [(k, odd) for k in range(-128, 129) for odd in (False, True) if k or odd]
    result = _bound(snapshot, x.min(axis=0), x.max(axis=0), images)
    old = local_correction_bound(g, tau=0.12, omitted_distance_lower=0.6, image_count=len(images))
    assert result.descriptors == tuple(images)
    assert result.image_count == 513 and result.cutoff_limited_images == 2
    assert 0 < result.velocity_upper < old.velocity_upper / 200
    assert 0 < result.gradient_upper < old.gradient_upper / 200
    assert result.velocity_upper >= sum(result.velocity_upper_by_image)
    assert result.gradient_upper >= sum(result.gradient_upper_by_image)
    assert type(result.distance_lower_bounds) is tuple
    with pytest.raises(FrozenInstanceError):
        result.image_count = 1


def test_vector_series_matches_scalar_sum_and_duplicates_are_charged():
    snapshot, x, _, _, _, _ = _source()
    images = [(0, False), (0, True), (1, True), (4, False), (4, False)]
    result = _bound(snapshot, x.min(axis=0), x.max(axis=0), images)
    # Using a single positive strength gives exactly the same validated scalar
    # input plus its scalar summation slack; compare tight arithmetic enclosure.
    g = np.array([[result.absolute_strength_upper, 0.0, 0.0]])
    for radius, u, j in zip(
        result.distance_lower_bounds,
        result.velocity_upper_by_image,
        result.gradient_upper_by_image,
        strict=True,
    ):
        scalar = local_correction_bound(g, tau=0.12, omitted_distance_lower=radius, image_count=1)
        assert u <= scalar.velocity_upper <= u * (1 + 1e-12)
        assert j <= scalar.gradient_upper <= j * (1 + 1e-12)
    assert result.velocity_upper_by_image[-1] == result.velocity_upper_by_image[-2]
    assert result.gradient_upper_by_image[-1] == result.gradient_upper_by_image[-2]


@pytest.mark.parametrize("scale", [2.0**-8, 1.0, 2.0**8])
def test_bound_dominates_actual_omitted_variable_core_correction_at_box_points(scale):
    snapshot, x, g, core, zmin, zmax = _source()
    x, core, zmin, zmax = x * scale, core * scale, zmin * scale, zmax * scale
    snapshot = prepare_tail_source(x, g, core, z_min=zmin, z_max=zmax)
    lower, upper = (
        np.array([-0.25 * scale, -0.3 * scale, zmin]),
        np.array([0.35 * scale, 0.3 * scale, zmax]),
    )
    images = [(k, odd) for k in range(-3, 4) for odd in (False, True)]
    cutoff, tau = 0.14 * scale, 0.12 * scale
    result = finite_image_correction_bound(
        snapshot, lower, upper, images=images, tau=tau, omitted_distance_lower=cutoff
    )
    rng = np.random.default_rng(94)
    targets = np.vstack((lower, upper, rng.uniform(lower, upper, (8, 3))))
    for q in targets:
        absolute_u = absolute_j = 0.0
        for k, odd in images:
            transformed, vector = x.copy(), g.copy()
            transformed[:, 2] = 2 * k * (zmax - zmin) + (
                2 * zmin - transformed[:, 2] if odd else transformed[:, 2]
            )
            if odd:
                vector[:, :2] *= -1
            for source, gamma, sigma in zip(transformed, vector, core, strict=True):
                displacement = q - source
                if np.linalg.norm(displacement) >= cutoff:
                    u, j = correction_fields(displacement, gamma, sigma, tau)
                    absolute_u += np.linalg.norm(u)
                    absolute_j += np.linalg.norm(j)
        assert absolute_u <= result.velocity_upper
        assert absolute_j <= result.gradient_upper


def test_snapshot_owns_values_caller_mutations_do_not_change_error_bound():
    snapshot, x, g, core, zmin, zmax = _source()
    lower, upper = x.min(axis=0), x.max(axis=0)
    images = [(0, True), (1, True)]
    before = _bound(snapshot, lower, upper, images)
    x[:] = 999.0
    g[:] = 111.0
    core[:] = 0.09
    after = _bound(snapshot, lower, upper, images)
    assert before == after
    with pytest.raises(ValueError, match="changed"):
        validate_source_values(snapshot, x, g, core, z_min=zmin, z_max=zmax)


def test_invalid_core_coverage_checked_even_for_zero_strength():
    x = np.zeros((1, 3))
    snapshot = prepare_tail_source(x, x, np.array([0.13]), z_min=-0.48, z_max=0.48)
    with pytest.raises(ValueError, match="actual source-only core"):
        _bound(snapshot, np.zeros(3), np.ones(3), [])


def test_empty_images_and_zero_sources_return_zero_without_dropping_descriptors():
    snapshot, x, _, _, _, _ = _source()
    result = _bound(snapshot, x.min(axis=0), x.max(axis=0), [])
    assert result.image_count == 0 and result.velocity_upper == result.gradient_upper == 0
    zero = prepare_tail_source(
        np.empty((0, 3)), np.empty((0, 3)), np.empty(0), z_min=-0.48, z_max=0.48
    )
    result = _bound(zero, np.zeros(3), np.ones(3), [(0, False), (1, True)])
    assert result.image_count == 2 and result.velocity_upper == result.gradient_upper == 0


def test_descriptor_snapshot_is_bounded_and_input_types_fail_closed():
    snapshot, x, _, _, _, _ = _source()
    visited = []

    def descriptors():
        for k in range(4):
            visited.append(k)
            yield k, True
        raise AssertionError("cannot consume beyond cap+1")

    with pytest.raises(ValueError, match="exceeds cap"):
        _bound(snapshot, x.min(axis=0), x.max(axis=0), descriptors(), max_images=3)
    assert visited == list(range(4))
    for images in ([(True, False)], [(1.5, True)], [(1, 1)], [(2**31, True)], [None]):
        with pytest.raises(ValueError, match="index|descriptor"):
            _bound(snapshot, x.min(axis=0), x.max(axis=0), images)
    for lower in (
        np.zeros(3, dtype=np.int64),
        np.zeros(3, dtype=np.longdouble),
        np.full(3, np.nan),
    ):
        with pytest.raises(ValueError, match="binary32/binary64"):
            _bound(snapshot, lower, x.max(axis=0), [(0, True)])
    with pytest.raises(TypeError, match="snapshot"):
        _bound(object(), np.zeros(3), np.ones(3), [(0, True)])


def test_interval_overflow_fails_closed_without_mutating_source():
    snapshot, _, _, _, _, _ = _source()
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(FloatingPointError):
        _bound(snapshot, np.full(3, 1e200), np.full(3, 2e200), [(0, True)])
