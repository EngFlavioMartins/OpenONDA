"""Outward cutoff validation: all four actual classifier omission paths."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
import math

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh.coordinates import finite_images
from source.solvers.vpm.physics.induction.gaussian_mesh.correction import _possibly_near
from source.solvers.vpm.physics.induction.gaussian_mesh.correction_distance import (
    correction_classification_bound,
    correction_omission_radius,
)
from source.solvers.vpm.physics.induction.gaussian_mesh.error_bounds import (
    finite_image_correction_bound,
)
from source.solvers.vpm.physics.induction.gaussian_tail.error_bounds import prepare_tail_source


def _snapshot(x, *, zmin=-0.5, zmax=0.5):
    x = np.asarray(x, np.float64).reshape(-1, 3)
    return prepare_tail_source(
        x, np.ones_like(x) * 0.1, np.full(len(x), 0.04), z_min=zmin, z_max=zmax
    )


def _device_omission(source, target, source_min, shape, cutoff, shift, odd):
    """Literal binary64 host rendition; separate paths remain inspectable.

    CUDA may fuse squared-norm operations; the proof accepts either ordering.
    This is not a replacement GPU test or a claim Python emulates atomics.
    """
    inverse = target.copy()
    inverse[2] = shift - target[2] if odd else target[2] - shift
    upper = source_min + (np.asarray(shape, np.float64) + 1.0) * cutoff
    if np.any(inverse < source_min - cutoff) or np.any(inverse > upper):
        return "early_grid"
    scale = max(1.0, abs(shift), *map(abs, target), *map(abs, source_min))
    margin = 64 * np.finfo(float).eps * scale
    assert np.isfinite(inverse).all() and margin <= cutoff / 8
    lower_key = np.maximum(0, np.floor(((inverse - cutoff) - margin - source_min) / cutoff)).astype(
        np.int64
    )
    upper_key = np.minimum(
        np.asarray(shape) - 1, np.floor(((inverse + cutoff) + margin - source_min) / cutoff)
    ).astype(np.int64)
    source_key = np.floor((source - source_min) / cutoff).astype(np.int64)
    if np.any(source_key < lower_key) or np.any(source_key > upper_key):
        return "cell_range"
    transformed = source.copy()
    transformed[2] = shift - source[2] if odd else shift + source[2]
    d = target - transformed
    squared = (d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]
    return "radius" if squared >= cutoff**2 else None


def _physical_squared(source, target, k, odd, zmin, zmax):
    transformed = list(map(Fraction, source))
    shift = 2 * k * (Fraction(zmax) - Fraction(zmin))
    transformed[2] = shift + (2 * Fraction(zmin) - transformed[2] if odd else transformed[2])
    return sum((Fraction(target[i]) - transformed[i]) ** 2 for i in range(3))


@pytest.mark.parametrize("offset", [0.0, 2.0**20, -(2.0**30)])
def test_every_classifier_omission_respects_exact_rational_physical_radius(offset):
    rng = np.random.default_rng(1997)
    zmin, zmax, cutoff = offset - 0.48, offset + 0.48, 0.6
    x = rng.uniform([-0.8, -0.7, -0.45], [0.9, 0.8, 0.45], (9, 3)) + offset
    snapshot = _snapshot(x, zmin=zmin, zmax=zmax)
    images = [(k, odd) for k in (-128, -2, 0, 1, 128) for odd in (False, True)]
    targets = []
    for k, odd in images:
        shift = 2 * k * (zmax - zmin) + (2 * zmin if odd else 0.0)
        center = x[k % len(x)].copy()
        center[2] = shift - center[2] if odd else shift + center[2]
        for axis in range(3):
            for value in (
                np.nextafter(cutoff, 0.0),
                cutoff,
                np.nextafter(cutoff, np.inf),
                0.99 * cutoff,
                1.01 * cutoff,
                3 * cutoff,
            ):
                q = center.copy()
                q[axis] += value
                targets.append(q)
    targets = np.asarray(targets)
    result = correction_classification_bound(
        snapshot, targets.min(0), targets.max(0), cutoff=cutoff, images=images
    )
    source_min, source_max = x.min(0), x.max(0)
    shape = np.floor((source_max - source_min) / cutoff).astype(np.int64) + 1
    paths = set()
    for (k, odd), (shift, _) in zip(images, result.world_images, strict=True):
        for q in targets:
            whole_image_skip = not _possibly_near(source_min, source_max, q, q, shift, odd, cutoff)
            for source in x:
                path = (
                    "host_aabb"
                    if whole_image_skip
                    else _device_omission(source, q, source_min, shape, cutoff, shift, odd)
                )
                if path:
                    paths.add(path)
                    assert Fraction(result.omitted_distance_lower) ** 2 <= _physical_squared(
                        source, q, k, odd, zmin, zmax
                    )
    assert {"host_aabb", "radius", "cell_range"} <= paths


def test_early_grid_and_cell_floor_decisions_cover_strict_interior_without_aabb():
    # Wide source bounds exercise cell keys on both sides of many integer
    # boundaries and the device early return independently of host exclusions.
    x = np.array([[-3.0, -0.7, -0.4], [3.0, 0.7, 0.4], [0.1, 0.2, 0.3]])
    snapshot = _snapshot(x)
    cutoff = 0.3
    source_min = x.min(0)
    shape = np.floor((x.max(0) - source_min) / cutoff).astype(np.int64) + 1
    targets = np.array([[v, 0.0, 0.0] for v in np.linspace(-4.0, 4.0, 65)])
    result = correction_classification_bound(
        snapshot, targets.min(0), targets.max(0), cutoff=cutoff, images=[(0, False)]
    )
    paths = set()
    for source in x:
        for q in targets:
            path = _device_omission(source, q, source_min, shape, cutoff, 0.0, False)
            if path:
                paths.add(path)
                assert Fraction(result.omitted_distance_lower) ** 2 <= _physical_squared(
                    source, q, 0, False, -0.5, 0.5
                )
    assert "early_grid" in paths and "cell_range" in paths


def test_world_shifts_match_real_operator_and_enclose_exact_slab_formula():
    zmin, zmax = 2.0**24 - 0.48, 2.0**24 + 0.49
    x = np.array([[0.1, 0.2, zmin + 0.1], [0.2, 0.3, zmax - 0.1]])
    snapshot = _snapshot(x, zmin=zmin, zmax=zmax)
    images = [(0, False), (0, True), (-128, True), (128, False), (128, True)]
    result = correction_classification_bound(
        snapshot, x.min(0), x.max(0), cutoff=0.6, images=images
    )
    _, actual_world, _ = finite_images(images, zmin, zmax, 29, 513, include_primary=True)
    assert result.world_images == actual_world
    for (k, odd), (represented, _) in zip(images, result.world_images, strict=True):
        exact = 2 * k * (Fraction(zmax) - Fraction(zmin)) + (2 * Fraction(zmin) if odd else 0)
        assert abs(Fraction(represented) - exact) <= Fraction(result.world_shift_error_upper)
    assert result.omitted_distance_lower < 0.6
    assert 0.6 - result.omitted_distance_lower < 1e-4


@pytest.mark.parametrize("offset", [0.0, 2.0**28, -(2.0**28)])
def test_adjacent_cell_indices_and_both_sides_of_cutoff_keep_every_proven_interior(offset):
    cutoff = 0.6
    rows = []
    for cell in range(4):
        boundary = np.float64(offset + cell * cutoff)
        for xx in (np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf)):
            rows.append([xx, offset + 0.1, offset + 0.13])
    x = np.asarray(rows)
    zmin, zmax = offset - 0.48, offset + 0.48
    snapshot = _snapshot(x, zmin=zmin, zmax=zmax)
    images = [(0, False), (0, True), (1, True), (-2, False)]
    source_min = x.min(0)
    shape = np.floor((x.max(0) - source_min) / cutoff).astype(np.int64) + 1
    qlo, qhi = x.min(0) - 5.0, x.max(0) + 5.0
    result = correction_classification_bound(snapshot, qlo, qhi, cutoff=cutoff, images=images)
    tested = 0
    for (k, odd), (shift, _) in zip(images, result.world_images, strict=True):
        for source in x:
            image = source.copy()
            image[2] = shift - source[2] if odd else shift + source[2]
            for sign in (-1, 1):
                for axis in range(3):
                    q = image.copy()
                    # Leave twice the validated arithmetic slack, not an
                    # arbitrary exact-equality assumption at a rounded face.
                    q[axis] += sign * (2 * result.omitted_distance_lower - cutoff)
                    if (
                        _physical_squared(source, q, k, odd, zmin, zmax)
                        < Fraction(result.omitted_distance_lower) ** 2
                    ):
                        tested += 1
                        assert _possibly_near(source_min, x.max(0), q, q, shift, odd, cutoff)
                        assert (
                            _device_omission(source, q, source_min, shape, cutoff, shift, odd)
                            is None
                        )
    assert tested > 200


def test_snapshot_and_query_mutation_cannot_change_owned_result_and_scalar_api_matches():
    x = np.array([[0.1, 0.2, 0.3]])
    lower, upper = np.array([-0.2, -0.2, -0.5]), np.array([0.2, 0.2, 0.5])
    snapshot = _snapshot(x)
    images = [(0, True), (1, True), (0, True)]
    result = correction_classification_bound(snapshot, lower, upper, cutoff=0.6, images=images)
    assert (
        correction_omission_radius(snapshot, lower, upper, cutoff=0.6, images=images)
        == result.omitted_distance_lower
    )
    saved = result.query_lower
    x[:] = 100
    lower[:] = -100
    images.clear()
    assert result.query_lower == saved and len(result.descriptors) == 3
    with pytest.raises(FrozenInstanceError):
        result.cutoff = 2.0


def test_validation_radius_composes_with_finite_correction_envelope():
    x = np.array([[0.1, 0.2, -0.46], [-0.2, 0.3, 0.46]])
    snapshot = _snapshot(x, zmin=-0.48, zmax=0.48)
    images = [(k, odd) for k in range(-128, 129) for odd in (False, True) if k or odd]
    result = correction_classification_bound(
        snapshot, x.min(0), x.max(0), cutoff=0.6, images=images
    )
    bound = finite_image_correction_bound(
        snapshot,
        x.min(0),
        x.max(0),
        tau=0.12,
        images=images,
        omitted_distance_lower=result.omitted_distance_lower,
    )
    exact_cutoff = finite_image_correction_bound(
        snapshot, x.min(0), x.max(0), tau=0.12, images=images, omitted_distance_lower=0.6
    )
    assert (
        exact_cutoff.velocity_upper <= bound.velocity_upper < 1.000001 * exact_cutoff.velocity_upper
    )
    assert (
        exact_cutoff.gradient_upper <= bound.gradient_upper < 1.000001 * exact_cutoff.gradient_upper
    )


@pytest.mark.parametrize(
    "cutoff", [0.0, -1.0, float("nan"), float("inf"), True, 1, np.longdouble(0.6), 1e-200, 1e200]
)
def test_invalid_or_unresolved_cutoff_declines(cutoff):
    snapshot = _snapshot([[0.0, 0.0, 0.0]])
    with pytest.raises((ValueError, FloatingPointError)):
        correction_omission_radius(
            snapshot, np.zeros(3), np.ones(3), cutoff=cutoff, images=[(0, True)]
        )


def test_unresolved_translation_and_excessive_grid_decline_before_integer_casts():
    huge = 2.0**50
    x = np.array([[huge, huge, 0.0]])
    snapshot = _snapshot(x)
    with pytest.raises(ValueError, match="unresolved"):
        correction_omission_radius(snapshot, x[0], x[0], cutoff=0.6, images=[(0, True)])
    x = np.array([[-1e4, -1e4, 0.0], [1e4, 1e4, 0.0]])
    snapshot = _snapshot(x)
    with pytest.raises(ValueError, match="cell-count"):
        correction_omission_radius(snapshot, x.min(0), x.max(0), cutoff=0.1, images=[(0, True)])


def test_image_iterator_is_bounded_and_invalid_descriptors_decline():
    snapshot = _snapshot([[0.0, 0.0, 0.0]])
    seen = []

    def images():
        for i in range(100):
            seen.append(i)
            yield (i, False)

    with pytest.raises(ValueError, match="cap"):
        correction_omission_radius(
            snapshot, np.zeros(3), np.ones(3), cutoff=0.6, images=images(), max_images=3
        )
    assert seen == [0, 1, 2, 3]
    for image in ([(True, False)], [(0, 1)], [(2**31, False)], [(math.nan, False)]):
        with pytest.raises(ValueError):
            correction_omission_radius(snapshot, np.zeros(3), np.ones(3), cutoff=0.6, images=image)
