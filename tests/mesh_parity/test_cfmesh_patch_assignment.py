"""Patch-assignment weights remain finite for faces exactly on a surface."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from source.solvers.fvm.mesh.cartesian.cfmesh_template import _patch_alignment_weight


@pytest.mark.parametrize(
    "maximum_squared,distance_squared,alignment",
    [
        (25.0, 0.0, 1.0),
        (25.0, 0.0, 0.0),
        (25.0, 1.0e-310, 0.5),
        (25.0, 1.0e-300, 0.25),
        (np.finfo(np.float64).max, 0.0, 1.0),
        (np.finfo(np.float64).max, np.finfo(np.float64).max, 1.0),
        (0.0, 0.0, 1.0),
        (np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0), 0.5),
        (9.0, 4.0, 0.75),
    ],
)
def test_alignment_weight_matches_high_precision_without_overflow(
    maximum_squared, distance_squared, alignment
):
    # Evaluate the original score at sufficient precision and exponent range
    # to distinguish a representable final result from an overflowing ratio.
    with localcontext() as context:
        context.prec = 100
        denominator = max(float(distance_squared), float(np.finfo(np.float64).tiny))
        expected = float(
            (Decimal.from_float(float(maximum_squared)) / Decimal.from_float(denominator)).sqrt()
            * Decimal.from_float(alignment)
        )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        weight = _patch_alignment_weight(maximum_squared, distance_squared, alignment)
    assert np.isfinite(weight)
    assert weight == pytest.approx(expected, rel=4.0e-16, abs=0.0)


def test_zero_distance_with_perpendicular_normal_does_not_poison_patch_selection():
    candidates = [(0, 0.0, 0.0), (1, 0.0, 0.5), (2, 25.0, 1.0)]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        best = max(candidates, key=lambda item: _patch_alignment_weight(25.0, item[1], item[2]))
    assert best[0] == 1


def test_all_zero_distances_retain_native_candidate_order_tie():
    candidates = [(3, 0.0, 0.0), (1, 0.0, 1.0)]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        best = max(candidates, key=lambda item: _patch_alignment_weight(0.0, item[1], item[2]))
    assert best[0] == 3
