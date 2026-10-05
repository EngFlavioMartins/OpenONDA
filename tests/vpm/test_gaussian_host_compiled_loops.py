"""Compiled host loops preserve ordered casts, collisions and strict cutoffs."""

import math

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh import host_fields
from source.solvers.vpm.physics.induction.gaussian_mesh.correction import (
    _correction_factors,
    correction_factors_host,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compiled_scatter_and_gather_preserve_ordered_cardinal_accumulation(dtype):
    rng = np.random.default_rng(722)
    first = rng.integers(0, 3, (23, 3))
    first[:5] = first[5]  # Repeated grid destinations require ordered additions.
    weights = rng.normal(size=(23, 3, 4)).astype(dtype)
    strength = rng.normal(size=(23, 3))
    expected = np.zeros((7, 7, 7), dtype=dtype)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                factor = weights[:, 0, a] * weights[:, 1, b] * weights[:, 2, c]
                np.add.at(
                    expected,
                    (first[:, 0] + a, first[:, 1] + b, first[:, 2] + c),
                    factor * strength[:, 1].astype(dtype),
                )
    actual = host_fields._scatter(first, weights, strength, (7, 7, 7), (7, 7, 7), dtype, 1)
    np.testing.assert_array_equal(actual, expected)
    gathered = np.zeros(len(first), dtype=np.float64)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                factor = (
                    weights[:, 0, a].astype(np.float64)
                    * weights[:, 1, b].astype(np.float64)
                    * weights[:, 2, c].astype(np.float64)
                )
                gathered += factor * expected[first[:, 0] + a, first[:, 1] + b, first[:, 2] + c]
    np.testing.assert_array_equal(host_fields._gather(actual, first, weights), gathered)


def _prior_pair_accumulation(position, strength, core, query, shift, odd, tau, cutoff, dtype):
    output = np.zeros((1, 12), dtype=dtype)
    accepted = []
    for index in range(len(position)):
        image = position[index].copy()
        image[2] = shift - image[2] if odd else image[2] + shift
        displacement = query - image
        radius = float(np.linalg.norm(displacement))
        if radius >= cutoff:
            continue
        vector = strength[index].copy()
        if odd:
            vector[:2] *= -1
        # This is the original Python formula, before JIT specialization.
        a, b = _correction_factors.py_func(radius, float(core[index]), tau)
        cross = np.array(
            [
                [0.0, -vector[2], vector[1]],
                [vector[2], 0.0, -vector[0]],
                [-vector[1], vector[0], 0.0],
            ]
        )
        output[0, :3] += (a * (cross @ displacement)).astype(dtype)
        output[0, 3:] += (
            (cross @ (a * np.eye(3) - b * np.outer(displacement, displacement)))
            .reshape(9)
            .astype(dtype)
        )
        accepted.append(index)
    return output, accepted


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("odd", [False, True])
def test_compiled_correction_matches_prior_coincidence_and_cutoff_expression(dtype, odd):
    cutoff, tau = 0.6, 0.12
    distances = [0.0, 0.004, 0.09, np.nextafter(cutoff, 0.0), cutoff, np.nextafter(cutoff, np.inf)]
    position = np.zeros((len(distances) + 12, 3))
    position[: len(distances), 0] = distances
    rng = np.random.default_rng(531)
    # Non-axis-aligned near-cutoff points also compare the norm/inclusion path.
    for i in range(12):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        radius = np.nextafter(cutoff, 0.0) if i % 2 else np.nextafter(cutoff, np.inf)
        position[len(distances) + i] = radius * direction
    strength = rng.normal(size=position.shape)
    core = np.resize([0.04, tau, np.nextafter(tau, 0.0), 0.07], len(position))
    query = np.zeros(3)
    expected, accepted = _prior_pair_accumulation(
        position, strength, core, query, 0.0, odd, tau, cutoff, dtype
    )
    output = np.zeros_like(expected)
    count = host_fields._accumulate_correction(
        output,
        0,
        np.arange(len(position)),
        0,
        position,
        strength,
        core,
        query,
        0.0,
        odd,
        tau,
        cutoff,
        dtype is np.float32,
    )
    assert 0 in accepted and 3 in accepted and 4 not in accepted and 5 not in accepted
    assert count == len(accepted)
    epsilon = np.finfo(dtype).eps
    np.testing.assert_allclose(output, expected, rtol=8 * epsilon, atol=8 * epsilon)


@pytest.mark.parametrize(
    "radius,sigma,tau",
    [
        (0.0, 0.04, 0.12),
        (0.004, 0.04, 0.12),
        (0.09, 0.04, 0.12),
        (0.09, 0.1199, 0.12),
        (0.09, 0.12, 0.12),
    ],
)
def test_shared_correction_arithmetic_and_public_validation(radius, sigma, tau):
    np.testing.assert_allclose(
        correction_factors_host(radius, sigma, tau),
        _correction_factors.py_func(radius, sigma, tau),
        rtol=4 * np.finfo(float).eps,
        atol=0.0,
    )
    for invalid in (-1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="finite r"):
            correction_factors_host(invalid, sigma, tau)


def test_compiled_loops_retain_bounds_checks_and_do_not_enable_fast_math():
    for kernel in (
        host_fields._scatter_into,
        host_fields._gather,
        host_fields._kernel_channels,
        host_fields._accumulate_correction,
    ):
        assert kernel.targetoptions["boundscheck"]
        assert not kernel.targetoptions.get("fastmath", False)
    assert not _correction_factors.targetoptions.get("fastmath", False)
    with pytest.raises(IndexError):
        host_fields._gather(
            np.zeros((1, 1, 1)), np.ones((1, 3), dtype=np.int64), np.ones((1, 3, 4))
        )
