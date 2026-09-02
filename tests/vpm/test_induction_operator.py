"""Regression tests for the frozen discrete VPM induction operator."""

import numpy as np

from studies.vpm.advection_stretching.assets.core import (
    contract,
    gradient,
    pair_rate,
    target_fields,
)


def _cloud() -> tuple[np.ndarray, np.ndarray]:
    position = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.7, 0.2, -0.1],
            [-0.4, 0.8, 0.3],
            [0.2, -0.5, 0.9],
        ]
    )
    vortex_strength = np.array(
        [
            [0.4, -0.2, 0.7],
            [-0.3, 0.8, 0.1],
            [0.6, 0.1, -0.5],
            [-0.2, -0.4, 0.3],
        ]
    )
    return position, vortex_strength


def test_equal_core_radii_make_pairwise_rate_equal_transposed_gradient_contraction():
    position, vortex_strength = _cloud()
    core_radius = np.full(len(position), 0.2)

    pairwise_rate = pair_rate(position, vortex_strength, core_radius, "TRANSPOSED")
    gradient_rate = contract(
        gradient(position, vortex_strength, core_radius), vortex_strength, "TRANSPOSED"
    )

    np.testing.assert_allclose(pairwise_rate, gradient_rate, rtol=1.0e-13, atol=1.0e-14)


def test_unequal_core_radii_use_the_symmetric_arithmetic_pair_radius():
    position, vortex_strength = _cloud()
    core_radius = np.array([0.1, 0.45, 0.2, 0.35])

    pairwise_rate = pair_rate(position, vortex_strength, core_radius, "TRANSPOSED")
    pair_radius_gradient_rate = contract(
        gradient(position, vortex_strength, core_radius), vortex_strength, "TRANSPOSED"
    )

    np.testing.assert_allclose(pairwise_rate, pair_radius_gradient_rate, rtol=1.0e-13, atol=1.0e-14)


def test_unequal_radii_distinguish_pairwise_rate_from_source_radius_gradient():
    position, vortex_strength = _cloud()
    core_radius = np.array([0.1, 0.45, 0.2, 0.35])

    pairwise_rate = pair_rate(position, vortex_strength, core_radius, "TRANSPOSED")
    _, _, source_radius_gradient = target_fields(position, position, vortex_strength, core_radius)
    source_radius_rate = contract(source_radius_gradient, vortex_strength, "TRANSPOSED")

    assert not np.allclose(pairwise_rate, source_radius_rate, rtol=1.0e-8, atol=1.0e-10)
    assert np.linalg.norm(pairwise_rate - source_radius_rate) > 0.1


def test_nonsymmetric_gradient_orientation_is_explicit():
    jacobian = np.array(
        [
            [
                [0.0, 2.0, -1.0],
                [3.0, 0.0, 4.0],
                [5.0, 6.0, 0.0],
            ]
        ]
    )
    vortex_strength = np.array([[0.4, -0.2, 0.7]])

    direct = contract(jacobian, vortex_strength, "DIRECT")
    transposed = contract(jacobian, vortex_strength, "TRANSPOSED")

    np.testing.assert_allclose(direct, jacobian @ vortex_strength[0])
    np.testing.assert_allclose(transposed, jacobian.transpose(0, 2, 1) @ vortex_strength[0])
    assert not np.allclose(direct, transposed)


def test_pairwise_transposed_rate_cancels_total_strength_for_unequal_radii():
    rng = np.random.default_rng(20260901)
    position = rng.normal(size=(16, 3))
    vortex_strength = rng.normal(size=(16, 3))
    core_radius = rng.uniform(0.05, 0.5, size=16)

    rate = pair_rate(position, vortex_strength, core_radius, "TRANSPOSED")

    np.testing.assert_allclose(rate.sum(axis=0), 0.0, rtol=0.0, atol=2.0e-14)


def test_self_interaction_is_excluded_from_particle_rate():
    position = np.array([[0.0, 0.0, 0.0]])
    vortex_strength = np.array([[0.4, -0.2, 0.7]])
    core_radius = np.array([0.2])

    rate = pair_rate(position, vortex_strength, core_radius, "TRANSPOSED")

    np.testing.assert_allclose(rate, 0.0, rtol=0.0, atol=0.0)
