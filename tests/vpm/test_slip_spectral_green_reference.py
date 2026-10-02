"""Bounded CPU mathematical qualification; no FFT backend or GPU imports."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0, k0

from tests.vpm._slip_periodic_gaussian_oracle import slip_periodic_gaussian
from tests.vpm._slip_spectral_green_reference import (
    _radial_mode,
    compact_padding_is_alias_free,
    slip_smooth_spectral_reference,
    smooth_xy_periodization_bounds,
    truncated_green_transform,
)


@pytest.mark.parametrize(
    "mu,s", [(0, 0), (0, 1e-7), (0, 0.8), (0, 5), (0.3, 0), (0.3, 0.8), (2, 5), (1e-4, 1e-4)]
)
def test_truncated_bessel_formula_matches_independent_radial_quadrature(mu, s):
    radius = 1.3
    if mu == 0:
        reference = quad(lambda r: r * np.log(radius / r) * j0(s * r), 0, radius, epsabs=1e-13)[0]
    else:
        reference = quad(lambda r: r * k0(mu * r) * j0(s * r), 0, radius, epsabs=1e-13)[0]
    np.testing.assert_allclose(
        truncated_green_transform(s, mu, radius), reference, rtol=2e-11, atol=2e-12
    )
    if mu == s == 0:
        assert truncated_green_transform(s, mu, radius) == radius**2 / 4


def test_smooth_continuous_fourier_field_matches_independent_infinite_image_oracle():
    position, strength = [[0.05, -0.05, 0.18]], [[0.2, -0.3, 0.7]]
    targets = [[0.3, 0.1, 0.27], [0.05, -0.05, 0.18]]
    tau = 0.22
    spectral = slip_smooth_spectral_reference(
        position, strength, targets, z_min=0, z_max=0.5, tau=tau, cutoff=1.6, modes=9
    )
    images = slip_periodic_gaussian(
        position, strength, [tau], targets, z_min=0, z_max=0.5, shells=8192
    )
    velocity_error = np.linalg.norm(spectral.velocity - images.velocity, axis=1)
    gradient_error = np.linalg.norm(spectral.gradient - images.gradient, axis=(1, 2))
    assert np.all(
        velocity_error
        <= spectral.cutoff_velocity_bound
        + spectral.mode_velocity_bound
        + images.velocity_tail_bound
        + 5 * spectral.quadrature_velocity_estimate
        + 2e-12
    )
    assert np.all(
        gradient_error
        <= spectral.cutoff_gradient_bound
        + spectral.mode_gradient_bound
        + images.gradient_tail_bound
        + 5 * spectral.quadrature_gradient_estimate
        + 2e-12
    )
    assert np.max(velocity_error) < 1e-8
    assert np.max(gradient_error) < 1e-8


def test_zero_mode_retains_non_neutral_analytical_gaussian_line_vortex():
    circulation, length, tau, distance = 0.7, 0.5, 0.13, 0.4
    result = slip_smooth_spectral_reference(
        [[0, 0, 0.17]],
        [[0, 0, circulation]],
        [[distance, 0, 0.31]],
        z_min=0,
        z_max=length,
        tau=tau,
        cutoff=2,
        modes=0,
    )
    exponential = np.exp(-((distance / tau) ** 2))
    factor = circulation / (2 * np.pi * length)
    expected_velocity = [0, factor * (1 - exponential) / distance, 0]
    expected_j = np.zeros((3, 3))
    expected_j[0, 1] = -factor * (1 - exponential) / distance**2
    expected_j[1, 0] = factor * (2 * exponential / tau**2 - (1 - exponential) / distance**2)
    np.testing.assert_allclose(result.velocity[0], expected_velocity, atol=2e-11)
    np.testing.assert_allclose(result.gradient[0], expected_j, atol=2e-10)


def test_gaussian_blurring_of_cutoff_is_not_exact_and_error_is_charged():
    args = ([[0, 0, 0.18]], [[0.2, -0.3, 0.7]], [[0.4, 0, 0.31]])
    common = {"z_min": 0, "z_max": 0.5, "tau": 0.2, "modes": 8}
    short = slip_smooth_spectral_reference(*args, cutoff=0.55, **common)
    long = slip_smooth_spectral_reference(*args, cutoff=2, **common)
    du = np.linalg.norm(short.velocity - long.velocity, axis=1)
    dj = np.linalg.norm(short.gradient - long.gradient, axis=(1, 2))
    assert du[0] > 1e-5
    assert np.all(du <= short.cutoff_velocity_bound + long.cutoff_velocity_bound + 1e-10)
    assert np.all(dj <= short.cutoff_gradient_bound + long.cutoff_gradient_bound + 1e-10)


@pytest.mark.parametrize("modes", [0, 1, 2])
def test_absolute_z_mode_tail_bounds_cover_missing_nonzero_modes(modes):
    args = ([[0, 0, 0.18]], [[0.2, -0.3, 0.7]], [[0.4, 0, 0.31]])
    common = {"z_min": 0, "z_max": 0.5, "tau": 0.4, "cutoff": 3}
    short = slip_smooth_spectral_reference(*args, modes=modes, **common)
    long = slip_smooth_spectral_reference(*args, modes=10, **common)
    assert (
        np.linalg.norm(short.velocity - long.velocity)
        <= short.mode_velocity_bound
        + long.mode_velocity_bound
        + short.cutoff_velocity_bound[0]
        + long.cutoff_velocity_bound[0]
        + 1e-10
    )
    assert (
        np.linalg.norm(short.gradient - long.gradient)
        <= short.mode_gradient_bound
        + long.mode_gradient_bound
        + short.cutoff_gradient_bound[0]
        + long.cutoff_gradient_bound[0]
        + 1e-10
    )


def test_compact_padding_condition_does_not_silently_assume_gaussian_compactness():
    # Both factors would need actual compact supports R=2 and B=0.5. A finite
    # Gaussian cutoff B itself still needs a separate omitted/alias estimate.
    assert compact_padding_is_alias_free([6, 6], [2, 2], 2, 0.5)
    assert not compact_padding_is_alias_free([4, 6], [2, 2], 2, 0.5)
    assert not compact_padding_is_alias_free([4.5, 6], [2, 2], 2, 0.5)
    with pytest.raises(ValueError, match="below cutoff"):
        slip_smooth_spectral_reference(
            [[0, 0, 0.1]], [[0, 0, 1]], [[2, 0, 0]], z_min=0, z_max=0.5, tau=0.2, cutoff=1, modes=2
        )


@pytest.mark.parametrize("mu", [0.0, 2.0])
def test_infinite_gaussian_periodization_bound_covers_explicit_neighbour_aliases(mu):
    cutoff, tau, period = 0.7, 0.3, np.array([2.0, 2.2])
    delta, extent = np.array([0.4, 0.1]), np.array([0.4, 0.1])
    vb, jb = smooth_xy_periodization_bounds(period, extent, mu, cutoff, tau)
    gradient_sum, hessian_sum, estimated_error = 0.0, 0.0, 0.0
    for nx in (-1, 0, 1):
        for ny in (-1, 0, 1):
            if nx == ny == 0:
                continue
            rho = float(np.linalg.norm(delta + period * [nx, ny]))
            (value, first, second), error = _radial_mode(rho, mu, cutoff, tau)
            gradient_sum += abs(first) + mu * abs(value)
            hessian_sum += abs(second) + abs(first) / rho + 2 * mu * abs(first) + mu**2 * abs(value)
            estimated_error += 5 * (1 + mu + mu**2 + 1 / rho) * error.sum()
    assert gradient_sum > 1e-8
    assert gradient_sum <= vb + estimated_error
    assert hessian_sum <= jb + estimated_error
    with pytest.raises(ValueError, match="padding requires"):
        smooth_xy_periodization_bounds([1, 1], extent, mu, cutoff, tau)
