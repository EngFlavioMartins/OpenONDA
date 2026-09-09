"""Unbounded energy must survive FFT box changes and obey the heat equation."""

import numpy as np
import pytest

from source.solvers.vpm.numerics.fourier_integrals import CartesianGrid, gaussian_fourier_integrals


def _integrals(position, strength, sigma, *, shape=(15, 17, 19)):
    return gaussian_fourier_integrals(
        np.asarray(position, dtype=float),
        np.asarray(strength, dtype=float),
        np.full(len(position), sigma),
        np.full(len(position), 0.1**3),
        effective_viscosity=np.full(len(position), 0.01),
        grid=CartesianGrid(np.full(3, -0.6), 0.1, shape),
        free_space=True,
    )


def test_single_blob_has_exact_unbounded_energy_and_viscous_power():
    sigma = 0.2
    strength = np.array([[0.3, -0.4, 0.7]])
    result = _integrals([[0, 0, 0]], strength, sigma)
    norm_sq = np.sum(strength**2)
    width = np.sqrt(2) * sigma
    assert result.total_kinetic_energy == pytest.approx(
        norm_sq / (6 * np.pi**1.5 * width), rel=1e-12
    )
    assert result.viscous_kinetic_energy_rate == pytest.approx(
        -0.01 * 2 * norm_sq / (3 * np.pi**1.5 * width**3), rel=1e-12
    )


def test_free_space_energy_does_not_change_when_the_fft_box_grows():
    position = [[0.023, -0.14, 0.012], [0.29, 0.15, 0.19]]
    strength = [[0.3, -0.4, 0.7], [0.1, 0.6, -0.2]]
    small = _integrals(position, strength, 0.2)
    large = _integrals(position, strength, 0.2, shape=(22, 25, 27))
    assert large.total_kinetic_energy == pytest.approx(small.total_kinetic_energy, rel=1e-12)
    assert large.viscous_kinetic_energy_rate == pytest.approx(
        small.viscous_kinetic_energy_rate, rel=1e-12
    )


def test_core_spreading_energy_derivative_matches_projected_viscous_power():
    position = [[0.023, -0.14, 0.012], [0.29, 0.15, 0.19]]
    strength = [[0.3, -0.4, 0.7], [0.1, 0.6, -0.2]]
    sigma = 0.2
    dt = 1e-5
    before = _integrals(position, strength, np.sqrt(sigma**2 - 4 * 0.01 * dt))
    after = _integrals(position, strength, np.sqrt(sigma**2 + 4 * 0.01 * dt))
    current = _integrals(position, strength, sigma)
    measured = (after.total_kinetic_energy - before.total_kinetic_energy) / (2 * dt)
    assert measured == pytest.approx(current.viscous_kinetic_energy_rate, rel=1e-6)


def test_free_space_mode_rejects_variable_core_radii():
    with pytest.raises(ValueError, match="common cores"):
        gaussian_fourier_integrals(
            np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            np.ones((2, 3)),
            np.array([0.1, 0.2]),
            np.full(2, 0.001),
            effective_viscosity=np.full(2, 0.01),
            free_space=True,
        )


def test_periodic_gaussian_enstrophy_filters_nyquist_modes():
    sigma = 0.08
    strength = np.array([[0.3, -0.4, 0.7]])
    result = gaussian_fourier_integrals(
        np.zeros((1, 3)),
        strength,
        np.array([sigma]),
        np.array([0.04**3]),
        grid=CartesianGrid(np.full(3, -0.4), 0.04, (21, 21, 21)),
    )
    exact = np.sum(strength**2) / ((2 * np.pi) ** 1.5 * sigma**3)
    # Periodic images are >20 core radii away; Gaussian tails at Nyquist
    # also lie below this tolerance for sigma/h=2.
    assert result.total_enstrophy == pytest.approx(exact, rel=1e-7)


def test_componentwise_fft_padding_preserves_mixed_core_and_viscosity_integrals(monkeypatch):
    """A common Fourier translation must cancel from energy, helicity and rates."""
    from dataclasses import asdict

    from scipy import fft

    from source.solvers.vpm.numerics.fourier_integrals import gaussian_fourier_integrals

    rng = np.random.default_rng(8921)
    position = rng.uniform([-0.6, -0.3, -0.2], [0.8, 0.4, 0.3], (24, 3))
    strength = rng.normal(size=(24, 3))
    radii = rng.uniform(0.15, 0.23, 24)
    volume = np.full(24, 0.1**3)
    viscosity = rng.uniform(0.01, 0.03, 24)
    arguments = {"effective_viscosity": viscosity, "spacing": 0.1}
    actual = gaussian_fourier_integrals(position, strength, radii, volume, **arguments)
    original_fft = fft.rfftn

    def centred_padding(values, *, s, **kwargs):
        padding = [
            (int(size - current) // 2, int(size - current + 1) // 2)
            for size, current in zip(s, values.shape, strict=True)
        ]
        return original_fft(np.pad(values, padding), **kwargs)

    monkeypatch.setattr(fft, "rfftn", centred_padding)
    centred = gaussian_fourier_integrals(position, strength, radii, volume, **arguments)
    for name, value in asdict(actual).items():
        expected = getattr(centred, name)
        if isinstance(value, float | np.floating):
            np.testing.assert_allclose(value, expected, rtol=2e-12, atol=1e-12, err_msg=name)
        else:
            assert value == expected
