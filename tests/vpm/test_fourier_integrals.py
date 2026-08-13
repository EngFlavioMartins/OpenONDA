"""Physics identities for the large-particle Fourier diagnostics."""

import numpy as np
import pytest

from source.solvers.VPM.numerics.fourier_integrals import gaussian_fourier_integrals

pytestmark = pytest.mark.unit


def test_variable_viscosity_dissipation_is_the_weighted_quadratic_form():
    rng = np.random.default_rng(20260811)
    count = 14
    position = rng.uniform(-0.5, 0.5, (count, 3))
    circulation = rng.normal(0.0, 0.08, (count, 3))
    radius = rng.uniform(0.14, 0.22, count)
    volume = np.full(count, 0.18**3)
    viscosity = np.linspace(0.2, 1.1, count)
    spacing = 0.18

    weighted = gaussian_fourier_integrals(
        position,
        circulation,
        radius,
        volume,
        viscosity=viscosity,
        spacing=spacing,
    )
    plus = gaussian_fourier_integrals(
        position,
        circulation * (1.0 + viscosity[:, None]),
        radius,
        volume,
        spacing=spacing,
    )
    minus = gaussian_fourier_integrals(
        position,
        circulation * (1.0 - viscosity[:, None]),
        radius,
        volume,
        spacing=spacing,
    )

    expected = -0.25 * (plus.enstrophy - minus.enstrophy)
    assert weighted.viscous_energy_dissipation == pytest.approx(expected, rel=2.0e-12)
    mean_viscosity_model = -float(np.mean(viscosity)) * weighted.enstrophy
    assert abs(expected - mean_viscosity_model) > 1.0e-3 * abs(expected)


def test_constant_viscosity_reduces_to_minus_nu_enstrophy():
    rng = np.random.default_rng(17)
    count = 10
    position = rng.uniform(-0.4, 0.4, (count, 3))
    circulation = rng.normal(0.0, 0.1, (count, 3))
    radius = np.full(count, 0.2)
    volume = np.full(count, 0.2**3)
    viscosity = np.full(count, 3.0e-3)

    result = gaussian_fourier_integrals(
        position,
        circulation,
        radius,
        volume,
        viscosity=viscosity,
        spacing=0.2,
    )

    assert result.viscous_energy_dissipation == pytest.approx(
        -viscosity[0] * result.enstrophy,
        rel=2.0e-12,
    )
