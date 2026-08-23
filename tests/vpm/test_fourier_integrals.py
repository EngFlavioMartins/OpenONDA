"""Physics identities for the large-particle Fourier diagnostics."""

import numpy as np
import pytest

from source.solvers.vpm.numerics.fourier_integrals import gaussian_fourier_integrals

pytestmark = pytest.mark.unit


def test_variable_effective_viscosity_dissipation_is_the_weighted_quadratic_form():
    rng = np.random.default_rng(20260811)
    count = 14
    position = rng.uniform(-0.5, 0.5, (count, 3))
    vortex_strength = rng.normal(0.0, 0.08, (count, 3))
    core_radius = rng.uniform(0.14, 0.22, count)
    particle_volume = np.full(count, 0.18**3)
    effective_viscosity = np.linspace(0.2, 1.1, count)
    spacing = 0.18

    weighted = gaussian_fourier_integrals(
        position,
        vortex_strength,
        core_radius,
        particle_volume,
        effective_viscosity=effective_viscosity,
        spacing=spacing,
    )
    plus = gaussian_fourier_integrals(
        position,
        vortex_strength * (1.0 + effective_viscosity[:, None]),
        core_radius,
        particle_volume,
        spacing=spacing,
    )
    minus = gaussian_fourier_integrals(
        position,
        vortex_strength * (1.0 - effective_viscosity[:, None]),
        core_radius,
        particle_volume,
        spacing=spacing,
    )

    expected = -0.25 * (plus.total_enstrophy - minus.total_enstrophy)
    assert weighted.viscous_kinetic_energy_rate == pytest.approx(expected, rel=2.0e-12)
    mean_effective_viscosity_model = -float(np.mean(effective_viscosity)) * weighted.total_enstrophy
    assert abs(expected - mean_effective_viscosity_model) > 1.0e-3 * abs(expected)


def test_constant_effective_viscosity_reduces_to_viscous_enstrophy_rate():
    rng = np.random.default_rng(17)
    count = 10
    position = rng.uniform(-0.4, 0.4, (count, 3))
    vortex_strength = rng.normal(0.0, 0.1, (count, 3))
    core_radius = np.full(count, 0.2)
    particle_volume = np.full(count, 0.2**3)
    effective_viscosity = np.full(count, 3.0e-3)

    result = gaussian_fourier_integrals(
        position,
        vortex_strength,
        core_radius,
        particle_volume,
        effective_viscosity=effective_viscosity,
        spacing=0.2,
    )

    assert result.viscous_kinetic_energy_rate == pytest.approx(
        -effective_viscosity[0] * result.total_enstrophy,
        rel=2.0e-12,
    )
