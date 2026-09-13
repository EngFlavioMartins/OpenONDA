"""Physical invariants for the isolated 3D stress/curl candidate.

Passing these tests does not qualify the candidate's integration into GBD.
There is deliberately no free-space/body-boundary implementation here.
"""

import numpy as np
import pytest

from studies.coupler_accuracy.stress_curl_3d import (
    centered_derivative,
    evaluate,
    gaussian_source_experiment,
    heat_source,
    manufactured,
    production_gbd_source,
    stress_source,
)
from studies.coupler_accuracy.variable_viscosity_3d import derivative


def test_smooth_3d_stress_matches_analytical_viscous_force():
    n = 15
    velocity, nu = manufactured(n, "smooth_variable")
    x, y, z = np.meshgrid(*([2 * np.pi * np.arange(n) / n] * 3), indexing="ij")
    grad_nu = -0.0003 * np.stack((
        np.sin(x) * np.cos(y) * np.cos(z),
        np.cos(x) * np.sin(y) * np.cos(z),
        np.cos(x) * np.cos(y) * np.sin(z),
    ), axis=-1)
    two_strain = np.zeros((*nu.shape, 3, 3))
    two_strain[..., 0, 1] = two_strain[..., 1, 0] = np.cos(x) - np.sin(y)
    two_strain[..., 1, 2] = two_strain[..., 2, 1] = np.cos(y) - np.sin(z)
    two_strain[..., 2, 0] = two_strain[..., 0, 2] = np.cos(z) - np.sin(x)
    # div(2 nu S) = nu laplacian(u) + 2 S grad(nu), with laplacian(u)=-u.
    analytical = -nu[..., None] * velocity + np.einsum("...ij,...j->...i", two_strain, grad_nu)
    actual, _, _ = stress_source(velocity, nu, derivative)
    np.testing.assert_allclose(actual, analytical, rtol=0, atol=2e-16)


@pytest.mark.parametrize("shape", [(11, 13, 15), (12, 14, 16)])
def test_arbitrary_3d_field_dissipates_energy_and_has_solenoidal_source(shape):
    rng = np.random.default_rng(723)
    velocity = rng.normal(size=(*shape, 3))
    nu = rng.uniform(0, 0.004, size=shape)
    h = 0.17

    def central(field, axis):
        return centered_derivative(field, axis, h)

    acceleration, source, dissipation = stress_source(velocity, nu, central)
    work = np.sum(velocity * acceleration)
    assert work < 0
    assert np.all(dissipation >= 0)
    np.testing.assert_allclose(work, -dissipation.sum(), rtol=2e-15)
    divergence = sum(central(source[..., i], i) for i in range(3))
    np.testing.assert_allclose(divergence, 0, rtol=0, atol=1e-13)
    np.testing.assert_allclose(source.sum(axis=(0, 1, 2)), 0, rtol=0, atol=1e-12)


def test_full_stress_converges_while_variable_heat_operator_keeps_model_error():
    rows = [evaluate(n, "smooth_variable") for n in (16, 32, 64)]
    errors = [row["candidate_relative_source_error"] for row in rows]
    assert errors[-1] < 0.0062
    assert 3.7 < errors[0] / errors[1] < 4.2
    assert 3.7 < errors[1] / errors[2] < 4.2
    assert 0.20 < rows[-1]["gbd_pde_relative_source_error"] < 0.21


def test_real_gbd_kernel_executes_the_componentwise_variable_heat_operator():
    velocity, nu = manufactured(16, "smooth_variable")
    h = 2 * np.pi / 16
    # curl(u)=u exactly for this fully 3D ABC field.
    actual, dt = production_gbd_source(velocity, nu, h)
    expected = heat_source(velocity, nu, h)
    assert float(nu.max()) * dt / h**2 < 1 / 12
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 2e-4


def test_gaussian_coefficient_mapping_avoids_reapplying_the_physical_smoothing():
    result = gaussian_source_experiment(n=15)
    assert result["direct_physical_to_coefficient_relative_error"] > 0.04
    assert result["mapped_source_relative_error"] < 2e-14
    assert result["velocity_and_coefficient_step_relative_difference"] < 2e-14
    assert result["maximum_energy_step_increase"] < 0
    assert result["worst_relative_work_identity_error"] < 2e-14


def test_stress_candidate_refuses_invalid_physical_inputs():
    with pytest.raises(ValueError, match="full 3D"):
        stress_source(np.zeros((8, 8, 3)), np.ones((8, 8)), derivative)
    with pytest.raises(ValueError, match="nonnegative"):
        stress_source(np.zeros((8, 8, 8, 3)), -np.ones((8, 8, 8)), derivative)
