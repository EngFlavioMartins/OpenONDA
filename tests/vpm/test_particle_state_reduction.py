"""Accepted-state limit metrics reduce the particle fields to their definitions."""

import numpy as np
import pytest
import taichi as ti

from openonda.vpm.diagnostics.particle_state import measure_particle_state
from openonda.vpm.particles.container import Particles

COUNT = 20_000
TIME_STEP_SIZE = 0.013


def _rotations(rng, count):
    q, r = np.linalg.qr(rng.normal(size=(count, 3, 3)))
    return q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]


def _velocity_gradient(rng, count):
    """Random Jacobians; a third have a repeated largest-magnitude strain eigenvalue."""
    gradient = rng.normal(size=(count, 3, 3))
    repeated = slice(0, count // 3)
    scale = rng.normal(size=(count // 3, 1))
    eigenvalues = np.hstack((-2.0 * scale, -2.0 * scale, 0.5 * scale))
    q = _rotations(rng, count // 3)
    strain = np.einsum("nij,nj,nkj->nik", q, eigenvalues, q)
    spin = gradient[repeated] - np.swapaxes(gradient[repeated], 1, 2)
    gradient[repeated] = strain + 0.5 * scale[:, :, None] * spin
    return gradient


@pytest.fixture(scope="module")
def particles():
    ti.init(arch=ti.cpu, default_fp=ti.f32, offline_cache=False)
    rng = np.random.default_rng(3)
    cloud = Particles(max_n_particles=COUNT, float_dtype="f32")
    cloud.replace_from_numpy(
        position=rng.uniform(-1.0, 1.0, size=(COUNT, 3)),
        velocity=rng.normal(size=(COUNT, 3)),
        vortex_strength=rng.normal(scale=1e-3, size=(COUNT, 3)),
        core_radius=rng.uniform(0.05, 0.1, size=COUNT),
        particle_volume=rng.uniform(1e-4, 1e-3, size=COUNT),
        kinematic_viscosity=np.full(COUNT, 1e-3),
        velocity_gradient=_velocity_gradient(rng, COUNT),
    )
    return cloud


def _float64_fields(particles):
    gradient = particles.velocity_gradient_cpu(use_cache=False).astype(np.float64)
    strength = particles.vortex_strength_cpu(use_cache=False).astype(np.float64)
    volume = particles.particle_volume_cpu(use_cache=False).astype(np.float64)
    return gradient, strength, volume


def test_extrema_match_strain_norms_strength_and_vorticity(particles):
    gradient, strength, volume = _float64_fields(particles)
    strain = 0.5 * (gradient + np.swapaxes(gradient, 1, 2))
    infinity = TIME_STEP_SIZE * np.abs(strain).sum(axis=2).max(axis=1)
    spectral = TIME_STEP_SIZE * np.linalg.norm(strain, ord=2, axis=(1, 2))
    magnitude = np.linalg.norm(strength, axis=1)

    state = measure_particle_state(particles, TIME_STEP_SIZE)
    metrics = state.metrics
    assert state.invalid_fields == ()
    # Float64 strain of float32 Jacobians is exact; its row sums are formed alike.
    assert metrics.strain_increment_infinity == infinity.max()
    assert metrics.strain_increment_infinity_particle == np.argmax(infinity)
    position = particles.position_cpu(use_cache=False)[np.argmax(infinity)]
    assert state.strain_limiting_position == tuple(position.astype(np.float64))
    # Jacobi eigenvalues and the singular value decomposition agree to round-off.
    assert metrics.strain_increment_spectral == pytest.approx(spectral.max(), rel=1e-14)
    assert spectral[metrics.strain_increment_spectral_particle] == pytest.approx(
        spectral.max(), rel=1e-14
    )
    assert metrics.maximum_particle_strength == pytest.approx(magnitude.max(), rel=1e-15)
    assert metrics.maximum_vorticity == pytest.approx((magnitude / volume).max(), rel=1e-15)


def test_spectral_norm_is_largest_absolute_strain_eigenvalue(particles):
    gradient, _, _ = _float64_fields(particles)
    strain = 0.5 * (gradient + np.swapaxes(gradient, 1, 2))
    largest = np.abs(np.linalg.eigvalsh(strain)).max(axis=1)
    repeated = slice(0, COUNT // 3)
    eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(strain[repeated])), axis=1)
    assert np.allclose(eigenvalues[:, 2], eigenvalues[:, 1], rtol=1e-5)

    state = measure_particle_state(particles, TIME_STEP_SIZE)
    assert state.metrics.strain_increment_spectral == pytest.approx(
        TIME_STEP_SIZE * largest.max(), rel=1e-14
    )


def test_velocity_curl_is_the_jacobian_antisymmetric_part(particles):
    gradient = particles.velocity_gradient_cpu(use_cache=False)
    curl = np.column_stack(
        (
            gradient[:, 2, 1] - gradient[:, 1, 2],
            gradient[:, 0, 2] - gradient[:, 2, 0],
            gradient[:, 1, 0] - gradient[:, 0, 1],
        )
    )
    assert np.array_equal(particles.velocity_curl_cpu(), curl)


def test_invalid_fields_and_nan_extrema(particles):
    particles.velocity[11] = [np.inf, 0.0, 0.0]
    particles.core_radius[12] = 0.0
    particles.particle_volume[13] = -1.0
    state = measure_particle_state(particles, TIME_STEP_SIZE)
    assert state.invalid_fields == ("velocity", "core_radius", "particle_volume")

    particles.velocity_gradient[COUNT - 2][0, 1] = np.nan
    particles.velocity_gradient[COUNT - 7][1, 0] = np.nan
    particles.vortex_strength[5] = [np.nan, 0.0, 0.0]
    state = measure_particle_state(particles, TIME_STEP_SIZE)
    metrics = state.metrics
    assert state.invalid_fields == (
        "velocity",
        "velocity_gradient",
        "vortex_strength",
        "core_radius",
        "particle_volume",
    )
    # NaN propagates and the first NaN particle is reported, as numpy.max/argmax do.
    assert np.isnan(metrics.strain_increment_infinity)
    assert np.isnan(metrics.strain_increment_spectral)
    assert np.isnan(metrics.maximum_particle_strength)
    assert np.isnan(metrics.maximum_vorticity)
    assert metrics.strain_increment_infinity_particle == COUNT - 7
    assert metrics.strain_increment_spectral_particle == COUNT - 7
