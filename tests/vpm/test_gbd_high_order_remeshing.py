"""Independent moment and translating-Gaussian qualifications for Lagrange6."""

import numpy as np
import pytest
import taichi as ti

from source.solvers.vpm.config.viscous import ViscousConfig
from source.solvers.vpm.physics.diffusion.grid import (
    _GridDiffusionMixin,
    _lagrange6_weights,
    _m4_prime_1d,
)


def test_six_point_weights_reproduce_polynomials_through_degree_five():
    fraction = np.r_[0.0, 1.0, np.random.default_rng(17).uniform(0, 1, 47)]
    w = _lagrange6_weights(fraction)
    for degree in range(6):
        np.testing.assert_allclose(
            w @ np.arange(-2, 4, dtype=float) ** degree, fraction**degree, atol=2e-14
        )


def test_device_scatter_matches_independent_cardinal_polynomial_deposition():
    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu)
    rng = np.random.default_rng(42)
    n = 7
    h = 0.1
    shape = (14, 14, 14)
    p = rng.uniform(0.35, 0.75, (n, 3)).astype("f")
    a = rng.normal(size=(n, 3)).astype("f")
    pf = ti.Vector.field(3, ti.f32, shape=n)
    af = ti.Vector.field(3, ti.f32, shape=n)
    grid = ti.Vector.field(3, ti.f32, shape=shape)
    pf.from_numpy(p)
    af.from_numpy(a)
    _GridDiffusionMixin()._lagrange6_scatter_gpu_kernel(
        pf, af, grid, 0.0, 0.0, 0.0, h, *shape, 0, n
    )
    expected = np.zeros((*shape, 3))
    for point, alpha in zip(p, a, strict=True):
        scaled = point.astype(float) / h
        base = np.floor(scaled).astype(int)
        from scipy.interpolate import lagrange

        weights = []
        for coordinate in scaled - base:
            weights.append([lagrange(np.arange(-2, 4), np.eye(6)[j])(coordinate) for j in range(6)])
        for i, j, k in np.ndindex(6, 6, 6):
            expected[tuple(base + np.array([i, j, k]) - 2)] += (
                weights[0][i] * weights[1][j] * weights[2][k] * alpha
            )
    np.testing.assert_allclose(grid.to_numpy(), expected, atol=2e-6, rtol=1e-4)


def test_translating_gaussian_excess_diffusion_is_reduced():
    h = 0.04
    dt = 0.015
    nu = np.pi / 3000
    steps = 100
    n = 100
    sigma = h
    x = -2 + np.arange(n) * h
    k = 2 * np.pi * np.fft.fftfreq(n, h)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    kx, ky = np.meshgrid(k, k, indexing="ij")
    initial = np.fft.fft2(np.exp(-(xx**2 + yy**2) / (0.01 - sigma * sigma)))
    smoothing = np.exp(-sigma * sigma * (kx * kx + ky * ky) / 4)
    exact = np.fft.ifft2(
        initial
        * smoothing
        * np.exp(-nu * dt * steps * (kx * kx + ky * ky) - 1j * dt * steps * (kx + 0.3 * ky))
    ).real
    errors = []
    for method in ["M4_PRIME", "LAGRANGE6"]:
        symbol = np.ones((n, n), dtype=complex)
        for velocity, kk in [(1.0, kx), (0.3, ky)]:
            shift = velocity * dt / h
            base = int(np.floor(shift))
            fraction = shift - base
            nodes = np.arange(-1, 3) if method == "M4_PRIME" else np.arange(-2, 4)
            weights = (
                _m4_prime_1d(nodes - fraction)
                if method == "M4_PRIME"
                else _lagrange6_weights(fraction)
            )
            symbol *= sum(
                w * np.exp(-1j * kk * (base + j) * h) for w, j in zip(weights, nodes, strict=True)
            )
        symbol *= 1 - 4 * nu * dt / h**2 * (np.sin(kx * h / 2) ** 2 + np.sin(ky * h / 2) ** 2)
        field = np.fft.ifft2(initial * smoothing * symbol**steps).real
        errors.append(np.linalg.norm(field - exact) / np.linalg.norm(exact))
    assert errors[1] < 0.03
    assert errors[1] < 0.2 * errors[0]


def test_wide_stencil_requires_room_for_diffusion_halo():
    with pytest.raises(ValueError, match="four grid cells"):
        ViscousConfig.gbd(particle_spacing=0.04, padding=3, remeshing_kernel="LAGRANGE6")
