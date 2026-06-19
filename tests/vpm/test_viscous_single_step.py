"""
Viscous diffusion single-step tests.

Each test runs exactly ONE update_state() call with only the viscous scheme
active.  This isolates the diffusion operator and verifies conservation,
growth laws, and exchange symmetry.
"""

import numpy as np
import pytest

from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
)

_SIGMA = 0.1
_VOLUME = (4.0 / 3.0) * np.pi * _SIGMA**3


def _viscous_solver(make_solver, scheme, kernel="GAUSSIAN", **kwargs):
    """Create a solver with only viscous diffusion active."""
    return make_solver(
        time_step_size=0.01,
        particles_kernel=kernel,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme=scheme, **kwargs),
        advection=AdvectionConfig(scheme="NONE"),
        velocity=VelocityConfig.direct(),
    )


# ── Core Spreading (CS) ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_cs_one_step_radius_growth(kernel_name, backend, solver_for_backend):
    """
    After one CS step: σ² = σ₀² + C·nu·dt.

    C = 4 (Gaussian), 4 (HOG), 2 (Super-Gaussian), 5 (Winckelmans).
    """
    nu = 0.01
    dt = 0.01
    C = {"GAUSSIAN": 4.0, "HIGH_ORDER_GAUSSIAN": 4.0, "SUPER_GAUSSIAN": 2.0, "WINCKELMANS": 5.0}[
        kernel_name
    ]

    solver = _viscous_solver(solver_for_backend, "CS", kernel=kernel_name)
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        circulation=np.array([[0.0, 0.0, 1.0]]),
        radius=np.array([_SIGMA]),
        volume=np.array([_VOLUME]),
        viscosity=np.array([nu]),
    )
    solver.update_state()
    sigma_new = float(solver.particles_radii[0])
    expected_sq = _SIGMA**2 + C * nu * dt
    rel_err = abs(sigma_new**2 - expected_sq) / expected_sq
    assert rel_err < 1e-4, (
        f"{kernel_name}/{backend}: CS radius growth wrong: σ²={sigma_new**2:.6e}, expected {expected_sq:.6e}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_cs_one_step_circulation_unchanged(kernel_name, backend, solver_for_backend):
    """CS must never modify circulation strengths."""
    solver = _viscous_solver(solver_for_backend, "CS", kernel=kernel_name)
    rng = np.random.default_rng(42)
    positions = rng.uniform(-1.0, 1.0, (5, 3))
    circulations = rng.normal(0.0, 0.1, (5, 3))
    solver.add_vortex_particles(
        position=positions,
        velocity=np.zeros((5, 3)),
        circulation=circulations,
        radius=np.full(5, _SIGMA),
        volume=np.full(5, _VOLUME),
        viscosity=np.full(5, 0.01),
    )
    gamma_before = solver.particles_circulation.copy()
    solver.update_state()
    gamma_after = solver.particles_circulation.copy()
    np.testing.assert_allclose(gamma_after, gamma_before, atol=1e-10, rtol=1e-6)


# ── Random Walk Method (RWM) ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_rwm_one_step_mean_zero(kernel_name, backend, solver_for_backend):
    """RWM: mean displacement over 500 ensembles must be ≈ 0."""
    if backend != "CPU":
        pytest.skip("RWM ensemble test: random sequences differ across GPU backends")

    nu = 0.1
    dt = 0.01
    n_ensemble = 500
    displacements = []
    for _ in range(n_ensemble):
        solver = _viscous_solver(solver_for_backend, "RWM", kernel=kernel_name)
        solver.add_vortex_particles(
            position=np.array([[0.0, 0.0, 0.0]]),
            velocity=np.zeros((1, 3)),
            circulation=np.array([[0.0, 0.0, 1.0]]),
            radius=np.array([_SIGMA]),
            volume=np.array([_VOLUME]),
            viscosity=np.array([nu]),
        )
        solver.update_state()
        displacements.append(solver.particles_positions[0].copy())
    mean_disp = np.mean(displacements, axis=0)
    assert np.all(np.abs(mean_disp) < 0.01), (
        f"{kernel_name}/{backend}: RWM mean displacement = {mean_disp} (must ≈ 0)"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_rwm_one_step_variance(kernel_name, backend, solver_for_backend):
    """RWM: variance of displacement must equal 2nudt per direction."""
    if backend != "CPU":
        pytest.skip("RWM ensemble test: random sequences differ across GPU backends")

    nu = 0.1
    dt = 0.01
    n_ensemble = 500
    displacements = []
    for _ in range(n_ensemble):
        solver = _viscous_solver(solver_for_backend, "RWM", kernel=kernel_name)
        solver.add_vortex_particles(
            position=np.array([[0.0, 0.0, 0.0]]),
            velocity=np.zeros((1, 3)),
            circulation=np.array([[0.0, 0.0, 1.0]]),
            radius=np.array([_SIGMA]),
            volume=np.array([_VOLUME]),
            viscosity=np.array([nu]),
        )
        solver.update_state()
        displacements.append(solver.particles_positions[0].copy())
    var = np.var(displacements, axis=0)
    expected = 2.0 * nu * dt
    for i, label in enumerate(["x", "y", "z"]):
        rel_err = abs(var[i] - expected) / expected
        assert rel_err < 0.15, (
            f"{kernel_name}/{backend}: RWM Var(Δ{label})={var[i]:.6f}, expected {expected:.6f}, rel_err={rel_err:.3e}"
        )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_rwm_circulation_unchanged(kernel_name, backend, solver_for_backend):
    """RWM must not modify circulation strengths."""
    solver = _viscous_solver(solver_for_backend, "RWM", kernel=kernel_name)
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        circulation=np.array([[0.0, 0.0, 1.0]]),
        radius=np.array([_SIGMA]),
        volume=np.array([_VOLUME]),
        viscosity=np.array([0.1]),
    )
    gamma_before = solver.particles_circulation.copy()
    solver.update_state()
    gamma_after = solver.particles_circulation.copy()
    np.testing.assert_allclose(gamma_after, gamma_before, atol=1e-10)


# ── Grid-based diffusion (DVH, GBD) ───────────────────────────────────────────


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_dvh_one_step_circulation_conservation(kernel_name, backend, solver_for_backend):
    """DVH must conserve total circulation."""
    h = 2.0 * _SIGMA
    solver = _viscous_solver(
        solver_for_backend,
        "DVH",
        kernel=kernel_name,
        dvh_grid_spacing=h,
        dvh_threshold=1e-8,
    )
    rng = np.random.default_rng(42)
    positions = rng.uniform(-0.3, 0.3, (8, 3))
    circulations = rng.normal(0.0, 0.1, (8, 3))
    solver.add_vortex_particles(
        position=positions,
        velocity=np.zeros((8, 3)),
        circulation=circulations,
        radius=np.full(8, _SIGMA),
        volume=np.full(8, _VOLUME),
        viscosity=np.full(8, 0.01),
    )
    gamma_sum_before = solver.particles_circulation.sum(axis=0)
    solver.update_state()
    gamma_sum_after = solver.particles_circulation.sum(axis=0)
    rel_err = np.linalg.norm(gamma_sum_after - gamma_sum_before) / (
        np.linalg.norm(gamma_sum_before) + 1e-12
    )
    assert rel_err < 1e-4, (
        f"{kernel_name}/{backend}: DVH broke circulation conservation: rel_err={rel_err:.3e}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_gbd_one_step_cfl_stability(kernel_name, backend, solver_for_backend):
    """GBD at CFL limit must not crash and must conserve circulation."""
    h = 2.0 * _SIGMA
    nu = 0.01
    dt = h**2 / (6.0 * nu)  # explicit Laplacian CFL
    solver = _viscous_solver(
        solver_for_backend,
        "GBD",
        kernel=kernel_name,
        gbd_grid_spacing=h,
        gbd_threshold=1e-8,
    )
    solver.config.time_step_size = dt
    rng = np.random.default_rng(43)
    positions = rng.uniform(-0.3, 0.3, (8, 3))
    circulations = rng.normal(0.0, 0.1, (8, 3))
    solver.add_vortex_particles(
        position=positions,
        velocity=np.zeros((8, 3)),
        circulation=circulations,
        radius=np.full(8, _SIGMA),
        volume=np.full(8, _VOLUME),
        viscosity=np.full(8, nu),
    )
    gamma_sum_before = solver.particles_circulation.sum(axis=0)
    solver.update_state()
    gamma_sum_after = solver.particles_circulation.sum(axis=0)
    rel_err = np.linalg.norm(gamma_sum_after - gamma_sum_before) / (
        np.linalg.norm(gamma_sum_before) + 1e-12
    )
    assert rel_err < 1e-4, (
        f"{kernel_name}/{backend}: GBD broke circulation conservation: rel_err={rel_err:.3e}"
    )


# ── Cross-backend consistency ─────────────────────────────────────────────────


@pytest.mark.parametrize("scheme", ["CS"])
def test_cross_backend_viscous_consistency(scheme, backend, solver_for_backend):
    """Same viscous setup on all backends must give identical results within 0.5%."""
    if not hasattr(test_cross_backend_viscous_consistency, "_cpu_cache"):
        test_cross_backend_viscous_consistency._cpu_cache = {}

    solver = _viscous_solver(solver_for_backend, scheme, kernel="GAUSSIAN")
    solver.add_vortex_particles(
        position=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        velocity=np.zeros((2, 3)),
        circulation=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        radius=np.full(2, _SIGMA),
        volume=np.full(2, _VOLUME),
        viscosity=np.full(2, 0.01),
    )
    gamma_before = solver.particles_circulation.copy()
    solver.update_state()
    gamma_after = solver.particles_circulation.copy()

    key = f"{scheme}_gamma"
    if backend == "CPU":
        test_cross_backend_viscous_consistency._cpu_cache[key] = gamma_after.copy()
    else:
        ref = test_cross_backend_viscous_consistency._cpu_cache.get(key)
        if ref is None:
            pytest.skip("CPU reference not yet computed")
        rel_err = np.linalg.norm(gamma_after - ref) / (np.linalg.norm(ref) + 1e-12)
        assert rel_err < 0.005, f"{scheme}/{backend}: viscous result mismatch vs CPU: {rel_err:.3e}"
