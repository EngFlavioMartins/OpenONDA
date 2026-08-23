"""
Viscous diffusion single-step tests.

Each test runs exactly ONE advance() call with only the viscous scheme
active.  This isolates the diffusion operator and verifies conservation,
growth laws, and exchange symmetry.
"""

import numpy as np
import pytest

from source.solvers.vpm.config.types import (
    AdvectionConfig,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
)

_SIGMA = 0.1
_VOLUME = (4.0 / 3.0) * np.pi * _SIGMA**3

# GPU DVH/GBD pre-allocate their grid, so they need explicit domain bounds.
_GRID_DIFFUSION_BOUNDS = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]


def _viscous_solver(
    make_solver,
    scheme,
    kernel="GAUSSIAN",
    time_step_size=0.01,
    domain_bounds=None,
    **kwargs,
):
    """Create a solver with only viscous diffusion active."""
    # vpm_domain_bounds belongs to the setup, not to ViscousConfig.
    setup_kwargs = {}
    if domain_bounds is not None:
        setup_kwargs["domain_bounds"] = domain_bounds
    return make_solver(
        time_step_size=time_step_size,
        particle_kernel=kernel,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme=scheme, **kwargs),
        advection=AdvectionConfig(scheme="NONE"),
        velocity=VelocityConfig.direct(),
        **setup_kwargs,
    )


# ── Core Spreading (CS) ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_cs_one_step_radius_growth(kernel_name, backend, solver_for_backend):
    """
    After one CS step: σ² = σ₀² + C·kinematic_viscosity·dt.

    C = 4 (Gaussian), 4 (HOG), 2 (Super-Gaussian), 4 (Winckelmans).
    """
    kinematic_viscosity = 0.01
    time_step_size = 0.01
    C = {
        "GAUSSIAN": 4.0,
        "HIGH_ORDER_GAUSSIAN": 4.0,
        "SUPER_GAUSSIAN": 2.0,
        "WINCKELMANS": 4.0,
    }[kernel_name]

    solver = _viscous_solver(solver_for_backend, "CS", kernel=kernel_name)
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        vortex_strength=np.array([[0.0, 0.0, 1.0]]),
        core_radius=np.array([_SIGMA]),
        particle_volume=np.array([_VOLUME]),
        kinematic_viscosity=np.array([kinematic_viscosity]),
    )
    solver.advance()
    sigma_new = float(solver.particle_core_radius[0])
    expected_sq = _SIGMA**2 + C * kinematic_viscosity * time_step_size
    rel_err = abs(sigma_new**2 - expected_sq) / expected_sq
    assert rel_err < 1e-4, (
        f"{kernel_name}/{backend}: CS radius growth wrong: σ²={sigma_new**2:.6e}, expected {expected_sq:.6e}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_cs_one_step_circulation_unchanged(kernel_name, backend, solver_for_backend):
    """CS must never modify circulation vortex_strength."""
    solver = _viscous_solver(solver_for_backend, "CS", kernel=kernel_name)
    rng = np.random.default_rng(42)
    position = rng.uniform(-1.0, 1.0, (5, 3))
    circulations = rng.normal(0.0, 0.1, (5, 3))
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros((5, 3)),
        vortex_strength=circulations,
        core_radius=np.full(5, _SIGMA),
        particle_volume=np.full(5, _VOLUME),
        kinematic_viscosity=np.full(5, 0.01),
    )
    gamma_before = solver.particle_vortex_strength.copy()
    solver.advance()
    gamma_after = solver.particle_vortex_strength.copy()
    np.testing.assert_allclose(gamma_after, gamma_before, atol=1e-10, rtol=1e-6)


# ── Random Walk Method (RWM) ──────────────────────────────────────────────────


def _rwm_displacements(make_solver, kernel_name, kinematic_viscosity, n_samples):
    solver = _viscous_solver(make_solver, "RWM", kernel=kernel_name)
    solver.add_vortex_particles(
        position=np.zeros((n_samples, 3)),
        velocity=np.zeros((n_samples, 3)),
        vortex_strength=np.tile([0.0, 0.0, 1.0], (n_samples, 1)),
        core_radius=np.full(n_samples, _SIGMA),
        particle_volume=np.full(n_samples, _VOLUME),
        kinematic_viscosity=np.full(n_samples, kinematic_viscosity),
    )
    solver.advance()
    return solver.particle_position.copy()


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_rwm_one_step_mean_zero(kernel_name, backend, solver_for_backend):
    """RWM: ensemble-mean displacement must be approximately zero."""
    if backend != "CPU":
        pytest.skip("RWM ensemble test: random sequences differ across GPU backends")

    kinematic_viscosity = 0.1
    n_ensemble = 2_000
    displacements = _rwm_displacements(
        solver_for_backend, kernel_name, kinematic_viscosity, n_ensemble
    )
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

    kinematic_viscosity = 0.1
    time_step_size = 0.01
    n_ensemble = 2_000
    displacements = _rwm_displacements(
        solver_for_backend, kernel_name, kinematic_viscosity, n_ensemble
    )
    var = np.var(displacements, axis=0)
    expected = 2.0 * kinematic_viscosity * time_step_size
    for i, label in enumerate(["x", "y", "z"]):
        rel_err = abs(var[i] - expected) / expected
        assert rel_err < 0.15, (
            f"{kernel_name}/{backend}: RWM Var(Δ{label})={var[i]:.6f}, expected {expected:.6f}, rel_err={rel_err:.3e}"
        )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_rwm_circulation_unchanged(kernel_name, backend, solver_for_backend):
    """RWM must not modify circulation vortex_strength."""
    solver = _viscous_solver(solver_for_backend, "RWM", kernel=kernel_name)
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        vortex_strength=np.array([[0.0, 0.0, 1.0]]),
        core_radius=np.array([_SIGMA]),
        particle_volume=np.array([_VOLUME]),
        kinematic_viscosity=np.array([0.1]),
    )
    gamma_before = solver.particle_vortex_strength.copy()
    solver.advance()
    gamma_after = solver.particle_vortex_strength.copy()
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
        domain_bounds=_GRID_DIFFUSION_BOUNDS,
        dvh_grid_spacing=h,
        dvh_threshold=1e-8,
    )
    rng = np.random.default_rng(42)
    position = rng.uniform(-0.3, 0.3, (8, 3))
    circulations = rng.normal(0.0, 0.1, (8, 3))
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros((8, 3)),
        vortex_strength=circulations,
        core_radius=np.full(8, _SIGMA),
        particle_volume=np.full(8, _VOLUME),
        kinematic_viscosity=np.full(8, 0.01),
    )
    gamma_sum_before = solver.particle_vortex_strength.sum(axis=0)
    solver.advance()
    gamma_sum_after = solver.particle_vortex_strength.sum(axis=0)
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
    kinematic_viscosity = 0.01
    time_step_size = h**2 / (6.0 * kinematic_viscosity)  # explicit Laplacian CFL
    solver = _viscous_solver(
        solver_for_backend,
        "GBD",
        kernel=kernel_name,
        time_step_size=time_step_size,
        domain_bounds=_GRID_DIFFUSION_BOUNDS,
        gbd_grid_spacing=h,
        gbd_threshold=1e-8,
    )
    rng = np.random.default_rng(43)
    position = rng.uniform(-0.3, 0.3, (8, 3))
    circulations = rng.normal(0.0, 0.1, (8, 3))
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros((8, 3)),
        vortex_strength=circulations,
        core_radius=np.full(8, _SIGMA),
        particle_volume=np.full(8, _VOLUME),
        kinematic_viscosity=np.full(8, kinematic_viscosity),
    )
    gamma_sum_before = solver.particle_vortex_strength.sum(axis=0)
    solver.advance()
    gamma_sum_after = solver.particle_vortex_strength.sum(axis=0)
    rel_err = np.linalg.norm(gamma_sum_after - gamma_sum_before) / (
        np.linalg.norm(gamma_sum_before) + 1e-12
    )
    assert rel_err < 1e-4, (
        f"{kernel_name}/{backend}: GBD broke circulation conservation: rel_err={rel_err:.3e}"
    )


def test_gbd_full_solver_subcycles_1000_particles_above_cfl(backend, solver_for_backend):
    """The complete GBD rebuild must remain finite at the archived cube failure alpha."""
    side = 10
    h = 0.1
    time_step_size = 0.05
    alpha = 0.425
    kinematic_viscosity = alpha * h**2 / time_step_size
    solver = _viscous_solver(
        solver_for_backend,
        "GBD",
        time_step_size=time_step_size,
        domain_bounds=_GRID_DIFFUSION_BOUNDS,
        gbd_grid_spacing=h,
        gbd_threshold=1.0e-10,
        gbd_threshold_mode="absolute",
    )

    axis = (np.arange(side, dtype=np.float32) - 0.5 * (side - 1)) * h
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    position = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)
    radius_sq = x**2 + y**2 + z**2
    circulations = np.zeros((side**3, 3), dtype=np.float32)
    circulations[:, 2] = (1.0e-3 * np.exp(-radius_sq / (2.0 * (2.0 * h) ** 2))).ravel()
    solver.add_vortex_particles(
        position=position,
        velocity=np.zeros_like(position),
        vortex_strength=circulations,
        core_radius=np.full(side**3, 1.1 * h, dtype=np.float32),
        particle_volume=np.full(side**3, h**3, dtype=np.float32),
        kinematic_viscosity=np.full(side**3, kinematic_viscosity, dtype=np.float32),
    )

    circulation_before = circulations.astype(np.float64).sum(axis=0)
    solver.advance()

    assert 0 < solver.particles.n_particles_total <= solver.particles.capacity
    assert np.isfinite(solver.particle_vortex_strength).all()
    circulation_after = solver.particle_vortex_strength.astype(np.float64).sum(axis=0)
    np.testing.assert_allclose(circulation_after, circulation_before, rtol=2e-4, atol=1e-8)


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
        vortex_strength=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        core_radius=np.full(2, _SIGMA),
        particle_volume=np.full(2, _VOLUME),
        kinematic_viscosity=np.full(2, 0.01),
    )
    solver.advance()
    gamma_after = solver.particle_vortex_strength.copy()

    key = f"{scheme}_gamma"
    if backend == "CPU":
        test_cross_backend_viscous_consistency._cpu_cache[key] = gamma_after.copy()
    else:
        ref = test_cross_backend_viscous_consistency._cpu_cache.get(key)
        if ref is None:
            pytest.skip("CPU reference not yet computed")
        rel_err = np.linalg.norm(gamma_after - ref) / (np.linalg.norm(ref) + 1e-12)
        assert rel_err < 0.005, f"{scheme}/{backend}: viscous result mismatch vs CPU: {rel_err:.3e}"
