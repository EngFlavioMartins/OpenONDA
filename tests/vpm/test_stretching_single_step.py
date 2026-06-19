"""
Vortex stretching single-step tests — symmetry, conservation, and scheme convergence.

Stretching updates circulation via dΓ/dt = (∇u)ᵀ·Γ (transpose) or
ω·∇u (classical).  These tests verify exact properties that hold for a
single time step with no advection or diffusion.
"""

import numpy as np
import pytest

from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
)

_SIGMA = 0.2
_VOLUME = (4.0 / 3.0) * np.pi * _SIGMA**3


def _solver_with_particles(make_solver, kernel_name, positions, circulations):
    """Create a solver with stretching enabled and given particles."""
    solver = make_solver(
        time_step_size=0.01,
        particles_kernel=kernel_name,
        stretching=StretchingConfig(scheme="RK3"),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
        velocity=VelocityConfig.direct(),
    )
    N = len(positions)
    solver.add_vortex_particles(
        position=np.array(positions),
        velocity=np.zeros((N, 3)),
        circulation=np.array(circulations),
        radius=np.full(N, _SIGMA),
        volume=np.full(N, _VOLUME),
        viscosity=np.zeros(N),
    )
    return solver


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
@pytest.mark.parametrize("scheme", ["EULER", "RK2", "RK3", "RK4"])
@pytest.mark.parametrize("mode", ["CLASSICAL", "TRANSPOSE", "MIXED"])
def test_single_blob_no_stretching(kernel_name, scheme, mode, backend, solver_for_backend):
    """
    A single blob has zero self-induced velocity gradient, therefore zero
    stretching.  Circulation must remain unchanged for every scheme and mode.

    Failure → missing self-exclusion in stretching kernel or non-zero
    self-gradient.
    """
    solver = _solver_with_particles(
        solver_for_backend,
        kernel_name,
        positions=[[0.0, 0.0, 0.0]],
        circulations=[[0.0, 0.0, 1.0]],
    )
    gamma_before = solver.particles.circulation_cpu().copy()
    solver.physics.vortex_stretching(solver.particles, dt=0.01, scheme=scheme, mode=mode)
    gamma_after = solver.particles.circulation_cpu()
    assert np.allclose(gamma_after, gamma_before, atol=1e-6), (
        f"{kernel_name}/{backend}/{scheme}/{mode}: single blob stretched: "
        f"before={gamma_before}, after={gamma_after}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
@pytest.mark.parametrize("scheme", ["EULER", "RK2", "RK3", "RK4"])
@pytest.mark.parametrize("mode", ["CLASSICAL", "TRANSPOSE", "MIXED"])
def test_two_parallel_vortices_2d_invariance(
    kernel_name, scheme, mode, backend, solver_for_backend
):
    """
    Two infinite parallel z-vortices constitute a 2-D flow.  Vortex stretching
    is identically zero in 2-D because ∂u_z/∂z = 0 and ω_x = ω_y = 0.

    Circulations must remain unchanged after one step.

    Failure → stretching kernel does not respect 2-D invariance.
    """
    solver = _solver_with_particles(
        solver_for_backend,
        kernel_name,
        positions=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
        circulations=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    )
    gamma_before = solver.particles.circulation_cpu().copy()
    solver.physics.vortex_stretching(solver.particles, dt=0.01, scheme=scheme, mode=mode)
    gamma_after = solver.particles.circulation_cpu()
    assert np.allclose(gamma_after, gamma_before, atol=1e-6), (
        f"{kernel_name}/{backend}/{scheme}/{mode}: 2-D vortices stretched: "
        f"before={gamma_before}, after={gamma_after}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
@pytest.mark.parametrize("mode", ["CLASSICAL", "TRANSPOSE", "MIXED"])
def test_circulation_conservation(kernel_name, mode, backend, solver_for_backend):
    """
    Stretching is a kinematic redistribution of vorticity; total circulation
    ΣΓ must be conserved.

    Failure → stretching scheme introduces a source/sink term.
    """
    solver = _solver_with_particles(
        solver_for_backend,
        kernel_name,
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        circulations=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    sum_before = np.sum(solver.particles.circulation_cpu(), axis=0)
    solver.physics.vortex_stretching(solver.particles, dt=0.01, scheme="RK3", mode=mode)
    sum_after = np.sum(solver.particles.circulation_cpu(), axis=0)
    assert np.allclose(sum_after, sum_before, atol=1e-5), (
        f"{kernel_name}/{backend}/{mode}: circulation not conserved: "
        f"before={sum_before}, after={sum_after}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
@pytest.mark.parametrize("mode", ["CLASSICAL", "TRANSPOSE"])
def test_small_dt_scheme_convergence(kernel_name, mode, backend, solver_for_backend):
    """
    For an infinitesimal time step all consistent schemes (Euler, RK2, RK3, RK4)
    must agree to first order.  Using dt = 1e-6 the relative difference between
    any two schemes must be < 1e-3.

    Failure → inconsistent Butcher tableau or bug in RK combine kernels.
    """
    solver = _solver_with_particles(
        solver_for_backend,
        kernel_name,
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        circulations=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    dt = 1e-6
    results = {}
    for scheme in ["EULER", "RK2", "RK3", "RK4"]:
        # Clone circulation state
        gamma0 = solver.particles.circulation_cpu().copy()
        solver.particles.set_field("circulation", gamma0)
        solver.physics.vortex_stretching(solver.particles, dt=dt, scheme=scheme, mode=mode)
        results[scheme] = solver.particles.circulation_cpu().copy()

    # Compare RK4 (highest order) against the others
    ref = results["RK4"]
    for scheme in ["EULER", "RK2", "RK3"]:
        rel = np.linalg.norm(results[scheme] - ref) / (np.linalg.norm(ref) + 1e-12)
        assert rel < 1e-3, (
            f"{kernel_name}/{backend}/{mode}: {scheme} deviates from RK4 at small dt: {rel:.3e}"
        )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_gradu_mode_zero_without_precomputed_gradient(kernel_name, backend, solver_for_backend):
    """
    GRADU mode reads particles.velocity_gradient.  If gradients have not been
    computed (default is zero), stretching must produce no change.

    Failure → GRADU mode accesses uninitialised memory or ignores the field.
    """
    solver = _solver_with_particles(
        solver_for_backend,
        kernel_name,
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        circulations=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    gamma_before = solver.particles.circulation_cpu().copy()
    solver.physics.vortex_stretching(solver.particles, dt=0.01, scheme="RK3", mode="GRADU")
    gamma_after = solver.particles.circulation_cpu()
    assert np.allclose(gamma_after, gamma_before, atol=1e-6), (
        f"{kernel_name}/{backend}: GRADU mode changed circulation with zero gradU"
    )
