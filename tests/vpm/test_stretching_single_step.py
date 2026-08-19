"""
Vortex stretching single-step tests — symmetry, conservation, and scheme convergence.

Stretching updates circulation via direct, transposed, or mixed pairwise
operators.  These tests verify exact properties that hold for a
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
@pytest.mark.parametrize("mode", ["DIRECT", "TRANSPOSED", "MIXED"])
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
    solver.physics.vortex_stretching(
        solver.particles, time_step_size=0.01, scheme=scheme, mode=mode
    )
    gamma_after = solver.particles.circulation_cpu()
    assert np.allclose(gamma_after, gamma_before, atol=1e-6), (
        f"{kernel_name}/{backend}/{scheme}/{mode}: single blob stretched: "
        f"before={gamma_before}, after={gamma_after}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
@pytest.mark.parametrize("scheme", ["EULER", "RK2", "RK3", "RK4"])
@pytest.mark.parametrize("mode", ["DIRECT", "TRANSPOSED", "MIXED"])
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
    solver.physics.vortex_stretching(
        solver.particles, time_step_size=0.01, scheme=scheme, mode=mode
    )
    gamma_after = solver.particles.circulation_cpu()
    assert np.allclose(gamma_after, gamma_before, atol=1e-6), (
        f"{kernel_name}/{backend}/{scheme}/{mode}: 2-D vortices stretched: "
        f"before={gamma_before}, after={gamma_after}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_transposed_circulation_conservation(kernel_name, backend, solver_for_backend):
    """
    The transposed pairwise stretching operator is antisymmetric, so the vector
    sum ΣΓ must be conserved.

    Failure → stretching scheme introduces a source/sink term.
    """
    solver = _solver_with_particles(
        solver_for_backend,
        kernel_name,
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        circulations=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    sum_before = np.sum(solver.particles.circulation_cpu(), axis=0)
    solver.physics.vortex_stretching(
        solver.particles, time_step_size=0.01, scheme="RK3", mode="TRANSPOSED"
    )
    sum_after = np.sum(solver.particles.circulation_cpu(), axis=0)
    assert np.allclose(sum_after, sum_before, atol=1e-5), (
        f"{kernel_name}/{backend}/TRANSPOSED: circulation not conserved: "
        f"before={sum_before}, after={sum_after}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
@pytest.mark.parametrize("mode", ["DIRECT", "MIXED"])
def test_stretching_rate_matches_velocity_gradient_form(
    kernel_name, mode, backend, solver_for_backend
):
    """
    The explicit pairwise stretching kernel must match the equivalent
    velocity-gradient form:

    * DIRECT: dΓ/dt = ∇u · Γ
    * MIXED:  dΓ/dt = S · Γ, with S = 0.5(∇u + ∇uᵀ)

    Failure → sign/order mismatch between the stretching kernel and the
    velocity-gradient kernel used elsewhere by LES and diagnostics.
    """
    solver = _solver_with_particles(
        solver_for_backend,
        kernel_name,
        positions=[
            [0.0, 0.0, 0.0],
            [0.7, 0.2, -0.1],
            [-0.2, 0.6, 0.3],
        ],
        circulations=[
            [0.8, -0.2, 0.4],
            [-0.3, 0.9, 0.2],
            [0.1, -0.5, 0.7],
        ],
    )

    gamma0 = solver.particles.circulation_cpu().copy()
    solver.physics.compute_velocity_gradients(solver.particles)
    grad_u = solver.particles.velocity_gradient_cpu()

    mode_int = 0 if mode == "DIRECT" else 2
    n_particles = len(gamma0)
    solver.physics._resize_temp_fields(n_particles)
    solver.physics._zero_temp_fields()
    solver.physics.compute_stretching_rate_kernel(
        solver.particles.position,
        solver.particles.circulation,
        solver.particles.radius,
        solver.physics.dstr_dt_temp,
        mode_int,
        n_particles,
    )
    actual_rate = solver.physics.dstr_dt_temp.to_numpy()[:n_particles]

    if mode == "DIRECT":
        expected_rate = np.einsum("nij,nj->ni", grad_u, gamma0)
    else:
        strain = 0.5 * (grad_u + np.swapaxes(grad_u, 1, 2))
        expected_rate = np.einsum("nij,nj->ni", strain, gamma0)

    np.testing.assert_allclose(
        actual_rate,
        expected_rate,
        rtol=3e-3,
        atol=3e-5,
        err_msg=f"{kernel_name}/{backend}/{mode}: stretching rate disagrees with ∇u form",
    )


def test_batched_direct_rate_matches_single_dispatch(backend, solver_for_backend):
    """Target batching must preserve each particle's source accumulation order."""
    solver = _solver_with_particles(
        solver_for_backend,
        "GAUSSIAN",
        positions=[
            [0.0, 0.0, 0.0],
            [0.7, 0.2, -0.1],
            [-0.2, 0.6, 0.3],
            [0.1, -0.4, 0.8],
        ],
        circulations=[
            [0.8, -0.2, 0.4],
            [-0.3, 0.9, 0.2],
            [0.1, -0.5, 0.7],
            [0.4, 0.2, -0.6],
        ],
    )
    p = solver.physics
    particles = solver.particles
    n_particles = len(particles)
    p._resize_temp_fields(n_particles)
    p._zero_temp_fields(n_particles)
    p.compute_stretching_rate_kernel(
        particles.position,
        particles.circulation,
        particles.radius,
        p.dstr_dt_temp,
        1,
        n_particles,
    )
    p.compute_stretching_rate_batch_kernel(
        particles.position,
        particles.circulation,
        particles.radius,
        p.dstr_dt_temp2,
        1,
        0,
        2,
        n_particles,
    )
    p.compute_stretching_rate_batch_kernel(
        particles.position,
        particles.circulation,
        particles.radius,
        p.dstr_dt_temp2,
        1,
        2,
        2,
        n_particles,
    )

    reference = p.dstr_dt_temp.to_numpy()[:n_particles]
    batched = p.dstr_dt_temp2.to_numpy()[:n_particles]
    np.testing.assert_array_equal(
        batched,
        reference,
        err_msg=f"{backend}: bounded stretching dispatch changed the direct rate",
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
@pytest.mark.parametrize("mode", ["DIRECT", "MIXED"])
def test_nonconservative_stretching_changes_circulation(
    kernel_name, mode, backend, solver_for_backend
):
    """
    Direct and mixed stretching are not antisymmetric pairwise operators, so
    they need not conserve ΣΓ for a generic 3-D particle arrangement.
    """
    solver = _solver_with_particles(
        solver_for_backend,
        kernel_name,
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        circulations=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    sum_before = np.sum(solver.particles.circulation_cpu(), axis=0)
    solver.physics.vortex_stretching(solver.particles, time_step_size=0.01, scheme="RK3", mode=mode)
    sum_after = np.sum(solver.particles.circulation_cpu(), axis=0)
    delta = np.linalg.norm(sum_after - sum_before)
    assert delta > 1e-5, (
        f"{kernel_name}/{backend}/{mode}: expected generic circulation drift, "
        f"before={sum_before}, after={sum_after}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
@pytest.mark.parametrize("mode", ["DIRECT", "TRANSPOSED"])
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
    time_step_size = 1e-6
    results = {}
    for scheme in ["EULER", "RK2", "RK3", "RK4"]:
        # Clone circulation state
        gamma0 = solver.particles.circulation_cpu().copy()
        solver.particles.set_field("circulation", gamma0)
        solver.physics.vortex_stretching(
            solver.particles, time_step_size=time_step_size, scheme=scheme, mode=mode
        )
        results[scheme] = solver.particles.circulation_cpu().copy()

    # Compare RK4 (highest order) against the others
    ref = results["RK4"]
    for scheme in ["EULER", "RK2", "RK3"]:
        rel = np.linalg.norm(results[scheme] - ref) / (np.linalg.norm(ref) + 1e-12)
        assert rel < 1e-3, (
            f"{kernel_name}/{backend}/{mode}: {scheme} deviates from RK4 at small dt: {rel:.3e}"
        )
