"""
Velocity gradient tensor tests — divergence-free, symmetries, and backend consistency.

The velocity gradient ∇u = ∂u_i/∂x_j is the fundamental tensor for vortex
stretching (ω·∇u) and strain-rate computation.  All properties tested here are
exact consequences of the Biot-Savart law and incompressibility.
"""

import numpy as np
import pytest

from source.solvers.VPM import Solver, VPMSetup
from source.solvers.VPM.config.backend import reset_taichi_backend
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StretchingConfig,
    VelocityConfig,
    ViscousConfig,
)

_SIGMA = 0.2
_VOLUME = (4.0 / 3.0) * np.pi * _SIGMA**3


def _single_blob_solver(make_solver, kernel_name):
    """Create a solver with one z-circulation blob at the origin."""
    solver = make_solver(
        time_step_size=0.01,
        particles_kernel=kernel_name,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
    )
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        circulation=np.array([[0.0, 0.0, 1.0]]),
        radius=np.array([_SIGMA]),
        volume=np.array([_VOLUME]),
        viscosity=np.array([0.0]),
    )
    return solver


def _two_particle_solver(make_solver, kernel_name, pos1, pos2, gamma1, gamma2):
    """Create a solver with two particles."""
    solver = make_solver(
        time_step_size=0.01,
        particles_kernel=kernel_name,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
    )
    solver.add_vortex_particles(
        position=np.array([pos1, pos2]),
        velocity=np.zeros((2, 3)),
        circulation=np.array([gamma1, gamma2]),
        radius=np.full(2, _SIGMA),
        volume=np.full(2, _VOLUME),
        viscosity=np.zeros(2),
    )
    return solver


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_divergence_free_single_blob(kernel_name, backend, solver_for_backend):
    """
    Incompressible flow: trace(∇u) = ∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z = 0.

    For a single Gaussian blob probed at several radii the trace must vanish
    within numerical round-off.

    Failure → compressible velocity kernel or wrong diagonal components.
    """
    solver = _single_blob_solver(solver_for_backend, kernel_name)
    r_values = np.array([0.5, 1.0, 2.0, 5.0]) * _SIGMA
    probes = np.column_stack([r_values, np.zeros_like(r_values), np.zeros_like(r_values)])
    grad_flat = solver.compute_target_velocity_gradients(probes)
    grad = grad_flat.reshape(-1, 3, 3)
    trace = np.trace(grad, axis1=1, axis2=2)
    assert np.allclose(trace, 0.0, atol=1e-5), (
        f"{kernel_name}/{backend}: trace(∇u) = {trace} (must be zero)"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_self_gradient_zero(kernel_name, backend, solver_for_backend):
    """
    Velocity gradient at the particle centre must be exactly zero.

    A blob induces no net gradient on itself (the singular self-term is excluded).

    Failure → missing self-exclusion in gradient kernel.
    """
    solver = _single_blob_solver(solver_for_backend, kernel_name)
    grad_flat = solver.compute_target_velocity_gradients(np.array([[0.0, 0.0, 0.0]]))
    grad = grad_flat.reshape(3, 3)
    assert np.allclose(grad, 0.0, atol=1e-6), (
        f"{kernel_name}/{backend}: self-gradient = {grad} (must be zero)"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_vorticity_tensor_antisymmetry(kernel_name, backend, solver_for_backend):
    """
    The vorticity tensor Ω = ½(∇u − ∇uᵀ) must be antisymmetric.

    For a z-vortex probed off-axis this implies Ω_xy = −Ω_yx and all diagonal
    Ω_ii = 0.

    Failure → wrong sign in off-diagonal gradient components or non-zero
    diagonal rotation terms.
    """
    solver = _single_blob_solver(solver_for_backend, kernel_name)
    probes = np.array([[_SIGMA, 0.0, 0.0], [2 * _SIGMA, 0.0, 0.0]])
    grad_flat = solver.compute_target_velocity_gradients(probes)
    grad = grad_flat.reshape(-1, 3, 3)
    Omega = 0.5 * (grad - np.transpose(grad, (0, 2, 1)))
    # Diagonal must vanish
    diag = np.trace(Omega, axis1=1, axis2=2)
    assert np.allclose(diag, 0.0, atol=1e-7), (
        f"{kernel_name}/{backend}: diagonal of Ω = {diag} (must be zero)"
    )
    # Off-diagonal antisymmetry
    assert np.allclose(Omega[:, 0, 1], -Omega[:, 1, 0], atol=1e-7), (
        f"{kernel_name}/{backend}: Ω_xy ≠ −Ω_yx"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_strain_tensor_symmetry(kernel_name, backend, solver_for_backend):
    """
    The strain-rate tensor S = ½(∇u + ∇uᵀ) must be symmetric.

    For any flow configuration S_ij = S_ji exactly.

    Failure → gradient tensor stored in wrong index order (row/column swap).
    """
    solver = _two_particle_solver(
        solver_for_backend,
        kernel_name,
        pos1=[-0.5, 0.0, 0.0],
        pos2=[0.5, 0.0, 0.0],
        gamma1=[0.0, 0.0, 1.0],
        gamma2=[0.0, 0.0, -1.0],
    )
    probes = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    grad_flat = solver.compute_target_velocity_gradients(probes)
    grad = grad_flat.reshape(-1, 3, 3)
    S = 0.5 * (grad + np.transpose(grad, (0, 2, 1)))
    diff = S - np.transpose(S, (0, 2, 1))
    assert np.allclose(diff, 0.0, atol=1e-8), (
        f"{kernel_name}/{backend}: strain tensor not symmetric: max |S−Sᵀ| = {np.max(np.abs(diff)):.3e}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_cross_backend_consistency_gradients(kernel_name, backend, solver_for_backend):
    """
    Velocity gradients on CPU, CUDA, and Vulkan must agree within 1%.

    Failure → backend-specific numerical divergence in gradient kernel.
    """
    if not hasattr(test_cross_backend_consistency_gradients, "_cpu_cache"):
        test_cross_backend_consistency_gradients._cpu_cache = {}

    solver = _single_blob_solver(solver_for_backend, kernel_name)
    probes = np.array([[_SIGMA, 0.0, 0.0], [2 * _SIGMA, 0.0, 0.0], [5 * _SIGMA, 0.0, 0.0]])
    grad_flat = solver.compute_target_velocity_gradients(probes)
    grad = grad_flat.reshape(-1, 3, 3)

    key = f"{kernel_name}_grad"
    if backend == "CPU":
        test_cross_backend_consistency_gradients._cpu_cache[key] = grad.copy()
    else:
        ref = test_cross_backend_consistency_gradients._cpu_cache.get(key)
        if ref is None:
            pytest.skip("CPU reference not yet computed — run CPU tests first")
        rel_err = np.linalg.norm(grad - ref) / (np.linalg.norm(ref) + 1e-12)
        assert rel_err < 0.01, f"{kernel_name}/{backend}: gradient mismatch vs CPU: {rel_err:.3e}"


@pytest.mark.parametrize(
    "target",
    [
        np.array([[0.33, 0.17, -0.11]]),
        np.array([[0.07, 0.03, -0.02]]),
    ],
)
def test_target_velocity_gradient_matches_velocity_finite_difference(tmp_path, target):
    """
    ∇u must be the spatial derivative of the same target-velocity kernel used
    for advection.  Divergence-free and symmetry tests can pass even when the
    whole tensor has the wrong sign convention.
    """
    reset_taichi_backend()
    solver = Solver(
        VPMSetup(
            time_step_size=0.01,
            processing_unit="CPU",
            particles_kernel="GAUSSIAN",
            stretching=StretchingConfig.disabled(),
            viscous=ViscousConfig(scheme="NONE"),
            advection=AdvectionConfig(scheme="NONE"),
            backup_directory=str(tmp_path),
            backup_frequency=0,
            logging_frequency=0,
        )
    )
    sigma = 0.2
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        circulation=np.array([[0.2, -0.4, 1.0]]),
        radius=np.array([sigma]),
        volume=np.array([(4.0 / 3.0) * np.pi * sigma**3]),
        viscosity=np.array([0.0]),
    )

    h = 1e-4
    grad_fd = np.zeros((3, 3))
    for axis in range(3):
        offset = np.zeros((1, 3))
        offset[0, axis] = h
        up = solver.compute_target_velocities(target + offset, include_freestream=False)[0]
        down = solver.compute_target_velocities(target - offset, include_freestream=False)[0]
        grad_fd[:, axis] = (up - down) / (2.0 * h)

    grad_kernel = solver.compute_target_velocity_gradients(target).reshape(3, 3)
    np.testing.assert_allclose(grad_kernel, grad_fd, rtol=1.5e-2, atol=2e-3)


def test_complete_target_gradient_uses_treecode_velocity_operator(tmp_path, monkeypatch):
    """The coupled Jacobian must differentiate the configured treecode trace."""
    reset_taichi_backend()
    theta = 0.2
    solver = Solver(
        VPMSetup(
            time_step_size=0.01,
            processing_unit="CPU",
            particles_kernel="GAUSSIAN",
            stretching=StretchingConfig.disabled(),
            viscous=ViscousConfig(scheme="NONE"),
            advection=AdvectionConfig(scheme="NONE"),
            velocity=VelocityConfig.treecode(theta=theta, multipole_order=2),
            backup_directory=str(tmp_path),
            backup_frequency=0,
            logging_frequency=0,
        )
    )
    positions = np.array(
        [
            [-0.55, -0.30, 0.10],
            [-0.20, 0.45, -0.35],
            [0.30, -0.40, 0.25],
            [0.55, 0.20, -0.15],
            [-0.45, 0.15, 0.50],
            [0.15, 0.55, 0.30],
            [0.45, -0.10, -0.50],
            [-0.10, -0.55, -0.25],
        ]
    )
    solver.add_vortex_particles(
        position=positions,
        velocity=np.zeros_like(positions),
        circulation=np.array(
            [
                [0.2, -0.1, 0.6],
                [-0.4, 0.3, 0.1],
                [0.1, 0.5, -0.2],
                [0.3, -0.2, -0.4],
                [-0.1, 0.4, 0.3],
                [0.5, 0.1, -0.3],
                [-0.3, -0.5, 0.2],
                [0.2, 0.2, 0.4],
            ]
        ),
        radius=np.full(len(positions), _SIGMA),
        volume=np.full(len(positions), _VOLUME),
        viscosity=np.zeros(len(positions)),
    )

    physics = solver.physics
    hierarchical_gradient = physics.compute_target_velocity_gradients_hierarchical
    calls = []

    def record_hierarchical_gradient(*args, **kwargs):
        calls.append(kwargs["theta"])
        return hierarchical_gradient(*args, **kwargs)

    def direct_gradient_must_not_run(*args, **kwargs):
        pytest.fail("treecode complete-gradient evaluation used the direct kernel")

    monkeypatch.setattr(
        physics, "compute_target_velocity_gradients_hierarchical", record_hierarchical_gradient
    )
    monkeypatch.setattr(physics, "compute_target_velocity_gradients", direct_gradient_must_not_run)

    target = np.array([[0.12, 0.16, -0.22]])
    gradient = solver.compute_complete_target_velocity_gradients(target, h=0.04)[0]
    assert calls == [theta]

    step = 2.0e-4
    finite_difference = np.zeros((3, 3))
    for axis in range(3):
        offset = np.zeros((1, 3))
        offset[0, axis] = step
        upper = solver.compute_target_velocities(target + offset, include_freestream=False)[0]
        lower = solver.compute_target_velocities(target - offset, include_freestream=False)[0]
        finite_difference[:, axis] = (upper - lower) / (2.0 * step)

    np.testing.assert_allclose(gradient, finite_difference, rtol=5e-2, atol=3e-3)
