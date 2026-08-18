"""
Single-blob physics tests — one particle, all backends, all kernels.

A single vortex blob is the simplest non-trivial VPM configuration.  Every
quantity (velocity, vorticity, energy, impulse) has a closed-form analytical
expression, making this the ideal smoke test for kernel correctness and
backend consistency.
"""

import numpy as np
import pytest

from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StretchingConfig,
    ViscousConfig,
)

# ── Analytical helpers ─────────────────────────────────────────────────────────

# Kernel-specific constants at ρ=0
_ZETA_0 = {
    "GAUSSIAN": 1.0 / (np.pi**1.5),
    "HIGH_ORDER_GAUSSIAN": 2.5 / (np.pi**1.5),
    "SUPER_GAUSSIAN": 2.5 * np.sqrt(2.0 / np.pi) / (4.0 * np.pi),
    "WINCKELMANS": 7.5 / (4.0 * np.pi),
}

_G_0 = {
    "GAUSSIAN": 1.0 / (2.0 * np.pi**1.5),  # limit of erf(ρ)/(4πρ) as ρ→0
    "HIGH_ORDER_GAUSSIAN": 3.0 / (4.0 * np.pi**1.5),
    "SUPER_GAUSSIAN": 1.5 * np.sqrt(2.0 / np.pi) / (4.0 * np.pi),
    "WINCKELMANS": 1.5 / (4.0 * np.pi),
}

# Kernel second moments m2 = int |q|^2 zeta(|q|) d^3q, which is what the angular
# impulse correction is: A = (1/3) sum x*(x*G) - (2/9) m2 sigma^2 G follows from
# int x*(x*omega) dV = d*(d*G) - (2/3) m2 sigma^2 G for a blob at d.  Verified
# against 3-D quadrature of int x*(x*omega) dV, not copied from the kernels --
# GAUSSIAN was 3.0 and SUPER_GAUSSIAN 1.875 in both places, and both were wrong.
# The two moment-cancelling polynomial kernels give exactly zero.
_ANG_CORR = {
    "GAUSSIAN": 1.5,
    "HIGH_ORDER_GAUSSIAN": 0.0,
    "SUPER_GAUSSIAN": 0.0,
    "WINCKELMANS": 1.5,
}

_SIGMA = 0.2
_ALPHA_Z = 1.0
_VOLUME = (4.0 / 3.0) * np.pi * _SIGMA**3


# ── Solver factory ─────────────────────────────────────────────────────────────


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
        circulation=np.array([[0.0, 0.0, _ALPHA_Z]]),
        radius=np.array([_SIGMA]),
        volume=np.array([_VOLUME]),
        viscosity=np.array([0.0]),
    )
    return solver


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_self_velocity_zero(kernel_name, backend, solver_for_backend):
    """
    Velocity at the particle centre must be exactly zero (q(0)=0).

    Failure → singular Biot-Savart kernel or missing self-interaction exclusion.
    """
    solver = _single_blob_solver(solver_for_backend, kernel_name)
    vel = solver.compute_target_velocities(np.array([[0.0, 0.0, 0.0]]), include_freestream=False)
    assert np.allclose(vel, 0.0, atol=1e-6), (
        f"{kernel_name}/{backend}: self-velocity = {vel} (must be zero)"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_vorticity_at_origin(kernel_name, backend, solver_for_backend):
    """
    Vorticity at the origin: ωz(0) = Γz · ζ(0) / σ³.

    Failure → wrong ζ normalisation or σ exponent.
    """
    solver = _single_blob_solver(solver_for_backend, kernel_name)
    omega = solver.compute_target_vorticities(np.array([[0.0, 0.0, 0.0]]))
    expected = _ALPHA_Z * _ZETA_0[kernel_name] / (_SIGMA**3)
    assert abs(float(omega[0, 2]) - expected) / expected < 0.02, (
        f"{kernel_name}/{backend}: ωz(0) = {omega[0, 2]:.6e}, expected {expected:.6e}"
    )


def test_target_gradient_layout_matches_velocity_finite_difference(backend, solver_for_backend):
    """Certify ``J[i,j]=du_i/dx_j`` before using it in coupling.

    A single regularized blob's reported ``zeta*Gamma`` field is not globally
    divergence-free, whereas Biot--Savart returns its solenoidal projection,
    so that separate field is not a valid pointwise curl oracle.  Differencing
    the actual target-velocity API certifies all nine entries without that
    assumption.
    """
    solver = _single_blob_solver(solver_for_backend, "GAUSSIAN")
    probes = np.array([[0.11, 0.07, 0.0], [0.23, -0.05, 0.0], [-0.16, 0.12, 0.0]])
    jacobian = solver.compute_target_velocity_gradients(probes).reshape(-1, 3, 3)
    step = 2.0e-4
    finite_difference = np.empty_like(jacobian)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step
        u_plus = solver.compute_target_velocities(probes + offset, include_freestream=False)
        u_minus = solver.compute_target_velocities(probes - offset, include_freestream=False)
        finite_difference[:, :, axis] = (u_plus - u_minus) / (2.0 * step)
    np.testing.assert_allclose(jacobian, finite_difference, rtol=2.0e-3, atol=2.0e-3)


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_vorticity_radial_decay(kernel_name, backend, solver_for_backend):
    """
    Vorticity at r = σ, 2σ, 3σ must match ζ(r/σ)/σ³.

    Failure → wrong radial profile or σ scaling.
    """
    solver = _single_blob_solver(solver_for_backend, kernel_name)
    r_values = np.array([1.0, 2.0, 3.0]) * _SIGMA
    probes = np.column_stack([r_values, np.zeros_like(r_values), np.zeros_like(r_values)])
    omega = solver.compute_target_vorticities(probes)[:, 2]

    rho = r_values / _SIGMA
    if kernel_name == "GAUSSIAN":
        zeta = np.exp(-(rho**2)) / (np.pi**1.5)
    elif kernel_name == "HIGH_ORDER_GAUSSIAN":
        zeta = (2.5 - rho**2) * np.exp(-(rho**2)) / (np.pi**1.5)
    elif kernel_name == "SUPER_GAUSSIAN":
        zeta = np.sqrt(2.0 / np.pi) * (2.5 - rho**2 / 2.0) * np.exp(-(rho**2) / 2.0) / (4.0 * np.pi)
    else:
        zeta = 7.5 / ((rho**2 + 1.0) ** 3.5 * 4.0 * np.pi)
    expected = _ALPHA_Z * zeta / _SIGMA**3
    assert np.allclose(omega, expected, rtol=0.02, atol=1e-7), (
        f"{kernel_name}/{backend}: radial vorticity = {omega}, expected {expected}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_far_field_velocity(kernel_name, backend, solver_for_backend):
    """
    Far from the core (r = 10σ) the velocity approaches the point-vortex limit:
    u_y → Γz / (4π r²).

    Failure → wrong far-field normalisation or kernel saturation.
    """
    solver = _single_blob_solver(solver_for_backend, kernel_name)
    r = 10.0 * _SIGMA
    probes = np.array([[r, 0.0, 0.0]])
    vel = solver.compute_target_velocities(probes, include_freestream=False)
    uy = float(vel[0, 1])
    expected = _ALPHA_Z / (4.0 * np.pi * r**2)
    rel_err = abs(uy - expected) / expected
    assert rel_err < 0.02, (
        f"{kernel_name}/{backend}: u_y({r}) = {uy:.6e}, expected {expected:.6e}, "
        f"rel_err={rel_err:.3e}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_kinetic_energy_self(kernel_name, backend, solver_for_backend):
    """
    Self-energy of one blob: E = ½ · g(0)/(σ√2) · Γ².

    The √2 is not a fudge: E = ½∫ω·ψ convolves the blob with its own mollified
    Green's function, and for a Gaussian ζ_σ * ζ_σ = ζ_{σ√2}.  Verified against
    direct quadrature of ½∫ω·ψ for σ=0.4, Γ_z=1.3: 0.13413031 versus
    0.13413031 from this formula (the pair-mean convention gives 0.18968890,
    41 % high).  See evaluation.py::compute_flow_integrals_kernel.

    Failure → wrong energy kernel g, wrong pair width, or missing
    self-interaction.
    """
    solver = _single_blob_solver(solver_for_backend, kernel_name)
    # Use the solver's flow integral evaluation
    solver.update_state()
    solver._update_all_flow_integrals()
    ke = solver.total_kinetic_energy
    expected = 0.5 * _G_0[kernel_name] / (_SIGMA * np.sqrt(2.0)) * _ALPHA_Z**2
    rel_err = abs(ke - expected) / expected
    assert rel_err < 0.05, (
        f"{kernel_name}/{backend}: KE = {ke:.6e}, expected {expected:.6e}, rel_err={rel_err:.3e}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_helicity_is_zero(kernel_name, backend, solver_for_backend):
    """
    Helicity of a single blob is zero (H = u · ω, and symmetry gives zero).

    Failure → sign error in helicity kernel or missing self-interaction.
    """
    solver = _single_blob_solver(solver_for_backend, kernel_name)
    solver.update_state()
    solver._update_all_flow_integrals()
    helicity = solver.total_helicity
    assert abs(helicity) < 1e-6, (
        f"{kernel_name}/{backend}: helicity = {helicity:.3e} (must be zero for single blob)"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_linear_impulse(kernel_name, backend, solver_for_backend):
    """
    Linear impulse: I = ½ Σ r × Γ.

    For a particle at (1,0,0) with Γ=(0,0,1): I = (0, -0.5, 0).

    Failure → wrong impulse formula or sign convention.
    """
    solver = solver_for_backend(
        time_step_size=0.01,
        particles_kernel=kernel_name,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
    )
    solver.add_vortex_particles(
        position=np.array([[1.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        circulation=np.array([[0.0, 0.0, _ALPHA_Z]]),
        radius=np.array([_SIGMA]),
        volume=np.array([_VOLUME]),
        viscosity=np.array([0.0]),
    )
    solver.update_state()
    solver._update_all_flow_integrals()
    impulse = solver.total_linear_impulse
    expected = np.array([0.0, -0.5, 0.0])
    assert np.allclose(impulse, expected, atol=1e-6), (
        f"{kernel_name}/{backend}: impulse = {impulse}, expected {expected}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_angular_impulse(kernel_name, backend, solver_for_backend):
    """
    Angular impulse: A = (1/3) Σ r × (r × Γ) − (2/9) C σ² Γ_total.

    For a particle at (1,0,0) with Γ=(0,0,1), σ=0.2:
        A = (1/3)(0, 0, -1) − (2/9) C (0.2)² (0,0,1)

    Failure → wrong angular impulse formula or correction constant.
    """
    solver = solver_for_backend(
        time_step_size=0.01,
        particles_kernel=kernel_name,
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig(scheme="NONE"),
        advection=AdvectionConfig(scheme="NONE"),
    )
    solver.add_vortex_particles(
        position=np.array([[1.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        circulation=np.array([[0.0, 0.0, _ALPHA_Z]]),
        radius=np.array([_SIGMA]),
        volume=np.array([_VOLUME]),
        viscosity=np.array([0.0]),
    )
    solver.update_state()
    solver._update_all_flow_integrals()
    ang = solver.total_angular_impulse
    c = _ANG_CORR[kernel_name]
    expected = np.array([0.0, 0.0, -1.0 / 3.0 - (2.0 / 9.0) * c * _SIGMA**2 * _ALPHA_Z])
    assert np.allclose(ang, expected, atol=1e-5), (
        f"{kernel_name}/{backend}: angular impulse = {ang}, expected {expected}"
    )


@pytest.mark.parametrize(
    "kernel_name", ["GAUSSIAN", "HIGH_ORDER_GAUSSIAN", "SUPER_GAUSSIAN", "WINCKELMANS"]
)
def test_cross_backend_consistency_single_blob(kernel_name, backend, solver_for_backend):
    """
    Meta-test: the same single-blob setup on CPU, CUDA, and Vulkan must give
    mutually consistent velocities and vorticity within 0.5%.

    This test is run per-backend; the assertion compares the current backend
    against a reference CPU run stored in a module-level cache.
    """
    # Module-level cache for CPU reference values
    if not hasattr(test_cross_backend_consistency_single_blob, "_cpu_cache"):
        test_cross_backend_consistency_single_blob._cpu_cache = {}

    solver = _single_blob_solver(solver_for_backend, kernel_name)
    probes = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    vel = solver.compute_target_velocities(probes, include_freestream=False)
    omega = solver.compute_target_vorticities(probes)

    key = f"{kernel_name}_vel"
    if backend == "CPU":
        test_cross_backend_consistency_single_blob._cpu_cache[key] = vel.copy()
        test_cross_backend_consistency_single_blob._cpu_cache[f"{kernel_name}_omega"] = omega.copy()
    else:
        ref_vel = test_cross_backend_consistency_single_blob._cpu_cache.get(key)
        ref_omega = test_cross_backend_consistency_single_blob._cpu_cache.get(
            f"{kernel_name}_omega"
        )
        if ref_vel is None or ref_omega is None:
            pytest.skip("CPU reference not yet computed — run CPU tests first")
        rel_err_vel = np.linalg.norm(vel - ref_vel) / (np.linalg.norm(ref_vel) + 1e-12)
        rel_err_omega = np.linalg.norm(omega - ref_omega) / (np.linalg.norm(ref_omega) + 1e-12)
        assert rel_err_vel < 0.005, (
            f"{kernel_name}/{backend}: velocity mismatch vs CPU: {rel_err_vel:.3e}"
        )
        assert rel_err_omega < 0.005, (
            f"{kernel_name}/{backend}: vorticity mismatch vs CPU: {rel_err_omega:.3e}"
        )
