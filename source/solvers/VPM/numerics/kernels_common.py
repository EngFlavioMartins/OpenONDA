"""
Factories for shared Taichi compute kernels: induced velocity, vorticity,
kinetic energy, enstrophy, time integration, and target-point evaluation.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ..config.constants import (
    DEFAULT_CUTOFF_RADIUS_FACTOR,
    EPSILON,
)


def _make_compute_velocities_kernel(q_):
    """Mini-factory: creates compute_velocities_kernel capturing q_."""

    @ti.kernel
    def compute_velocities_kernel(
        positions: ti.template(),
        strengths: ti.template(),
        radii: ti.template(),
        velocities: ti.template(),
        background_velocity: ti.template(),
        num_particles: ti.i32,
    ):  # type: ignore
        """Compute self-induced velocity + background velocity for all particles."""
        N = num_particles
        U_inf = background_velocity[None]
        for i in range(N):
            vel = ti.Vector([0.0, 0.0, 0.0])
            pos_i = positions[i]
            radii_i = radii[i]

            for j in range(N):
                pos_j = positions[j]
                strength_j = strengths[j]
                r_ij = pos_i - pos_j
                r_sq = r_ij.dot(r_ij)
                r_mag = ti.sqrt(r_sq)

                if r_mag > EPSILON:
                    sigma = 0.5 * (radii_i + radii[j])
                    r_sigma = r_mag / sigma
                    vel += q_(r_sigma) * (r_ij.cross(strength_j)) / (r_sq * r_mag)

            velocities[i] = -vel + U_inf

    return compute_velocities_kernel


def _make_compute_vorticities_kernel(zeta_):
    """Mini-factory: creates compute_vorticities_kernel capturing zeta_."""

    @ti.kernel
    def compute_vorticities_kernel(
        positions: ti.template(),
        strengths: ti.template(),
        radii: ti.template(),
        vorticities: ti.template(),
        num_particles: ti.i32,
    ):  # type: ignore
        N = num_particles
        for i in range(N):
            vort = ti.Vector([0.0, 0.0, 0.0])
            pos_i = positions[i]
            radii_i = radii[i]

            for j in range(N):
                pos_j = positions[j]
                strength_j = strengths[j]
                r_ij = pos_i - pos_j
                r_mag = ti.sqrt(r_ij.dot(r_ij))
                sigma = 0.5 * (radii_i + radii[j])
                r_sigma = r_mag / sigma
                if r_sigma < DEFAULT_CUTOFF_RADIUS_FACTOR:
                    # Unlike velocity, regularised vorticity has a finite,
                    # physically essential self contribution at r = 0.
                    vort += zeta_(r_sigma) / (sigma * sigma * sigma) * strength_j

            vorticities[i] = vort

    return compute_vorticities_kernel


def _make_kinetic_energy_kernel(g_):
    """Mini-factory: creates compute_kinetic_energy_kernel capturing g_."""

    @ti.kernel
    def compute_kinetic_energy_kernel(
        positions: ti.template(),
        strengths: ti.template(),
        radii: ti.template(),
        kinetic_energy: ti.template(),
        num_particles: ti.i32,
    ):  # type: ignore
        N = num_particles
        for i in range(N):
            energy_sum = ti.cast(0.0, ti.f32)
            str_i = strengths[i]
            pos_i = positions[i]

            for j in range(N):
                pos_j = positions[j]
                str_j = strengths[j]
                sigma = 0.5 * (radii[i] + radii[j])
                r_ij = pos_i - pos_j
                r_mag = ti.sqrt(r_ij.dot(r_ij))
                r_sigma = r_mag / sigma
                if r_sigma < DEFAULT_CUTOFF_RADIUS_FACTOR:
                    # E = ½∫ω·ψ convolves two blobs, so the pair width is
                    # σ_e = sqrt(σ_i²+σ_j²), not the pair mean σ (see
                    # evaluation.py::compute_flow_integrals_kernel).
                    # g(0) is finite: retain the continuum blob self energy.
                    sigma_e = ti.sqrt(radii[i] * radii[i] + radii[j] * radii[j])
                    energy_sum += g_(r_mag / sigma_e) / sigma_e * str_j.dot(str_i) * 0.5

            kinetic_energy[i] = energy_sum

    return compute_kinetic_energy_kernel


def _make_update_position_euler_kernel():
    """Mini-factory: creates update_position_euler_kernel (no captures)."""

    @ti.kernel
    def update_position_euler_kernel(
        positions: ti.template(), velocities: ti.template(), dt: ti.f32, num_particles: ti.i32
    ):  # type: ignore
        N = num_particles
        for i in range(N):
            positions[i] += dt * velocities[i]

    return update_position_euler_kernel


def _make_step_euler_forward_strengths_kernel():
    """Mini-factory: creates step_euler_forward_strengths_kernel (no captures)."""

    @ti.kernel
    def step_euler_forward_strengths_kernel(
        str_in: ti.template(),
        dstr_dt: ti.template(),
        str_out: ti.template(),
        dt: ti.f32,
        num_particles: ti.i32,
        growth_limit: ti.f32,
    ):
        """Euler step for strength update with per-particle clipping to avoid runaway growth."""
        for i in range(num_particles):
            delta = dt * dstr_dt[i]
            gamma = str_in[i]
            mag = ti.sqrt(gamma.dot(gamma))
            max_allowed = ti.max(growth_limit * mag, 1e-12)
            dnorm = ti.sqrt(delta.dot(delta))
            dnorm_safe = ti.max(dnorm, 1e-12)
            scale = ti.min(max_allowed / dnorm_safe, 1.0)
            str_out[i] = gamma + delta * scale

    return step_euler_forward_strengths_kernel


def _make_target_velocity_kernel(q_):
    """Mini-factory: creates compute_target_velocity_kernel capturing q_."""

    @ti.kernel
    def compute_target_velocity_kernel(
        target_positions: ti.template(),
        positions: ti.template(),
        strengths: ti.template(),
        radii: ti.template(),
        target_velocities: ti.template(),
        background_velocity: ti.template(),
        num_targets: ti.i32,
        num_particles: ti.i32,
    ):  # type: ignore
        """Compute self-induced velocity + background velocity at target positions."""
        M = num_targets
        N = num_particles
        U_inf = background_velocity[None]
        for i in range(M):
            vel = ti.Vector([0.0, 0.0, 0.0])
            target_pos = target_positions[i]
            for j in range(N):
                pos_j = positions[j]
                strength_j = strengths[j]
                r_ij = target_pos - pos_j
                r_sq = r_ij.dot(r_ij)
                r_mag = ti.sqrt(r_sq)
                if r_mag > EPSILON:
                    sigma = radii[j]
                    r_sigma = r_mag / sigma
                    vel += q_(r_sigma) * (r_ij.cross(strength_j)) / (r_sq * r_mag)
            target_velocities[i] = -vel + U_inf

    return compute_target_velocity_kernel


def _make_target_source_velocity_kernel(q_):
    """Mini-factory: creates compute_target_source_velocity_kernel capturing q_."""

    @ti.kernel
    def compute_target_source_velocity_kernel(
        target_positions: ti.template(),
        source_positions: ti.template(),
        source_strengths: ti.template(),
        source_radii: ti.template(),
        target_velocities: ti.template(),
        num_targets: ti.i32,
        num_sources: ti.i32,
    ):  # type: ignore
        """Compute velocity induced by source particles at target positions."""
        M = num_targets
        N = num_sources
        for i in range(M):
            vel = ti.Vector([0.0, 0.0, 0.0])
            target_pos = target_positions[i]
            for j in range(N):
                pos_j = source_positions[j]
                strength_j = source_strengths[j]
                radii_j = source_radii[j]
                r_ij = target_pos - pos_j
                r_sq = r_ij.dot(r_ij)
                r_mag = ti.sqrt(r_sq)
                if r_mag > EPSILON:
                    sigma = radii_j
                    r_sigma = r_mag / sigma
                    vel += q_(r_sigma) * strength_j * r_ij / (r_sq * r_mag)
            target_velocities[i] += vel

    return compute_target_source_velocity_kernel


def _make_target_vorticity_kernel(zeta_):
    """Mini-factory: creates compute_target_vorticity_kernel capturing zeta_."""

    @ti.kernel
    def compute_target_vorticity_kernel(
        target_positions: ti.template(),
        positions: ti.template(),
        strengths: ti.template(),
        radii: ti.template(),
        target_vorticities: ti.template(),
        num_targets: ti.i32,
        num_particles: ti.i32,
    ):  # type: ignore
        M = num_targets
        N = num_particles
        for i in range(M):
            vort = ti.Vector([0.0, 0.0, 0.0])
            target_pos = target_positions[i]
            for j in range(N):
                pos_j = positions[j]
                strength_j = strengths[j]
                r_ij = target_pos - pos_j
                r_mag = ti.sqrt(r_ij.dot(r_ij))
                sigma = radii[j]
                r_sigma = r_mag / sigma
                # A target may coincide with a source: zeta(0) is finite.
                vort += zeta_(r_sigma) / (sigma * sigma * sigma) * strength_j
            target_vorticities[i] = vort

    return compute_target_vorticity_kernel


def _make_csm_kernel(diffusivity_constant_):
    """Mini-factory: creates update_radius_csm_kernel capturing diffusivity_constant_."""

    @ti.kernel
    def update_radius_csm_kernel(
        radii: ti.template(), viscosities_eff: ti.template(), dt: ti.f32, num_particles: ti.i32
    ):  # type: ignore
        N = num_particles
        for i in range(N):
            nu_eff = viscosities_eff[i]
            current_radius = radii[i]
            diffusion_term = diffusivity_constant_() * nu_eff * dt
            new_rad_sq = current_radius * current_radius + diffusion_term

            # Guard against invalid radius
            radii[i] = ti.sqrt(new_rad_sq)

    return update_radius_csm_kernel


def _make_rwm_kernel():
    """Mini-factory: creates update_position_rwm_kernel (no captures)."""

    @ti.kernel
    def update_position_rwm_kernel(
        positions: ti.template(), viscosities_eff: ti.template(), dt: ti.f32, num_particles: ti.i32
    ):  # type: ignore
        """Apply the Random Walk Method (RWM) with Brownian displacements.

        Uses a manual Box-Muller transform instead of ti.randn() to avoid
        NaN values produced by ti.randn on the Metal backend (macOS).
        """
        two_pi: ti.f32 = 6.28318530717959
        N = num_particles
        for i in range(N):
            nu_eff = viscosities_eff[i]
            displacement_factor = ti.sqrt(2.0 * nu_eff * dt)
            # Box-Muller transform: two uniform samples -> two standard normals
            u1 = ti.max(ti.random(ti.f32), 1e-7)
            u2 = ti.random(ti.f32)
            u3 = ti.max(ti.random(ti.f32), 1e-7)
            u4 = ti.random(ti.f32)
            mag1 = ti.sqrt(-2.0 * ti.log(u1))
            mag2 = ti.sqrt(-2.0 * ti.log(u3))
            dx = displacement_factor * mag1 * ti.cos(two_pi * u2)
            dy = displacement_factor * mag1 * ti.sin(two_pi * u2)
            dz = displacement_factor * mag2 * ti.cos(two_pi * u4)
            positions[i] += ti.Vector([dx, dy, dz])

    return update_position_rwm_kernel


@ti.func
def _stretching_contribution(
    str_i: ti.math.vec3,
    str_j: ti.math.vec3,
    r_ij: ti.math.vec3,
    q_val: ti.f32,
    zeta_val: ti.f32,
    sigma: ti.f32,
    r_sigma: ti.f32,
    mode: ti.i32,
) -> ti.math.vec3:
    """Compute stretching contribution from particle j to particle i.

    Uses numerically stable formulation with protected denominators.
    """
    dstr = ti.Vector([0.0, 0.0, 0.0])

    # The regularized kernels are finite as r/sigma -> 0.  Only protect the
    # exact self denominator; do not mask near-core particle interactions.
    r_sigma_safe = ti.max(r_sigma, EPSILON)
    sigma_safe = ti.max(sigma, EPSILON)

    # Compute denominators with protection
    denom_coeff1 = sigma_safe**3 * r_sigma_safe**3
    denom_coeff2 = sigma_safe**5 * r_sigma_safe**5

    coeff2 = (3.0 * q_val - zeta_val * r_sigma_safe * r_sigma_safe * r_sigma_safe) / denom_coeff2
    r_cross_Gj = r_ij.cross(str_j)
    Gi_dot_r = str_i.dot(r_ij)
    Gi_dot_rCrossGj = str_i.dot(r_cross_Gj)

    if mode == 0:
        coeff1 = -q_val / denom_coeff1
        dstr = coeff1 * str_i.cross(str_j) + coeff2 * Gi_dot_r * r_cross_Gj

    elif mode == 1:
        coeff1 = q_val / denom_coeff1
        dstr = coeff1 * str_i.cross(str_j) + coeff2 * Gi_dot_rCrossGj * r_ij

    elif mode == 2:
        dstr = 0.5 * coeff2 * (Gi_dot_r * r_cross_Gj + Gi_dot_rCrossGj * r_ij)

    else:
        # Pairwise conservative transposed form.  The exchange is
        # antisymmetric (so ΣΓ is invariant) and its tangential term satisfies
        #
        #   r_ij × dΓ_ij = -(u_ij × Γ_i + u_ji × Γ_j),
        #
        # so the complete coupled x/Γ system conserves linear impulse without
        # an a-posteriori projection.  The radial transposed term does not
        # affect this identity.
        coeff1 = -q_val / denom_coeff1
        dstr = coeff1 * str_i.cross(str_j) + coeff2 * Gi_dot_rCrossGj * r_ij

    return dstr


def _make_stretching_rate_kernel(q_, zeta_):
    """Mini-factory: creates compute_stretching_rate_kernel capturing q_ and zeta_."""

    @ti.kernel
    def compute_stretching_rate_kernel(
        positions: ti.template(),
        strengths: ti.template(),
        radii: ti.template(),
        dstr_dt_out: ti.template(),
        mode: ti.i32,
        num_particles: ti.i32,
    ):  # type: ignore
        """Conservative direct pair-wise vortex stretching."""
        N = num_particles
        for i in range(N):
            str_i = strengths[i]
            pos_i = positions[i]
            radii_i = radii[i]
            dstr_dt = ti.Vector([0.0, 0.0, 0.0])

            for j in range(N):
                pos_j = positions[j]
                str_j = strengths[j]
                r_ij = pos_i - pos_j
                r_mag = ti.sqrt(r_ij.dot(r_ij))

                if r_mag > EPSILON:
                    sigma = 0.5 * (radii_i + radii[j])
                    r_sigma = r_mag / sigma

                    q_val = q_(r_sigma)
                    zeta_val = zeta_(r_sigma)
                    dstr_dt += _stretching_contribution(
                        str_i, str_j, r_ij, q_val, zeta_val, sigma, r_sigma, mode
                    )

            dstr_dt_out[i] = dstr_dt

    return compute_stretching_rate_kernel


def _create_basic_kernels(kernel_functions):
    """Create basic velocity and vorticity computation kernels."""
    q_ = kernel_functions["q_"]
    zeta_ = kernel_functions["zeta_"]

    @ti.kernel
    def add_background_velocity_kernel(
        velocities: ti.template(), background_velocity: ti.template(), num_particles: int
    ):  # type: ignore
        """Add background velocity to all active particles."""
        for i in range(num_particles):
            velocities[i][0] += background_velocity[0]
            velocities[i][1] += background_velocity[1]
            velocities[i][2] += background_velocity[2]

    return {
        "compute_velocities_kernel": _make_compute_velocities_kernel(q_),
        "add_background_velocity_kernel": add_background_velocity_kernel,
        "compute_vorticities_kernel": _make_compute_vorticities_kernel(zeta_),
    }


def _create_gradient_kernels(kernel_functions):
    """Create velocity gradient and strain tensor computation kernels."""
    q_ = kernel_functions["q_"]
    zeta_ = kernel_functions["zeta_"]

    @ti.func
    def skew(v):
        return ti.Matrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    @ti.kernel
    def compute_velocity_gradients_kernel(
        positions: ti.template(),
        strengths: ti.template(),
        radii: ti.template(),
        gradU: ti.template(),
        Sij: ti.template(),
        num_particles: ti.i32,
    ):  # type: ignore
        N = num_particles
        for i in range(N):
            pos_i = positions[i]
            radii_i = radii[i]
            gradu = ti.Matrix.zero(ti.f32, 3, 3)

            for j in range(N):
                pos_j = positions[j]
                str_j = strengths[j]
                r_ij = pos_i - pos_j
                r_sq = r_ij.dot(r_ij)
                r_mag = ti.sqrt(r_sq)

                if r_mag > EPSILON:
                    sigma = 0.5 * (radii_i + radii[j])
                    r_sigma = r_mag / sigma

                    if r_sigma < DEFAULT_CUTOFF_RADIUS_FACTOR:
                        q_val = q_(r_sigma)
                        zeta_val = zeta_(r_sigma) / (sigma * sigma * sigma)
                        r_cb = r_sq * r_mag  # r³ = r² · r

                        term1 = q_val / r_cb
                        term2 = 3.0 * q_val / (r_cb * r_sq) - zeta_val / r_sq

                        gradu += term1 * skew(str_j) + term2 * (
                            (r_ij.cross(str_j)).outer_product(r_ij)
                        )

            gradU[i] = gradu

            # Symmetric strain tensor for Mixed Scheme
            Sij[i] = 0.5 * (gradu + gradu.transpose())

    @ti.kernel
    def compute_velocity_and_gradient_kernel(
        positions: ti.template(),
        strengths: ti.template(),
        radii: ti.template(),
        velocities: ti.template(),
        gradU: ti.template(),
        Sij: ti.template(),
        background_velocity: ti.template(),
        num_particles: ti.i32,
    ):  # type: ignore
        """Fused direct evaluation: u, ∇u and S in a single j-loop.

        The solver needs both u (advection) and ∇u (stretching) each RK stage;
        sharing the one j-loop reuses r_ij / r_mag / sigma / q per pair instead of
        recomputing them in a second O(N²) sweep.  Velocity and near-core
        gradient evaluation both use the regularized kernel directly, while
        the far gradient cutoff matches the separate gradient kernel."""
        N = num_particles
        U_inf = background_velocity[None]
        for i in range(N):
            vel = ti.Vector([0.0, 0.0, 0.0])
            gradu = ti.Matrix.zero(ti.f32, 3, 3)
            pos_i = positions[i]
            radii_i = radii[i]
            for j in range(N):
                str_j = strengths[j]
                r_ij = pos_i - positions[j]
                r_sq = r_ij.dot(r_ij)
                r_mag = ti.sqrt(r_sq)
                if r_mag > EPSILON:
                    sigma = 0.5 * (radii_i + radii[j])
                    r_sigma = r_mag / sigma
                    q_val = q_(r_sigma)
                    vel += q_val * (r_ij.cross(str_j)) / (r_sq * r_mag)
                    if r_sigma < DEFAULT_CUTOFF_RADIUS_FACTOR:
                        zeta_val = zeta_(r_sigma) / (sigma * sigma * sigma)
                        r_cb = r_sq * r_mag
                        term1 = q_val / r_cb
                        term2 = 3.0 * q_val / (r_cb * r_sq) - zeta_val / r_sq
                        gradu += term1 * skew(str_j) + term2 * (
                            (r_ij.cross(str_j)).outer_product(r_ij)
                        )
            velocities[i] = -vel + U_inf
            gradU[i] = gradu
            Sij[i] = 0.5 * (gradu + gradu.transpose())

    return {
        "compute_velocity_gradients_kernel": compute_velocity_gradients_kernel,
        "compute_velocity_and_gradient_kernel": compute_velocity_and_gradient_kernel,
    }


def _create_energy_kernels(kernel_functions):
    """Create the kinetic-energy and helicity computation kernels."""
    q_ = kernel_functions["q_"]
    g_ = kernel_functions["g_"]

    @ti.kernel
    def compute_helicity_kernel(
        positions: ti.template(),
        strengths: ti.template(),
        radii: ti.template(),
        helicity: ti.template(),
        num_particles: ti.i32,
    ):  # type: ignore
        N = num_particles
        for i in range(N):
            hel = ti.cast(0.0, ti.f32)
            str_i = strengths[i]
            pos_i = positions[i]
            radii_i = radii[i]
            for j in range(N):
                pos_j = positions[j]
                str_j = strengths[j]
                sigma = 0.5 * (radii_i + radii[j])
                r_ij = pos_i - pos_j
                r_sq = r_ij.dot(r_ij)
                r_mag = ti.sqrt(r_sq)
                if r_mag > EPSILON:
                    r_sigma = r_mag / sigma
                    if r_sigma < DEFAULT_CUTOFF_RADIUS_FACTOR:
                        hel += q_(r_sigma) * r_ij.dot(str_i.cross(str_j)) / (r_sq * r_mag)
            helicity[i] = hel

    return {
        "compute_kinetic_energy_kernel": _make_kinetic_energy_kernel(g_),
        "compute_helicity_kernel": compute_helicity_kernel,
    }


def _create_position_update_kernels(kernel_functions):
    """Create time integration kernels for position and strength updates."""
    kernel_functions["q_"]

    @ti.kernel
    def step_euler_forward_kernel(
        pos_in: ti.template(),
        vel_in: ti.template(),
        pos_out: ti.template(),
        dt: ti.f32,
        num_particles: ti.i32,
    ):  # type: ignore
        """Simple advection: pos_out = pos_in + dt * vel_in"""
        for i in range(num_particles):
            pos_out[i] = pos_in[i] + dt * vel_in[i]

    @ti.kernel
    def linear_combination_kernel(
        dest: ti.template(),
        src1: ti.template(),
        src2: ti.template(),
        w1: ti.f32,
        w2: ti.f32,
        num_particles: ti.i32,
    ):  # type: ignore
        """Compute dest = w1 * src1 + w2 * src2 element-wise."""
        for i in range(num_particles):
            dest[i] = w1 * src1[i] + w2 * src2[i]

    @ti.kernel
    def step_rk2_combine_kernel(
        positions: ti.template(),
        k1: ti.template(),
        k2: ti.template(),
        dt: ti.f32,
        num_particles: ti.i32,
    ):  # type: ignore
        """RK2 Heun's method: pos += (dt/2) * (k1 + k2).

        Heun's method (explicit trapezoidal) is more stable than the midpoint
        method for particle methods near boundaries or with stiff interactions.
        """
        for i in range(num_particles):
            positions[i] += (dt / 2.0) * (k1[i] + k2[i])

    @ti.kernel
    def step_rk3_ssp_combine_kernel(
        positions: ti.template(),
        k1: ti.template(),
        k2: ti.template(),
        k3: ti.template(),
        dt: ti.f32,
        num_particles: ti.i32,
    ):  # type: ignore
        """SSP-RK3 (Strong Stability Preserving): pos += (dt/6) * (k1 + k2 + 4*k3).

        The Shu-Osher SSP-RK3 scheme has optimal stability properties for
        convection-dominated problems and preserves monotonicity (TVD property).
        """
        for i in range(num_particles):
            positions[i] += (dt / 6.0) * (k1[i] + k2[i] + 4.0 * k3[i])

    @ti.kernel
    def step_rk4_combine_kernel(
        positions: ti.template(),
        k1: ti.template(),
        k2: ti.template(),
        k3: ti.template(),
        k4: ti.template(),
        dt: ti.f32,
        num_particles: ti.i32,
    ):  # type: ignore
        """Classic RK4: pos += (dt/6) * (k1 + 2*k2 + 2*k3 + k4).

        The classic 4th-order Runge-Kutta scheme provides high accuracy
        for smooth problems. Requires 4 function evaluations per step.
        """
        for i in range(num_particles):
            positions[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])

    @ti.kernel
    def fixed_point_residual_kernel(
        a: ti.template(),
        b: ti.template(),
        scale: ti.template(),
        num_particles: ti.i32,
    ) -> ti.f32:  # type: ignore
        """Max |a-b| over particles, and max |b|, for a relative convergence test.

        Drives the implicit-midpoint fixed-point iteration.  ``scale[None]``
        receives max|b| so the caller can form a relative residual without a
        second pass.
        """
        scale[None] = 0.0
        residual = 0.0
        for i in range(num_particles):
            ti.atomic_max(residual, (a[i] - b[i]).norm())
            ti.atomic_max(scale[None], b[i].norm())
        return residual

    return {
        "update_position_euler_kernel": _make_update_position_euler_kernel(),
        "step_euler_forward_kernel": step_euler_forward_kernel,
        "step_euler_forward_strengths_kernel": _make_step_euler_forward_strengths_kernel(),
        "linear_combination_kernel": linear_combination_kernel,
        "step_rk2_combine_kernel": step_rk2_combine_kernel,
        "step_rk3_ssp_combine_kernel": step_rk3_ssp_combine_kernel,
        "step_rk4_combine_kernel": step_rk4_combine_kernel,
        "fixed_point_residual_kernel": fixed_point_residual_kernel,
    }


def _create_target_eval_kernels(kernel_functions):
    """Create kernels for evaluating flow fields at arbitrary target positions."""
    q_ = kernel_functions["q_"]
    zeta_ = kernel_functions["zeta_"]

    @ti.func
    def skew(v):
        return ti.Matrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    @ti.kernel
    def compute_target_velocity_gradient_kernel(
        target_positions: ti.template(),
        positions: ti.template(),
        strengths: ti.template(),
        radii: ti.template(),
        target_gradU: ti.template(),
        num_targets: ti.i32,
        num_particles: ti.i32,
    ):  # type: ignore
        """Compute velocity gradient tensor at arbitrary target positions."""
        M = num_targets
        N = num_particles
        for i in range(M):
            target_pos = target_positions[i]
            gradu = ti.Matrix.zero(ti.f32, 3, 3)

            for j in range(N):
                pos_j = positions[j]
                str_j = strengths[j]
                r_ij = target_pos - pos_j
                r_sq = r_ij.dot(r_ij)
                r_mag = ti.sqrt(r_sq)

                if r_mag > EPSILON:
                    sigma = radii[j]
                    r_sigma = r_mag / sigma

                    q_val = q_(r_sigma)
                    zeta_val = zeta_(r_sigma) / (sigma * sigma * sigma)
                    r_cb = r_sq * r_mag  # r³ = r² · r

                    term1 = q_val / r_cb
                    term2 = 3.0 * q_val / (r_cb * r_sq) - zeta_val / r_sq

                    gradu += term1 * skew(str_j) + term2 * ((r_ij.cross(str_j)).outer_product(r_ij))

            target_gradU[i] = gradu

    return {
        "compute_target_velocity_kernel": _make_target_velocity_kernel(q_),
        "compute_target_source_velocity_kernel": _make_target_source_velocity_kernel(q_),
        "compute_target_vorticity_kernel": _make_target_vorticity_kernel(zeta_),
        "compute_target_velocity_gradient_kernel": compute_target_velocity_gradient_kernel,
    }


def _create_diffusion_kernels(kernel_functions):
    """Create kernels for viscous diffusion (CSM, RWM)."""
    diffusivity_constant_ = kernel_functions["diffusivity_constant_"]

    return {
        "update_radius_csm_kernel": _make_csm_kernel(diffusivity_constant_),
        "update_position_rwm_kernel": _make_rwm_kernel(),
    }


def _make_gradient_contraction_kernel():
    """Mini-factory: rate dΓ/dt from an already-computed velocity gradient ∇u.

    The pairwise stretching kernel forms ∇u·Γ implicitly; given the velocity
    gradient tensor J (as computed by the treecode) the stretching rate is a
    purely local contraction — O(N) instead of O(N²).  Matches the direct
    pairwise kernel's three modes:

      mode 0 (DIRECT):     dΓ = (Γ·∇)u   = J · Γ
      mode 1 (TRANSPOSED): dΓ = (Γ·∇')u  = Jᵀ · Γ
      mode 2 (MIXED):      dΓ = ½(J + Jᵀ)·Γ = S · Γ
    """

    @ti.kernel
    def gradient_contraction_rate_kernel(
        grad: ti.template(),  # velocity gradient tensor J per particle
        strengths: ti.template(),  # Γ per particle
        dstr_dt_out: ti.template(),
        mode: ti.i32,
        num_particles: ti.i32,
    ):  # type: ignore
        for i in range(num_particles):
            J = grad[i]
            g = strengths[i]
            rate = ti.Vector([0.0, 0.0, 0.0])
            if mode == 0:
                rate = J @ g
            elif mode == 1:
                rate = J.transpose() @ g
            else:
                rate = (0.5 * (J + J.transpose())) @ g
            dstr_dt_out[i] = rate

    return gradient_contraction_rate_kernel


def _create_stretching_kernels(kernel_functions):
    """Create kernels for vortex stretching computations."""
    q_ = kernel_functions["q_"]
    zeta_ = kernel_functions["zeta_"]

    return {
        "compute_stretching_rate_kernel": _make_stretching_rate_kernel(q_, zeta_),
        "gradient_contraction_rate_kernel": _make_gradient_contraction_kernel(),
    }


def _create_utility_kernels(kernel_functions):
    """Create utility kernels for circulation centroids."""

    @ti.kernel
    def compute_circulation_centroids_kernel(
        positions: ti.template(), strengths: ti.template(), num_particles: ti.i32
    ) -> ti.types.vector(3, ti.f32):  # type: ignore
        total_strength_scalar = 0.0
        total_strength_vector = ti.Vector([0.0, 0.0, 0.0])
        weighted_position = ti.Vector([0.0, 0.0, 0.0])
        n = num_particles
        for i in range(n):
            if i < positions.shape[0] and i < strengths.shape[0]:
                strength_vec = strengths[i]
                strength_mag = strength_vec.norm()
                if strength_mag > EPSILON:
                    total_strength_scalar += strength_mag
                    total_strength_vector += strength_vec
                    pos = positions[i]
                    weighted_position += pos * strength_mag
        centroid = ti.Vector([0.0, 0.0, 0.0])
        if total_strength_scalar > EPSILON:
            centroid = weighted_position / total_strength_scalar
        return centroid

    return {
        "compute_circulation_centroids_kernel": compute_circulation_centroids_kernel,
    }


def create_kernels(kernel_functions):
    """Create all Taichi kernels for VPM physics.

    This factory function orchestrates the creation of all physics kernels
    by delegating to specialized helper functions for each category.
    """
    # Assemble kernels from modular factories
    kernels = {}
    kernels.update(_create_basic_kernels(kernel_functions))
    kernels.update(_create_gradient_kernels(kernel_functions))
    kernels.update(_create_energy_kernels(kernel_functions))
    kernels.update(_create_position_update_kernels(kernel_functions))
    kernels.update(_create_target_eval_kernels(kernel_functions))
    kernels.update(_create_diffusion_kernels(kernel_functions))
    kernels.update(_create_stretching_kernels(kernel_functions))
    kernels.update(_create_utility_kernels(kernel_functions))

    return kernels
