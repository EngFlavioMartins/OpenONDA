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
        position: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        velocity: ti.template(),
        freestream_velocity: ti.template(),
        n_particles_total: ti.i32,
    ):  # type: ignore
        """Compute self-induced velocity + background velocity for all particles."""
        N = n_particles_total
        freestream_vec = freestream_velocity[None]
        for i in range(N):
            vel = position[i] * 0.0
            pos_i = position[i]
            radii_i = core_radius[i]

            for j in range(N):
                pos_j = position[j]
                strength_j = vortex_strength[j]
                r_ij = pos_i - pos_j
                r_sq = r_ij.dot(r_ij)
                r_mag = ti.sqrt(r_sq)

                if r_mag > EPSILON:
                    sigma = 0.5 * (radii_i + core_radius[j])
                    r_sigma = r_mag / sigma
                    vel += q_(r_sigma) * (r_ij.cross(strength_j)) / (r_sq * r_mag)

            velocity[i] = -vel + freestream_vec

    return compute_velocities_kernel


def _make_compute_vorticities_kernel(zeta_):
    """Mini-factory: creates compute_vorticities_kernel capturing zeta_."""

    @ti.kernel
    def compute_vorticities_kernel(
        position: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        vorticity: ti.template(),
        n_particles_total: ti.i32,
    ):  # type: ignore
        N = n_particles_total
        for i in range(N):
            vort = vortex_strength[i] * 0.0
            pos_i = position[i]
            radii_i = core_radius[i]

            for j in range(N):
                pos_j = position[j]
                strength_j = vortex_strength[j]
                r_ij = pos_i - pos_j
                r_mag = ti.sqrt(r_ij.dot(r_ij))
                sigma = 0.5 * (radii_i + core_radius[j])
                r_sigma = r_mag / sigma
                if r_sigma < DEFAULT_CUTOFF_RADIUS_FACTOR:
                    # Unlike velocity, regularised vorticity has a finite,
                    # physically essential self contribution at r = 0.
                    vort += zeta_(r_sigma) / (sigma * sigma * sigma) * strength_j

            vorticity[i] = vort

    return compute_vorticities_kernel


def _make_kinetic_energy_kernel(g_):
    """Mini-factory: creates compute_kinetic_energy_kernel capturing g_."""

    @ti.kernel
    def compute_kinetic_energy_kernel(
        position: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        particle_kinetic_energy: ti.template(),
        n_particles_total: ti.i32,
    ):  # type: ignore
        N = n_particles_total
        for i in range(N):
            str_i = vortex_strength[i]
            energy_sum = str_i.dot(str_i) * 0.0
            pos_i = position[i]

            for j in range(N):
                pos_j = position[j]
                str_j = vortex_strength[j]
                sigma = 0.5 * (core_radius[i] + core_radius[j])
                r_ij = pos_i - pos_j
                r_mag = ti.sqrt(r_ij.dot(r_ij))
                r_sigma = r_mag / sigma
                if r_sigma < DEFAULT_CUTOFF_RADIUS_FACTOR:
                    # E = ½∫ω·ψ convolves two blobs, so the pair width is
                    # σ_e = sqrt(σ_i²+σ_j²), not the pair mean σ (see
                    # evaluation.py::compute_flow_integrals_kernel).
                    # g(0) is finite: retain the continuum blob self energy.
                    sigma_e = ti.sqrt(
                        core_radius[i] * core_radius[i] + core_radius[j] * core_radius[j]
                    )
                    energy_sum += g_(r_mag / sigma_e) / sigma_e * str_j.dot(str_i) * 0.5

            particle_kinetic_energy[i] = energy_sum

    return compute_kinetic_energy_kernel


def _make_update_position_euler_kernel():
    """Mini-factory: creates update_position_euler_kernel (no captures)."""

    @ti.kernel
    def update_position_euler_kernel(
        position: ti.template(),
        velocity: ti.template(),
        time_step_size: ti.template(),
        n_particles_total: ti.i32,
    ):  # type: ignore
        N = n_particles_total
        for i in range(N):
            position[i] += time_step_size * velocity[i]

    return update_position_euler_kernel


def _make_step_euler_forward_strengths_kernel():
    """Mini-factory: creates step_euler_forward_strengths_kernel (no captures)."""

    @ti.kernel
    def step_euler_forward_strengths_kernel(
        str_in: ti.template(),
        dstr_dt: ti.template(),
        str_out: ti.template(),
        time_step_size: ti.template(),
        n_particles_total: ti.i32,
        growth_limit: ti.template(),
    ):
        """Euler step for strength update with per-particle clipping to avoid runaway growth."""
        for i in range(n_particles_total):
            delta = time_step_size * dstr_dt[i]
            vortex_strength = str_in[i]
            mag = ti.sqrt(vortex_strength.dot(vortex_strength))
            max_allowed = ti.max(growth_limit * mag, 1e-12)
            dnorm = ti.sqrt(delta.dot(delta))
            dnorm_safe = ti.max(dnorm, 1e-12)
            scale = ti.min(max_allowed / dnorm_safe, 1.0)
            str_out[i] = vortex_strength + delta * scale

    return step_euler_forward_strengths_kernel


def _make_target_velocity_kernel(q_):
    """Mini-factory: creates compute_target_velocity_kernel capturing q_."""

    @ti.kernel
    def compute_target_velocity_kernel(
        target_position: ti.template(),
        position: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        target_velocity: ti.template(),
        freestream_velocity: ti.template(),
        n_targets: ti.i32,
        n_particles_total: ti.i32,
    ):  # type: ignore
        """Compute self-induced velocity + background velocity at target position."""
        M = n_targets
        N = n_particles_total
        freestream_vec = freestream_velocity[None]
        for i in range(M):
            target_pos = target_position[i]
            vel = target_pos * 0.0
            for j in range(N):
                pos_j = position[j]
                strength_j = vortex_strength[j]
                r_ij = target_pos - pos_j
                r_sq = r_ij.dot(r_ij)
                r_mag = ti.sqrt(r_sq)
                if r_mag > EPSILON:
                    sigma = core_radius[j]
                    r_sigma = r_mag / sigma
                    vel += q_(r_sigma) * (r_ij.cross(strength_j)) / (r_sq * r_mag)
            target_velocity[i] = -vel + freestream_vec

    return compute_target_velocity_kernel


def _make_target_source_velocity_kernel(q_):
    """Mini-factory: creates compute_target_source_velocity_kernel capturing q_."""

    @ti.kernel
    def compute_target_source_velocity_kernel(
        target_position: ti.template(),
        source_position: ti.template(),
        source_strength: ti.template(),
        source_core_radius: ti.template(),
        target_velocity: ti.template(),
        n_targets: ti.i32,
        n_sources: ti.i32,
    ):  # type: ignore
        """Compute velocity induced by source particles at target position."""
        M = n_targets
        N = n_sources
        for i in range(M):
            target_pos = target_position[i]
            vel = target_pos * 0.0
            for j in range(N):
                pos_j = source_position[j]
                strength_j = source_strength[j]
                radii_j = source_core_radius[j]
                r_ij = target_pos - pos_j
                r_sq = r_ij.dot(r_ij)
                r_mag = ti.sqrt(r_sq)
                if r_mag > EPSILON:
                    sigma = radii_j
                    r_sigma = r_mag / sigma
                    vel += q_(r_sigma) * strength_j * r_ij / (r_sq * r_mag)
            target_velocity[i] += vel

    return compute_target_source_velocity_kernel


def _make_target_vorticity_kernel(zeta_):
    """Mini-factory: creates compute_target_vorticity_kernel capturing zeta_."""

    @ti.kernel
    def compute_target_vorticity_kernel(
        target_position: ti.template(),
        position: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        target_vorticity: ti.template(),
        n_targets: ti.i32,
        n_particles_total: ti.i32,
    ):  # type: ignore
        M = n_targets
        N = n_particles_total
        for i in range(M):
            target_pos = target_position[i]
            vort = target_pos * 0.0
            for j in range(N):
                pos_j = position[j]
                strength_j = vortex_strength[j]
                r_ij = target_pos - pos_j
                r_mag = ti.sqrt(r_ij.dot(r_ij))
                sigma = core_radius[j]
                r_sigma = r_mag / sigma
                # A target may coincide with a source: zeta(0) is finite.
                vort += zeta_(r_sigma) / (sigma * sigma * sigma) * strength_j
            target_vorticity[i] = vort

    return compute_target_vorticity_kernel


def _make_csm_kernel(diffusivity_constant_):
    """Mini-factory: creates update_radius_csm_kernel capturing diffusivity_constant_."""

    @ti.kernel
    def update_radius_csm_kernel(
        core_radius: ti.template(),
        viscosities_eff: ti.template(),
        time_step_size: ti.template(),
        n_particles_total: ti.i32,
    ):  # type: ignore
        N = n_particles_total
        for i in range(N):
            effective_viscosity = viscosities_eff[i]
            current_radius = core_radius[i]
            diffusion_term = diffusivity_constant_() * effective_viscosity * time_step_size
            new_rad_sq = current_radius * current_radius + diffusion_term

            # Guard against invalid radius
            core_radius[i] = ti.sqrt(new_rad_sq)

    return update_radius_csm_kernel


def _make_rwm_kernel():
    """Create the counter-based Random Walk Method kernel.

    The random variates are a pure function of the declared seed, accepted
    step, particle index, and draw index.  They therefore do not depend on an
    opaque backend RNG cursor and survive fresh processes and restarts exactly.
    """

    @ti.func
    def hash_u32(value: ti.u32) -> ti.u32:
        value ^= value >> 16
        value *= ti.u32(0x7FEB352D)
        value ^= value >> 15
        value *= ti.u32(0x846CA68B)
        value ^= value >> 16
        return value

    @ti.func
    def uniform_01(random_seed: ti.i32, accepted_step: ti.i32, particle: ti.i32, draw: ti.i32):
        key = ti.cast(random_seed, ti.u32)
        key ^= ti.cast(accepted_step, ti.u32) * ti.u32(0x9E3779B9)
        key ^= ti.cast(particle + 1, ti.u32) * ti.u32(0x85EBCA6B)
        key ^= ti.cast(draw + 1, ti.u32) * ti.u32(0xC2B2AE35)
        # Midpoint conversion avoids exact zero and one, which keeps the
        # Box--Muller logarithm finite on every backend.
        return (ti.cast(hash_u32(key), ti.f32) + 0.5) * ti.f32(1.0 / 4294967296.0)

    @ti.kernel
    def update_position_rwm_kernel(
        position: ti.template(),
        viscosities_eff: ti.template(),
        time_step_size: ti.template(),
        n_particles_total: ti.i32,
        random_seed: ti.i32,
        accepted_step: ti.i32,
    ):  # type: ignore
        """Apply the Random Walk Method (RWM) with Brownian displacements.

        Uses a stateless integer hash plus a manual Box--Muller transform.  The
        result is reproducible across repeated calls, fresh processes, and
        save/restart continuation for a fixed backend and precision.
        """
        two_pi: ti.f32 = 6.28318530717959
        N = n_particles_total
        for i in range(N):
            effective_viscosity = viscosities_eff[i]
            displacement_factor = ti.sqrt(2.0 * effective_viscosity * time_step_size)
            # Box-Muller transform: two uniform samples -> two standard normals
            u1 = uniform_01(random_seed, accepted_step, i, 0)
            u2 = uniform_01(random_seed, accepted_step, i, 1)
            u3 = uniform_01(random_seed, accepted_step, i, 2)
            u4 = uniform_01(random_seed, accepted_step, i, 3)
            mag1 = ti.sqrt(-2.0 * ti.log(u1))
            mag2 = ti.sqrt(-2.0 * ti.log(u3))
            dx = displacement_factor * mag1 * ti.cos(two_pi * u2)
            dy = displacement_factor * mag1 * ti.sin(two_pi * u2)
            dz = displacement_factor * mag2 * ti.cos(two_pi * u4)
            position[i] += ti.Vector([dx, dy, dz])

    return update_position_rwm_kernel


@ti.func
def _stretching_contribution(
    str_i,
    str_j,
    r_ij,
    q_val,
    zeta_val,
    sigma,
    r_sigma,
    mode,
):
    """Compute stretching contribution from particle j to particle i.

    Uses numerically stable formulation with protected denominators.
    """
    dstr = str_i * 0.0

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

    else:
        # Symmetric direct/transposed formulation (MIXED).
        dstr = 0.5 * coeff2 * (Gi_dot_r * r_cross_Gj + Gi_dot_rCrossGj * r_ij)

    return dstr


def _make_stretching_rate_kernel(q_, zeta_):
    """Mini-factory: creates compute_stretching_rate_kernel capturing q_ and zeta_."""

    @ti.kernel
    def compute_stretching_rate_kernel(
        position: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        dstr_dt_out: ti.template(),
        mode: ti.i32,
        n_particles_total: ti.i32,
    ):  # type: ignore
        """Direct pair-wise vortex stretching."""
        N = n_particles_total
        for i in range(N):
            str_i = vortex_strength[i]
            pos_i = position[i]
            radii_i = core_radius[i]
            dstr_dt = str_i * 0.0

            for j in range(N):
                pos_j = position[j]
                str_j = vortex_strength[j]
                r_ij = pos_i - pos_j
                r_mag = ti.sqrt(r_ij.dot(r_ij))

                if r_mag > EPSILON:
                    sigma = 0.5 * (radii_i + core_radius[j])
                    r_sigma = r_mag / sigma

                    q_val = q_(r_sigma)
                    zeta_val = zeta_(r_sigma)
                    dstr_dt += _stretching_contribution(
                        str_i, str_j, r_ij, q_val, zeta_val, sigma, r_sigma, mode
                    )

            dstr_dt_out[i] = dstr_dt

    return compute_stretching_rate_kernel


def _make_stretching_rate_batch_kernel(q_, zeta_):
    """Create the bounded-dispatch form of the direct stretching kernel."""

    @ti.kernel
    def compute_stretching_rate_batch_kernel(
        position: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        dstr_dt_out: ti.template(),
        mode: ti.i32,
        start_target: ti.i32,
        target_count: ti.i32,
        n_particles_total: ti.i32,
    ):  # type: ignore
        """Direct pair-wise stretching for a bounded target interval."""
        N = n_particles_total
        for local_target in range(target_count):
            i = start_target + local_target
            str_i = vortex_strength[i]
            pos_i = position[i]
            radii_i = core_radius[i]
            dstr_dt = str_i * 0.0

            for j in range(N):
                pos_j = position[j]
                str_j = vortex_strength[j]
                r_ij = pos_i - pos_j
                r_mag = ti.sqrt(r_ij.dot(r_ij))

                if r_mag > EPSILON:
                    sigma = 0.5 * (radii_i + core_radius[j])
                    r_sigma = r_mag / sigma

                    q_val = q_(r_sigma)
                    zeta_val = zeta_(r_sigma)
                    dstr_dt += _stretching_contribution(
                        str_i, str_j, r_ij, q_val, zeta_val, sigma, r_sigma, mode
                    )

            dstr_dt_out[i] = dstr_dt

    return compute_stretching_rate_batch_kernel


def _create_basic_kernels(kernel_functions):
    """Create basic velocity and vorticity computation kernels."""
    q_ = kernel_functions["q_"]
    zeta_ = kernel_functions["zeta_"]

    @ti.kernel
    def add_freestream_velocity_kernel(
        velocity: ti.template(), freestream_velocity: ti.template(), n_particles_total: int
    ):  # type: ignore
        """Add background velocity to all active particles."""
        for i in range(n_particles_total):
            velocity[i][0] += freestream_velocity[0]
            velocity[i][1] += freestream_velocity[1]
            velocity[i][2] += freestream_velocity[2]

    return {
        "compute_velocities_kernel": _make_compute_velocities_kernel(q_),
        "add_freestream_velocity_kernel": add_freestream_velocity_kernel,
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
        position: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        velocity_gradient: ti.template(),
        strain_rate: ti.template(),
        n_particles_total: ti.i32,
    ):  # type: ignore
        N = n_particles_total
        for i in range(N):
            pos_i = position[i]
            radii_i = core_radius[i]
            gradu = pos_i.outer_product(pos_i) * 0.0

            for j in range(N):
                pos_j = position[j]
                str_j = vortex_strength[j]
                r_ij = pos_i - pos_j
                r_sq = r_ij.dot(r_ij)
                r_mag = ti.sqrt(r_sq)

                if r_mag > EPSILON:
                    sigma = 0.5 * (radii_i + core_radius[j])
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

            velocity_gradient[i] = gradu

            # Symmetric strain tensor for Mixed Scheme
            strain_rate[i] = 0.5 * (gradu + gradu.transpose())

    @ti.kernel
    def compute_velocity_and_gradient_kernel(
        position: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        velocity: ti.template(),
        velocity_gradient: ti.template(),
        strain_rate: ti.template(),
        freestream_velocity: ti.template(),
        n_particles_total: ti.i32,
    ):  # type: ignore
        """Fused direct evaluation: u, ∇u and S in a single j-loop.

        The solver needs both u (advection) and ∇u (stretching) each RK stage;
        sharing the one j-loop reuses r_ij / r_mag / sigma / q per pair instead of
        recomputing them in a second O(N²) sweep.  Velocity and near-core
        gradient evaluation both use the regularized kernel directly, while
        the far gradient cutoff matches the separate gradient kernel."""
        N = n_particles_total
        freestream_vec = freestream_velocity[None]
        for i in range(N):
            pos_i = position[i]
            vel = pos_i * 0.0
            gradu = pos_i.outer_product(pos_i) * 0.0
            radii_i = core_radius[i]
            for j in range(N):
                str_j = vortex_strength[j]
                r_ij = pos_i - position[j]
                r_sq = r_ij.dot(r_ij)
                r_mag = ti.sqrt(r_sq)
                if r_mag > EPSILON:
                    sigma = 0.5 * (radii_i + core_radius[j])
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
            velocity[i] = -vel + freestream_vec
            velocity_gradient[i] = gradu
            strain_rate[i] = 0.5 * (gradu + gradu.transpose())

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
        position: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        particle_helicity: ti.template(),
        n_particles_total: ti.i32,
    ):  # type: ignore
        N = n_particles_total
        for i in range(N):
            str_i = vortex_strength[i]
            hel = str_i.dot(str_i) * 0.0
            pos_i = position[i]
            radii_i = core_radius[i]
            for j in range(N):
                pos_j = position[j]
                str_j = vortex_strength[j]
                sigma = 0.5 * (radii_i + core_radius[j])
                r_ij = pos_i - pos_j
                r_sq = r_ij.dot(r_ij)
                r_mag = ti.sqrt(r_sq)
                if r_mag > EPSILON:
                    r_sigma = r_mag / sigma
                    if r_sigma < DEFAULT_CUTOFF_RADIUS_FACTOR:
                        hel += q_(r_sigma) * r_ij.dot(str_i.cross(str_j)) / (r_sq * r_mag)
            particle_helicity[i] = hel

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
        time_step_size: ti.template(),
        n_particles_total: ti.i32,
    ):  # type: ignore
        """Simple advection: pos_out = pos_in + dt * vel_in"""
        for i in range(n_particles_total):
            pos_out[i] = pos_in[i] + time_step_size * vel_in[i]

    @ti.kernel
    def linear_combination_kernel(
        dest: ti.template(),
        src1: ti.template(),
        src2: ti.template(),
        w1: ti.template(),
        w2: ti.template(),
        n_particles_total: ti.i32,
    ):  # type: ignore
        """Compute dest = w1 * src1 + w2 * src2 element-wise."""
        for i in range(n_particles_total):
            dest[i] = w1 * src1[i] + w2 * src2[i]

    @ti.kernel
    def step_rk2_combine_kernel(
        position: ti.template(),
        k1: ti.template(),
        k2: ti.template(),
        time_step_size: ti.template(),
        n_particles_total: ti.i32,
    ):  # type: ignore
        """RK2 Heun's method: pos += (dt/2) * (k1 + k2).

        Heun's method (explicit trapezoidal) is more stable than the midpoint
        method for particle methods near boundaries or with stiff interactions.
        """
        for i in range(n_particles_total):
            position[i] += (time_step_size / 2.0) * (k1[i] + k2[i])

    @ti.kernel
    def step_rk3_ssp_combine_kernel(
        position: ti.template(),
        k1: ti.template(),
        k2: ti.template(),
        k3: ti.template(),
        time_step_size: ti.template(),
        n_particles_total: ti.i32,
    ):  # type: ignore
        """SSP-RK3 (Strong Stability Preserving): pos += (dt/6) * (k1 + k2 + 4*k3).

        The Shu-Osher SSP-RK3 scheme has optimal stability properties for
        convection-dominated problems and preserves monotonicity (TVD property).
        """
        for i in range(n_particles_total):
            position[i] += (time_step_size / 6.0) * (k1[i] + k2[i] + 4.0 * k3[i])

    @ti.kernel
    def step_rk4_combine_kernel(
        position: ti.template(),
        k1: ti.template(),
        k2: ti.template(),
        k3: ti.template(),
        k4: ti.template(),
        time_step_size: ti.template(),
        n_particles_total: ti.i32,
    ):  # type: ignore
        """Classic RK4: pos += (dt/6) * (k1 + 2*k2 + 2*k3 + k4).

        The classic 4th-order Runge-Kutta scheme provides high accuracy
        for smooth problems. Requires 4 function evaluations per step.
        """
        for i in range(n_particles_total):
            position[i] += (time_step_size / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])

    return {
        "update_position_euler_kernel": _make_update_position_euler_kernel(),
        "step_euler_forward_kernel": step_euler_forward_kernel,
        "step_euler_forward_strengths_kernel": _make_step_euler_forward_strengths_kernel(),
        "linear_combination_kernel": linear_combination_kernel,
        "step_rk2_combine_kernel": step_rk2_combine_kernel,
        "step_rk3_ssp_combine_kernel": step_rk3_ssp_combine_kernel,
        "step_rk4_combine_kernel": step_rk4_combine_kernel,
    }


def _create_target_eval_kernels(kernel_functions):
    """Create kernels for evaluating flow fields at arbitrary target position."""
    q_ = kernel_functions["q_"]
    zeta_ = kernel_functions["zeta_"]

    @ti.func
    def skew(v):
        return ti.Matrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    @ti.kernel
    def compute_target_velocity_gradient_kernel(
        target_position: ti.template(),
        position: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        target_velocity_gradient: ti.template(),
        n_targets: ti.i32,
        n_particles_total: ti.i32,
    ):  # type: ignore
        """Compute velocity gradient tensor at arbitrary target position."""
        M = n_targets
        N = n_particles_total
        for i in range(M):
            target_pos = target_position[i]
            gradu = target_pos.outer_product(target_pos) * 0.0

            for j in range(N):
                pos_j = position[j]
                str_j = vortex_strength[j]
                r_ij = target_pos - pos_j
                r_sq = r_ij.dot(r_ij)
                r_mag = ti.sqrt(r_sq)

                if r_mag > EPSILON:
                    sigma = core_radius[j]
                    r_sigma = r_mag / sigma

                    q_val = q_(r_sigma)
                    zeta_val = zeta_(r_sigma) / (sigma * sigma * sigma)
                    r_cb = r_sq * r_mag  # r³ = r² · r

                    term1 = q_val / r_cb
                    term2 = 3.0 * q_val / (r_cb * r_sq) - zeta_val / r_sq

                    gradu += term1 * skew(str_j) + term2 * ((r_ij.cross(str_j)).outer_product(r_ij))

            target_velocity_gradient[i] = gradu

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
        vortex_strength: ti.template(),  # Γ per particle
        dstr_dt_out: ti.template(),
        mode: ti.i32,
        n_particles_total: ti.i32,
    ):  # type: ignore
        for i in range(n_particles_total):
            J = grad[i]
            g = vortex_strength[i]
            rate = g * 0.0
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
        "compute_stretching_rate_batch_kernel": _make_stretching_rate_batch_kernel(q_, zeta_),
        "gradient_contraction_rate_kernel": _make_gradient_contraction_kernel(),
    }


def _create_vortex_centroid_kernels(kernel_functions):
    """Create utility kernels for weighted vortex centroids."""

    @ti.kernel
    def compute_vortex_centroid_kernel(
        position: ti.template(), vortex_strength: ti.template(), n_particles_total: ti.i32
    ) -> ti.types.vector(3, ti.f32):  # type: ignore
        total_strength_scalar = 0.0
        total_strength_vector = ti.Vector([0.0, 0.0, 0.0])
        weighted_position = ti.Vector([0.0, 0.0, 0.0])
        n = n_particles_total
        for i in range(n):
            if i < position.shape[0] and i < vortex_strength.shape[0]:
                strength_vec = vortex_strength[i]
                vortex_strength_magnitude = strength_vec.norm()
                if vortex_strength_magnitude > EPSILON:
                    total_strength_scalar += vortex_strength_magnitude
                    total_strength_vector += strength_vec
                    pos = position[i]
                    weighted_position += pos * vortex_strength_magnitude
        vortex_centroid = ti.Vector([0.0, 0.0, 0.0])
        if total_strength_scalar > EPSILON:
            vortex_centroid = weighted_position / total_strength_scalar
        return vortex_centroid

    return {
        "compute_vortex_centroid_kernel": compute_vortex_centroid_kernel,
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
    kernels.update(_create_vortex_centroid_kernels(kernel_functions))

    return kernels
