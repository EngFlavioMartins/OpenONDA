"""
Factories for shared Taichi compute kernels: induced velocity, vorticity,
kinetic energy, enstrophy, time integration, and target-point evaluation.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from collections.abc import Callable

import taichi as ti

from ..config.constants import (
    DEFAULT_CUTOFF_RADIUS_FACTOR,
)


def _make_compute_velocities_kernel(radial_factors_: Callable) -> Callable:
    """Create direct velocity using finite blob factors at all separations."""

    @ti.kernel
    def compute_velocities_kernel(
        position: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        velocity: ti.template(),
        freestream_velocity: ti.template(),
        n_particles_total: ti.i32,
    ):
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

                sigma = 0.5 * (radii_i + core_radius[j])
                factors = radial_factors_(r_mag / sigma, sigma, False)
                vel += factors[0] * r_ij.cross(strength_j)

            velocity[i] = -vel + freestream_vec

    return compute_velocities_kernel


def _make_compute_vorticities_kernel(zeta_: Callable) -> Callable:
    """Mini-factory: creates compute_vorticities_kernel capturing zeta_."""

    @ti.kernel
    def compute_vorticities_kernel(
        position: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        vorticity: ti.template(),
        n_particles_total: ti.i32,
    ):
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


def _make_target_velocity_kernel(radial_factors_: Callable) -> Callable:
    """Create target velocity using source-core factors at all separations."""

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
    ):
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
                sigma = core_radius[j]
                factors = radial_factors_(r_mag / sigma, sigma, False)
                vel += factors[0] * r_ij.cross(strength_j)
            target_velocity[i] = -vel + freestream_vec

    return compute_target_velocity_kernel


def _make_transport_target_velocity_kernel(radial_factors_: Callable) -> Callable:
    """Build a direct finite-target probe kernel with symmetric blob radii."""

    @ti.kernel
    def compute_transport_target_velocity_kernel(
        targets: ti.template(),
        target_core: ti.template(),
        position: ti.template(),
        strength: ti.template(),
        core: ti.template(),
        velocity: ti.template(),
        background: ti.template(),
        n_targets: ti.i32,
        n_sources: ti.i32,
    ):
        """Write full pair sums in O(MN) work and O(M+N) field storage."""
        for i in range(n_targets):
            value = targets[i] * 0.0
            for j in range(n_sources):
                displacement = targets[i] - position[j]
                distance_squared = displacement.dot(displacement)
                distance = ti.sqrt(distance_squared)
                sigma = 0.5 * (target_core[i] + core[j])
                factors = radial_factors_(distance / sigma, sigma, False)
                value += factors[0] * strength[j].cross(displacement)
            velocity[i] = value + background[None]

    return compute_transport_target_velocity_kernel


def _make_target_source_velocity_kernel(radial_factors_: Callable) -> Callable:
    """Create the source-core radial velocity kernel with a finite centre limit."""

    @ti.kernel
    def compute_target_source_velocity_kernel(
        target_position: ti.template(),
        source_position: ti.template(),
        source_strength: ti.template(),
        source_core_radius: ti.template(),
        target_velocity: ti.template(),
        n_targets: ti.i32,
        n_sources: ti.i32,
    ):
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
                sigma = radii_j
                factors = radial_factors_(r_mag / sigma, sigma, False)
                vel += factors[0] * strength_j * r_ij
            target_velocity[i] += vel

    return compute_target_source_velocity_kernel


def _make_target_source_velocity_gradient_kernel(radial_factors_: Callable) -> Callable:
    """Create the Jacobian kernel for radial source-particle induction."""

    @ti.kernel
    def compute_target_source_velocity_gradient_kernel(
        target_position: ti.template(),
        source_position: ti.template(),
        source_strength: ti.template(),
        source_core_radius: ti.template(),
        target_velocity_gradient: ti.template(),
        n_targets: ti.i32,
        n_sources: ti.i32,
    ):
        for i in range(n_targets):
            # Derive the accumulator type from the destination field so the
            # gradient path does not silently demote f64 VPM runs to f32.
            gradient = target_velocity_gradient[i] * 0.0
            target_pos = target_position[i]
            for j in range(n_sources):
                r_vec = target_pos - source_position[j]
                r_sq = r_vec.dot(r_vec)
                r_mag = ti.sqrt(r_sq)
                sigma = source_core_radius[j]
                factors = radial_factors_(r_mag / sigma, sigma, True)
                for row in ti.static(range(3)):
                    for column in ti.static(range(3)):
                        identity = 1.0 if row == column else 0.0
                        gradient[row, column] += source_strength[j] * (
                            identity * factors[0] - factors[1] * r_vec[row] * r_vec[column]
                        )
            target_velocity_gradient[i] = gradient

    return compute_target_source_velocity_gradient_kernel


def _make_target_vorticity_kernel(zeta_: Callable) -> Callable:
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
    ):
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


def _make_csm_kernel(diffusivity_constant_: Callable) -> Callable:
    """Mini-factory: creates update_radius_csm_kernel capturing diffusivity_constant_."""

    @ti.kernel
    def update_radius_csm_kernel(
        core_radius: ti.template(),
        viscosities_eff: ti.template(),
        time_step_size: ti.template(),
        n_particles_total: ti.i32,
    ):
        N = n_particles_total
        for i in range(N):
            effective_viscosity = viscosities_eff[i]
            current_radius = core_radius[i]
            diffusion_term = diffusivity_constant_() * effective_viscosity * time_step_size
            new_rad_sq = current_radius * current_radius + diffusion_term

            # Guard against invalid radius
            core_radius[i] = ti.sqrt(new_rad_sq)

    return update_radius_csm_kernel


def _make_rwm_kernel() -> Callable:
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
    ):
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
    radial_factors,
    mode,
):
    """Contract the finite pair Jacobian with a target strength [m³/s]."""
    dstr = str_i * 0.0
    coeff1 = radial_factors[0]
    coeff2 = radial_factors[1]
    r_cross_Gj = r_ij.cross(str_j)
    Gi_dot_r = str_i.dot(r_ij)
    Gi_dot_rCrossGj = str_i.dot(r_cross_Gj)

    if mode == 0:
        dstr = -coeff1 * str_i.cross(str_j) + coeff2 * Gi_dot_r * r_cross_Gj

    elif mode == 1:
        dstr = coeff1 * str_i.cross(str_j) + coeff2 * Gi_dot_rCrossGj * r_ij

    else:
        # Symmetric direct/transposed formulation (MIXED).
        dstr = 0.5 * coeff2 * (Gi_dot_r * r_cross_Gj + Gi_dot_rCrossGj * r_ij)

    return dstr


def _make_stretching_rate_kernel(radial_factors_: Callable) -> Callable:
    """Create direct stretching from finite kernel-specific radial factors."""

    @ti.kernel
    def compute_stretching_rate_kernel(
        position: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        dstr_dt_out: ti.template(),
        mode: ti.i32,
        n_particles_total: ti.i32,
    ):
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

                sigma = 0.5 * (radii_i + core_radius[j])
                factors = radial_factors_(r_mag / sigma, sigma, True)
                dstr_dt += _stretching_contribution(str_i, str_j, r_ij, factors, mode)

            dstr_dt_out[i] = dstr_dt

    return compute_stretching_rate_kernel


def _make_stretching_rate_batch_kernel(radial_factors_: Callable) -> Callable:
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
    ):
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

                sigma = 0.5 * (radii_i + core_radius[j])
                factors = radial_factors_(r_mag / sigma, sigma, True)
                dstr_dt += _stretching_contribution(str_i, str_j, r_ij, factors, mode)

            dstr_dt_out[i] = dstr_dt

    return compute_stretching_rate_batch_kernel


def _make_velocity_and_stretching_rate_kernel(radial_factors_: Callable) -> Callable:
    """Fuse direct-summation velocity and the selected stretching in one pair walk."""

    @ti.kernel
    def compute_velocity_and_stretching_rate_kernel(
        position: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        velocity: ti.template(),
        vortex_strength_rate: ti.template(),
        freestream_velocity: ti.template(),
        stretching_mode: ti.i32,
        n_particles_total: ti.i32,
    ):
        for i in range(n_particles_total):
            position_i = position[i]
            strength_i = vortex_strength[i]
            radius_i = core_radius[i]
            induced_velocity = position_i * 0.0
            strength_rate = strength_i * 0.0
            for j in range(n_particles_total):
                displacement = position_i - position[j]
                radius_sq = displacement.dot(displacement)
                radius = ti.sqrt(radius_sq)
                sigma = 0.5 * (radius_i + core_radius[j])
                factors = radial_factors_(radius / sigma, sigma, True)
                induced_velocity += factors[0] * displacement.cross(vortex_strength[j])
                strength_rate += _stretching_contribution(
                    strength_i, vortex_strength[j], displacement, factors, stretching_mode
                )
            velocity[i] = -induced_velocity + freestream_velocity[None]
            vortex_strength_rate[i] = strength_rate

    return compute_velocity_and_stretching_rate_kernel


def _create_basic_kernels(kernel_functions: dict[str, Callable]) -> dict[str, Callable]:
    """Create basic velocity and vorticity computation kernels."""
    radial_factors_ = kernel_functions["radial_factors_"]
    zeta_ = kernel_functions["zeta_"]

    return {
        "compute_velocities_kernel": _make_compute_velocities_kernel(radial_factors_),
        "compute_vorticities_kernel": _make_compute_vorticities_kernel(zeta_),
    }


def _create_gradient_kernels(kernel_functions: dict[str, Callable]) -> dict[str, Callable]:
    """Create velocity gradient and strain tensor computation kernels."""
    radial_factors_ = kernel_functions["radial_factors_"]

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
    ):
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

                sigma = 0.5 * (radii_i + core_radius[j])
                factors = radial_factors_(r_mag / sigma, sigma, True)
                gradu += factors[0] * skew(str_j) + factors[1] * (
                    r_ij.cross(str_j).outer_product(r_ij)
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
    ):
        """Fused direct evaluation: u, ∇u and S in a single j-loop.

        The solver needs both u (advection) and ∇u (stretching) each RK stage;
        sharing the one j-loop reuses r_ij / r_mag / sigma / q per pair instead of
        recomputing them in a second O(N²) sweep. Both paths include the finite
        source-centre Jacobian and the algebraic far field."""
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
                sigma = 0.5 * (radii_i + core_radius[j])
                factors = radial_factors_(r_mag / sigma, sigma, True)
                vel += factors[0] * r_ij.cross(str_j)
                gradu += factors[0] * skew(str_j) + factors[1] * (
                    r_ij.cross(str_j).outer_product(r_ij)
                )
            velocity[i] = -vel + freestream_vec
            velocity_gradient[i] = gradu
            strain_rate[i] = 0.5 * (gradu + gradu.transpose())

    return {
        "compute_velocity_gradients_kernel": compute_velocity_gradients_kernel,
        "compute_velocity_and_gradient_kernel": compute_velocity_and_gradient_kernel,
    }


def _create_target_eval_kernels(kernel_functions: dict[str, Callable]) -> dict[str, Callable]:
    """Create kernels for evaluating flow fields at arbitrary target position."""
    radial_factors_ = kernel_functions["radial_factors_"]
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
    ):
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

                sigma = core_radius[j]
                factors = radial_factors_(r_mag / sigma, sigma, True)
                gradu += factors[0] * skew(str_j) + factors[1] * (
                    r_ij.cross(str_j).outer_product(r_ij)
                )

            target_velocity_gradient[i] = gradu

    return {
        "compute_target_velocity_kernel": _make_target_velocity_kernel(radial_factors_),
        "compute_transport_target_velocity_kernel": _make_transport_target_velocity_kernel(
            radial_factors_
        ),
        "compute_target_source_velocity_kernel": _make_target_source_velocity_kernel(
            radial_factors_
        ),
        "compute_target_source_velocity_gradient_kernel": _make_target_source_velocity_gradient_kernel(
            radial_factors_
        ),
        "compute_target_vorticity_kernel": _make_target_vorticity_kernel(zeta_),
        "compute_target_velocity_gradient_kernel": compute_target_velocity_gradient_kernel,
    }


def _create_diffusion_kernels(kernel_functions: dict[str, Callable]) -> dict[str, Callable]:
    """Create kernels for viscous diffusion (CSM, RWM)."""
    diffusivity_constant_ = kernel_functions["diffusivity_constant_"]

    return {
        "update_radius_csm_kernel": _make_csm_kernel(diffusivity_constant_),
        "update_position_rwm_kernel": _make_rwm_kernel(),
    }


def _create_stretching_kernels(kernel_functions: dict[str, Callable]) -> dict[str, Callable]:
    """Create kernels for vortex stretching computations."""
    radial_factors_ = kernel_functions["radial_factors_"]

    return {
        "compute_stretching_rate_kernel": _make_stretching_rate_kernel(radial_factors_),
        "compute_stretching_rate_batch_kernel": _make_stretching_rate_batch_kernel(radial_factors_),
        "compute_velocity_and_stretching_rate_kernel": _make_velocity_and_stretching_rate_kernel(
            radial_factors_
        ),
    }


def create_kernels(kernel_functions: dict[str, Callable]) -> dict[str, Callable]:
    """Create the Taichi kernels of the VPM physics for one particle kernel.

    ``kernel_functions`` holds the device radial functions of the particle
    kernel; each category factory closes over the ones it needs.
    """
    kernels = {}
    kernels.update(_create_basic_kernels(kernel_functions))
    kernels.update(_create_gradient_kernels(kernel_functions))
    kernels.update(_create_target_eval_kernels(kernel_functions))
    kernels.update(_create_diffusion_kernels(kernel_functions))
    kernels.update(_create_stretching_kernels(kernel_functions))
    return kernels
