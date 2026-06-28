"""
Physics Engine Module for VPM Solver.
=====================================
Unified physics engine combining advection, diffusion, and stretching.

This module provides the main PhysicsEngine class that orchestrates
all physics operations for the VPM solver. It uses composition over
multiple inheritance for cleaner architecture.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import taichi as ti

from ..config.constants import MAX_PARTICLES
from .base import PhysicsBase
from .diffusion import _GridDiffusionMixin


@ti.data_oriented
class PhysicsEngine(PhysicsBase, _GridDiffusionMixin):
    """
    Unified Physics Engine for VPM.

    This class orchestrates all physics operations by inheriting from
    PhysicsBase (for field evaluation) and _GridDiffusionMixin (for DVH),
    and delegating to specialized handlers for advection, etc.
    """

    def __init__(
        self,
        particles_kernel: str = "GAUSSIAN",
        max_particles: int = MAX_PARTICLES,
        accumulator_dtype: ti.types = ti.f32,
    ):
        """Initialize the unified physics engine."""
        # Initialize base classes
        super().__init__(particles_kernel, max_particles, accumulator_dtype)
        self._init_grid_diffusion()

        # Optional advection velocity override: fn(pos (N,3), vel_bs (N,3)) -> (N,3).
        # Applied at every RK stage after Biot-Savart; used by the FVM-VPM coupler
        # to blend in the FVM near-body velocity for overlap-region particles.
        self.velocity_override = None

        # Create specialized physics handlers
        self._advection = _AdvectionHandler(self)
        self._diffusion = _DiffusionHandler(self)
        self._stretching = _StretchingHandler(self)
        self.stretching_rate_limiter = None

    def __str__(self):
        return (
            f"  Kernel                   : {self.particles_kernel}\n"
            f"  Max Particles            : {self.max_particles}"
        )

    # =========================================================================
    # ADVECTION INTERFACE
    # =========================================================================

    def update_positions(self, particles, dt: float, scheme: str = "RK4"):
        """
        Update particle positions using specified time integration scheme.

        Delegates to advection handler.

        Args:
            particles: Particle container
            dt: Time step size [s]
            scheme: 'NONE', 'EULER', 'RK2', 'RK3', or 'RK4'
        """
        self._advection.update_positions(particles, dt, scheme)

    # =========================================================================
    # DIFFUSION INTERFACE
    # =========================================================================

    def core_spreading_diffusion(self, particles, dt: float):
        """
        Apply Core Spreading Method diffusion.

        Delegates to diffusion handler.

        Args:
            particles: Particle container
            dt: Time step size [s]
        """
        self._diffusion.core_spreading_diffusion(particles, dt)

    def random_walk_method_diffusion(self, particles, dt: float):
        """
        Apply Random Walk Method diffusion.

        Delegates to diffusion handler.

        Args:
            particles: Particle container
            dt: Time step size [s]
        """
        self._diffusion.random_walk_method_diffusion(particles, dt)

    def update_volumes(self, particles, dt: float):
        """
        Update volumes from velocity divergence.

        Delegates to diffusion handler.

        Args:
            particles: Particle container
            dt: Time step size [s]
        """
        self._diffusion.update_volumes(particles, dt)

    # =========================================================================
    # STRETCHING INTERFACE
    # =========================================================================

    def vortex_stretching(
        self,
        particles,
        dt: float,
        scheme: str = "RK3",
        mode: str = "TRANSPOSE",
        rvpm_f: float = 0.0,
        rvpm_g: float = 0.2,
    ):
        """
        Apply vortex stretching.

        Delegates to stretching handler.

        Args:
            particles: Particle container
            dt: Time step size [s]
            scheme: 'EULER', 'RK2', 'RK3', or 'RK4'
            mode: 'CLASSICAL', 'TRANSPOSE', 'MIXED', 'GRADU', or 'RVPM'
            rvpm_f, rvpm_g: rVPM reformulation parameters (mode='RVPM' only);
                c_r = (g+f)/(1/3+f), c_σ = (g+f)/(1+3f).  Defaults f=0, g=1/5.
        """
        self._stretching.vortex_stretching(particles, dt, scheme, mode, rvpm_f, rvpm_g)

    def save_strength_magnitudes(self, particles):
        """
        Save strength magnitudes for splitting detection.

        Delegates to stretching handler.

        Args:
            particles: Particle container
        """
        self._stretching.save_strength_magnitudes(particles)


# =============================================================================
# INTERNAL HANDLER CLASSES (share parent's temp fields and kernels)
# =============================================================================


class _AdvectionHandler:
    """
    Lightweight advection handler that uses parent's resources.

    This avoids duplicate temp field allocation by referencing
    the parent PhysicsEngine's fields and kernels.
    """

    def __init__(self, parent: PhysicsEngine):
        self._parent = parent

    def update_positions(self, particles, dt: float, scheme: str = "RK4"):
        """Advance particle positions by dt with a single step of the given scheme.

        The advance is one full step of the chosen scheme (EULER/RK2/RK3/RK4) over
        the macro time-step.  Every velocity evaluation routes through
        ``parent.velocity_self`` so the configured velocity method (direct or
        treecode) is applied consistently at every stage.  Self-contained: does
        not depend on a pre-populated ``particles.velocity`` (k1 is computed fresh).

        Args:
            particles:  particle container.
            dt:         macro time-step [s].
            scheme:     "NONE" | "EULER" | "RK2" | "RK3" | "RK4".
        """
        N = len(particles)
        scheme = scheme.upper()
        if N == 0 or dt == 0.0 or scheme == "NONE":
            return

        self._parent._resize_temp_fields(N)
        self._step(particles, dt, scheme, N)

    def _vel(self, particles, pos_field, out_field, N):
        """Self-induced velocity at ``pos_field`` → ``out_field`` (honors method)."""
        self._parent.velocity_self(
            pos_field,
            particles.circulation,
            particles.radius,
            out_field,
            particles.velocity_background,
            N,
        )
        if self._parent.velocity_override is not None:
            pos_np = pos_field.to_numpy()
            vel_np = out_field.to_numpy()
            override = self._parent.velocity_override
            if hasattr(override, "blend_into"):
                override.blend_into(pos_np[:N], vel_np[:N], vel_np[:N])
            else:
                vel_np[:N] = override(pos_np[:N], vel_np[:N])
            out_field.from_numpy(vel_np)

    def _step(self, particles, dt, scheme, N):
        """One full step of the chosen scheme over dt."""
        if scheme == "EULER":
            self._euler(particles, dt, N)
        elif scheme == "RK2":
            self._rk2(particles, dt, N)
        elif scheme == "RK3":
            self._rk3(particles, dt, N)
        elif scheme == "RK4":
            self._rk4(particles, dt, N)
        else:
            raise ValueError(
                f"Unknown advection scheme: {scheme}. Use NONE, EULER, RK2, RK3, or RK4."
            )

    def _euler(self, particles, dt, N):
        """x_{n+1} = x_n + dt·v(x_n)."""
        p = self._parent
        self._vel(particles, particles.position, particles.velocity, N)  # k1
        p.step_euler_forward_kernel(
            particles.position, particles.velocity, particles.position, dt, N
        )

    def _rk2(self, particles, dt, N):
        """Heun's method: x_{n+1} = x_n + dt/2·(k1 + k2)."""
        p = self._parent
        self._vel(particles, particles.position, particles.velocity, N)  # k1
        # x_pred = x_n + dt·k1
        p.step_euler_forward_kernel(particles.position, particles.velocity, p.pos_temp, dt, N)
        self._vel(particles, p.pos_temp, p.vel_temp, N)  # k2 = v(x_pred)
        p.step_rk2_combine_kernel(particles.position, particles.velocity, p.vel_temp, dt, N)

    def _rk3(self, particles, dt, N):
        """SSP-RK3: x_{n+1} = x_n + dt/6·(k1 + k2 + 4·k3)."""
        p = self._parent
        self._vel(particles, particles.position, particles.velocity, N)  # k1
        # x1 = x_n + dt·k1
        p.step_euler_forward_kernel(particles.position, particles.velocity, p.pos_temp, dt, N)
        self._vel(particles, p.pos_temp, p.vel_temp, N)  # k2 = v(x1)
        # x2 = x_n + dt/4·(k1 + k2)
        p.linear_combination_kernel(
            p.pos_temp2, particles.velocity, p.vel_temp, 0.25 * dt, 0.25 * dt, N
        )
        p.step_euler_forward_kernel(particles.position, p.pos_temp2, p.pos_temp2, 1.0, N)
        self._vel(particles, p.pos_temp2, p.vel_temp2, N)  # k3 = v(x2)
        p.step_rk3_ssp_combine_kernel(
            particles.position, particles.velocity, p.vel_temp, p.vel_temp2, dt, N
        )

    def _rk4(self, particles, dt, N):
        """Classic RK4: x_{n+1} = x_n + dt/6·(k1 + 2·k2 + 2·k3 + k4)."""
        p = self._parent
        self._vel(particles, particles.position, particles.velocity, N)  # k1 → particles.velocity
        # k2 = v(x_n + 0.5·dt·k1)
        p.step_euler_forward_kernel(particles.position, particles.velocity, p.pos_temp, 0.5 * dt, N)
        self._vel(particles, p.pos_temp, p.vel_temp, N)
        # k3 = v(x_n + 0.5·dt·k2)
        p.step_euler_forward_kernel(particles.position, p.vel_temp, p.pos_temp, 0.5 * dt, N)
        self._vel(particles, p.pos_temp, p.vel_temp2, N)
        # k4 = v(x_n + dt·k3)  (stored in pos_temp2)
        p.step_euler_forward_kernel(particles.position, p.vel_temp2, p.pos_temp, dt, N)
        self._vel(particles, p.pos_temp, p.pos_temp2, N)
        p.step_rk4_combine_kernel(
            particles.position, particles.velocity, p.vel_temp, p.vel_temp2, p.pos_temp2, dt, N
        )


class _DiffusionHandler:
    """
    Lightweight diffusion handler that uses parent's resources.
    """

    def __init__(self, parent: PhysicsEngine):
        self._parent = parent

    def core_spreading_diffusion(self, particles, dt: float):
        """Core spreading diffusion."""
        N = len(particles)
        if N == 0 or dt <= 0.0:
            return
        p = self._parent
        p._resize_temp_fields(N)
        p.update_radius_csm_kernel(particles.radius, particles.viscosity_effective, dt, N)

    def random_walk_method_diffusion(self, particles, dt: float):
        """Random walk diffusion."""
        N = len(particles)
        if N == 0 or dt <= 0.0:
            return
        p = self._parent
        p._resize_temp_fields(N)
        p.update_position_rwm_kernel(particles.position, particles.viscosity_effective, dt, N)

    def update_volumes(self, particles, dt: float):
        """Volume update from divergence."""
        N = len(particles)
        if N == 0 or dt == 0.0:
            return
        p = self._parent
        p._resize_temp_fields(N)
        p.update_volume_divergence_kernel(
            particles.volume, particles.radius, particles.velocity_gradient, dt, N
        )


class _StretchingHandler:
    """
    Lightweight stretching handler that uses parent's resources.
    """

    def __init__(self, parent: PhysicsEngine):
        self._parent = parent

    def vortex_stretching(
        self,
        particles,
        dt: float,
        scheme: str = "RK3",
        mode: str = "TRANSPOSE",
        rvpm_f: float = 0.0,
        rvpm_g: float = 0.2,
    ):
        """Vortex stretching using parent's temp fields."""
        N = len(particles)
        if N == 0 or dt == 0.0:
            return

        p = self._parent
        p._resize_temp_fields(N)
        p._zero_temp_fields()

        # GradU/RVPM modes: local O(N) stretching using pre-computed velocity
        # gradients.  RVPM (Alvarez & Ning reformulation) removes the fraction
        # c_r = (g+f)/(1/3+f) of the parallel stretching rate from the strength
        # and absorbs it into the core size (σ ← σ·exp(−c_σ·S_∥·dt),
        # c_σ = (g+f)/(1+3f)), conserving the element volume measure σ²|α|.
        mode_str = mode.upper()
        if mode_str in ("GRADU", "RVPM"):
            c_r = 0.0
            if mode_str == "RVPM":
                c_r = (rvpm_g + rvpm_f) / (1.0 / 3.0 + rvpm_f)
            self._gradu_stretching(particles, dt, scheme.upper(), N, c_r)
            if mode_str == "RVPM":
                c_sigma = (rvpm_g + rvpm_f) / (1.0 + 3.0 * rvpm_f)
                p.update_radius_rvpm_kernel(
                    particles.circulation,
                    particles.velocity_gradient,
                    particles.radius,
                    c_sigma,
                    dt,
                    N,
                )
            return

        # Convert mode
        if mode_str == "CLASSICAL":
            mode_int = 0
        elif mode_str in ("TRANSPOSE", "TRANSPOSED"):
            mode_int = 1
        elif mode_str == "MIXED":
            mode_int = 2
        else:
            raise ValueError(f"Unknown mode: {mode}")

        scheme = scheme.upper()

        if scheme == "EULER":
            p.compute_stretching_rate_kernel(
                particles.position,
                particles.circulation,
                particles.radius,
                p.dstr_dt_temp,
                mode_int,
                N,
            )
            self._limit_rate(particles.position, particles.circulation, p.dstr_dt_temp, dt, N)
            p.step_euler_forward_kernel(
                particles.circulation, p.dstr_dt_temp, particles.circulation, dt, N
            )

        elif scheme == "RK2":
            self._stretching_rk2(particles, dt, mode_int, N)

        elif scheme == "RK3":
            self._stretching_rk3(particles, dt, mode_int, N)

        elif scheme == "RK4":
            self._stretching_rk4(particles, dt, mode_int, N)

        else:
            raise ValueError(f"Unknown scheme: {scheme}")

    def _stretching_rk2(self, particles, dt, mode_int, N):
        """RK2 stretching."""
        p = self._parent
        p.compute_stretching_rate_kernel(
            particles.position, particles.circulation, particles.radius, p.dstr_dt_temp, mode_int, N
        )
        self._limit_rate(particles.position, particles.circulation, p.dstr_dt_temp, dt, N)
        p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp, p.str_temp, dt, N)
        p.compute_stretching_rate_kernel(
            particles.position, p.str_temp, particles.radius, p.dstr_dt_temp2, mode_int, N
        )
        self._limit_rate(particles.position, p.str_temp, p.dstr_dt_temp2, dt, N)
        p.step_rk2_combine_kernel(particles.circulation, p.dstr_dt_temp, p.dstr_dt_temp2, dt, N)

    def _stretching_rk3(self, particles, dt, mode_int, N):
        """RK3 stretching."""
        p = self._parent
        p.compute_stretching_rate_kernel(
            particles.position, particles.circulation, particles.radius, p.dstr_dt_temp, mode_int, N
        )
        self._limit_rate(particles.position, particles.circulation, p.dstr_dt_temp, dt, N)
        p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp, p.str_temp, dt, N)
        p.compute_stretching_rate_kernel(
            particles.position, p.str_temp, particles.radius, p.dstr_dt_temp2, mode_int, N
        )
        self._limit_rate(particles.position, p.str_temp, p.dstr_dt_temp2, dt, N)
        p.linear_combination_kernel(
            p.str_temp2, p.dstr_dt_temp, p.dstr_dt_temp2, 0.25 * dt, 0.25 * dt, N
        )
        p.step_euler_forward_kernel(particles.circulation, p.str_temp2, p.str_temp2, 1.0, N)
        p.compute_stretching_rate_kernel(
            particles.position, p.str_temp2, particles.radius, p.dstr_dt_temp3, mode_int, N
        )
        self._limit_rate(particles.position, p.str_temp2, p.dstr_dt_temp3, dt, N)
        p.step_rk3_ssp_combine_kernel(
            particles.circulation, p.dstr_dt_temp, p.dstr_dt_temp2, p.dstr_dt_temp3, dt, N
        )

    def _stretching_rk4(self, particles, dt, mode_int, N):
        """RK4 stretching."""
        p = self._parent
        p.compute_stretching_rate_kernel(
            particles.position, particles.circulation, particles.radius, p.dstr_dt_temp, mode_int, N
        )
        self._limit_rate(particles.position, particles.circulation, p.dstr_dt_temp, dt, N)
        p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp, p.str_temp, 0.5 * dt, N)
        p.compute_stretching_rate_kernel(
            particles.position, p.str_temp, particles.radius, p.dstr_dt_temp2, mode_int, N
        )
        self._limit_rate(particles.position, p.str_temp, p.dstr_dt_temp2, dt, N)
        p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp2, p.str_temp, 0.5 * dt, N)
        p.compute_stretching_rate_kernel(
            particles.position, p.str_temp, particles.radius, p.dstr_dt_temp3, mode_int, N
        )
        self._limit_rate(particles.position, p.str_temp, p.dstr_dt_temp3, dt, N)
        p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp3, p.str_temp, dt, N)
        p.compute_stretching_rate_kernel(
            particles.position, p.str_temp, particles.radius, p.vel_temp, mode_int, N
        )
        self._limit_rate(particles.position, p.str_temp, p.vel_temp, dt, N)
        p.step_rk4_combine_kernel(
            particles.circulation,
            p.dstr_dt_temp,
            p.dstr_dt_temp2,
            p.dstr_dt_temp3,
            p.vel_temp,
            dt,
            N,
        )

    def save_strength_magnitudes(self, particles):
        """Save magnitudes for splitting detection."""
        N = len(particles)
        if N == 0:
            return
        p = self._parent
        p._resize_temp_fields(N)
        p.compute_strength_magnitudes_kernel(particles.circulation, p.str_mag_before, N)
        ti.sync()

    # ----- GradU-based stretching (O(N) per sub-step) -----

    def _limit_rate(self, positions, strengths, rates, dt: float, N: int) -> None:
        limiter = getattr(self._parent, "stretching_rate_limiter", None)
        if limiter is not None:
            limiter.apply_to_rate(positions, strengths, rates, dt, N)

    def _gradu_rate(self, strengths, gradU, out, N, c_r: float = 0.0):
        """Compute dα/dt = (∇u)ᵀ·α (c_r=0) or the rVPM rate Jᵀα − c_r(α̂·Jᵀα)α̂."""
        if c_r != 0.0:
            self._parent.compute_rvpm_stretching_rate_kernel(strengths, gradU, out, c_r, N)
        else:
            self._parent.compute_gradu_stretching_rate_kernel(strengths, gradU, out, N)

    def _gradu_stretching(self, particles, dt: float, scheme: str, N: int, c_r: float = 0.0):
        """GradU/rVPM stretching with RK time integration."""
        p = self._parent
        gradU = particles.velocity_gradient

        if scheme == "EULER":
            self._gradu_rate(particles.circulation, gradU, p.dstr_dt_temp, N, c_r)
            self._limit_rate(particles.position, particles.circulation, p.dstr_dt_temp, dt, N)
            p.step_euler_forward_kernel(
                particles.circulation, p.dstr_dt_temp, particles.circulation, dt, N
            )

        elif scheme == "RK2":
            # k1
            self._gradu_rate(particles.circulation, gradU, p.dstr_dt_temp, N, c_r)
            self._limit_rate(particles.position, particles.circulation, p.dstr_dt_temp, dt, N)
            p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp, p.str_temp, dt, N)
            # k2
            self._gradu_rate(p.str_temp, gradU, p.dstr_dt_temp2, N, c_r)
            self._limit_rate(particles.position, p.str_temp, p.dstr_dt_temp2, dt, N)
            p.step_rk2_combine_kernel(particles.circulation, p.dstr_dt_temp, p.dstr_dt_temp2, dt, N)

        elif scheme == "RK3":
            # k1
            self._gradu_rate(particles.circulation, gradU, p.dstr_dt_temp, N, c_r)
            self._limit_rate(particles.position, particles.circulation, p.dstr_dt_temp, dt, N)
            p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp, p.str_temp, dt, N)
            # k2
            self._gradu_rate(p.str_temp, gradU, p.dstr_dt_temp2, N, c_r)
            self._limit_rate(particles.position, p.str_temp, p.dstr_dt_temp2, dt, N)
            p.linear_combination_kernel(
                p.str_temp2, p.dstr_dt_temp, p.dstr_dt_temp2, 0.25 * dt, 0.25 * dt, N
            )
            p.step_euler_forward_kernel(particles.circulation, p.str_temp2, p.str_temp2, 1.0, N)
            # k3
            self._gradu_rate(p.str_temp2, gradU, p.dstr_dt_temp3, N, c_r)
            self._limit_rate(particles.position, p.str_temp2, p.dstr_dt_temp3, dt, N)
            p.step_rk3_ssp_combine_kernel(
                particles.circulation, p.dstr_dt_temp, p.dstr_dt_temp2, p.dstr_dt_temp3, dt, N
            )

        elif scheme == "RK4":
            # k1
            self._gradu_rate(particles.circulation, gradU, p.dstr_dt_temp, N, c_r)
            self._limit_rate(particles.position, particles.circulation, p.dstr_dt_temp, dt, N)
            p.step_euler_forward_kernel(
                particles.circulation, p.dstr_dt_temp, p.str_temp, 0.5 * dt, N
            )
            # k2
            self._gradu_rate(p.str_temp, gradU, p.dstr_dt_temp2, N, c_r)
            self._limit_rate(particles.position, p.str_temp, p.dstr_dt_temp2, dt, N)
            p.step_euler_forward_kernel(
                particles.circulation, p.dstr_dt_temp2, p.str_temp, 0.5 * dt, N
            )
            # k3
            self._gradu_rate(p.str_temp, gradU, p.dstr_dt_temp3, N, c_r)
            self._limit_rate(particles.position, p.str_temp, p.dstr_dt_temp3, dt, N)
            p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp3, p.str_temp, dt, N)
            # k4
            self._gradu_rate(p.str_temp, gradU, p.vel_temp, N, c_r)
            self._limit_rate(particles.position, p.str_temp, p.vel_temp, dt, N)
            p.step_rk4_combine_kernel(
                particles.circulation,
                p.dstr_dt_temp,
                p.dstr_dt_temp2,
                p.dstr_dt_temp3,
                p.vel_temp,
                dt,
                N,
            )

        else:
            raise ValueError(f"Unknown scheme for GRADU stretching: {scheme}")
