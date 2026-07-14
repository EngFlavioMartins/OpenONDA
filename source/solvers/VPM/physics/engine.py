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

    # ADVECTION INTERFACE

    def update_positions(
        self, particles, dt: float, scheme: str = "RK3", precomputed_k1: bool = False
    ):
        """
        Update particle positions using specified time integration scheme.

        Delegates to advection handler.

        Args:
            particles: Particle container
            dt: Time step size [s]
            scheme: 'NONE', 'EULER', 'RK2', 'RK3', or 'RK4'
            precomputed_k1: when True, ``particles.velocity`` already holds
                v(x_n) (e.g. from a fused velocity+gradient pass at t_n), so the
                integrator's first stage reuses it instead of recomputing.
        """
        self._advection.update_positions(particles, dt, scheme, precomputed_k1)

    # DIFFUSION INTERFACE

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

    # STRETCHING INTERFACE

    def vortex_stretching(
        self,
        particles,
        dt: float,
        scheme: str = "RK3",
        mode: str = "TRANSPOSED",
        use_treecode: bool = False,
        treecode_theta: float = 0.3,
    ):
        """
        Apply vortex stretching.

        Delegates to stretching handler.

        Args:
            particles: Particle container
            dt: Time step size [s]
            scheme: 'EULER', 'RK2', 'RK3', or 'RK4'
            mode: 'DIRECT', 'TRANSPOSED', or 'MIXED'
            use_treecode: evaluate the rate from the O(N log N) treecode gradient
                instead of the direct O(N²) pairwise kernel (large N).
            treecode_theta: Barnes–Hut opening angle for the treecode gradient.
        """
        self._stretching.vortex_stretching(
            particles, dt, scheme, mode, use_treecode, treecode_theta
        )

    def save_strength_magnitudes(self, particles):
        """
        Save strength magnitudes for splitting detection.

        Delegates to stretching handler.

        Args:
            particles: Particle container
        """
        self._stretching.save_strength_magnitudes(particles)


# =========================================================
# INTERNAL HANDLER CLASSES (share parent's temp fields and kernels)
# =========================================================


class _AdvectionHandler:
    """
    Lightweight advection handler that uses parent's resources.

    This avoids duplicate temp field allocation by referencing
    the parent PhysicsEngine's fields and kernels.
    """

    def __init__(self, parent: PhysicsEngine):
        self._parent = parent

    def update_positions(
        self, particles, dt: float, scheme: str = "RK3", precomputed_k1: bool = False
    ):
        """Advance particle positions by dt with a single step of the given scheme.

        The advance is one full step of the chosen scheme (EULER/RK2/RK3/RK4) over
        the macro time-step.  Every velocity evaluation routes through
        ``parent.velocity_self`` so the configured velocity method (direct or
        treecode) is applied consistently at every stage.

        Args:
            particles:  particle container.
            dt:         macro time-step [s].
            scheme:     "NONE" | "EULER" | "RK2" | "RK3" | "RK4".
            precomputed_k1: when True the first stage reuses ``particles.velocity``
                (already holds v(x_n)) instead of recomputing it — set by the
                solver when a fused velocity+gradient pass populated it at t_n.
        """
        N = len(particles)
        scheme = scheme.upper()
        if N == 0 or dt == 0.0 or scheme == "NONE":
            return

        self._parent._resize_temp_fields(N)
        self._step(particles, dt, scheme, N, precomputed_k1)

    def _vel(self, particles, pos_field, out_field, N, reuse_tree=False):
        """Self-induced velocity at ``pos_field`` → ``out_field`` (honors method).

        ``reuse_tree=True`` (RK stages ≥ 2) refits the stage-1 LBVH topology to
        the displaced positions instead of rebuilding it — same physics, ~half
        the per-stage tree cost.  Not used when a velocity override (FVM blend)
        is active, since that path round-trips through the CPU anyway.
        """
        self._parent.velocity_self(
            pos_field,
            particles.circulation,
            particles.radius,
            out_field,
            particles.velocity_background,
            N,
            reuse_tree=reuse_tree and self._parent.velocity_override is None,
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

    def _step(self, particles, dt, scheme, N, precomputed_k1=False):
        """One full step of the chosen scheme over dt."""
        if scheme == "EULER":
            self._euler(particles, dt, N, precomputed_k1)
        elif scheme == "RK2":
            self._rk2(particles, dt, N, precomputed_k1)
        elif scheme == "RK3":
            self._rk3(particles, dt, N, precomputed_k1)
        elif scheme == "RK4":
            self._rk4(particles, dt, N, precomputed_k1)
        else:
            raise ValueError(
                f"Unknown advection scheme: {scheme}. Use NONE, EULER, RK2, RK3, or RK4."
            )

    def _k1(self, particles, N, precomputed_k1):
        """Stage-1 velocity v(x_n) → particles.velocity (reused if precomputed)."""
        if not precomputed_k1:
            self._vel(particles, particles.position, particles.velocity, N)

    def _euler(self, particles, dt, N, precomputed_k1=False):
        """x_{n+1} = x_n + dt·v(x_n)."""
        p = self._parent
        self._k1(particles, N, precomputed_k1)
        p.step_euler_forward_kernel(
            particles.position, particles.velocity, particles.position, dt, N
        )

    def _rk2(self, particles, dt, N, precomputed_k1=False):
        """Heun's method: x_{n+1} = x_n + dt/2·(k1 + k2)."""
        p = self._parent
        self._k1(particles, N, precomputed_k1)
        # x_pred = x_n + dt·k1
        p.step_euler_forward_kernel(particles.position, particles.velocity, p.pos_temp, dt, N)
        self._vel(particles, p.pos_temp, p.vel_temp, N, reuse_tree=True)  # k2 = v(x_pred)
        p.step_rk2_combine_kernel(particles.position, particles.velocity, p.vel_temp, dt, N)

    def _rk3(self, particles, dt, N, precomputed_k1=False):
        """SSP-RK3: x_{n+1} = x_n + dt/6·(k1 + k2 + 4·k3)."""
        p = self._parent
        self._k1(particles, N, precomputed_k1)
        # x1 = x_n + dt·k1
        p.step_euler_forward_kernel(particles.position, particles.velocity, p.pos_temp, dt, N)
        self._vel(particles, p.pos_temp, p.vel_temp, N, reuse_tree=True)  # k2 = v(x1)
        # x2 = x_n + dt/4·(k1 + k2)
        p.linear_combination_kernel(
            p.pos_temp2, particles.velocity, p.vel_temp, 0.25 * dt, 0.25 * dt, N
        )
        p.step_euler_forward_kernel(particles.position, p.pos_temp2, p.pos_temp2, 1.0, N)
        self._vel(particles, p.pos_temp2, p.vel_temp2, N, reuse_tree=True)  # k3 = v(x2)
        p.step_rk3_ssp_combine_kernel(
            particles.position, particles.velocity, p.vel_temp, p.vel_temp2, dt, N
        )

    def _rk4(self, particles, dt, N, precomputed_k1=False):
        """Classic RK4: x_{n+1} = x_n + dt/6·(k1 + 2·k2 + 2·k3 + k4)."""
        p = self._parent
        self._k1(particles, N, precomputed_k1)  # k1 → particles.velocity
        # k2 = v(x_n + 0.5·dt·k1)
        p.step_euler_forward_kernel(particles.position, particles.velocity, p.pos_temp, 0.5 * dt, N)
        self._vel(particles, p.pos_temp, p.vel_temp, N, reuse_tree=True)
        # k3 = v(x_n + 0.5·dt·k2)
        p.step_euler_forward_kernel(particles.position, p.vel_temp, p.pos_temp, 0.5 * dt, N)
        self._vel(particles, p.pos_temp, p.vel_temp2, N, reuse_tree=True)
        # k4 = v(x_n + dt·k3)  (stored in pos_temp2)
        p.step_euler_forward_kernel(particles.position, p.vel_temp2, p.pos_temp, dt, N)
        self._vel(particles, p.pos_temp, p.pos_temp2, N, reuse_tree=True)
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
        # Set per-call by vortex_stretching(); defaults keep _rate() safe if it
        # is ever reached before the first stretching call.
        self._use_treecode = False
        self._treecode_theta = 0.3

    def vortex_stretching(
        self,
        particles,
        dt: float,
        scheme: str = "RK3",
        mode: str = "TRANSPOSED",
        use_treecode: bool = False,
        treecode_theta: float = 0.3,
    ):
        """Vortex stretching using parent's temp fields.

        ``use_treecode=True`` evaluates the stretching rate from the O(N log N)
        treecode velocity gradient instead of the direct O(N²) pairwise kernel
        (see _rate); numerically identical up to the Barnes–Hut tolerance.
        """
        N = len(particles)
        if N == 0 or dt == 0.0:
            return

        p = self._parent
        # Treecode stretching needs the actual treecode velocity method; if the
        # solver is in DIRECT velocity mode there is no tree, so fall back.
        self._use_treecode = bool(use_treecode) and p.velocity_method == "TREECODE"
        self._treecode_theta = float(treecode_theta)
        p._resize_temp_fields(N)
        p._zero_temp_fields()

        mode_str = mode.upper()
        if mode_str == "DIRECT":
            mode_int = 0
        elif mode_str == "TRANSPOSED":
            mode_int = 1
        elif mode_str == "MIXED":
            mode_int = 2
        else:
            raise ValueError(f"Unknown stretching mode: {mode}. Use DIRECT, TRANSPOSED, or MIXED.")

        scheme = scheme.upper()

        if scheme == "EULER":
            self._rate(
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

    def _rate(self, pos, strg, rad, out, mode_int, N):
        """Stretching rate dΓ/dt at (pos, strg): direct pairwise or treecode.

        Direct: the O(N²) pairwise kernel.  Treecode: build the LBVH at
        (pos, strg), evaluate the velocity gradient J = ∇u (O(N log N)), and
        contract it locally — J·Γ (DIRECT), Jᵀ·Γ (TRANSPOSED) or S·Γ (MIXED).
        The two agree up to the Barnes–Hut opening-angle tolerance.
        """
        p = self._parent
        if self._use_treecode:
            tree = p._get_or_create_treecode(N, self._treecode_theta)
            tree.build(pos, strg, rad, N)
            tree.compute_velocity_gradients_gpu()
            p.gradient_contraction_rate_kernel(tree.velocity_gradients, strg, out, mode_int, N)
        else:
            p.compute_stretching_rate_kernel(pos, strg, rad, out, mode_int, N)

    def _stretching_rk2(self, particles, dt, mode_int, N):
        """RK2 stretching."""
        p = self._parent
        self._rate(
            particles.position, particles.circulation, particles.radius, p.dstr_dt_temp, mode_int, N
        )
        self._limit_rate(particles.position, particles.circulation, p.dstr_dt_temp, dt, N)
        p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp, p.str_temp, dt, N)
        self._rate(particles.position, p.str_temp, particles.radius, p.dstr_dt_temp2, mode_int, N)
        self._limit_rate(particles.position, p.str_temp, p.dstr_dt_temp2, dt, N)
        p.step_rk2_combine_kernel(particles.circulation, p.dstr_dt_temp, p.dstr_dt_temp2, dt, N)

    def _stretching_rk3(self, particles, dt, mode_int, N):
        """RK3 stretching."""
        p = self._parent
        self._rate(
            particles.position, particles.circulation, particles.radius, p.dstr_dt_temp, mode_int, N
        )
        self._limit_rate(particles.position, particles.circulation, p.dstr_dt_temp, dt, N)
        p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp, p.str_temp, dt, N)
        self._rate(particles.position, p.str_temp, particles.radius, p.dstr_dt_temp2, mode_int, N)
        self._limit_rate(particles.position, p.str_temp, p.dstr_dt_temp2, dt, N)
        p.linear_combination_kernel(
            p.str_temp2, p.dstr_dt_temp, p.dstr_dt_temp2, 0.25 * dt, 0.25 * dt, N
        )
        p.step_euler_forward_kernel(particles.circulation, p.str_temp2, p.str_temp2, 1.0, N)
        self._rate(particles.position, p.str_temp2, particles.radius, p.dstr_dt_temp3, mode_int, N)
        self._limit_rate(particles.position, p.str_temp2, p.dstr_dt_temp3, dt, N)
        p.step_rk3_ssp_combine_kernel(
            particles.circulation, p.dstr_dt_temp, p.dstr_dt_temp2, p.dstr_dt_temp3, dt, N
        )

    def _stretching_rk4(self, particles, dt, mode_int, N):
        """RK4 stretching."""
        p = self._parent
        self._rate(
            particles.position, particles.circulation, particles.radius, p.dstr_dt_temp, mode_int, N
        )
        self._limit_rate(particles.position, particles.circulation, p.dstr_dt_temp, dt, N)
        p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp, p.str_temp, 0.5 * dt, N)
        self._rate(particles.position, p.str_temp, particles.radius, p.dstr_dt_temp2, mode_int, N)
        self._limit_rate(particles.position, p.str_temp, p.dstr_dt_temp2, dt, N)
        p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp2, p.str_temp, 0.5 * dt, N)
        self._rate(particles.position, p.str_temp, particles.radius, p.dstr_dt_temp3, mode_int, N)
        self._limit_rate(particles.position, p.str_temp, p.dstr_dt_temp3, dt, N)
        p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp3, p.str_temp, dt, N)
        self._rate(particles.position, p.str_temp, particles.radius, p.vel_temp, mode_int, N)
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

    def _limit_rate(self, positions, strengths, rates, dt: float, N: int) -> None:
        limiter = getattr(self._parent, "stretching_rate_limiter", None)
        if limiter is not None:
            limiter.apply_to_rate(positions, strengths, rates, dt, N)
