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

import numpy as np
import taichi as ti

from ..config.constants import MAX_PARTICLES
from .base import PhysicsBase
from .diffusion.core_spreading import apply_core_spreading
from .diffusion.grid import _GridDiffusionMixin
from .diffusion.random_walk import apply_random_walk

_DIRECT_STRETCHING_BATCH_SIZE = 4096


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
        max_targets: int = 200000,
    ):
        """Initialize the unified physics engine."""
        # Initialize base classes
        super().__init__(particles_kernel, max_particles, accumulator_dtype, max_targets)
        self._init_grid_diffusion()

        # Optional advection velocity override: fn(pos (N,3), vel_bs (N,3)) -> (N,3).
        # Applied at every RK stage after Biot-Savart; used by the FVM-VPM coupler
        # to blend in the FVM near-body velocity for overlap-region particles.
        self.velocity_override = None
        self.body_velocity = None

        # Create specialized physics handlers
        self._advection = _AdvectionHandler(self)
        self._diffusion = _DiffusionHandler(self)
        self._stretching = _StretchingHandler(self)
        self._coupled = _CoupledAdvectionStretchingHandler(self)

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

    def update_positions_and_strengths(
        self,
        particles,
        dt: float,
        scheme: str = "RK3",
        mode: str = "TRANSPOSED",
        use_treecode: bool = False,
        treecode_theta: float = 0.3,
        conserve_moments: bool = False,
        conserve_energy: bool = False,
        axisymmetric_axis: int = -1,
        precomputed_velocity_k1: bool = False,
    ):
        """Advance particle positions and vortex strengths at common RK stages.

        A fractional ``x``-then-``Gamma`` update evaluates the stretching
        equation on positions from a different time level than the velocity
        equation.  This coupled method instead treats ``(x, Gamma)`` as one ODE
        state and evaluates both right-hand sides at every RK stage.
        """
        self._coupled.update(
            particles,
            dt,
            scheme,
            mode,
            use_treecode,
            treecode_theta,
            conserve_moments,
            conserve_energy,
            axisymmetric_axis,
            precomputed_velocity_k1,
        )

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

    def _vel(
        self,
        particles,
        pos_field,
        out_field,
        N,
        reuse_tree=False,
        strength_field=None,
    ):
        """Self-induced velocity at ``pos_field`` → ``out_field`` (honors method).

        ``reuse_tree=True`` (RK stages ≥ 2) refits the stage-1 LBVH topology to
        the displaced positions instead of rebuilding it — same physics, ~half
        the per-stage tree cost.  Not used when a velocity override (FVM blend)
        is active, since that path round-trips through the CPU anyway.
        """
        self._parent.velocity_self(
            pos_field,
            particles.circulation if strength_field is None else strength_field,
            particles.radius,
            out_field,
            particles.velocity_background,
            N,
            reuse_tree=reuse_tree and self._parent.velocity_override is None,
        )
        body_velocity = self._parent.body_velocity
        if body_velocity is not None or self._parent.velocity_override is not None:
            pos_np = pos_field.to_numpy()
            vel_np = out_field.to_numpy()
            if body_velocity is not None:
                vel_np[:N] += np.asarray(body_velocity(pos_np[:N]), dtype=vel_np.dtype).reshape(
                    N, 3
                )
            override = self._parent.velocity_override
            if override is not None:
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


class _CoupledAdvectionStretchingHandler:
    """Run advection and stretching as a single method-of-lines system."""

    def __init__(self, parent: PhysicsEngine):
        self._parent = parent

    @staticmethod
    def _mode_int(mode: str) -> int:
        value = mode.upper()
        if value == "DIRECT":
            return 0
        if value == "TRANSPOSED":
            return 1
        if value == "MIXED":
            return 2
        raise ValueError(f"Unknown stretching mode: {mode}. Use DIRECT, TRANSPOSED, or MIXED.")

    def update(
        self,
        particles,
        dt: float,
        scheme: str,
        mode: str,
        use_treecode: bool,
        treecode_theta: float,
        conserve_moments: bool,
        conserve_energy: bool,
        axisymmetric_axis: int,
        precomputed_velocity_k1: bool,
    ) -> None:
        N = len(particles)
        scheme = scheme.upper()
        if N == 0 or dt == 0.0:
            return
        if scheme not in {"RK2", "RK3"}:
            raise ValueError("Coupled advection/stretching supports RK2 and RK3.")

        p = self._parent
        p._resize_temp_fields(N)
        p._zero_temp_fields(N)
        p._stretching._use_treecode = bool(use_treecode) and p.velocity_method == "TREECODE"
        p._stretching._treecode_theta = float(treecode_theta)
        mode_int = self._mode_int(mode)

        # k1 = f(x_n, Gamma_n)
        self._stage_rhs(
            particles,
            particles.position,
            particles.circulation,
            particles.velocity,
            p.dstr_dt_temp,
            mode_int,
            N,
            precomputed=precomputed_velocity_k1,
            conserve_moments=conserve_moments,
            conserve_energy=conserve_energy,
            axisymmetric_axis=axisymmetric_axis,
        )

        # y1 = y_n + dt*k1
        p.step_euler_forward_kernel(particles.position, particles.velocity, p.pos_temp, dt, N)
        p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp, p.str_temp, dt, N)

        # k2 = f(y1).  A tree topology cannot merely be refitted here because
        # coupled stages change both positions and circulations.
        self._stage_rhs(
            particles,
            p.pos_temp,
            p.str_temp,
            p.vel_temp,
            p.dstr_dt_temp2,
            mode_int,
            N,
            precomputed=False,
            conserve_moments=conserve_moments,
            conserve_energy=conserve_energy,
            axisymmetric_axis=axisymmetric_axis,
        )

        if scheme == "RK2":
            p.step_rk2_combine_kernel(particles.position, particles.velocity, p.vel_temp, dt, N)
            p.step_rk2_combine_kernel(particles.circulation, p.dstr_dt_temp, p.dstr_dt_temp2, dt, N)
            return

        # SSP-RK3 stage y2 = y_n + dt/4*(k1+k2)
        p.linear_combination_kernel(
            p.pos_temp2, particles.velocity, p.vel_temp, 0.25 * dt, 0.25 * dt, N
        )
        p.step_euler_forward_kernel(particles.position, p.pos_temp2, p.pos_temp2, 1.0, N)
        p.linear_combination_kernel(
            p.str_temp2, p.dstr_dt_temp, p.dstr_dt_temp2, 0.25 * dt, 0.25 * dt, N
        )
        p.step_euler_forward_kernel(particles.circulation, p.str_temp2, p.str_temp2, 1.0, N)

        # k3 = f(y2)
        self._stage_rhs(
            particles,
            p.pos_temp2,
            p.str_temp2,
            p.vel_temp2,
            p.dstr_dt_temp3,
            mode_int,
            N,
            precomputed=False,
            conserve_moments=conserve_moments,
            conserve_energy=conserve_energy,
            axisymmetric_axis=axisymmetric_axis,
        )

        p.step_rk3_ssp_combine_kernel(
            particles.position,
            particles.velocity,
            p.vel_temp,
            p.vel_temp2,
            dt,
            N,
        )
        p.step_rk3_ssp_combine_kernel(
            particles.circulation,
            p.dstr_dt_temp,
            p.dstr_dt_temp2,
            p.dstr_dt_temp3,
            dt,
            N,
        )

    def _stage_rhs(
        self,
        particles,
        position,
        strength,
        velocity_out,
        rate_out,
        mode_int: int,
        N: int,
        *,
        precomputed: bool,
        conserve_moments: bool,
        conserve_energy: bool,
        axisymmetric_axis: int,
    ) -> None:
        """Evaluate velocity and stretching from one common particle state."""
        p = self._parent
        tree_stretching = p._stretching._use_treecode
        matching_evaluator = tree_stretching == (p.velocity_method == "TREECODE")
        no_velocity_hook = p.velocity_override is None and p.body_velocity is None
        # The direct pairwise TRANSPOSED kernel is antisymmetric to roundoff;
        # contracting the separately accumulated (currently f32) gradient is
        # mathematically equivalent but loses that algebraic cancellation.
        # Keep the exact pair kernel unless the moment projection will restore
        # the invariants anyway.  Treecode stretching has no pairwise
        # antisymmetry to preserve and always benefits from the fused traversal.
        can_share_gradient = (
            matching_evaluator
            and no_velocity_hook
            and (p.velocity_method == "TREECODE" or conserve_moments or conserve_energy)
        )

        gradient = None
        if precomputed and can_share_gradient:
            gradient = particles.velocity_gradient
        elif can_share_gradient:
            if p.velocity_method == "TREECODE":
                theta = min(p.velocity_theta, p._stretching._treecode_theta)
                tree = p._get_or_create_treecode(N, theta)
                tree.build(position, strength, particles.radius, N)
                p._target_tree_key = None
                background = particles.velocity_background
                background_np = np.array(
                    [background[None][0], background[None][1], background[None][2]],
                    dtype=np.float32,
                )
                tree.compute_velocity_and_gradient_gpu(background_np)
                p._copy_vec3(tree.velocities, velocity_out, N)
                gradient = tree.velocity_gradients
            else:
                p.compute_velocity_and_gradient_kernel(
                    position,
                    strength,
                    particles.radius,
                    velocity_out,
                    particles.velocity_gradient,
                    particles.strain_rate,
                    particles.velocity_background,
                    N,
                )
                gradient = particles.velocity_gradient
        else:
            if not precomputed:
                p._advection._vel(
                    particles,
                    position,
                    velocity_out,
                    N,
                    strength_field=strength,
                )
            p._stretching._rate(
                position,
                strength,
                particles.radius,
                rate_out,
                mode_int,
                N,
            )

        if gradient is not None:
            p.gradient_contraction_rate_kernel(gradient, strength, rate_out, mode_int, N)
        if axisymmetric_axis >= 0:
            p.average_axisymmetric_no_swirl_rhs(
                position,
                velocity_out,
                rate_out,
                particles.zone_id,
                axisymmetric_axis,
                N,
            )
        if conserve_moments or conserve_energy:
            p.conserve_rate_moments(
                position,
                strength,
                particles.radius,
                velocity_out,
                rate_out,
                N,
                conserve_energy=conserve_energy,
            )
        if axisymmetric_axis >= 0 and (conserve_moments or conserve_energy):
            # The invariant projection is rotationally equivariant in exact
            # arithmetic. Re-average its f32 reduction noise so later stages
            # cannot leave the declared symmetry manifold.
            p.average_axisymmetric_no_swirl_rhs(
                position,
                velocity_out,
                rate_out,
                particles.zone_id,
                axisymmetric_axis,
                N,
            )


class _DiffusionHandler:
    """
    Lightweight diffusion handler that uses parent's resources.
    """

    def __init__(self, parent: PhysicsEngine):
        self._parent = parent

    def core_spreading_diffusion(self, particles, dt: float):
        """Core spreading diffusion."""
        apply_core_spreading(self._parent, particles, dt)

    def random_walk_method_diffusion(self, particles, dt: float):
        """Random walk diffusion."""
        apply_random_walk(self._parent, particles, dt)


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
        p._zero_temp_fields(N)

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

    def _rate(self, pos, strg, rad, out, mode_int, N, *, reuse_tree: bool = False):
        """Stretching rate dΓ/dt at (pos, strg): direct pairwise or treecode.

        Direct: the O(N²) pairwise kernel.  Treecode: build the LBVH at
        (pos, strg), evaluate the velocity gradient J = ∇u (O(N log N)), and
        contract it locally — J·Γ (DIRECT), Jᵀ·Γ (TRANSPOSED) or S·Γ (MIXED).
        The two agree up to the Barnes–Hut opening-angle tolerance.
        """
        p = self._parent
        if self._use_treecode:
            tree = p._get_or_create_treecode(N, self._treecode_theta)
            if reuse_tree and p.reuse_tree_topology:
                tree.refit_circulation(strg, N)
            else:
                tree.build(pos, strg, rad, N)
            p._target_tree_key = None
            tree.compute_velocity_gradients_gpu()
            p.gradient_contraction_rate_kernel(tree.velocity_gradients, strg, out, mode_int, N)
        else:
            # A single N-target direct dispatch becomes a multi-second Vulkan
            # kernel for coupled clouds.  Bound each submission and drain it so
            # the driver cannot overlap successive O(N²) RK stages or trip its
            # watchdog.  Target batches are independent and preserve the exact
            # per-target source accumulation order.
            for start in range(0, N, _DIRECT_STRETCHING_BATCH_SIZE):
                count = min(_DIRECT_STRETCHING_BATCH_SIZE, N - start)
                p.compute_stretching_rate_batch_kernel(
                    pos, strg, rad, out, mode_int, start, count, N
                )
                ti.sync()

    def _stretching_rk2(self, particles, dt, mode_int, N):
        """RK2 stretching."""
        p = self._parent
        self._rate(
            particles.position, particles.circulation, particles.radius, p.dstr_dt_temp, mode_int, N
        )
        p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp, p.str_temp, dt, N)
        # The next rate reads ``str_temp``.  Keep that RK dependency explicit
        # across backends; on Vulkan it also separates two expensive direct
        # stretching dispatch sequences.
        ti.sync()
        self._rate(
            particles.position,
            p.str_temp,
            particles.radius,
            p.dstr_dt_temp2,
            mode_int,
            N,
            reuse_tree=True,
        )
        p.step_rk2_combine_kernel(particles.circulation, p.dstr_dt_temp, p.dstr_dt_temp2, dt, N)

    def _stretching_rk3(self, particles, dt, mode_int, N):
        """RK3 stretching."""
        p = self._parent
        self._rate(
            particles.position, particles.circulation, particles.radius, p.dstr_dt_temp, mode_int, N
        )
        p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp, p.str_temp, dt, N)
        ti.sync()
        self._rate(
            particles.position,
            p.str_temp,
            particles.radius,
            p.dstr_dt_temp2,
            mode_int,
            N,
            reuse_tree=True,
        )
        p.linear_combination_kernel(
            p.str_temp2, p.dstr_dt_temp, p.dstr_dt_temp2, 0.25 * dt, 0.25 * dt, N
        )
        p.step_euler_forward_kernel(particles.circulation, p.str_temp2, p.str_temp2, 1.0, N)
        ti.sync()
        self._rate(
            particles.position,
            p.str_temp2,
            particles.radius,
            p.dstr_dt_temp3,
            mode_int,
            N,
            reuse_tree=True,
        )
        p.step_rk3_ssp_combine_kernel(
            particles.circulation, p.dstr_dt_temp, p.dstr_dt_temp2, p.dstr_dt_temp3, dt, N
        )

    def _stretching_rk4(self, particles, dt, mode_int, N):
        """RK4 stretching."""
        p = self._parent
        self._rate(
            particles.position, particles.circulation, particles.radius, p.dstr_dt_temp, mode_int, N
        )
        p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp, p.str_temp, 0.5 * dt, N)
        ti.sync()
        self._rate(
            particles.position,
            p.str_temp,
            particles.radius,
            p.dstr_dt_temp2,
            mode_int,
            N,
            reuse_tree=True,
        )
        p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp2, p.str_temp, 0.5 * dt, N)
        ti.sync()
        self._rate(
            particles.position,
            p.str_temp,
            particles.radius,
            p.dstr_dt_temp3,
            mode_int,
            N,
            reuse_tree=True,
        )
        p.step_euler_forward_kernel(particles.circulation, p.dstr_dt_temp3, p.str_temp, dt, N)
        ti.sync()
        self._rate(
            particles.position,
            p.str_temp,
            particles.radius,
            p.vel_temp,
            mode_int,
            N,
            reuse_tree=True,
        )
        p.step_rk4_combine_kernel(
            particles.circulation,
            p.dstr_dt_temp,
            p.dstr_dt_temp2,
            p.dstr_dt_temp3,
            p.vel_temp,
            dt,
            N,
        )
