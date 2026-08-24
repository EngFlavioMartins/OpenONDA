"""
Pressure Gradient Physics Module for VPM Solver.
=================================================
Computes pressure gradients from VPM particle fields using the incompressible
Navier-Stokes momentum equation.

Physics Background
------------------
For incompressible flow, the pressure gradient is derived from:

    ∂u/∂t + (u·∇)u = -∇p/ρ + kinematic_viscosity∇²u

Rearranging:

    ∇p = -ρ [ ∂u/∂t + (u·∇)u - kinematic_viscosity∇²u ]

Three-Term Decomposition
------------------------
1. TEMPORAL TERM (∂u/∂t) - Analytical sum over particles:
   ∂u/∂t = Σᵢ [ -(∇uᵢ)·vᵢ + K(x, xᵢ, dαᵢ/dt) ]

   - First term: Motion contribution from particle advection
   - Second term: Biot-Savart velocity induced by stretching rate dαᵢ/dt
   - Stretching rate: dαᵢ/dt = (∇u)ᵀ · αᵢ (transpose mode)

2. ADVECTIVE TERM ((u·∇)u) - Field operation:
   Computed from total velocity u and velocity gradient ∇u at each point.

3. VISCOUS TERM (kinematic_viscosity∇²u) - Laplacian via finite differences:
   ∇²u ≈ [u(x+h) - 2u(x) + u(x-h)] / h² (for each direction)

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np
from scipy.special import erf
import taichi as ti

from ..config.constants import MAX_N_PARTICLES
from .base import PhysicsBase

# Minimum normalized distance for stable kernel evaluation
MIN_R_SIGMA_PRESSURE = 0.5
ONE_OVER_FOUR_PI = 0.0795774715459
TWO_OVER_SQRT_PI = 1.1283791671


def _q_kernel(normalized_distance: np.ndarray) -> np.ndarray:
    """Velocity kernel q(r/σ) - Gaussian regularized Biot-Savart.

    Args:
        normalized_distance: Normalized distance r/σ (can be array)

    Returns:
        Kernel values (same shape as input)
    """
    normalized_distance = np.atleast_1d(normalized_distance)
    result = np.zeros_like(normalized_distance, dtype=np.float64)

    # Small normalized_distance: series, since the closed form below cancels to nothing.
    # The coefficient is 4/(3 sqrt(pi)); it read 4/(3 sqrt(pi^3)) here — a factor
    # of pi, the same defect that was in the treecode's copy of this kernel.
    small = normalized_distance < 1e-4
    result[small] = (
        (4.0 / (3.0 * np.sqrt(np.pi))) * (normalized_distance[small] ** 3) * ONE_OVER_FOUR_PI
    )

    # Normal case: erf-based formula
    large = ~small
    erf_term = erf(normalized_distance[large])
    exp_term = (
        TWO_OVER_SQRT_PI * normalized_distance[large] * np.exp(-(normalized_distance[large] ** 2))
    )
    result[large] = (erf_term - exp_term) * ONE_OVER_FOUR_PI

    return result


def _zeta_kernel(normalized_distance: np.ndarray) -> np.ndarray:
    """Vorticity kernel ζ(r/σ) - Gaussian distribution.

    Args:
        normalized_distance: Normalized distance r/σ

    Returns:
        Kernel values
    """
    ONE_OVER_PI_15 = 0.179587122125
    return ONE_OVER_PI_15 * np.exp(-(normalized_distance**2))


@ti.data_oriented
class PressurePhysics(PhysicsBase):
    """
    Pressure gradient physics handler for VPM.

    Computes pressure gradients at particle locations or arbitrary target points
    using the Navier-Stokes momentum equation. The temporal term (∂u/∂t) is
    computed analytically from particle motion and stretching.

    Inherits from PhysicsBase:
    - Field evaluation (velocity, vorticity, gradients)
    - Temporary field management
    - Kernel initialization

    Methods:
        compute_pressure_gradients: Pressure gradient at particle locations
        compute_target_pressure_gradient: Pressure gradient at arbitrary points
    """

    def __init__(
        self,
        particle_kernel: str = "GAUSSIAN",
        max_n_particles: int = MAX_N_PARTICLES,
        accumulator_dtype: ti.types = ti.f32,
    ):  # type: ignore
        """
        Initialize pressure physics module.

        Args:
            particle_kernel: Kernel type for particle interactions
            max_n_particles: Maximum number of particles
            accumulator_dtype: Data type for accumulation
        """
        super().__init__(particle_kernel, max_n_particles, accumulator_dtype)

        # Additional temporary fields for pressure gradient computation
        self._initialize_pressure_fields()

    def _initialize_pressure_fields(self, size: int = None):
        """Initialize temporary fields for pressure gradient computation."""
        if size is None:
            size = self.max_n_particles

        # Pressure gradient output field
        self.pressure_gradient = ti.Vector.field(3, dtype=self.accumulator_dtype, shape=(size,))

        # Stretching rate temporary field (dα/dt for each particle)
        self.dalpha_dt_field = ti.Vector.field(3, dtype=self.accumulator_dtype, shape=(size,))

        # Target fields for pressure gradient at arbitrary points
        self.target_pressure_gradient = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(size,)
        )

        # Target position for GPU kernel
        self.pressure_target_position = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(size,)
        )

        # Temporal term output (du/dt at targets)
        self.temporal_term_field = ti.Vector.field(3, dtype=self.accumulator_dtype, shape=(size,))

        self._pressure_field_size = size

    def _resize_pressure_fields(self, N: int):
        """Validate that pressure evaluation fits the startup allocation."""
        if self._pressure_field_size >= N:
            return
        raise ValueError(
            f"Pressure evaluation requires {N} slots, but allocated capacity is "
            f"{self._pressure_field_size}. Increase max_n_particles before constructing "
            "the pressure evaluator."
        )

    # PRESSURE GRADIENT AT TARGET POSITIONS (Main Interface)

    def compute_target_pressure_gradient(
        self,
        particles,
        target_position: np.ndarray,
        density: float = 1.0,
        kinematic_viscosity: float = 1e-5,
        include_viscous: bool = True,
        include_temporal: bool = True,
        laplacian_spacing: float = None,
        include_freestream: bool = True,
        temporal_method: str = "lagrangian",
        velocity_previous: np.ndarray | None = None,
        time_step_size: float | None = None,
        return_velocity: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Compute pressure gradient at arbitrary target position.

        Uses the momentum equation:
            ∇p = -ρ [ ∂u/∂t + (u·∇)u - kinematic_viscosity∇²u ]

        Args:
            particles: Particle container
            target_position: Array of shape (M, 3) with target coordinates
            density: Fluid density [kg/m³]
            kinematic_viscosity: Kinematic viscosity [m²/s]
            include_viscous: Include viscous term (default True)
            include_temporal: Include temporal term (default True)
            laplacian_spacing: Step size for Laplacian. If None, uses average particle core radius.
            include_freestream: Include background velocity in computations
            temporal_method: 'lagrangian' (default, particle-based with motion term) or
                           'eulerian' (fixed-point backward differences)
            velocity_previous: Previous velocity field for Eulerian method (M, 3).
                             Required if temporal_method='eulerian'
            time_step_size: Time step for Eulerian method. Required if temporal_method='eulerian'
            return_velocity: If True, also return the internally computed target_velocity.
                           This allows callers to obtain a velocity snapshot from
                           the same code path used for the pressure gradient,
                           avoiding code-path mismatches when building an Eulerian
                           temporal term across consecutive calls.

        Returns:
            If return_velocity is False (default):
                np.ndarray: Pressure gradient at each target point [M, 3], units [Pa/m]
            If return_velocity is True:
                tuple[np.ndarray, np.ndarray]: (pressure_gradient [M,3], target_velocity [M,3])
        """
        N = particles.n_particles_total
        M = len(target_position)

        if N == 0 or M == 0:
            z = np.zeros((M, 3), dtype=np.float64)
            return (z, z) if return_velocity else z

        target_position = np.asarray(target_position, dtype=np.float64).reshape(-1, 3)

        # Default step size
        if laplacian_spacing is None:
            laplacian_spacing = float(np.mean(particles.core_radius_cpu()))

        # STEP 1: Compute velocity at target points
        target_velocity = self.compute_target_velocity(
            particles, target_position, include_freestream=include_freestream
        )

        # STEP 2: Compute velocity gradient at target points
        target_velocity_gradient = self.compute_target_velocity_gradient(
            particles, target_position
        ).reshape(M, 3, 3)

        # STEP 3: Compute advective term (u·∇)u at target
        # (u·∇)u_a = u_b * ∂u_a/∂x_b = u_b * velocity_gradient[a,b]
        advective = np.einsum("mb,mab->ma", target_velocity, target_velocity_gradient)

        # STEP 4: Compute temporal term ∂u/∂t
        temporal = self._resolve_temporal_term(
            particles,
            target_position,
            target_velocity,
            M,
            include_temporal,
            temporal_method,
            velocity_previous,
            time_step_size,
            include_freestream,
        )

        # STEP 5: Compute viscous term kinematic_viscosity∇²u (finite differences)
        if include_viscous and kinematic_viscosity > 0:
            viscous = self._compute_viscous_term(
                particles,
                target_position,
                kinematic_viscosity,
                laplacian_spacing,
                include_freestream,
            )
        else:
            viscous = np.zeros((M, 3), dtype=np.float64)

        # STEP 6: Combine: ∇p = -ρ [ ∂u/∂t + (u·∇)u - kinematic_viscosity∇²u ]
        pressure_gradient = -density * (temporal + advective - viscous)

        if return_velocity:
            return pressure_gradient, np.asarray(target_velocity, dtype=np.float64)
        return pressure_gradient

    def compute_target_pressure_gradient_components(
        self,
        particles,
        target_position: np.ndarray,
        density: float = 1.0,
        kinematic_viscosity: float = 1e-5,
        include_viscous: bool = True,
        include_temporal: bool = True,
        laplacian_spacing: float = None,
        include_freestream: bool = True,
        temporal_method: str = "lagrangian",
        velocity_previous: np.ndarray | None = None,
        time_step_size: float | None = None,
        return_velocity: bool = False,
    ) -> dict | tuple[dict, np.ndarray]:
        """
        Compute pressure gradient and its individual components at target position.

        Returns each term of the momentum equation separately for diagnostics:
            ∇p = -ρ [ ∂u/∂t + (u·∇)u - kinematic_viscosity∇²u ]

        Args:
            particles: Particle container
            target_position: Array of shape (M, 3) with target coordinates
            density: Fluid density [kg/m³]
            kinematic_viscosity: Kinematic viscosity [m²/s]
            include_viscous: Include viscous term (default True)
            include_temporal: Include temporal term (default True)
            laplacian_spacing: Step size for Laplacian. If None, uses average particle core radius.
            include_freestream: Include background velocity in computations
            temporal_method: 'lagrangian' (default) or 'eulerian'
            velocity_previous: Previous velocity field for Eulerian method (M, 3)
            time_step_size: Time step for Eulerian method
            return_velocity: If True, also return the internally computed target_velocity.

        Returns:
            If return_velocity is False (default):
                dict with keys: 'pressure_gradient', 'convective_pressure_gradient', 'viscous_pressure_gradient', 'temporal_pressure_gradient'
            If return_velocity is True:
                tuple[dict, np.ndarray]: (components_dict, target_velocity [M, 3])
        """
        N = particles.n_particles_total
        M = len(target_position)

        if N == 0 or M == 0:
            zeros = np.zeros((M, 3), dtype=np.float64)
            result = {
                "pressure_gradient": zeros.copy(),
                "convective_pressure_gradient": zeros.copy(),
                "viscous_pressure_gradient": zeros.copy(),
                "temporal_pressure_gradient": zeros.copy(),
            }
            return (result, zeros.copy()) if return_velocity else result

        target_position = np.asarray(target_position, dtype=np.float64).reshape(-1, 3)

        if laplacian_spacing is None:
            laplacian_spacing = float(np.mean(particles.core_radius_cpu()))

        # Velocity at targets
        target_velocity = self.compute_target_velocity(
            particles, target_position, include_freestream=include_freestream
        )

        # Velocity gradient at targets
        target_velocity_gradient = self.compute_target_velocity_gradient(
            particles, target_position
        ).reshape(M, 3, 3)

        # Advective term: (u·∇)u
        advective = np.einsum("mb,mab->ma", target_velocity, target_velocity_gradient)

        # Temporal term
        if include_temporal:
            if temporal_method == "eulerian":
                if velocity_previous is None or time_step_size is None:
                    raise ValueError(
                        "temporal_method='eulerian' requires velocity_previous and time_step_size"
                    )
                temporal = (target_velocity - velocity_previous) / time_step_size
            else:
                temporal = self._compute_temporal_term_with_particles(
                    particles, target_position, include_freestream
                )
        else:
            temporal = np.zeros((M, 3), dtype=np.float64)

        # Viscous term
        if include_viscous and kinematic_viscosity > 0:
            viscous = self._compute_viscous_term(
                particles,
                target_position,
                kinematic_viscosity,
                laplacian_spacing,
                include_freestream,
            )
        else:
            viscous = np.zeros((M, 3), dtype=np.float64)

        # Sign convention: each component is the contribution to ∇p
        convective_pressure_gradient = -density * advective  # -ρ(u·∇)u
        viscous_pressure_gradient = density * viscous  # +ρkinematic_viscosity∇²u
        temporal_pressure_gradient = -density * temporal  # -ρ∂u/∂t

        pressure_gradient = (
            convective_pressure_gradient + viscous_pressure_gradient + temporal_pressure_gradient
        )

        result = {
            "pressure_gradient": pressure_gradient,
            "convective_pressure_gradient": convective_pressure_gradient,
            "viscous_pressure_gradient": viscous_pressure_gradient,
            "temporal_pressure_gradient": temporal_pressure_gradient,
        }
        return (result, target_velocity) if return_velocity else result

    def compute_pressure_gradients(
        self,
        particles,
        density: float = 1.0,
        kinematic_viscosity: float = 1e-5,
        include_viscous: bool = True,
        include_temporal: bool = True,
        laplacian_spacing: float = None,
    ) -> np.ndarray:
        """
        Compute pressure gradient at all particle position.

        This is a convenience method that calls compute_target_pressure_gradient
        with particle position as target points.

        Args:
            particles: Particle container with velocity and velocity_gradient populated
            density: Fluid density [kg/m³]
            kinematic_viscosity: Kinematic viscosity [m²/s]
            include_viscous: Include viscous term kinematic_viscosity∇²u (default True)
            include_temporal: Include temporal term ∂u/∂t (default True)
            laplacian_spacing: Step size for Laplacian finite difference.
                        If None, uses average particle core radius.

        Returns:
            np.ndarray: Pressure gradient at each particle [N, 3], units [Pa/m]
        """
        N = particles.n_particles_total
        if N == 0:
            return np.zeros((0, 3), dtype=np.float64)

        target_position = particles.position_cpu()

        return self.compute_target_pressure_gradient(
            particles,
            target_position,
            density=density,
            kinematic_viscosity=kinematic_viscosity,
            include_viscous=include_viscous,
            include_temporal=include_temporal,
            laplacian_spacing=laplacian_spacing,
            include_freestream=True,
        )

    def compute_target_pressure_gradient_hierarchical(
        self,
        particles,
        target_position: np.ndarray,
        density: float = 1.0,
        kinematic_viscosity: float = 1e-5,
        include_viscous: bool = True,
        include_temporal: bool = True,
        include_freestream: bool = True,
        temporal_method: str = "eulerian",
        velocity_previous: np.ndarray | None = None,
        time_step_size: float | None = None,
        particle_spacing: float | None = None,
        return_velocity: bool = False,
        theta: float = 0.5,
        freestream_velocity: np.ndarray | None = None,
        body_fn=None,
    ) -> dict | tuple[dict, np.ndarray]:
        """
        Compute pressure-gradient terms at target points using a Barnes-Hut treecode.

        This is the O(N log N) hierarchical variant of
        ``compute_target_pressure_gradient``. It requires the Eulerian temporal
        method (``velocity_previous`` and ``time_step_size``) and evaluates the velocity at the
        targets plus the finite-difference offsets used for the viscous term in a
        single treecode pass.

        Args:
            particles: Particle container
            target_position: Array of shape (M, 3) with target coordinates
            density: Fluid density [kg/m³]
            kinematic_viscosity: Kinematic viscosity [m²/s]
            include_viscous: Include viscous term (default True)
            include_temporal: Include temporal term (default True)
            include_freestream: Include background velocity in computations
            temporal_method: Must be 'eulerian' (fixed-point backward differences)
            velocity_previous: Previous velocity field [M, 3]. Required if
                temporal_method='eulerian'
            time_step_size: Time step for Eulerian method. Required if temporal_method='eulerian'
            particle_spacing: Step size for the Laplacian finite difference. If None, uses average
                particle core radius
            return_velocity: If True, also return the internally computed velocity
            theta: Opening angle parameter for the treecode (smaller = more accurate)
            freestream_velocity: Freestream velocity [3] used when there are no
                particles
            body_fn: Callable(position) -> velocity for body-induced velocity

        Returns:
            dict with keys ``pressure_gradient``, ``convective``, ``viscous``, ``temporal``, or
            a ``(dict, velocity)`` tuple when ``return_velocity`` is True.
        """
        if temporal_method != "eulerian":
            raise ValueError("Treecode pressure gradients require temporal_method='eulerian'")
        N = particles.n_particles_total
        points = np.asarray(target_position, dtype=np.float64).reshape(-1, 3)
        count = len(points)
        targets = points
        if include_viscous:
            if particle_spacing is None:
                particle_spacing = float(np.mean(particles.core_radius_cpu())) if N > 0 else 1.0
            offsets = np.eye(3, dtype=np.float64) * float(particle_spacing)
            targets = np.concatenate(
                [
                    points,
                    *(points + offsets[j] for j in range(3)),
                    *(points - offsets[j] for j in range(3)),
                ]
            )
        velocity_samples = self.compute_target_velocity_hierarchical(
            particles,
            targets,
            theta=float(theta),
            include_freestream=include_freestream,
        ).astype(np.float64)
        if N == 0 and include_freestream:
            velocity_samples[:] = freestream_velocity
        velocity = velocity_samples[:count]
        gradient = self.compute_target_velocity_gradient_hierarchical(
            particles, points, theta=float(theta)
        ).reshape(count, 3, 3)
        if body_fn is not None:
            velocity_samples += np.asarray(body_fn(targets), dtype=velocity_samples.dtype).reshape(
                velocity_samples.shape
            )
            velocity = velocity_samples[:count]
            gradient_h = (
                float(particle_spacing)
                if particle_spacing is not None
                else (float(np.mean(particles.core_radius_cpu())) if N > 0 else 0.05)
            )
            for axis in range(3):
                offset = np.zeros(3, dtype=np.float64)
                offset[axis] = gradient_h
                plus = np.asarray(body_fn(points + offset), dtype=np.float64)
                minus = np.asarray(body_fn(points - offset), dtype=np.float64)
                gradient[:, :, axis] += (plus - minus) / (2.0 * gradient_h)
        advective = np.einsum("mb,mab->ma", velocity, gradient)
        temporal = np.zeros_like(velocity)
        if include_temporal:
            if velocity_previous is None or time_step_size is None:
                raise ValueError(
                    "Treecode pressure gradients require velocity_previous and time_step_size"
                )
            temporal = (velocity - velocity_previous) / float(time_step_size)
        viscous = np.zeros_like(velocity)
        if include_viscous and kinematic_viscosity > 0.0:
            plus = velocity_samples[count : 4 * count].reshape(3, count, 3)
            minus = velocity_samples[4 * count :].reshape(3, count, 3)
            viscous = (
                float(kinematic_viscosity)
                * np.sum(plus + minus - 2.0 * velocity[None, :, :], axis=0)
                / float(particle_spacing) ** 2
            )
        result = {
            "pressure_gradient": density * (-temporal - advective + viscous),
            "convective_pressure_gradient": -density * advective,
            "viscous_pressure_gradient": density * viscous,
            "temporal_pressure_gradient": -density * temporal,
        }
        return (result, velocity) if return_velocity else result

    # TEMPORAL TERM COMPUTATION (Analytical VPM formulation)

    def _resolve_temporal_term(
        self,
        particles,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        M: int,
        include_temporal: bool,
        temporal_method: str,
        velocity_previous: np.ndarray | None,
        time_step_size: float | None,
        include_freestream: bool,
    ) -> np.ndarray:
        """Select and compute the temporal term ∂u/∂t."""
        if not include_temporal:
            return np.zeros((M, 3), dtype=np.float64)
        if temporal_method == "eulerian":
            if velocity_previous is None or time_step_size is None:
                raise ValueError(
                    "temporal_method='eulerian' requires velocity_previous and time_step_size"
                )
            return (target_velocity - velocity_previous) / time_step_size
        return self._compute_temporal_term_with_particles(
            particles, target_position, include_freestream
        )

    def _compute_temporal_term_with_particles(
        self, particles, target_position: np.ndarray, include_freestream: bool
    ) -> np.ndarray:
        """
        Compute ∂u/∂t using particle container directly (GPU-accelerated).

        This version reads directly from particle fields on GPU for better performance.

        Args:
            particles: Particle container
            target_position: Evaluation points (M, 3)
            include_freestream: Include freestream velocity (unused, kept for API)

        Returns:
            np.ndarray: du/dt at each target point (M, 3)
        """
        M = len(target_position)
        N = particles.n_particles_total

        if N == 0:
            return np.zeros((M, 3), dtype=np.float64)

        # Ensure fields are sized correctly
        self._resize_pressure_fields(max(M, N))

        # Copy target position through fixed-shape external buffers so
        # diagnostics with changing sample counts do not accumulate staging
        # allocations on Vulkan/Metal.
        self._upload_vector_array(target_position, self.pressure_target_position, M)

        # Compute stretching rate: dα/dt = (∇u)ᵀ · α on GPU
        self._compute_stretching_rate_kernel(
            particles.vortex_strength, particles.velocity_gradient, self.dalpha_dt_field, N
        )

        # Compute temporal term on GPU
        self._compute_temporal_term_kernel_direct(
            self.pressure_target_position,
            particles.position,
            particles.velocity,
            particles.vortex_strength,
            particles.core_radius,
            self.dalpha_dt_field,
            self.temporal_term_field,
            M,
            N,
        )

        return self._download_vector_field(self.temporal_term_field, M).astype(np.float64)

    @ti.kernel
    def _compute_stretching_rate_kernel(
        self,
        vortex_strength: ti.template(),  # type: ignore
        velocity_gradient: ti.template(),  # type: ignore
        dalpha_dt_out: ti.template(),  # type: ignore
        N: ti.i32,  # type: ignore
    ):
        """Compute dα/dt = (∇u)ᵀ · α using pre-computed velocity gradients.

        The transpose formulation (∇u)ᵀ · α conserves vortex strength better than
        the classical (∇u) · α formulation.
        """
        for i in range(N):
            alpha_i = vortex_strength[i]
            grad_u_i = velocity_gradient[i]

            # Compute (∇u)ᵀ · α = J^T * α
            # velocity_gradient[a, b] = ∂u_a/∂x_b
            # (velocity_gradient^T)[a, b] = ∂u_b/∂x_a
            dalpha = ti.Vector([0.0, 0.0, 0.0])
            for a in ti.static(range(3)):
                for b in ti.static(range(3)):
                    # (J^T)_{ab} * alpha_b = velocity_gradient[b,a] * alpha_b
                    dalpha[a] += grad_u_i[b, a] * alpha_i[b]

            dalpha_dt_out[i] = dalpha

    @ti.kernel
    def _compute_temporal_term_kernel_direct(
        self,
        target_position: ti.template(),  # type: ignore
        position: ti.template(),  # type: ignore
        velocity: ti.template(),  # type: ignore
        vortex_strength: ti.template(),  # type: ignore
        core_radius: ti.template(),  # type: ignore
        dalpha_dt: ti.template(),  # type: ignore
        du_dt_out: ti.template(),  # type: ignore
        M: ti.i32,  # type: ignore
        N: ti.i32,  # type: ignore
    ):
        """
        Compute temporal term ∂u/∂t at target position (GPU kernel).

        For each target m, sums contributions from all particles:
            du/dt = Σᵢ [ -(∇uᵢ)·vᵢ + K(x, xᵢ, dαᵢ/dt) ]

        Complexity: O(M×N) on GPU (parallel over M targets)
        """
        # Constants
        ONE_OVER_FOUR_PI = 0.0795774715
        TWO_OVER_SQRT_PI = 1.1283791671
        ONE_OVER_PI_15 = 0.179587122125
        MIN_RHO = 0.5
        EPS = 1e-10

        for m in range(M):
            target_pos = target_position[m]
            du_dt = ti.Vector([0.0, 0.0, 0.0])

            for i in range(N):
                r_vec = target_pos - position[i]
                r_mag = r_vec.norm()
                sigma = core_radius[i]

                if r_mag > EPS:
                    normalized_distance = r_mag / sigma

                    if normalized_distance > MIN_RHO:
                        # Compute kernel values
                        # q(normalized_distance) = [erf(normalized_distance) - (2/√π) * normalized_distance * exp(-normalized_distance²)] / (4π)
                        erf_val = self._erf_approx(normalized_distance)
                        exp_val = ti.exp(-normalized_distance * normalized_distance)
                        q_val = (
                            erf_val - TWO_OVER_SQRT_PI * normalized_distance * exp_val
                        ) * ONE_OVER_FOUR_PI

                        # zeta(normalized_distance) = exp(-normalized_distance²) / π^1.5
                        zeta_val = ONE_OVER_PI_15 * exp_val

                        r_mag_cubed = r_mag * r_mag * r_mag
                        r_mag_fifth = r_mag_cubed * r_mag * r_mag

                        # Coefficients for velocity gradient
                        term1 = q_val / r_mag_cubed
                        term2 = 3.0 * q_val / r_mag_fifth - zeta_val / (
                            sigma * sigma * sigma * r_mag * r_mag
                        )

                        vortex_strength_i = vortex_strength[i]
                        vel_i = velocity[i]
                        dalpha_i = dalpha_dt[i]

                        # r × Γ
                        r_cross_vortex_strength = r_vec.cross(vortex_strength_i)

                        # ---------------------------------------------------------
                        # MOTION CONTRIBUTION: -(∇uᵢ)·vᵢ
                        # Build velocity gradient and multiply by -v_i in one step
                        # ---------------------------------------------------------
                        # Skew matrix contribution: velocity_gradient[a,b] from cross product terms
                        # velocity_gradient[0,1] = -Γz * term1,  velocity_gradient[0,2] = Γy * term1
                        # velocity_gradient[1,0] = Γz * term1,   velocity_gradient[1,2] = -Γx * term1
                        # velocity_gradient[2,0] = -Γy * term1,  velocity_gradient[2,1] = Γx * term1

                        # Compute -(velocity_gradient) @ v_i component by component
                        motion = ti.Vector([0.0, 0.0, 0.0])

                        # Row 0: velocity_gradient[0,:] @ v = 0*v[0] + (-Γz*term1)*v[1] + (Γy*term1)*v[2]
                        #        + term2 * (r×Γ)[0] * (r·v)
                        r_dot_v = r_vec[0] * vel_i[0] + r_vec[1] * vel_i[1] + r_vec[2] * vel_i[2]

                        motion[0] = (-vortex_strength_i[2] * term1) * vel_i[1] + (
                            vortex_strength_i[1] * term1
                        ) * vel_i[2]
                        motion[0] += term2 * r_cross_vortex_strength[0] * r_dot_v

                        # Row 1: velocity_gradient[1,:] @ v = (Γz*term1)*v[0] + 0*v[1] + (-Γx*term1)*v[2]
                        motion[1] = (vortex_strength_i[2] * term1) * vel_i[0] + (
                            -vortex_strength_i[0] * term1
                        ) * vel_i[2]
                        motion[1] += term2 * r_cross_vortex_strength[1] * r_dot_v

                        # Row 2: velocity_gradient[2,:] @ v = (-Γy*term1)*v[0] + (Γx*term1)*v[1] + 0*v[2]
                        motion[2] = (-vortex_strength_i[1] * term1) * vel_i[0] + (
                            vortex_strength_i[0] * term1
                        ) * vel_i[1]
                        motion[2] += term2 * r_cross_vortex_strength[2] * r_dot_v

                        du_dt -= motion  # Negative sign: -(∇u)·v

                        # ---------------------------------------------------------
                        # STRETCHING CONTRIBUTION: K(x, xᵢ, dαᵢ/dt)
                        # Biot-Savart velocity using stretching rate as strength
                        # u_stretch = -q/r³ * (r × dα/dt)
                        # ---------------------------------------------------------
                        r_cross_dalpha = r_vec.cross(dalpha_i)
                        du_dt -= (q_val / r_mag_cubed) * r_cross_dalpha

            du_dt_out[m] = du_dt

    @ti.func
    def _erf_approx(self, x: ti.f32) -> ti.f32:  # type: ignore
        """Abramowitz & Stegun approximation for error function (Taichi func)."""
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        approximation_coefficient = 0.327591100

        sign = 1.0
        x_abs = x
        if x < 0:
            sign = -1.0
            x_abs = -x
        t = 1.0 / (1.0 + approximation_coefficient * x_abs)
        y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * ti.exp(-x_abs * x_abs))
        return sign * y

    # VISCOUS TERM COMPUTATION (via finite differences)

    def _compute_viscous_term(
        self,
        particles,
        target_position: np.ndarray,
        kinematic_viscosity: float,
        particle_spacing: float,
        include_freestream: bool = True,
    ) -> np.ndarray:
        """
        Compute viscous term kinematic_viscosity∇²u at target position using finite differences.

        The Laplacian is approximated using a 6-point stencil:
            ∂²u/∂x² ≈ [u(x+h) - 2u(x) + u(x-h)] / h²

        Args:
            particles: Particle container
            target_position: Evaluation points [M, 3]
            kinematic_viscosity: Kinematic viscosity [m²/s]
            particle_spacing: Finite difference step size [m]
            include_freestream: Include background velocity

        Returns:
            np.ndarray: Viscous term kinematic_viscosity∇²u of shape [M, 3]
        """
        M = len(target_position)
        if M == 0:
            return np.zeros((0, 3), dtype=np.float64)

        target_position = np.asarray(target_position, dtype=np.float64)
        particle_spacing_sq = particle_spacing * particle_spacing

        # Central velocity
        centre_velocity = self.compute_target_velocity(
            particles, target_position, include_freestream=include_freestream
        )

        laplacian = np.zeros((M, 3), dtype=np.float64)

        # X-direction
        pts_xp = target_position.copy()
        pts_xp[:, 0] += particle_spacing
        pts_xm = target_position.copy()
        pts_xm[:, 0] -= particle_spacing
        u_xp = self.compute_target_velocity(
            particles, pts_xp, include_freestream=include_freestream
        )
        u_xm = self.compute_target_velocity(
            particles, pts_xm, include_freestream=include_freestream
        )
        laplacian += (u_xp - 2 * centre_velocity + u_xm) / particle_spacing_sq

        # Y-direction
        pts_yp = target_position.copy()
        pts_yp[:, 1] += particle_spacing
        pts_ym = target_position.copy()
        pts_ym[:, 1] -= particle_spacing
        u_yp = self.compute_target_velocity(
            particles, pts_yp, include_freestream=include_freestream
        )
        u_ym = self.compute_target_velocity(
            particles, pts_ym, include_freestream=include_freestream
        )
        laplacian += (u_yp - 2 * centre_velocity + u_ym) / particle_spacing_sq

        # Z-direction
        pts_zp = target_position.copy()
        pts_zp[:, 2] += particle_spacing
        pts_zm = target_position.copy()
        pts_zm[:, 2] -= particle_spacing
        u_zp = self.compute_target_velocity(
            particles, pts_zp, include_freestream=include_freestream
        )
        u_zm = self.compute_target_velocity(
            particles, pts_zm, include_freestream=include_freestream
        )
        laplacian += (u_zp - 2 * centre_velocity + u_zm) / particle_spacing_sq

        return kinematic_viscosity * laplacian
