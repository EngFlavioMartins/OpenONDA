"""
Physics Base Module for VPM Solver.
====================================
Contains shared functionality for all physics modules:
- Field evaluation (velocity, vorticity, gradients)
- Temporary field management
- Kernel initialization

This module eliminates code duplication between the advection, diffusion,
and stretching handlers (in engine.py) by providing a common base class.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import numpy as np
import taichi as ti

from ..config.constants import DEFAULT_CUTOFF_RADIUS_FACTOR, EPSILON, MAX_N_PARTICLES

# Smallest treecode capacity worth allocating.  Below this the fixed per-node
# cost is irrelevant and the doubling would just add rebuilds.
_TREECODE_MIN_CAPACITY = 8192

# Fixed host-side transfer shape for NumPy <-> Taichi ndarray kernels.
#
# Vulkan and Metal both route ndarray arguments through backend staging buffers.
# Passing a fresh external array shape for every sampler/particle count can make
# those staging allocations accumulate on Taichi 1.7.x.  Keep all hot transfers
# at this shape and pass the active count separately.
_HOST_TRANSFER_CHUNK_SIZE = 65536

# =========================================================
# PHYSICS BASE CLASS
# =========================================================


@ti.data_oriented
class PhysicsBase:
    """
    Base class for VPM physics modules providing shared functionality.

    This class contains:
    - Temporary field initialization and management
    - Target field management (for at-point evaluations)
    - Kernel initialization and binding
    - Field evaluation methods (velocity, vorticity, gradients)

    The concrete PhysicsEngine delegates to internal handlers for advection,
    diffusion, and stretching; the unified self-induced velocity operator
    (compute_self_induced_velocity / configure_velocity) lives here so every handler shares it.
    """

    def __init__(
        self,
        particle_kernel: str = "GAUSSIAN",
        max_n_particles: int = MAX_N_PARTICLES,
        accumulator_dtype: ti.types = ti.f32,
        max_evaluation_points: int = 200000,
    ):
        """
        Initialize physics base with kernel type and field allocation.

        Args:
            particle_kernel: Kernel type for particle interactions
                            Options: 'GAUSSIAN', 'SUPER_GAUSSIAN', 'WINCKELMANS'
            max_n_particles: Maximum number of particles for field allocation
            accumulator_dtype: Data type for accumulation (ti.f32 or ti.f64)
            max_evaluation_points: Fixed capacity for at-point field evaluations
        """
        self.particle_kernel = particle_kernel.upper()
        self.max_n_particles = max_n_particles
        self.accumulator_dtype = accumulator_dtype
        self.np_dtype = np.float32 if accumulator_dtype == ti.f32 else np.float64
        self.max_evaluation_points = int(max_evaluation_points)
        if self.max_evaluation_points < 1:
            raise ValueError("max_evaluation_points must be at least 1")

        # Determine compute dtype from accumulator dtype
        compute_dtype = accumulator_dtype

        # Zero velocity field for non-freestream computations
        self._zero_velocity = ti.Vector.field(3, dtype=compute_dtype, shape=())
        self._zero_velocity[None] = [0.0, 0.0, 0.0]

        # Track allocated sizes to avoid unnecessary reallocations
        self._temp_field_size = 0
        self._target_field_size = 0

        # Cached treecode instance (to avoid memory leak from repeated allocations)
        self._treecode = None
        self._treecode_max_particles = 0
        # A target-query phase often asks for panel, blending, and coupling-face
        # fields from one immutable particle state.  Keep one source-tree key so
        # those traversals share the LBVH instead of rebuilding it per caller.
        self._target_tree_key = None

        # Velocity-evaluation method — the single source of truth for how the
        # self-induced velocity is computed (see compute_self_induced_velocity()).  Set once by
        # the solver via configure_velocity(); defaults to direct O(N²) summation.
        # DIRECT: exact O(N²); use VelocityConfig.treecode(theta=0.5) for N ≳ 5 000 (~5% error)
        self.velocity_method = "DIRECT"  # "DIRECT" | "TREECODE"
        self.velocity_theta = 0.3  # Barnes-Hut opening angle (treecode only)
        self.treecode_multipole_order = 1
        self.treecode_sort_particle_targets = False
        self.treecode_traversal_block_dim = 128

        # Reuse the stage-1 LBVH topology across the later RK advection stages
        # (refit vs full rebuild).  On by default; a safe escape hatch / bench
        # toggle — setting it False reverts to a full tree build at every stage.
        self.reuse_tree_topology = True

        # Cached filtered particle fields for zone-aware BC computation
        self._filtered_field_size = self.max_n_particles
        self._filtered_pos = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_n_particles,)
        )
        self._filtered_vortex_strength = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_n_particles,)
        )
        self._filtered_rad = ti.field(dtype=self.accumulator_dtype, shape=(self.max_n_particles,))

        self._host_vector_chunks = {}
        self._host_scalar_chunks = {}
        self._host_matrix_chunks = {}

        # The first nine rows constrain vector vortex strength, linear impulse, and
        # kernel-corrected angular impulse.  Row ten optionally constrains the
        # exact inviscid rate of the discrete blob energy.
        self._rate_constraint_defect = ti.field(dtype=compute_dtype, shape=(10,))
        self._rate_constraint_gram = ti.field(dtype=compute_dtype, shape=(10, 10))
        self._rate_constraint_multiplier = ti.field(dtype=compute_dtype, shape=(10,))
        self._rate_projection_original_norm_sq = ti.field(dtype=compute_dtype, shape=())
        self._rate_energy_gradient = ti.Vector.field(
            3, dtype=compute_dtype, shape=(self.max_n_particles,)
        )
        self.rate_projection_correction_ratio = 0.0
        self.rate_projection_max_correction_ratio = 0.0
        self._axisymmetric_vector_sum_a = ti.Vector.field(
            3, dtype=compute_dtype, shape=(self.max_n_particles,)
        )
        self._axisymmetric_vector_sum_b = ti.Vector.field(
            3, dtype=compute_dtype, shape=(self.max_n_particles,)
        )
        self._axisymmetric_scalar_sum = ti.field(dtype=compute_dtype, shape=(self.max_n_particles,))
        self._axisymmetric_orbit_count = ti.field(dtype=ti.i32, shape=(self.max_n_particles,))
        self._angular_core_coefficient = (
            0.0
            if self.particle_kernel
            in {
                "HIGH_ORDER_GAUSSIAN",
                "SUPER_GAUSSIAN",
            }
            else 1.0 / 3.0
        )

        # Initialize Taichi fields
        self._initialize_temp_fields()
        self._initialize_target_fields(self.max_evaluation_points)

        # Bind kernel functions based on particle kernel type
        self._define_taichi_kernels()

    # TEMPORARY FIELD MANAGEMENT

    def _initialize_temp_fields(self):
        """
        Initialize temporary Taichi fields for intermediate calculations.

        These fields are used for:
        - Multi-stage time integrators (RK2, RK3, RK4)
        - Intermediate velocity/vorticity storage
        - Strength rate computations
        """
        # Position/velocity temporaries for advection
        self.pos_temp = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_n_particles,)
        )
        self.pos_temp2 = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_n_particles,)
        )
        self.vel_temp = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_n_particles,)
        )
        self.vel_temp2 = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_n_particles,)
        )

        # Strength temporaries for stretching/diffusion
        self.str_temp = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_n_particles,)
        )
        self.str_temp2 = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_n_particles,)
        )

        # Strength rate temporaries for RK integration
        self.dstr_dt_temp = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_n_particles,)
        )
        self.dstr_dt_temp2 = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_n_particles,)
        )
        self.dstr_dt_temp3 = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_n_particles,)
        )

        # Mark initial size
        self._temp_field_size = self.max_n_particles

    @ti.kernel
    def _zero_temp_fields_kernel(self, N: ti.i32):
        for i in range(N):
            self.pos_temp[i] = ti.Vector([0.0, 0.0, 0.0])
            self.pos_temp2[i] = ti.Vector([0.0, 0.0, 0.0])
            self.vel_temp[i] = ti.Vector([0.0, 0.0, 0.0])
            self.vel_temp2[i] = ti.Vector([0.0, 0.0, 0.0])
            self.str_temp[i] = ti.Vector([0.0, 0.0, 0.0])
            self.str_temp2[i] = ti.Vector([0.0, 0.0, 0.0])
            self.dstr_dt_temp[i] = ti.Vector([0.0, 0.0, 0.0])
            self.dstr_dt_temp2[i] = ti.Vector([0.0, 0.0, 0.0])
            self.dstr_dt_temp3[i] = ti.Vector([0.0, 0.0, 0.0])

    def _zero_temp_fields(self, N: int | None = None):
        """Zero the temporaries over the active range only.

        ``field.fill(0)`` covers the whole startup allocation (``max_n_particles``),
        which is 9 vec3 fields — 54 MB of writes at the 500k default — regardless
        of how many particles exist.  Only entries below N are ever read back.
        """
        count = self._temp_field_size if N is None else min(int(N), self._temp_field_size)
        if count > 0:
            self._zero_temp_fields_kernel(count)

    def _resize_temp_fields(self, N: int):
        """Validate that a particle operation fits the startup allocation."""
        if self._temp_field_size >= N:
            return
        raise ValueError(
            f"Particle operation requires {N} slots, but max_n_particles="
            f"{self._temp_field_size}. Increase VPMSetup.max_n_particles before "
            "constructing the solver."
        )

    @ti.kernel
    def _reset_rate_moments(self):
        self._rate_projection_original_norm_sq[None] = 0.0
        for row in range(10):
            self._rate_constraint_defect[row] = 0.0
            self._rate_constraint_multiplier[row] = 0.0
            for column in range(10):
                self._rate_constraint_gram[row, column] = 0.0

    @ti.kernel
    def _reduce_rate_moments(
        self,
        position: ti.template(),
        strength: ti.template(),
        core_radius: ti.template(),
        velocity: ti.template(),
        strength_rate: ti.template(),
        count: ti.i32,
    ):
        for i in range(count):
            x = position[i]
            vortex_strength = strength[i]
            u = velocity[i]
            rate = strength_rate[i]
            ti.atomic_add(self._rate_projection_original_norm_sq[None], rate.dot(rate))
            rows = ti.Matrix.zero(self.accumulator_dtype, 10, 3)
            for component in ti.static(range(3)):
                rows[component, component] = 1.0

            rows[3, 1], rows[3, 2] = -0.5 * x[2], 0.5 * x[1]
            rows[4, 0], rows[4, 2] = 0.5 * x[2], -0.5 * x[0]
            rows[5, 0], rows[5, 1] = -0.5 * x[1], 0.5 * x[0]

            radius = core_radius[i]
            core_term = ti.cast(self._angular_core_coefficient, self.accumulator_dtype) * radius**2
            x_sq = x.dot(x)
            for row, column in ti.static(ti.ndrange(3, 3)):
                rows[6 + row, column] = x[row] * x[column] / 3.0
                if ti.static(row == column):
                    rows[6 + row, column] -= x_sq / 3.0 + core_term

            impulse_rate = 0.5 * (u.cross(vortex_strength) + x.cross(rate))
            angular_rate = (
                u.cross(x.cross(vortex_strength))
                + x.cross(u.cross(vortex_strength))
                + x.cross(x.cross(rate))
            ) / 3.0 - core_term * rate
            for component in ti.static(range(3)):
                ti.atomic_add(self._rate_constraint_defect[component], rate[component])
                ti.atomic_add(self._rate_constraint_defect[3 + component], impulse_rate[component])
                ti.atomic_add(self._rate_constraint_defect[6 + component], angular_rate[component])
            for row in ti.static(range(9)):
                for column in ti.static(range(9)):
                    value = ti.cast(0.0, self.accumulator_dtype)
                    for component in ti.static(range(3)):
                        value += rows[row, component] * rows[column, component]
                    ti.atomic_add(self._rate_constraint_gram[row, column], value)

    def _define_rate_energy_kernel(self, q_, g_) -> None:
        """Bind the exact discrete-energy row of the rate projection."""

        @ti.kernel
        def reduce_rate_energy(
            position: ti.template(),
            strength: ti.template(),
            core_radius: ti.template(),
            velocity: ti.template(),
            strength_rate: ti.template(),
            count: ti.i32,
        ):
            for i in range(count):
                x_i = position[i]
                vortex_strength_i = strength[i]
                vector_potential = ti.Vector.zero(self.accumulator_dtype, 3)
                energy_position_gradient = ti.Vector.zero(self.accumulator_dtype, 3)
                for j in range(count):
                    displacement = x_i - position[j]
                    distance = displacement.norm()
                    pair_radius = 0.5 * (core_radius[i] + core_radius[j])
                    if distance / pair_radius < DEFAULT_CUTOFF_RADIUS_FACTOR:
                        convolved_radius = ti.sqrt(core_radius[i] ** 2 + core_radius[j] ** 2)
                        rho = distance / convolved_radius
                        vector_potential += g_(rho) / convolved_radius * strength[j]
                        if distance > EPSILON:
                            energy_position_gradient -= (
                                q_(rho)
                                * vortex_strength_i.dot(strength[j])
                                / distance**3
                                * displacement
                            )

                self._rate_energy_gradient[i] = vector_potential
                energy_rate = vector_potential.dot(strength_rate[i]) + energy_position_gradient.dot(
                    velocity[i]
                )
                ti.atomic_add(self._rate_constraint_defect[9], energy_rate)

                rows = ti.Matrix.zero(self.accumulator_dtype, 9, 3)
                for component in ti.static(range(3)):
                    rows[component, component] = 1.0
                rows[3, 1], rows[3, 2] = -0.5 * x_i[2], 0.5 * x_i[1]
                rows[4, 0], rows[4, 2] = 0.5 * x_i[2], -0.5 * x_i[0]
                rows[5, 0], rows[5, 1] = -0.5 * x_i[1], 0.5 * x_i[0]
                core_term = (
                    ti.cast(self._angular_core_coefficient, self.accumulator_dtype)
                    * core_radius[i] ** 2
                )
                x_sq = x_i.dot(x_i)
                for row, column in ti.static(ti.ndrange(3, 3)):
                    rows[6 + row, column] = x_i[row] * x_i[column] / 3.0
                    if ti.static(row == column):
                        rows[6 + row, column] -= x_sq / 3.0 + core_term

                for row in ti.static(range(9)):
                    cross_gram = ti.cast(0.0, self.accumulator_dtype)
                    for component in ti.static(range(3)):
                        cross_gram += rows[row, component] * vector_potential[component]
                    ti.atomic_add(self._rate_constraint_gram[row, 9], cross_gram)
                    ti.atomic_add(self._rate_constraint_gram[9, row], cross_gram)
                ti.atomic_add(
                    self._rate_constraint_gram[9, 9], vector_potential.dot(vector_potential)
                )

        self._reduce_rate_energy = reduce_rate_energy

    @ti.kernel
    def _apply_rate_moment_correction(
        self,
        position: ti.template(),
        core_radius: ti.template(),
        strength_rate: ti.template(),
        count: ti.i32,
    ):
        for i in range(count):
            x = position[i]
            rows = ti.Matrix.zero(self.accumulator_dtype, 10, 3)
            for component in ti.static(range(3)):
                rows[component, component] = 1.0
            rows[3, 1], rows[3, 2] = -0.5 * x[2], 0.5 * x[1]
            rows[4, 0], rows[4, 2] = 0.5 * x[2], -0.5 * x[0]
            rows[5, 0], rows[5, 1] = -0.5 * x[1], 0.5 * x[0]
            radius = core_radius[i]
            core_term = ti.cast(self._angular_core_coefficient, self.accumulator_dtype) * radius**2
            x_sq = x.dot(x)
            for row, column in ti.static(ti.ndrange(3, 3)):
                rows[6 + row, column] = x[row] * x[column] / 3.0
                if ti.static(row == column):
                    rows[6 + row, column] -= x_sq / 3.0 + core_term
            rows[9, 0] = self._rate_energy_gradient[i][0]
            rows[9, 1] = self._rate_energy_gradient[i][1]
            rows[9, 2] = self._rate_energy_gradient[i][2]

            correction = ti.Vector.zero(self.accumulator_dtype, 3)
            for row, component in ti.static(ti.ndrange(10, 3)):
                correction[component] += (
                    self._rate_constraint_multiplier[row] * rows[row, component]
                )
            strength_rate[i] += correction

    def conserve_rate_moments(
        self,
        position,
        strength,
        radius,
        velocity,
        strength_rate,
        count: int,
        *,
        conserve_energy: bool = False,
    ) -> None:
        """Make the inviscid rates of vortex strength and both impulses vanish.

        For ``G = sum(vortex_strength_i)`` and
        ``I = 0.5 sum(x_i cross vortex_strength_i)``, the uncorrected defect is

        ``dI/dt = 0.5 sum(u_i cross vortex_strength_i + x_i cross
        d(vortex_strength_i)/dt)``.

        Angular impulse uses the same finite-core correction as the flow
        diagnostics.  The nine moment constraints, and optionally the exact
        discrete-energy-rate constraint, are solved simultaneously for the
        minimum Euclidean-norm strength-rate correction.  No particle ordering,
        remeshing, clipping, or dissipative filtering is involved.  Viscous core
        spreading remains outside this projection, so its modeled energy and
        angular rates are retained.
        """
        if count <= 0:
            return
        self._reset_rate_moments()
        self._reduce_rate_moments(position, strength, radius, velocity, strength_rate, count)
        if conserve_energy:
            self._reduce_rate_energy(position, strength, radius, velocity, strength_rate, count)
        ti.sync()

        defect = self._rate_constraint_defect.to_numpy().astype(np.float64)
        gram = self._rate_constraint_gram.to_numpy().astype(np.float64)
        multipliers = np.linalg.pinv(gram, rcond=1.0e-12) @ (-defect)
        if not np.isfinite(multipliers).all():
            raise RuntimeError("VPM moment projection produced a non-finite correction")

        correction_norm_sq = max(float(multipliers @ gram @ multipliers), 0.0)
        original_norm_sq = max(float(self._rate_projection_original_norm_sq[None]), 0.0)
        ratio = np.sqrt(correction_norm_sq / max(original_norm_sq, np.finfo(float).tiny))
        self.rate_projection_correction_ratio = float(ratio)
        self.rate_projection_max_correction_ratio = max(
            self.rate_projection_max_correction_ratio,
            self.rate_projection_correction_ratio,
        )

        self._rate_constraint_multiplier.from_numpy(multipliers.astype(self.np_dtype))
        self._apply_rate_moment_correction(position, radius, strength_rate, count)

    @ti.kernel
    def _reset_axisymmetric_accumulators(self, count: ti.i32):
        for i in range(count):
            self._axisymmetric_vector_sum_a[i] = ti.Vector([0.0, 0.0, 0.0])
            self._axisymmetric_vector_sum_b[i] = ti.Vector([0.0, 0.0, 0.0])
            self._axisymmetric_scalar_sum[i] = 0.0
            self._axisymmetric_orbit_count[i] = ti.cast(0, ti.i32)

    @ti.func
    def _cylindrical_components(self, position, value, axis: ti.i32):
        b = ti.cast((axis + 1) % 3, ti.i32)
        c = ti.cast((axis + 2) % 3, ti.i32)
        radius = ti.sqrt(position[b] ** 2 + position[c] ** 2)
        inverse_radius = 1.0 / ti.max(radius, ti.cast(1.0e-20, self.accumulator_dtype))
        return ti.Vector(
            [
                value[axis],
                (position[b] * value[b] + position[c] * value[c]) * inverse_radius,
                (-position[c] * value[b] + position[b] * value[c]) * inverse_radius,
            ]
        )

    @ti.func
    def _cartesian_from_cylindrical(self, position, value, axis: ti.i32):
        b = ti.cast((axis + 1) % 3, ti.i32)
        c = ti.cast((axis + 2) % 3, ti.i32)
        radius = ti.sqrt(position[b] ** 2 + position[c] ** 2)
        inverse_radius = 1.0 / ti.max(radius, ti.cast(1.0e-20, self.accumulator_dtype))
        result = ti.Vector.zero(self.accumulator_dtype, 3)
        result[axis] = value[0]
        result[b] = (value[1] * position[b] - value[2] * position[c]) * inverse_radius
        result[c] = (value[1] * position[c] + value[2] * position[b]) * inverse_radius
        return result

    @ti.kernel
    def _reduce_axisymmetric_vectors(
        self,
        position: ti.template(),
        vector_a: ti.template(),
        vector_b: ti.template(),
        orbit_id: ti.template(),
        axis: ti.i32,
        count: ti.i32,
    ):
        for i in range(count):
            orbit = ti.cast(orbit_id[i], ti.i32)
            a = self._cylindrical_components(position[i], vector_a[i], axis)
            b = self._cylindrical_components(position[i], vector_b[i], axis)
            for component in ti.static(range(3)):
                ti.atomic_add(self._axisymmetric_vector_sum_a[orbit][component], a[component])
                ti.atomic_add(self._axisymmetric_vector_sum_b[orbit][component], b[component])
            ti.atomic_add(self._axisymmetric_orbit_count[orbit], ti.cast(1, ti.i32))

    @ti.kernel
    def _apply_axisymmetric_vectors(
        self,
        position: ti.template(),
        vector_a: ti.template(),
        vector_b: ti.template(),
        orbit_id: ti.template(),
        axis: ti.i32,
        count: ti.i32,
    ):
        for i in range(count):
            orbit = ti.cast(orbit_id[i], ti.i32)
            denominator = ti.cast(self._axisymmetric_orbit_count[orbit], self.accumulator_dtype)
            mean_a = self._axisymmetric_vector_sum_a[orbit] / denominator
            mean_b = self._axisymmetric_vector_sum_b[orbit] / denominator
            # vector_a is polar velocity: no swirl means u_theta = 0.
            # vector_b is an axial vortex strength rate: reflection symmetry leaves
            # only its azimuthal component.
            mean_a[2] = 0.0
            mean_b[0] = 0.0
            mean_b[1] = 0.0
            vector_a[i] = self._cartesian_from_cylindrical(position[i], mean_a, axis)
            vector_b[i] = self._cartesian_from_cylindrical(position[i], mean_b, axis)

    def average_axisymmetric_no_swirl_rhs(
        self, position, vector_a, vector_b, orbit_id, axis: int, count: int
    ) -> None:
        """Average velocity/rate over the axisymmetric no-swirl manifold."""
        if count <= 0:
            return
        self._reset_axisymmetric_accumulators(count)
        self._reduce_axisymmetric_vectors(position, vector_a, vector_b, orbit_id, axis, count)
        self._apply_axisymmetric_vectors(position, vector_a, vector_b, orbit_id, axis, count)

    @ti.kernel
    def _reduce_axisymmetric_scalar(
        self,
        scalar: ti.template(),
        orbit_id: ti.template(),
        count: ti.i32,
    ):
        for i in range(count):
            orbit = ti.cast(orbit_id[i], ti.i32)
            ti.atomic_add(self._axisymmetric_scalar_sum[orbit], scalar[i])
            ti.atomic_add(self._axisymmetric_orbit_count[orbit], ti.cast(1, ti.i32))

    @ti.kernel
    def _apply_axisymmetric_scalar(
        self,
        scalar: ti.template(),
        orbit_id: ti.template(),
        count: ti.i32,
    ):
        for i in range(count):
            orbit = ti.cast(orbit_id[i], ti.i32)
            scalar[i] = self._axisymmetric_scalar_sum[orbit] / ti.cast(
                self._axisymmetric_orbit_count[orbit], self.accumulator_dtype
            )

    def average_axisymmetric_scalar(self, scalar, orbit_id, count: int) -> None:
        """Average one scalar field over declared particle orbits."""
        if count <= 0:
            return
        self._reset_axisymmetric_accumulators(count)
        self._reduce_axisymmetric_scalar(scalar, orbit_id, count)
        self._apply_axisymmetric_scalar(scalar, orbit_id, count)

    # TARGET FIELD MANAGEMENT (for at-point evaluations)

    def _initialize_target_fields(self, size: int = 50000):
        """Allocate the fixed-capacity fields used for at-point evaluation."""
        self.target_position = ti.Vector.field(3, dtype=self.accumulator_dtype, shape=(size,))
        self.target_velocity = ti.Vector.field(3, dtype=self.accumulator_dtype, shape=(size,))
        self.target_vorticity = ti.Vector.field(3, dtype=self.accumulator_dtype, shape=(size,))
        self.target_velocity_gradient = ti.Matrix.field(
            3, 3, dtype=self.accumulator_dtype, shape=(size,)
        )
        self._target_field_size = size

    def _resize_target_fields(self, N: int):
        """Validate that an at-point query fits the startup allocation."""
        if self._target_field_size >= N:
            return
        raise ValueError(
            f"Target query requires {N} points but max_evaluation_points="
            f"{self._target_field_size}. Increase VPMSetup.max_evaluation_points "
            "before constructing the solver; runtime Taichi field growth is "
            "disabled because replaced fields retain device memory."
        )

    @ti.kernel
    def _copy_ndarray_to_vec3_field(
        self, src: ti.types.ndarray(), dst: ti.template(), start_idx: ti.i32, n: ti.i32
    ):  # type: ignore
        """Copy n vec3 entries from a fixed-size NumPy buffer to a Taichi field."""
        for i in range(n):
            for k in ti.static(range(3)):
                dst[start_idx + i][k] = src[i, k]

    @ti.kernel
    def _copy_ndarray_to_scalar_field(
        self, src: ti.types.ndarray(), dst: ti.template(), start_idx: ti.i32, n: ti.i32
    ):  # type: ignore
        """Copy n scalar entries from a fixed-size NumPy buffer to a Taichi field."""
        for i in range(n):
            dst[start_idx + i] = src[i]

    @ti.kernel
    def _extract_vec3_field_prefix(
        self, src: ti.template(), dst: ti.types.ndarray(), start_idx: ti.i32, n: ti.i32
    ):  # type: ignore
        """Copy n vec3 entries from a Taichi field to a fixed-size NumPy buffer."""
        for i in range(n):
            for k in ti.static(range(3)):
                dst[i, k] = src[start_idx + i][k]

    @ti.kernel
    def _extract_mat3_field_prefix(
        self, src: ti.template(), dst: ti.types.ndarray(), start_idx: ti.i32, n: ti.i32
    ):  # type: ignore
        """Copy n mat3 entries from a Taichi field to a fixed-size NumPy buffer."""
        for i in range(n):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dst[i, j, k] = src[start_idx + i][j, k]

    def _host_transfer_buffer(self, family: str, field, direction: str) -> np.ndarray:
        """Return a fixed staging array unique to a field and direction."""
        key = (direction, id(field))
        if family == "vector":
            buffers = self._host_vector_chunks
            shape = (_HOST_TRANSFER_CHUNK_SIZE, 3)
        elif family == "scalar":
            buffers = self._host_scalar_chunks
            shape = (_HOST_TRANSFER_CHUNK_SIZE,)
        elif family == "matrix":
            buffers = self._host_matrix_chunks
            shape = (_HOST_TRANSFER_CHUNK_SIZE, 3, 3)
        else:
            raise ValueError(f"Unknown host transfer buffer family {family!r}")
        if key not in buffers:
            buffers[key] = np.empty(shape, dtype=self.np_dtype)
        return buffers[key]

    def _upload_vector_array(self, src: np.ndarray, dst, n: int | None = None):
        """Upload a vec3 array through fixed-size ndarray chunks."""
        arr = np.ascontiguousarray(src, dtype=self.np_dtype)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Expected vector array with shape (N, 3), got {arr.shape}")
        count = arr.shape[0] if n is None else n
        buf = self._host_transfer_buffer("vector", dst, "upload")
        for lo in range(0, count, _HOST_TRANSFER_CHUNK_SIZE):
            hi = min(lo + _HOST_TRANSFER_CHUNK_SIZE, count)
            n_chunk = hi - lo
            buf[:n_chunk] = arr[lo:hi]
            self._copy_ndarray_to_vec3_field(buf, dst, lo, n_chunk)
            ti.sync()

    def _upload_scalar_array(self, src: np.ndarray, dst, n: int | None = None):
        """Upload a scalar array through fixed-size ndarray chunks."""
        arr = np.ascontiguousarray(src, dtype=self.np_dtype)
        if arr.ndim != 1:
            raise ValueError(f"Expected scalar array with shape (N,), got {arr.shape}")
        count = arr.shape[0] if n is None else n
        buf = self._host_transfer_buffer("scalar", dst, "upload")
        for lo in range(0, count, _HOST_TRANSFER_CHUNK_SIZE):
            hi = min(lo + _HOST_TRANSFER_CHUNK_SIZE, count)
            n_chunk = hi - lo
            buf[:n_chunk] = arr[lo:hi]
            self._copy_ndarray_to_scalar_field(buf, dst, lo, n_chunk)
            ti.sync()

    def _download_vector_field(self, src, n: int) -> np.ndarray:
        """Download the active vec3 prefix without exposing variable ndarray shapes."""
        if n == 0:
            return np.empty((0, 3), dtype=self.np_dtype)
        out = np.empty((n, 3), dtype=self.np_dtype)
        buf = self._host_transfer_buffer("vector", src, "download")
        for lo in range(0, n, _HOST_TRANSFER_CHUNK_SIZE):
            count = min(_HOST_TRANSFER_CHUNK_SIZE, n - lo)
            self._extract_vec3_field_prefix(src, buf, lo, count)
            ti.sync()
            out[lo : lo + count] = buf[:count]
        return out

    def _download_matrix_field(self, src, n: int) -> np.ndarray:
        """Download the active mat3 prefix without exposing variable ndarray shapes."""
        if n == 0:
            return np.empty((0, 3, 3), dtype=self.np_dtype)
        out = np.empty((n, 3, 3), dtype=self.np_dtype)
        buf = self._host_transfer_buffer("matrix", src, "download")
        for lo in range(0, n, _HOST_TRANSFER_CHUNK_SIZE):
            count = min(_HOST_TRANSFER_CHUNK_SIZE, n - lo)
            self._extract_mat3_field_prefix(src, buf, lo, count)
            ti.sync()
            out[lo : lo + count] = buf[:count]
        return out

    def extract_target_velocity(self, n: int) -> np.ndarray:
        """Return first n target velocity as a NumPy array (no full alloc transfer)."""
        return self._download_vector_field(self.target_velocity, n)

    def _resize_filtered_fields(self, N: int):
        """
        Validate filtered particle capacity for zone-aware field evaluation.

        Args:
            N: Required number of filtered particles
        """
        if self._filtered_field_size >= N:
            return

        raise ValueError(
            f"Filtered evaluation requires {N} particles, but max_n_particles="
            f"{self._filtered_field_size}. Increase VPMSetup.max_n_particles "
            "before constructing the solver."
        )

    # TREECODE MANAGEMENT (for hierarchical methods)

    def _get_or_create_treecode(self, required_size: int, theta: float = 0.5):
        """
        Get or create cached treecode instance for hierarchical methods.

        Taichi fields cannot be garbage collected, so creating a new TaichiTreecode
        instance on every call leads to memory accumulation and eventual OOM.
        This method caches and reuses a single instance, sized once to the run's
        declared particle ceiling so a growing particle count never forces a
        rebuild.

        Args:
            required_size: Required max_n_particles capacity
            theta: Opening angle for MAC (updates cached tree's theta)

        Returns:
            TaichiTreecode: Cached or newly created treecode instance
        """
        from ..acceleration.treecode_gpu import TaichiTreecode

        # Create new treecode only if we need more capacity
        if self._treecode is None or required_size > self._treecode_max_particles:
            ceiling = min(int(self.max_n_particles), MAX_N_PARTICLES)
            alloc_size = max(_TREECODE_MIN_CAPACITY, required_size, ceiling)
            self._treecode = TaichiTreecode(
                max_n_particles=alloc_size,
                max_nodes=2 * alloc_size,
                theta=theta,
                kernel_type=self.particle_kernel,
                multipole_order=self.treecode_multipole_order,
                sort_particle_targets=self.treecode_sort_particle_targets,
                traversal_block_dim=self.treecode_traversal_block_dim,
            )
            self._treecode_max_particles = alloc_size
        else:
            # Update theta in case it changed
            self._treecode.theta = theta
            self._treecode.theta_sq = theta * theta
            self._treecode.set_kernel_type(self.particle_kernel)
            self._treecode.set_multipole_order(self.treecode_multipole_order)
            self._treecode.set_sort_particle_targets(self.treecode_sort_particle_targets)
            self._treecode.traversal_block_dim = self.treecode_traversal_block_dim

        return self._treecode

    # KERNEL INITIALIZATION

    def _define_taichi_kernels(self):
        """
        Import and bind Taichi kernels based on selected particle kernel type.

        Dynamically loads kernel functions from the appropriate module
        (gaussian, super_gaussian, or winckelmans).
        """
        # Import kernel factory
        from ..numerics.kernels_common import create_kernels

        # Select kernel module based on type
        if self.particle_kernel == "GAUSSIAN":
            from ..kernels.gaussian import create_gaussian_kernels

            kernel_functions = create_gaussian_kernels(self.accumulator_dtype)
        elif self.particle_kernel == "HIGH_ORDER_GAUSSIAN":
            from ..kernels.high_order_gaussian import create_high_order_gaussian_kernels

            kernel_functions = create_high_order_gaussian_kernels(self.accumulator_dtype)
        elif self.particle_kernel == "SUPER_GAUSSIAN":
            from ..kernels.super_gaussian import create_super_gaussian_kernels

            kernel_functions = create_super_gaussian_kernels(self.accumulator_dtype)
        elif self.particle_kernel == "WINCKELMANS":
            from ..kernels.winckelmans import create_winckelmans_kernels

            kernel_functions = create_winckelmans_kernels(self.accumulator_dtype)
        else:
            raise ValueError(f"Unknown particle kernel: {self.particle_kernel}")

        # Create all kernels from factory
        self.kernels = create_kernels(kernel_functions)
        self._define_rate_energy_kernel(kernel_functions["q_"], kernel_functions["g_"])

        # Bind kernels as instance methods for easy access
        for name, fn in self.kernels.items():
            setattr(self, name, fn)

    # UNIFIED SELF-INDUCED VELOCITY OPERATOR

    def configure_velocity(
        self,
        method: str,
        theta: float = 0.5,
        multipole_order: int = 1,
        sort_particle_targets: bool = False,
        traversal_block_dim: int = 128,
    ) -> None:
        """Set how the self-induced velocity is evaluated (single source of truth).

        Args:
            method: "DIRECT" (O(N²) GPU summation) or "TREECODE" (Barnes-Hut O(N log N)).
            theta:  Barnes-Hut opening angle (treecode only; smaller = more accurate).
        """
        self.velocity_method = method.upper()
        self.velocity_theta = theta
        if multipole_order not in (1, 2, 3):
            raise ValueError(f"treecode multipole_order must be 1, 2 or 3, got {multipole_order}")
        if traversal_block_dim < 0:
            raise ValueError(
                f"treecode traversal_block_dim must be >= 0, got {traversal_block_dim}"
            )
        self.treecode_multipole_order = int(multipole_order)
        self.treecode_sort_particle_targets = bool(sort_particle_targets)
        self.treecode_traversal_block_dim = int(traversal_block_dim)

    @ti.kernel
    def _copy_vec3(self, src: ti.template(), dst: ti.template(), N: ti.i32):
        """Copy the first N entries of one vec3 field into another."""
        for i in range(N):
            dst[i] = src[i]

    @ti.kernel
    def _copy_mat3(self, src: ti.template(), dst: ti.template(), N: ti.i32):
        """Copy the first N entries of one 3×3 matrix field into another."""
        for i in range(N):
            dst[i] = src[i]

    def compute_self_induced_velocity(
        self,
        position,
        vortex_strength,
        core_radius,
        velocity,
        background_velocity,
        n_particles_total: int,
        reuse_tree: bool = False,
    ) -> None:
        """Self-induced velocity of a particle set, evaluated at its own position.

        Writes the result into the ``velocity`` Taichi vec3 field. Honors the method
        set by :meth:`configure_velocity` — this is the ONLY place that decides
        between direct summation and the treecode.  Every advection integrator
        (Euler, RK2/3/4) routes through here, so the configured velocity method
        is applied consistently at every stage.

        In vortex advection all particles move together, so at each RK stage the
        sources and evaluation points are the same displaced set; the treecode is
        built from ``position`` and evaluated at ``position``.

        Args:
            position: Particle position.
            vortex_strength: Particle vortex-strength vectors.
            core_radius: Particle core radius.
            velocity: Field receiving the induced velocity.
            background_velocity: Background or freestream velocity.
            n_particles_total: Number of active particles.
            reuse_tree: when True, reuse the LBVH topology from the previous
                build and only refit its position-dependent multipoles (valid
                for RK stages ≥ 2, where vortex_strength/core_radius are unchanged and
                particles have moved < h).  Falls back to a full build if no
                compatible tree exists.
        """
        if n_particles_total == 0:
            return
        if self.velocity_method == "TREECODE":
            tree = self._get_or_create_treecode(n_particles_total, self.velocity_theta)
            # Reuse the stage-1 topology when asked; otherwise (or on any
            # mismatch) do a full build.  VortexStrengths/core_radius are advection-
            # invariant, so a refit needs only the displaced position.
            if reuse_tree and self.reuse_tree_topology:
                try:
                    tree.refit(position, n_particles_total)
                except RuntimeError:
                    tree.build(position, vortex_strength, core_radius, n_particles_total)
            else:
                tree.build(position, vortex_strength, core_radius, n_particles_total)
            # The shared tree now represents a possibly intermediate RK state.
            # A target caller must rebuild from the published particle state
            # instead of trusting a pre-existing revision key.
            self._target_tree_key = None
            # On-device traversal + field-to-field copy.  The freestream is passed
            # as a field so nothing crosses to the host inside an RK stage.
            tree.compute_velocities_gpu(background_field=background_velocity)
            self._copy_vec3(tree.velocity, velocity, n_particles_total)
        else:
            self.compute_velocities_kernel(
                position,
                vortex_strength,
                core_radius,
                velocity,
                background_velocity,
                n_particles_total,
            )

    def _ensure_target_tree_current(self, particles, capacity: int, theta: float):
        """Return a tree built for the current particle source revision.

        ``Particles.state_revision`` changes only when a Biot--Savart source
        field (position, vortex strength, radius, or population) changes.  A
        missing revision deliberately disables reuse so lightweight external
        particle adapters cannot accidentally observe a stale hierarchy.
        """
        N = len(particles)
        tree = self._get_or_create_treecode(capacity, theta)
        revision = getattr(particles, "state_revision", None)
        key = None if revision is None else (id(tree), id(particles), int(revision), int(N))
        if key is None or self._target_tree_key != key:
            tree.build(particles.position, particles.vortex_strength, particles.core_radius, N)
            self._target_tree_key = key
        return tree

    # FIELD EVALUATION METHODS

    def compute_velocities(self, particles):
        """
        Compute self-induced velocity at all particle position.

        Uses the Biot-Savart law with regularization kernel:
            u(x) = Σ_j K(x - x_j, σ_j) × Γ_j

        Args:
            particles: Particle container with position, vortex strength, radius fields
        """
        N = len(particles)
        if N == 0:
            return

        self._resize_temp_fields(N)
        background_velocity = particles.velocity_background

        self.compute_velocities_kernel(
            particles.position,
            particles.vortex_strength,
            particles.core_radius,
            particles.velocity,
            background_velocity,
            N,
        )

    def compute_target_velocity(
        self,
        particles,
        target_position,
        target_velocity=None,
        include_freestream: bool = True,
        zone_mask: np.ndarray | None = None,
    ):
        """
        Compute velocity at arbitrary target position.

        Args:
            particles: Particle container
            target_position: Array of shape (M, 3) with target coordinates, or Taichi field
            target_velocity: Optional Taichi field to write results to directly
            include_freestream: If True, add background velocity
            zone_mask: Optional boolean mask to filter contributing particles

        Returns:
            np.ndarray: Velocities at target position, shape (M, 3) if target_velocity is None
        """
        N = len(particles)

        # Select velocity field based on include_freestream flag
        background_velocity = (
            particles.velocity_background if include_freestream else self._zero_velocity
        )

        # If zone_mask is provided, use filtered computation
        if zone_mask is not None and N > 0:
            return self._compute_target_velocity_filtered(
                particles, target_position, zone_mask, include_freestream
            )

        # Handle Taichi field input
        if not isinstance(target_position, np.ndarray):
            return self._compute_target_velocity_taichi(
                particles,
                target_position,
                target_velocity,
                include_freestream,
                background_velocity,
                N,
            )

        # Handle Numpy input
        M = len(target_position)
        if M == 0:
            return np.zeros((0, 3), dtype=self.np_dtype)

        if N == 0:
            result = np.zeros((M, 3), dtype=self.np_dtype)
            if include_freestream:
                bg = particles.velocity_background_cpu()
                result += bg
            return result

        # Target evaluation must honor the same velocity method selected for
        # particle advection.  Falling through to the direct M-by-N kernel
        # here made coupled boundary queries launch almost one billion pair
        # interactions in a single Vulkan dispatch, tripping integrated-GPU
        # watchdogs even though particle self-evaluation used the treecode.
        if self.velocity_method == "TREECODE":
            return self.compute_target_velocity_hierarchical(
                particles,
                target_position,
                theta=self.velocity_theta,
                include_freestream=include_freestream,
            )

        # Resize target fields if needed
        self._resize_target_fields(M)

        # Copy target position to GPU through fixed-shape external buffers.
        self._upload_vector_array(target_position, self.target_position, M)

        # Compute velocity at targets
        self.compute_target_velocity_kernel(
            self.target_position,
            particles.position,
            particles.vortex_strength,
            particles.core_radius,
            self.target_velocity,
            background_velocity,
            M,
            N,
        )

        return self.extract_target_velocity(M)

    def _compute_target_velocity_taichi(
        self,
        particles,
        target_position,
        target_velocity,
        include_freestream: bool,
        background_velocity,
        N: int,
    ):
        """Handle velocity computation when target_position is a Taichi field."""
        M = target_position.shape[0]
        if N == 0:
            if include_freestream:
                freestream_velocity = np.array(
                    [
                        particles.velocity_background[None][0],
                        particles.velocity_background[None][1],
                        particles.velocity_background[None][2],
                    ],
                    dtype=self.np_dtype,
                )
            else:
                freestream_velocity = np.zeros(3, dtype=self.np_dtype)
            if target_velocity is not None:
                for i in range(M):
                    target_velocity[i] = freestream_velocity
                return None
            return np.tile(freestream_velocity, (M, 1))

        out_field = target_velocity if target_velocity is not None else self.target_velocity
        if target_velocity is None:
            self._resize_target_fields(M)
            out_field = self.target_velocity

        self.compute_target_velocity_kernel(
            target_position,
            particles.position,
            particles.vortex_strength,
            particles.core_radius,
            out_field,
            background_velocity,
            M,
            N,
        )

        if target_velocity is None:
            return self._download_vector_field(out_field, M)
        return None

    def _compute_target_velocity_filtered(
        self, particles, target_position, zone_mask, include_freestream: bool = True
    ):
        """
        Compute velocity at target position using only selected particles.

        This is a CPU-based fallback for zone-aware BC computation that filters
        particles before velocity computation. Only particles where zone_mask[i]=True
        contribute to the induced velocity.

        Args:
            particles: Full particle set
            target_position: np.ndarray (M, 3) evaluation points
            zone_mask: Boolean mask (N,) - only True particles contribute
            include_freestream: If True, adds background velocity

        Returns:
            Velocities at target position (M, 3)
        """
        # Ensure numpy input
        if not isinstance(target_position, np.ndarray):
            target_position = target_position.to_numpy()

        M = len(target_position)
        if M == 0:
            return np.zeros((0, 3), dtype=self.np_dtype)

        # Get particle data and filter by zone_mask
        position = particles.position_cpu()
        vortex_strength = particles.vortex_strength_cpu()
        core_radius = particles.core_radius_cpu()

        # Apply zone mask
        filtered_pos = position[zone_mask]
        filtered_vortex_strength = vortex_strength[zone_mask]
        filtered_rad = core_radius[zone_mask]

        N_filtered = len(filtered_pos)

        # If no particles pass the filter, return freestream only
        if N_filtered == 0:
            if include_freestream:
                bg = particles.velocity_background_cpu()
                return np.tile(bg, (M, 1))
            return np.zeros((M, 3), dtype=self.np_dtype)

        # Resize target and filtered fields (cached to prevent memory leak)
        self._resize_target_fields(M)
        self._resize_filtered_fields(N_filtered)

        # Copy filtered data through fixed-shape external buffers.
        self._upload_vector_array(filtered_pos, self._filtered_pos, N_filtered)
        self._upload_vector_array(
            filtered_vortex_strength, self._filtered_vortex_strength, N_filtered
        )
        self._upload_scalar_array(filtered_rad, self._filtered_rad, N_filtered)

        # Copy target position to GPU
        self._upload_vector_array(target_position, self.target_position, M)

        # Select background velocity
        background_velocity = (
            particles.velocity_background if include_freestream else self._zero_velocity
        )

        # Compute with filtered particles
        self.compute_target_velocity_kernel(
            self.target_position,
            self._filtered_pos,
            self._filtered_vortex_strength,
            self._filtered_rad,
            self.target_velocity,
            background_velocity,
            M,
            N_filtered,
        )

        return self.extract_target_velocity(M)

    def compute_velocities_from_arrays(
        self,
        source_position: np.ndarray,
        source_vortex_strength: np.ndarray,
        source_core_radius: np.ndarray,
        target_position: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Biot-Savart velocity from raw NumPy arrays (GPU-accelerated).

        This is a lightweight alternative to ``compute_target_velocity``
        that does **not** require a full ``Particles`` container.  It reuses
        the cached ``_filtered_*`` and ``target_*`` Taichi fields to avoid
        memory leaks and reallocations.

        Designed for lightweight velocity evaluation where the
        correction-sheet particles exist only as NumPy arrays and their
        induced velocity must be evaluated at target position.

        No freestream / background velocity is added.

        Parameters
        ----------
        source_position : ndarray (N, 3)
            Positions of source particles.
        source_vortex_strength : ndarray (N, 3)
            Vortex strength (vortex strength α) of source particles.
        source_core_radius : ndarray (N,)
            Core core_radius of source particles.
        target_position : ndarray (M, 3)
            Evaluation points.

        Returns
        -------
        ndarray (M, 3)
            Induced velocity at each target point.
        """
        N = len(source_position)
        M = len(target_position)

        if N == 0 or M == 0:
            return np.zeros((M, 3), dtype=self.np_dtype)

        # Resize cached Taichi fields (grow-only, no GC leak)
        self._resize_target_fields(M)
        self._resize_filtered_fields(N)

        # Upload source data to GPU through fixed-shape external buffers.
        self._upload_vector_array(source_position, self._filtered_pos, N)
        self._upload_vector_array(source_vortex_strength, self._filtered_vortex_strength, N)
        self._upload_scalar_array(source_core_radius, self._filtered_rad, N)

        # Upload target position
        self._upload_vector_array(target_position, self.target_position, M)

        # Call Taichi kernel (no background velocity)
        self.compute_target_velocity_kernel(
            self.target_position,
            self._filtered_pos,
            self._filtered_vortex_strength,
            self._filtered_rad,
            self.target_velocity,
            self._zero_velocity,
            M,
            N,
        )

        return self.extract_target_velocity(M)

    def compute_vorticities(self, particles):
        """
        Compute vorticity field at all particle position.

        Uses kernel superposition:
            ω(x) = Σ_j ζ(x - x_j, σ_j) * Γ_j / V_j

        Args:
            particles: Particle container
        """
        N = len(particles)
        if N == 0:
            return

        self._resize_temp_fields(N)

        self.compute_vorticities_kernel(
            particles.position,
            particles.vortex_strength,
            particles.core_radius,
            particles.vorticity,
            N,
        )

    def compute_target_vorticity(self, particles, target_position: np.ndarray) -> np.ndarray:
        """
        Compute vorticity field at arbitrary target position.

        Args:
            particles: Particle container
            target_position: Array of shape (M, 3) with target coordinates

        Returns:
            np.ndarray: Vorticities at target position, shape (M, 3)
        """
        N = len(particles)
        M = len(target_position)

        if N == 0 or M == 0:
            return np.zeros((M, 3), dtype=self.np_dtype)

        # Resize target fields if needed
        self._resize_target_fields(M)

        # Copy target position to GPU
        self._upload_vector_array(target_position, self.target_position, M)

        # Compute vorticity at targets
        self.compute_target_vorticity_kernel(
            self.target_position,
            particles.position,
            particles.vortex_strength,
            particles.core_radius,
            self.target_vorticity,
            M,
            N,
        )

        return self._download_vector_field(self.target_vorticity, M)

    def compute_velocity_gradients(self, particles):
        """
        Compute velocity gradient tensor at all particle position.

        Computes ∇u = (∂u_i/∂x_j) tensor and derives:
        - Strain rate tensor: S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i) / 2
        - Rotation rate: Ω_ij = (∂u_i/∂x_j - ∂u_j/∂x_i) / 2

        Args:
            particles: Particle container
        """
        N = len(particles)
        if N == 0:
            return

        self._resize_temp_fields(N)

        self.compute_velocity_gradients_kernel(
            particles.position,
            particles.vortex_strength,
            particles.core_radius,
            particles.velocity_gradient,
            particles.strain_rate,
            N,
        )

    def compute_target_velocity_gradient(
        self, particles, target_position: np.ndarray
    ) -> np.ndarray:
        """
        Compute velocity gradient tensors at arbitrary target position.

        Args:
            particles: Particle container
            target_position: Array of shape (M, 3) with target coordinates

        Returns:
            np.ndarray: Velocity gradients at targets, shape (M, 9) [flattened 3x3]
        """
        N = len(particles)
        M = len(target_position)

        if N == 0 or M == 0:
            return np.zeros((M, 9), dtype=self.np_dtype)

        # Resize target fields if needed
        self._resize_target_fields(M)

        # Copy target position to GPU
        self._upload_vector_array(target_position, self.target_position, M)

        # Compute velocity gradients at targets
        self.compute_target_velocity_gradient_kernel(
            self.target_position,
            particles.position,
            particles.vortex_strength,
            particles.core_radius,
            self.target_velocity_gradient,
            M,
            N,
        )

        # Return flattened gradients
        grads = self._download_matrix_field(self.target_velocity_gradient, M)
        return grads.reshape(M, 9)

    def compute_velocities_hierarchical(self, particles, theta: float = 0.5):
        """
        Compute velocity using Barnes-Hut treecode for O(N log N) complexity.

        Args:
            particles: Particle container
            theta: Opening angle parameter for MAC (smaller = more accurate)
                   Recommended: 0.3-0.5 for high accuracy, 0.7-1.0 for speed
        """
        N = len(particles)
        if N == 0:
            return

        tree = self._get_or_create_treecode(N, theta)
        # Build from GPU fields directly
        tree.build(particles.position, particles.vortex_strength, particles.core_radius, N)
        self._target_tree_key = None
        bg = particles.velocity_background
        bg_arr = np.array([bg[None][0], bg[None][1], bg[None][2]], dtype=np.float32)
        velocity = tree.compute_velocities(bg_arr)

        particles.set_field("velocity", velocity)

    def compute_target_velocity_hierarchical(
        self,
        particles,
        target_position: np.ndarray,
        theta: float = 0.5,
        include_freestream: bool = True,
    ) -> np.ndarray:
        """
        Compute velocity at targets using Barnes-Hut treecode.

        Args:
            particles: Particle container
            target_position: Target coordinates [M, 3]
            theta: Opening angle parameter (smaller = more accurate)
            include_freestream: Include background velocity (default True)

        Returns:
            np.ndarray: Velocities at target position [M, 3]
        """
        N = len(particles)
        M = len(target_position)

        if N == 0 or M == 0:
            return np.zeros((M, 3), dtype=self.np_dtype)

        max_size = max(N, M)
        tree = self._ensure_target_tree_current(particles, max_size, theta)

        background_vel = None
        if include_freestream:
            bg = particles.velocity_background
            background_vel = np.array([bg[None][0], bg[None][1], bg[None][2]], dtype=np.float32)

        return tree.compute_target_velocity(target_position, background_vel)

    def compute_target_velocity_gradient_hierarchical(
        self, particles, target_position: np.ndarray, theta: float = 0.5
    ) -> np.ndarray:
        """
        Compute velocity gradients at targets using Barnes-Hut treecode.

        Args:
            particles: Particle container
            target_position: Target coordinates [M, 3]
            theta: Opening angle parameter (smaller = more accurate)

        Returns:
            np.ndarray: Velocity gradients at targets [M, 9] (flattened 3x3)
        """
        N = len(particles)
        M = len(target_position)

        if N == 0 or M == 0:
            return np.zeros((M, 9), dtype=self.np_dtype)

        max_size = max(N, M)
        tree = self._ensure_target_tree_current(particles, max_size, theta)

        grads = tree.compute_target_velocity_gradient(target_position)
        return grads.reshape(M, 9)

    def compute_target_velocity_and_gradients_hierarchical(
        self,
        particles,
        target_position: np.ndarray,
        theta: float = 0.5,
        include_freestream: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate target velocity and Jacobian from one Barnes-Hut tree build.

        The returned Jacobian has shape ``(M, 9)`` in row-major order,
        ``J[i, j] = d(u_i)/d(x_j)``.
        """
        N = len(particles)
        M = len(target_position)
        if M == 0:
            return (
                np.zeros((0, 3), dtype=self.np_dtype),
                np.zeros((0, 9), dtype=self.np_dtype),
            )
        if N == 0:
            velocity = np.zeros((M, 3), dtype=self.np_dtype)
            if include_freestream:
                velocity += particles.velocity_background_cpu()
            return velocity, np.zeros((M, 9), dtype=self.np_dtype)

        max_size = max(N, M)
        tree = self._ensure_target_tree_current(particles, max_size, theta)
        background = None
        if include_freestream:
            bg = particles.velocity_background
            background = np.array([bg[None][0], bg[None][1], bg[None][2]], dtype=np.float32)
        velocity, gradient = tree.compute_target_velocity_and_gradients(target_position, background)
        return velocity, gradient.reshape(M, 9)

    def compute_velocity_gradients_hierarchical(self, particles, theta: float = 0.5):
        """
        Compute velocity gradients using Barnes-Hut treecode for O(N log N) complexity.

        Args:
            particles: Particle container
            theta: Opening angle parameter for MAC (smaller = more accurate)
                   Recommended: 0.3-0.4 for gradients (tighter than velocity)
        """
        N = len(particles)
        if N == 0:
            return

        tree = self._get_or_create_treecode(N, theta)
        tree.build(particles.position, particles.vortex_strength, particles.core_radius, N)
        self._target_tree_key = None

        # On-device traversal + field-to-field copy: ∇u and S
        tree.compute_velocity_gradients_gpu()
        self._copy_mat3(tree.velocity_gradient, particles.velocity_gradient, N)
        self._copy_mat3(tree.strain_rate, particles.strain_rate, N)

    def compute_velocity_and_gradient_hierarchical(self, particles, theta: float = 0.5) -> None:
        """Fused treecode evaluation of u, ∇u and S in one build.

        Writes ``velocity`` (= v(x_n), reusable as the advection k1), plus
        ``velocity_gradient`` and ``strain_rate`` — replacing a separate velocity
        pass and a separate gradient pass at the same t_n configuration."""
        N = len(particles)
        if N == 0:
            return
        tree = self._get_or_create_treecode(N, theta)
        tree.build(particles.position, particles.vortex_strength, particles.core_radius, N)
        self._target_tree_key = None
        bg = particles.velocity_background
        bg_arr = np.array([bg[None][0], bg[None][1], bg[None][2]], dtype=np.float32)
        tree.compute_velocity_and_gradient_gpu(bg_arr)
        self._copy_vec3(tree.velocity, particles.velocity, N)
        self._copy_mat3(tree.velocity_gradient, particles.velocity_gradient, N)
        self._copy_mat3(tree.strain_rate, particles.strain_rate, N)

    def compute_velocity_and_gradient(self, particles) -> None:
        """Fused direct (O(N²)) evaluation of u, ∇u and S in a single j-loop.

        DIRECT counterpart of :meth:`compute_velocity_and_gradient_hierarchical`;
        writes ``velocity``, ``velocity_gradient`` and ``strain_rate`` together."""
        N = len(particles)
        if N == 0:
            return
        self._resize_temp_fields(N)
        self.compute_velocity_and_gradient_kernel(
            particles.position,
            particles.vortex_strength,
            particles.core_radius,
            particles.velocity,
            particles.velocity_gradient,
            particles.strain_rate,
            particles.velocity_background,
            N,
        )

    # DIAGNOSTICS METHODS

    def compute_kinetic_energy(self, particles):
        """
        Compute kinetic energy at each particle.

        Evaluates E = ½‖u‖² per particle by calling a Taichi kernel that
        gathers velocity from the Biot–Savart sum and stores the result
        in ``particles.particle_kinetic_energy``.

        Args:
            particles: Particle container with ``position``, ``vortex strength``,
                ``core_radius``, and ``particle_kinetic_energy`` fields.
        """
        N = len(particles)
        if N == 0:
            return
        self._resize_temp_fields(N)
        self.compute_kinetic_energy_kernel(
            particles.position,
            particles.vortex_strength,
            particles.core_radius,
            particles.particle_kinetic_energy,
            N,
        )

    def compute_helicity(self, particles):
        """
        Compute helicity at each particle.

        Evaluates H = u · ω per particle by calling a Taichi kernel that
        combines the velocity and vorticity fields and stores the result
        in ``particles.particle_helicity``.

        Args:
            particles: Particle container with ``position``, ``vortex strength``,
                ``core_radius``, and ``particle_helicity`` fields.
        """
        N = len(particles)
        if N == 0:
            return
        self._resize_temp_fields(N)
        self.compute_helicity_kernel(
            particles.position,
            particles.vortex_strength,
            particles.core_radius,
            particles.particle_helicity,
            N,
        )
