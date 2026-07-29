"""
Physics Base Module for VPM Solver.
====================================
Contains shared functionality for all physics modules:
- Field evaluation (velocities, vorticities, gradients)
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

from ..config.constants import MAX_PARTICLES

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
    - Field evaluation methods (velocities, vorticities, gradients)

    The concrete PhysicsEngine delegates to internal handlers for advection,
    diffusion, and stretching; the unified self-induced velocity operator
    (velocity_self / configure_velocity) lives here so every handler shares it.
    """

    def __init__(
        self,
        particles_kernel: str = "GAUSSIAN",
        max_particles: int = MAX_PARTICLES,
        accumulator_dtype: ti.types = ti.f32,
        max_targets: int = 200000,
    ):
        """
        Initialize physics base with kernel type and field allocation.

        Args:
            particles_kernel: Kernel type for particle interactions
                            Options: 'GAUSSIAN', 'SUPER_GAUSSIAN', 'WINCKELMANS'
            max_particles: Maximum number of particles for field allocation
            accumulator_dtype: Data type for accumulation (ti.f32 or ti.f64)
            max_targets: Fixed capacity for at-point field evaluations
        """
        self.particles_kernel = particles_kernel.upper()
        self.max_particles = max_particles
        self.accumulator_dtype = accumulator_dtype
        self.np_dtype = np.float32 if accumulator_dtype == ti.f32 else np.float64
        self.max_targets = int(max_targets)
        if self.max_targets < 1:
            raise ValueError("max_targets must be at least 1")

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

        # Velocity-evaluation method — the single source of truth for how the
        # self-induced velocity is computed (see velocity_self()).  Set once by
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
        self._filtered_field_size = self.max_particles
        self._filtered_pos = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_particles,)
        )
        self._filtered_circ = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_particles,)
        )
        self._filtered_rad = ti.field(dtype=self.accumulator_dtype, shape=(self.max_particles,))

        self._host_vector_chunk = None
        self._host_scalar_chunk = None
        self._host_matrix_chunk = None

        # Initialize Taichi fields
        self._initialize_temp_fields()
        self._initialize_target_fields(self.max_targets)

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
            3, dtype=self.accumulator_dtype, shape=(self.max_particles,)
        )
        self.pos_temp2 = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_particles,)
        )
        self.vel_temp = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_particles,)
        )
        self.vel_temp2 = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_particles,)
        )

        # Strength temporaries for stretching/diffusion
        self.str_temp = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_particles,)
        )
        self.str_temp2 = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_particles,)
        )

        # Strength rate temporaries for RK integration
        self.dstr_dt_temp = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_particles,)
        )
        self.dstr_dt_temp2 = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_particles,)
        )
        self.dstr_dt_temp3 = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_particles,)
        )

        # Mark initial size
        self._temp_field_size = self.max_particles

    def _zero_temp_fields(self):
        """Zero all temporary fields to prevent contamination between operations."""
        self.pos_temp.fill(0)
        self.pos_temp2.fill(0)
        self.vel_temp.fill(0)
        self.vel_temp2.fill(0)
        self.str_temp.fill(0)
        self.str_temp2.fill(0)
        self.dstr_dt_temp.fill(0)
        self.dstr_dt_temp2.fill(0)
        self.dstr_dt_temp3.fill(0)

    def _resize_temp_fields(self, N: int):
        """Validate that a particle operation fits the startup allocation."""
        if self._temp_field_size >= N:
            return
        raise ValueError(
            f"Particle operation requires {N} slots, but max_particles="
            f"{self._temp_field_size}. Increase VPMSetup.max_particles before "
            "constructing the solver."
        )

    # TARGET FIELD MANAGEMENT (for at-point evaluations)

    def _initialize_target_fields(self, size: int = 50000):
        """Allocate the fixed-capacity fields used for at-point evaluation."""
        self.target_positions = ti.Vector.field(3, dtype=self.accumulator_dtype, shape=(size,))
        self.target_velocities = ti.Vector.field(3, dtype=self.accumulator_dtype, shape=(size,))
        self.target_vorticities = ti.Vector.field(3, dtype=self.accumulator_dtype, shape=(size,))
        self.target_velocity_gradients = ti.Matrix.field(
            3, 3, dtype=self.accumulator_dtype, shape=(size,)
        )
        self._target_field_size = size

    def _resize_target_fields(self, N: int):
        """Validate that an at-point query fits the startup allocation."""
        if self._target_field_size >= N:
            return
        raise ValueError(
            f"Target query requires {N} points but max_targets="
            f"{self._target_field_size}. Increase VPMSetup.max_targets "
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

    def _ensure_host_transfer_buffers(self):
        """Allocate reusable fixed-shape buffers for ndarray kernel transfers."""
        if self._host_vector_chunk is not None:
            return
        self._host_vector_chunk = np.empty((_HOST_TRANSFER_CHUNK_SIZE, 3), dtype=self.np_dtype)
        self._host_scalar_chunk = np.empty((_HOST_TRANSFER_CHUNK_SIZE,), dtype=self.np_dtype)
        self._host_matrix_chunk = np.empty((_HOST_TRANSFER_CHUNK_SIZE, 3, 3), dtype=self.np_dtype)

    def _upload_vector_array(self, src: np.ndarray, dst, n: int | None = None):
        """Upload a vec3 array through fixed-size ndarray chunks."""
        arr = np.ascontiguousarray(src, dtype=self.np_dtype)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Expected vector array with shape (N, 3), got {arr.shape}")
        count = arr.shape[0] if n is None else n
        self._ensure_host_transfer_buffers()
        buf = self._host_vector_chunk
        for lo in range(0, count, _HOST_TRANSFER_CHUNK_SIZE):
            hi = min(lo + _HOST_TRANSFER_CHUNK_SIZE, count)
            n_chunk = hi - lo
            buf[:n_chunk] = arr[lo:hi]
            self._copy_ndarray_to_vec3_field(buf, dst, lo, n_chunk)

    def _upload_scalar_array(self, src: np.ndarray, dst, n: int | None = None):
        """Upload a scalar array through fixed-size ndarray chunks."""
        arr = np.ascontiguousarray(src, dtype=self.np_dtype)
        if arr.ndim != 1:
            raise ValueError(f"Expected scalar array with shape (N,), got {arr.shape}")
        count = arr.shape[0] if n is None else n
        self._ensure_host_transfer_buffers()
        buf = self._host_scalar_chunk
        for lo in range(0, count, _HOST_TRANSFER_CHUNK_SIZE):
            hi = min(lo + _HOST_TRANSFER_CHUNK_SIZE, count)
            n_chunk = hi - lo
            buf[:n_chunk] = arr[lo:hi]
            self._copy_ndarray_to_scalar_field(buf, dst, lo, n_chunk)

    def _download_vector_field(self, src, n: int) -> np.ndarray:
        """Download the active vec3 prefix without exposing variable ndarray shapes."""
        if n == 0:
            return np.empty((0, 3), dtype=self.np_dtype)
        out = np.empty((n, 3), dtype=self.np_dtype)
        self._ensure_host_transfer_buffers()
        buf = self._host_vector_chunk
        for lo in range(0, n, _HOST_TRANSFER_CHUNK_SIZE):
            count = min(_HOST_TRANSFER_CHUNK_SIZE, n - lo)
            self._extract_vec3_field_prefix(src, buf, lo, count)
            out[lo : lo + count] = buf[:count]
        return out

    def _download_matrix_field(self, src, n: int) -> np.ndarray:
        """Download the active mat3 prefix without exposing variable ndarray shapes."""
        if n == 0:
            return np.empty((0, 3, 3), dtype=self.np_dtype)
        out = np.empty((n, 3, 3), dtype=self.np_dtype)
        self._ensure_host_transfer_buffers()
        buf = self._host_matrix_chunk
        for lo in range(0, n, _HOST_TRANSFER_CHUNK_SIZE):
            count = min(_HOST_TRANSFER_CHUNK_SIZE, n - lo)
            self._extract_mat3_field_prefix(src, buf, lo, count)
            out[lo : lo + count] = buf[:count]
        return out

    def extract_target_velocities(self, n: int) -> np.ndarray:
        """Return first n target velocities as a NumPy array (no full alloc transfer)."""
        return self._download_vector_field(self.target_velocities, n)

    def _resize_filtered_fields(self, N: int):
        """
        Validate filtered particle capacity for zone-aware field evaluation.

        Args:
            N: Required number of filtered particles
        """
        if self._filtered_field_size >= N:
            return

        raise ValueError(
            f"Filtered evaluation requires {N} particles, but max_particles="
            f"{self._filtered_field_size}. Increase VPMSetup.max_particles "
            "before constructing the solver."
        )

    # TREECODE MANAGEMENT (for hierarchical methods)

    def _get_or_create_treecode(self, required_size: int, theta: float = 0.5):
        """
        Get or create cached treecode instance for hierarchical methods.

        Taichi fields cannot be garbage collected, so creating a new TaichiTreecode
        instance on every call leads to memory accumulation and eventual OOM.
        This method caches and reuses a single instance, only recreating if the
        required size exceeds the current allocation.

        Args:
            required_size: Required max_particles capacity
            theta: Opening angle for MAC (updates cached tree's theta)

        Returns:
            TaichiTreecode: Cached or newly created treecode instance
        """
        from ..acceleration.treecode_gpu import TaichiTreecode

        # Create new treecode only if we need more capacity
        if self._treecode is None or required_size > self._treecode_max_particles:
            # Use MAX_PARTICLES as the floor, NOT self.max_particles.
            # self.max_particles can be inflated to millions of particles by
            # _resize_temp_fields when a diffusion scheme (DVH/GBD) scatter
            # first creates a dense particle grid.  Using it here would cause
            # TaichiTreecode to allocate traversal_stack fields of shape
            # (N_scatter, 48) — e.g. (3.36M, 48) — consuming ~1.2 GB of
            # GPU memory for stacks alone instead of the expected ~190 MB,
            # which exhausts the CUDA memory pool and causes
            # CUDA_ERROR_ILLEGAL_ADDRESS on the first subsequent GPU sync.
            alloc_size = max(required_size, MAX_PARTICLES)
            self._treecode = TaichiTreecode(
                max_particles=alloc_size,
                max_nodes=2 * alloc_size,
                theta=theta,
                kernel_type=self.particles_kernel,
                multipole_order=self.treecode_multipole_order,
                sort_particle_targets=self.treecode_sort_particle_targets,
                traversal_block_dim=self.treecode_traversal_block_dim,
            )
            self._treecode_max_particles = alloc_size
        else:
            # Update theta in case it changed
            self._treecode.theta = theta
            self._treecode.theta_sq = theta * theta
            self._treecode.set_kernel_type(self.particles_kernel)
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
        if self.particles_kernel == "GAUSSIAN":
            from ..kernels.gaussian import create_gaussian_kernels

            kernel_functions = create_gaussian_kernels(self.accumulator_dtype)
        elif self.particles_kernel == "HIGH_ORDER_GAUSSIAN":
            from ..kernels.high_order_gaussian import create_high_order_gaussian_kernels

            kernel_functions = create_high_order_gaussian_kernels(self.accumulator_dtype)
        elif self.particles_kernel == "SUPER_GAUSSIAN":
            from ..kernels.super_gaussian import create_super_gaussian_kernels

            kernel_functions = create_super_gaussian_kernels(self.accumulator_dtype)
        elif self.particles_kernel == "WINCKELMANS":
            from ..kernels.winckelmans import create_winckelmans_kernels

            kernel_functions = create_winckelmans_kernels(self.accumulator_dtype)
        else:
            raise ValueError(f"Unknown particle kernel: {self.particles_kernel}")

        # Create all kernels from factory
        self.kernels = create_kernels(kernel_functions)

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

    def velocity_self(self, pos, strg, rad, out, bg, N: int, reuse_tree: bool = False) -> None:
        """Self-induced velocity of a particle set, evaluated at its own positions.

        Writes the result into the ``out`` taichi vec3 field.  Honors the method
        set by :meth:`configure_velocity` — this is the ONLY place that decides
        between direct summation and the treecode.  Every advection integrator
        (Euler, RK2/3/4) routes through here, so the configured velocity method
        is applied consistently at every stage.

        In vortex advection all particles move together, so at each RK stage the
        sources and evaluation points are the same displaced set; the treecode is
        built from ``pos`` and evaluated at ``pos``.

        Args:
            pos:  vec3 field of evaluation/source positions (length ≥ N).
            strg: vec3 field of circulations Γ (length ≥ N).
            rad:  scalar field of core radii (length ≥ N).
            out:  vec3 field receiving the velocities (length ≥ N).
            bg:   0-d vec3 field with the background/freestream velocity.
            N:    number of active particles.
            reuse_tree: when True, reuse the LBVH topology from the previous
                build and only refit its position-dependent multipoles (valid
                for RK stages ≥ 2, where circulations/radii are unchanged and
                particles have moved < h).  Falls back to a full build if no
                compatible tree exists.
        """
        if N == 0:
            return
        if self.velocity_method == "TREECODE":
            tree = self._get_or_create_treecode(N, self.velocity_theta)
            # Reuse the stage-1 topology when asked; otherwise (or on any
            # mismatch) do a full build.  Circulations/radii are advection-
            # invariant, so a refit needs only the displaced positions.
            if reuse_tree and self.reuse_tree_topology:
                try:
                    tree.refit(pos, N)
                except RuntimeError:
                    tree.build(pos, strg, rad, N)
            else:
                tree.build(pos, strg, rad, N)
            # Background velocity: extract single 3-vector (cheap, 3 floats).
            bg_arr = np.array([bg[None][0], bg[None][1], bg[None][2]], dtype=np.float32)
            # On-device traversal + field-to-field copy
            tree.compute_velocities_gpu(bg_arr)
            self._copy_vec3(tree.velocities, out, N)
        else:
            self.compute_velocities_kernel(pos, strg, rad, out, bg, N)

    # FIELD EVALUATION METHODS

    def compute_velocities(self, particles):
        """
        Compute self-induced velocities at all particle positions.

        Uses the Biot-Savart law with regularization kernel:
            u(x) = Σ_j K(x - x_j, σ_j) × Γ_j

        Args:
            particles: Particle container with position, circulation, radius fields
        """
        N = len(particles)
        if N == 0:
            return

        self._resize_temp_fields(N)
        bg_vel = particles.velocity_background

        self.compute_velocities_kernel(
            particles.position,
            particles.circulation,
            particles.radius,
            particles.velocity,
            bg_vel,
            N,
        )

    def compute_target_velocities(
        self,
        particles,
        target_positions,
        target_velocities=None,
        include_freestream: bool = True,
        zone_mask: np.ndarray | None = None,
    ):
        """
        Compute velocities at arbitrary target positions.

        Args:
            particles: Particle container
            target_positions: Array of shape (M, 3) with target coordinates, or Taichi field
            target_velocities: Optional Taichi field to write results to directly
            include_freestream: If True, add background velocity
            zone_mask: Optional boolean mask to filter contributing particles

        Returns:
            np.ndarray: Velocities at target positions, shape (M, 3) if target_velocities is None
        """
        N = len(particles)

        # Select velocity field based on include_freestream flag
        bg_vel = particles.velocity_background if include_freestream else self._zero_velocity

        # If zone_mask is provided, use filtered computation
        if zone_mask is not None and N > 0:
            return self._compute_target_velocities_filtered(
                particles, target_positions, zone_mask, include_freestream
            )

        # Handle Taichi field input
        if not isinstance(target_positions, np.ndarray):
            return self._compute_target_velocities_taichi(
                particles, target_positions, target_velocities, include_freestream, bg_vel, N
            )

        # Handle Numpy input
        M = len(target_positions)
        if M == 0:
            return np.zeros((0, 3), dtype=self.np_dtype)

        if N == 0:
            result = np.zeros((M, 3), dtype=self.np_dtype)
            if include_freestream:
                bg = particles.velocity_background_cpu()
                result += bg
            return result

        # Resize target fields if needed
        self._resize_target_fields(M)

        # Copy target positions to GPU through fixed-shape external buffers.
        self._upload_vector_array(target_positions, self.target_positions, M)

        # Compute velocities at targets
        self.compute_target_velocity_kernel(
            self.target_positions,
            particles.position,
            particles.circulation,
            particles.radius,
            self.target_velocities,
            bg_vel,
            M,
            N,
        )

        return self.extract_target_velocities(M)

    def _compute_target_velocities_taichi(
        self,
        particles,
        target_positions,
        target_velocities,
        include_freestream: bool,
        bg_vel,
        N: int,
    ):
        """Handle velocity computation when target_positions is a Taichi field."""
        M = target_positions.shape[0]
        if N == 0:
            if include_freestream:
                U_inf = np.array(
                    [
                        particles.velocity_background[None][0],
                        particles.velocity_background[None][1],
                        particles.velocity_background[None][2],
                    ],
                    dtype=self.np_dtype,
                )
            else:
                U_inf = np.zeros(3, dtype=self.np_dtype)
            if target_velocities is not None:
                for i in range(M):
                    target_velocities[i] = U_inf
                return None
            return np.tile(U_inf, (M, 1))

        out_field = target_velocities if target_velocities is not None else self.target_velocities
        if target_velocities is None:
            self._resize_target_fields(M)
            out_field = self.target_velocities

        self.compute_target_velocity_kernel(
            target_positions,
            particles.position,
            particles.circulation,
            particles.radius,
            out_field,
            bg_vel,
            M,
            N,
        )

        if target_velocities is None:
            return self._download_vector_field(out_field, M)
        return None

    def _compute_target_velocities_filtered(
        self, particles, target_positions, zone_mask, include_freestream: bool = True
    ):
        """
        Compute velocities at target positions using only selected particles.

        This is a CPU-based fallback for zone-aware BC computation that filters
        particles before velocity computation. Only particles where zone_mask[i]=True
        contribute to the induced velocity.

        Args:
            particles: Full particle set
            target_positions: np.ndarray (M, 3) evaluation points
            zone_mask: Boolean mask (N,) - only True particles contribute
            include_freestream: If True, adds background velocity

        Returns:
            Velocities at target positions (M, 3)
        """
        # Ensure numpy input
        if not isinstance(target_positions, np.ndarray):
            target_positions = target_positions.to_numpy()

        M = len(target_positions)
        if M == 0:
            return np.zeros((0, 3), dtype=self.np_dtype)

        # Get particle data and filter by zone_mask
        positions = particles.position_cpu()
        circulations = particles.circulation_cpu()
        radii = particles.radius_cpu()

        # Apply zone mask
        filtered_pos = positions[zone_mask]
        filtered_circ = circulations[zone_mask]
        filtered_rad = radii[zone_mask]

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
        self._upload_vector_array(filtered_circ, self._filtered_circ, N_filtered)
        self._upload_scalar_array(filtered_rad, self._filtered_rad, N_filtered)

        # Copy target positions to GPU
        self._upload_vector_array(target_positions, self.target_positions, M)

        # Select background velocity
        bg_vel = particles.velocity_background if include_freestream else self._zero_velocity

        # Compute with filtered particles
        self.compute_target_velocity_kernel(
            self.target_positions,
            self._filtered_pos,
            self._filtered_circ,
            self._filtered_rad,
            self.target_velocities,
            bg_vel,
            M,
            N_filtered,
        )

        return self.extract_target_velocities(M)

    def compute_velocities_from_arrays(
        self,
        source_pos: np.ndarray,
        source_circ: np.ndarray,
        source_rad: np.ndarray,
        target_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Biot-Savart velocities from raw NumPy arrays (GPU-accelerated).

        This is a lightweight alternative to ``compute_target_velocities``
        that does **not** require a full ``Particles`` container.  It reuses
        the cached ``_filtered_*`` and ``target_*`` Taichi fields to avoid
        memory leaks and reallocations.

        Designed for lightweight velocity evaluation where the
        correction-sheet particles exist only as NumPy arrays and their
        induced velocity must be evaluated at target positions.

        No freestream / background velocity is added.

        Parameters
        ----------
        source_pos : ndarray (N, 3)
            Positions of source particles.
        source_circ : ndarray (N, 3)
            Circulation (vortex strength α) of source particles.
        source_rad : ndarray (N,)
            Core radii of source particles.
        target_pos : ndarray (M, 3)
            Evaluation points.

        Returns
        -------
        ndarray (M, 3)
            Induced velocity at each target point.
        """
        N = len(source_pos)
        M = len(target_pos)

        if N == 0 or M == 0:
            return np.zeros((M, 3), dtype=self.np_dtype)

        # Resize cached Taichi fields (grow-only, no GC leak)
        self._resize_target_fields(M)
        self._resize_filtered_fields(N)

        # Upload source data to GPU through fixed-shape external buffers.
        self._upload_vector_array(source_pos, self._filtered_pos, N)
        self._upload_vector_array(source_circ, self._filtered_circ, N)
        self._upload_scalar_array(source_rad, self._filtered_rad, N)

        # Upload target positions
        self._upload_vector_array(target_pos, self.target_positions, M)

        # Call Taichi kernel (no background velocity)
        self.compute_target_velocity_kernel(
            self.target_positions,
            self._filtered_pos,
            self._filtered_circ,
            self._filtered_rad,
            self.target_velocities,
            self._zero_velocity,
            M,
            N,
        )

        return self.extract_target_velocities(M)

    def compute_vorticities(self, particles):
        """
        Compute vorticity field at all particle positions.

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
            particles.position, particles.circulation, particles.radius, particles.vorticity, N
        )

    def compute_target_vorticities(self, particles, target_positions: np.ndarray) -> np.ndarray:
        """
        Compute vorticity field at arbitrary target positions.

        Args:
            particles: Particle container
            target_positions: Array of shape (M, 3) with target coordinates

        Returns:
            np.ndarray: Vorticities at target positions, shape (M, 3)
        """
        N = len(particles)
        M = len(target_positions)

        if N == 0 or M == 0:
            return np.zeros((M, 3), dtype=self.np_dtype)

        # Resize target fields if needed
        self._resize_target_fields(M)

        # Copy target positions to GPU
        self._upload_vector_array(target_positions, self.target_positions, M)

        # Compute vorticities at targets
        self.compute_target_vorticity_kernel(
            self.target_positions,
            particles.position,
            particles.circulation,
            particles.radius,
            self.target_vorticities,
            M,
            N,
        )

        return self._download_vector_field(self.target_vorticities, M)

    def compute_velocity_gradients(self, particles):
        """
        Compute velocity gradient tensor at all particle positions.

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
            particles.circulation,
            particles.radius,
            particles.velocity_gradient,
            particles.strain_rate,
            N,
        )

    def compute_target_velocity_gradients(
        self, particles, target_positions: np.ndarray
    ) -> np.ndarray:
        """
        Compute velocity gradient tensors at arbitrary target positions.

        Args:
            particles: Particle container
            target_positions: Array of shape (M, 3) with target coordinates

        Returns:
            np.ndarray: Velocity gradients at targets, shape (M, 9) [flattened 3x3]
        """
        N = len(particles)
        M = len(target_positions)

        if N == 0 or M == 0:
            return np.zeros((M, 9), dtype=self.np_dtype)

        # Resize target fields if needed
        self._resize_target_fields(M)

        # Copy target positions to GPU
        self._upload_vector_array(target_positions, self.target_positions, M)

        # Compute velocity gradients at targets
        self.compute_target_velocity_gradient_kernel(
            self.target_positions,
            particles.position,
            particles.circulation,
            particles.radius,
            self.target_velocity_gradients,
            M,
            N,
        )

        # Return flattened gradients
        grads = self._download_matrix_field(self.target_velocity_gradients, M)
        return grads.reshape(M, 9)

    def compute_velocities_hierarchical(self, particles, theta: float = 0.5):
        """
        Compute velocities using Barnes-Hut treecode for O(N log N) complexity.

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
        tree.build(particles.position, particles.circulation, particles.radius, N)
        bg = particles.velocity_background
        bg_arr = np.array([bg[None][0], bg[None][1], bg[None][2]], dtype=np.float32)
        velocities = tree.compute_velocities(bg_arr)

        particles.set_field("velocity", velocities)

    def compute_target_velocities_hierarchical(
        self,
        particles,
        target_positions: np.ndarray,
        theta: float = 0.5,
        include_freestream: bool = True,
    ) -> np.ndarray:
        """
        Compute velocities at targets using Barnes-Hut treecode.

        Args:
            particles: Particle container
            target_positions: Target coordinates [M, 3]
            theta: Opening angle parameter (smaller = more accurate)
            include_freestream: Include background velocity (default True)

        Returns:
            np.ndarray: Velocities at target positions [M, 3]
        """
        N = len(particles)
        M = len(target_positions)

        if N == 0 or M == 0:
            return np.zeros((M, 3), dtype=self.np_dtype)

        max_size = max(N, M)
        tree = self._get_or_create_treecode(max_size, theta)
        # Build from GPU fields directly.
        tree.build(particles.position, particles.circulation, particles.radius, N)

        background_vel = None
        if include_freestream:
            bg = particles.velocity_background
            background_vel = np.array([bg[None][0], bg[None][1], bg[None][2]], dtype=np.float32)

        return tree.compute_target_velocities(target_positions, background_vel)

    def compute_target_velocity_gradients_hierarchical(
        self, particles, target_positions: np.ndarray, theta: float = 0.5
    ) -> np.ndarray:
        """
        Compute velocity gradients at targets using Barnes-Hut treecode.

        Args:
            particles: Particle container
            target_positions: Target coordinates [M, 3]
            theta: Opening angle parameter (smaller = more accurate)

        Returns:
            np.ndarray: Velocity gradients at targets [M, 9] (flattened 3x3)
        """
        N = len(particles)
        M = len(target_positions)

        if N == 0 or M == 0:
            return np.zeros((M, 9), dtype=self.np_dtype)

        max_size = max(N, M)
        tree = self._get_or_create_treecode(max_size, theta)
        tree.build(particles.position, particles.circulation, particles.radius, N)

        grads = tree.compute_target_velocity_gradients(target_positions)
        return grads.reshape(M, 9)

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
        tree.build(particles.position, particles.circulation, particles.radius, N)

        # On-device traversal + field-to-field copy: ∇u and S
        tree.compute_velocity_gradients_gpu()
        self._copy_mat3(tree.velocity_gradients, particles.velocity_gradient, N)
        self._copy_mat3(tree.strain_rates, particles.strain_rate, N)

    def compute_velocity_and_gradient_hierarchical(self, particles, theta: float = 0.5) -> None:
        """Fused treecode evaluation of u, ∇u and S in one build.

        Writes ``velocity`` (= v(x_n), reusable as the advection k1), plus
        ``velocity_gradient`` and ``strain_rate`` — replacing a separate velocity
        pass and a separate gradient pass at the same t_n configuration."""
        N = len(particles)
        if N == 0:
            return
        tree = self._get_or_create_treecode(N, theta)
        tree.build(particles.position, particles.circulation, particles.radius, N)
        bg = particles.velocity_background
        bg_arr = np.array([bg[None][0], bg[None][1], bg[None][2]], dtype=np.float32)
        tree.compute_velocity_and_gradient_gpu(bg_arr)
        self._copy_vec3(tree.velocities, particles.velocity, N)
        self._copy_mat3(tree.velocity_gradients, particles.velocity_gradient, N)
        self._copy_mat3(tree.strain_rates, particles.strain_rate, N)

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
            particles.circulation,
            particles.radius,
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
        in ``particles.kinetic_energy``.

        Args:
            particles: Particle container with ``position``, ``circulation``,
                ``radius``, and ``kinetic_energy`` fields.
        """
        N = len(particles)
        if N == 0:
            return
        self._resize_temp_fields(N)
        self.compute_kinetic_energy_kernel(
            particles.position, particles.circulation, particles.radius, particles.kinetic_energy, N
        )

    def compute_helicity(self, particles):
        """
        Compute helicity at each particle.

        Evaluates H = u · ω per particle by calling a Taichi kernel that
        combines the velocity and vorticity fields and stores the result
        in ``particles.helicity``.

        Args:
            particles: Particle container with ``position``, ``circulation``,
                ``radius``, and ``helicity`` fields.
        """
        N = len(particles)
        if N == 0:
            return
        self._resize_temp_fields(N)
        self.compute_helicity_kernel(
            particles.position, particles.circulation, particles.radius, particles.helicity, N
        )
