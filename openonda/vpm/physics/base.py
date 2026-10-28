"""
Physics Base Module for VPM Solver.
====================================
Contains shared functionality for all physics modules:
- Field evaluation (velocity, vorticity, gradients)
- Temporary field management
- Kernel initialization

This module owns the shared device fields and low-level kernels used by the
induction methods, diffusion operators, target queries, and diagnostics.
"""

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Protocol, overload, runtime_checkable

import numpy as np
import taichi as ti

from openonda.core.slip_slab_images import slab_image_particles

from ..config.constants import MAX_N_PARTICLES
from ..kernels.registry import kernel_definition
from .induction.base import InductionMethod, TaichiDataType, TaichiField

if TYPE_CHECKING:
    from ..particles.container import Particles
    from .induction.treecode.lbvh import TaichiTreecode

# A compiled Taichi kernel; its arguments are Taichi fields and scalars.
TaichiKernel = Callable[..., Any]


class ParticleSources(Protocol):
    """Biot-Savart sources read by target queries: the particle container or a stage view."""

    @property
    def position(self) -> TaichiField: ...

    @property
    def vortex_strength(self) -> TaichiField: ...

    @property
    def core_radius(self) -> TaichiField: ...

    @property
    def velocity_background(self) -> TaichiField: ...

    def velocity_background_cpu(self) -> np.ndarray: ...

    def __len__(self) -> int: ...


@runtime_checkable
class HostParticleSources(ParticleSources, Protocol):
    """Particle sources that also provide host copies of their fields."""

    def position_cpu(self) -> np.ndarray: ...

    def vortex_strength_cpu(self) -> np.ndarray: ...

    def core_radius_cpu(self) -> np.ndarray: ...


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

# Fraction of a particle's circulation its core may leave beyond the image reach
_VORTICITY_TAIL = 1.0e-4


def _kernel_reach(kernel: str) -> float:
    """Core radii beyond which a kernel misses at most ``_VORTICITY_TAIL`` of its circulation.

    The enclosed circulation of a high-order core overshoots before it settles,
    so the reach is the last radius where the defect still exceeds the tail.
    """
    radius = np.linspace(0.0, 64.0, 64001)
    defect = np.abs(1.0 - 4.0 * np.pi * kernel_definition(kernel).q(radius))
    return float(radius[np.flatnonzero(defect > _VORTICITY_TAIL)[-1] + 1])


# PHYSICS BASE CLASS


@ti.data_oriented
class PhysicsBase:
    """
    Base class for VPM physics modules providing shared functionality.

    This class contains:
    - Temporary field initialization and management
    - Target field management (for at-point evaluations)
    - Kernel initialization and binding
    - Field evaluation methods (velocity, vorticity, gradients)

    The concrete PhysicsEngine provides shared fields and physical operators;
    coupled inviscid rates are selected through the induction conditions.
    """

    # Taichi kernels bound for the particle kernel by _define_taichi_kernels.
    compute_velocities_kernel: TaichiKernel
    compute_vorticities_kernel: TaichiKernel
    compute_velocity_gradients_kernel: TaichiKernel
    compute_velocity_and_gradient_kernel: TaichiKernel
    compute_target_velocity_kernel: TaichiKernel
    compute_transport_target_velocity_kernel: TaichiKernel
    compute_target_source_velocity_kernel: TaichiKernel
    compute_target_source_velocity_gradient_kernel: TaichiKernel
    compute_target_vorticity_kernel: TaichiKernel
    compute_target_velocity_gradient_kernel: TaichiKernel
    update_radius_csm_kernel: TaichiKernel
    update_position_rwm_kernel: TaichiKernel
    compute_velocity_and_stretching_rate_kernel: TaichiKernel

    # Slip-slab z bounds, set only by a bound slip-slab induction.
    _slip_slab_bounds: tuple[float, float]

    def __init__(
        self,
        particle_kernel: str = "gaussian",
        max_n_particles: int = MAX_N_PARTICLES,
        accumulator_dtype: TaichiDataType = ti.f32,
        max_evaluation_points: int = _HOST_TRANSFER_CHUNK_SIZE,
    ):
        """
        Initialize physics base with kernel type and field allocation.

        Args:
            particle_kernel: Kernel type for particle interactions
                            Options: 'gaussian', 'super_gaussian', 'winckelmans'
            max_n_particles: Maximum number of particles for field allocation
            accumulator_dtype: Data type for accumulation (ti.f32 or ti.f64)
            max_evaluation_points: Fixed capacity for at-point field evaluations
        """
        self.particle_kernel = particle_kernel.lower()
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
        self._treecode: TaichiTreecode | None = None
        self._treecode_max_particles = 0
        self._vorticity_tree: TaichiTreecode | None = None
        # A target-query phase often asks for blending and coupling-face
        # fields from one immutable particle state.  Keep one source-tree key so
        # those traversals share the LBVH instead of rebuilding it per caller.
        self._target_tree_key: tuple[int, int, int, int] | None = None

        # Velocity-evaluation method — the single source of truth for how the
        # target-query fields are computed. Bound once through configure_velocity();
        # standalone workspaces default to direct O(N²) summation.
        # DIRECT: exact O(N²); select TreecodeInduction for large clouds.
        self.velocity_method = "direct"  # "direct" | "treecode"
        self.velocity_theta = 0.3  # Barnes-Hut opening angle (treecode only)
        self.treecode_multipole_order = 1
        self.treecode_sort_particle_targets = False
        self.treecode_traversal_block_dim = 128

        # The solver binds the selected induction backend here after all
        # device fields are initialized.  Target queries use this same
        # conditions when available; standalone PhysicsEngine users retain the
        # direct-kernel fallback below.
        self.induction: InductionMethod | None = None

        # Cached filtered particle fields for zone-aware BC computation
        self._filtered_field_size = self.max_n_particles
        self._filtered_pos = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_n_particles,)
        )
        self._filtered_vortex_strength = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_n_particles,)
        )
        self._filtered_rad = ti.field(dtype=self.accumulator_dtype, shape=(self.max_n_particles,))

        # Fixed host staging arrays keyed by (direction, field identity).
        self._host_vector_chunks: dict[tuple[str, int], np.ndarray] = {}
        self._host_scalar_chunks: dict[tuple[str, int], np.ndarray] = {}
        self._host_matrix_chunks: dict[tuple[str, int], np.ndarray] = {}

        self._axisymmetric_vector_sum_a = ti.Vector.field(
            3, dtype=compute_dtype, shape=(self.max_n_particles,)
        )
        self._axisymmetric_vector_sum_b = ti.Vector.field(
            3, dtype=compute_dtype, shape=(self.max_n_particles,)
        )
        self._axisymmetric_scalar_sum = ti.field(dtype=compute_dtype, shape=(self.max_n_particles,))
        self._axisymmetric_orbit_count = ti.field(dtype=ti.i32, shape=(self.max_n_particles,))
        self._angular_core_coefficient = kernel_definition(
            self.particle_kernel
        ).angular_core_coefficient

        # Initialize Taichi fields
        self._initialize_temp_fields()
        self._initialize_target_fields(self.max_evaluation_points)

        # Bind kernel functions based on particle kernel type
        self._define_taichi_kernels()

    # TEMPORARY FIELD MANAGEMENT

    def _initialize_temp_fields(self) -> None:
        """
        Initialize temporary Taichi fields for intermediate calculations.

        These fields are used for:
        - Multi-stage time integrators (RK2, RK3, RK4)
        - Intermediate velocity/vorticity storage
        - Strength rate computations
        """
        # One shared strength-rate destination is required by the stage RHS.
        # The old second position/velocity/strength buffers were never read;
        # allocating them at the default 500k-particle capacity consumed tens
        # of megabytes before an induction backend had allocated anything.
        self.dstr_dt_temp = ti.Vector.field(
            3, dtype=self.accumulator_dtype, shape=(self.max_n_particles,)
        )
        # Mark initial size
        self._temp_field_size = self.max_n_particles

    def _resize_temp_fields(self, N: int) -> None:
        """Validate that a particle operation fits the startup allocation."""
        if self._temp_field_size >= N:
            return
        raise ValueError(
            f"Particle operation requires {N} slots, but max_n_particles="
            f"{self._temp_field_size}. Increase Numerics.max_n_particles before "
            "constructing the solver."
        )

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
        cylindrical_radius = ti.sqrt(position[b] ** 2 + position[c] ** 2)
        inverse_radius = 1.0 / ti.max(cylindrical_radius, ti.cast(1.0e-20, self.accumulator_dtype))
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
        cylindrical_radius = ti.sqrt(position[b] ** 2 + position[c] ** 2)
        inverse_radius = 1.0 / ti.max(cylindrical_radius, ti.cast(1.0e-20, self.accumulator_dtype))
        result = ti.Vector.zero(self.accumulator_dtype, 3)
        result[axis] = value[0]
        result[b] = (value[1] * position[b] - value[2] * position[c]) * inverse_radius
        result[c] = (value[1] * position[c] + value[2] * position[b]) * inverse_radius
        return result

    @ti.kernel
    def _reduce_axisymmetric_vectors(
        self,
        position: ti.template(),  # type: ignore[invalid-annotation]
        vector_a: ti.template(),  # type: ignore[invalid-annotation]
        vector_b: ti.template(),  # type: ignore[invalid-annotation]
        orbit_id: ti.template(),  # type: ignore[invalid-annotation]
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
        position: ti.template(),  # type: ignore[invalid-annotation]
        vector_a: ti.template(),  # type: ignore[invalid-annotation]
        vector_b: ti.template(),  # type: ignore[invalid-annotation]
        orbit_id: ti.template(),  # type: ignore[invalid-annotation]
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
        self,
        position: TaichiField,
        vector_a: TaichiField,
        vector_b: TaichiField,
        orbit_id: TaichiField,
        axis: int,
        count: int,
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
        scalar: ti.template(),  # type: ignore[invalid-annotation]
        orbit_id: ti.template(),  # type: ignore[invalid-annotation]
        count: ti.i32,
    ):
        for i in range(count):
            orbit = ti.cast(orbit_id[i], ti.i32)
            ti.atomic_add(self._axisymmetric_scalar_sum[orbit], scalar[i])
            ti.atomic_add(self._axisymmetric_orbit_count[orbit], ti.cast(1, ti.i32))

    @ti.kernel
    def _apply_axisymmetric_scalar(
        self,
        scalar: ti.template(),  # type: ignore[invalid-annotation]
        orbit_id: ti.template(),  # type: ignore[invalid-annotation]
        count: ti.i32,
    ):
        for i in range(count):
            orbit = ti.cast(orbit_id[i], ti.i32)
            scalar[i] = self._axisymmetric_scalar_sum[orbit] / ti.cast(
                self._axisymmetric_orbit_count[orbit], self.accumulator_dtype
            )

    def average_axisymmetric_scalar(
        self, scalar: TaichiField, orbit_id: TaichiField, count: int
    ) -> None:
        """Average one scalar field over declared particle orbits."""
        if count <= 0:
            return
        self._reset_axisymmetric_accumulators(count)
        self._reduce_axisymmetric_scalar(scalar, orbit_id, count)
        self._apply_axisymmetric_scalar(scalar, orbit_id, count)

    # TARGET FIELD MANAGEMENT (for at-point evaluations)

    def _initialize_target_fields(self, size: int = 50000) -> None:
        """Allocate the fixed-capacity fields used for at-point evaluation."""
        self.target_position = ti.Vector.field(3, dtype=self.accumulator_dtype, shape=(size,))
        self.target_velocity = ti.Vector.field(3, dtype=self.accumulator_dtype, shape=(size,))
        self.target_vorticity = ti.Vector.field(3, dtype=self.accumulator_dtype, shape=(size,))
        self.target_velocity_gradient = ti.Matrix.field(
            3, 3, dtype=self.accumulator_dtype, shape=(size,)
        )
        self._target_field_size = size

    def _resize_target_fields(self, N: int) -> None:
        """Validate that an at-point query fits the startup allocation."""
        if self._target_field_size >= N:
            return
        raise ValueError(
            f"Target batch requires {N} points but its fixed workspace holds "
            f"{self._target_field_size}; batch the query before evaluation."
        )

    def _target_batch_slices(self, count: int) -> Iterator[tuple[int, int]]:
        """Yield bounded target ranges without growing Taichi fields."""
        for start in range(0, count, self._target_field_size):
            yield start, min(start + self._target_field_size, count)

    @ti.kernel
    def _copy_ndarray_to_vec3_field(
        self,
        src: ti.types.ndarray(),  # type: ignore[invalid-annotation]
        dst: ti.template(),  # type: ignore[invalid-annotation]
        start_idx: ti.i32,
        n: ti.i32,
    ):
        """Copy n vec3 entries from a fixed-size NumPy buffer to a Taichi field."""
        for i in range(n):
            for k in ti.static(range(3)):
                dst[start_idx + i][k] = src[i, k]

    @ti.kernel
    def _copy_ndarray_to_scalar_field(
        self,
        src: ti.types.ndarray(),  # type: ignore[invalid-annotation]
        dst: ti.template(),  # type: ignore[invalid-annotation]
        start_idx: ti.i32,
        n: ti.i32,
    ):
        """Copy n scalar entries from a fixed-size NumPy buffer to a Taichi field."""
        for i in range(n):
            dst[start_idx + i] = src[i]

    @ti.kernel
    def _copy_ndarray_to_mat3_field(
        self,
        src: ti.types.ndarray(),  # type: ignore[invalid-annotation]
        dst: ti.template(),  # type: ignore[invalid-annotation]
        start_idx: ti.i32,
        n: ti.i32,
    ):
        """Copy n mat3 entries from a fixed-size NumPy buffer to a Taichi field."""
        for i in range(n):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dst[start_idx + i][j, k] = src[i, j, k]

    @ti.kernel
    def _extract_vec3_field_prefix(
        self,
        src: ti.template(),  # type: ignore[invalid-annotation]
        dst: ti.types.ndarray(),  # type: ignore[invalid-annotation]
        start_idx: ti.i32,
        n: ti.i32,
    ):
        """Copy n vec3 entries from a Taichi field to a fixed-size NumPy buffer."""
        for i in range(n):
            for k in ti.static(range(3)):
                dst[i, k] = src[start_idx + i][k]

    @ti.kernel
    def _extract_mat3_field_prefix(
        self,
        src: ti.template(),  # type: ignore[invalid-annotation]
        dst: ti.types.ndarray(),  # type: ignore[invalid-annotation]
        start_idx: ti.i32,
        n: ti.i32,
    ):
        """Copy n mat3 entries from a Taichi field to a fixed-size NumPy buffer."""
        for i in range(n):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dst[i, j, k] = src[start_idx + i][j, k]

    @ti.kernel
    def _extract_scalar_field_prefix(
        self,
        src: ti.template(),  # type: ignore[invalid-annotation]
        dst: ti.types.ndarray(),  # type: ignore[invalid-annotation]
        start_idx: ti.i32,
        n: ti.i32,
    ):
        """Copy n scalar entries from a Taichi field to a fixed-size NumPy buffer."""
        for i in range(n):
            dst[i] = src[start_idx + i]

    def _host_transfer_buffer(self, family: str, field: TaichiField, direction: str) -> np.ndarray:
        """Return a fixed staging array unique to a field and direction."""
        key = (direction, id(field))
        shape: tuple[int, ...]
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
            dtype = self.np_dtype
            if direction == "download":
                dtype = np.float32 if field.dtype == ti.f32 else np.float64
            buffers[key] = np.empty(shape, dtype=dtype)
        return buffers[key]

    def upload_vector_array(self, src: np.ndarray, dst: TaichiField, n: int | None = None) -> None:
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

    def _upload_scalar_array(self, src: np.ndarray, dst: TaichiField, n: int | None = None) -> None:
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

    def _upload_matrix_array(self, src: np.ndarray, dst: TaichiField, n: int | None = None) -> None:
        """Upload a mat3 array through fixed-size ndarray chunks."""
        arr = np.ascontiguousarray(src, dtype=self.np_dtype)
        if arr.ndim != 3 or arr.shape[1:] != (3, 3):
            raise ValueError(f"Expected matrix array with shape (N, 3, 3), got {arr.shape}")
        count = arr.shape[0] if n is None else n
        buf = self._host_transfer_buffer("matrix", dst, "upload")
        for lo in range(0, count, _HOST_TRANSFER_CHUNK_SIZE):
            hi = min(lo + _HOST_TRANSFER_CHUNK_SIZE, count)
            n_chunk = hi - lo
            buf[:n_chunk] = arr[lo:hi]
            self._copy_ndarray_to_mat3_field(buf, dst, lo, n_chunk)
            ti.sync()

    def download_vector_field(self, src: TaichiField, n: int) -> np.ndarray:
        """Download the active vec3 prefix without exposing variable ndarray shapes."""
        dtype = np.float32 if src.dtype == ti.f32 else np.float64
        if n == 0:
            return np.empty((0, 3), dtype=dtype)
        out = np.empty((n, 3), dtype=dtype)
        buf = self._host_transfer_buffer("vector", src, "download")
        for lo in range(0, n, _HOST_TRANSFER_CHUNK_SIZE):
            count = min(_HOST_TRANSFER_CHUNK_SIZE, n - lo)
            self._extract_vec3_field_prefix(src, buf, lo, count)
            ti.sync()
            out[lo : lo + count] = buf[:count]
        return out

    def _download_matrix_field(self, src: TaichiField, n: int) -> np.ndarray:
        """Download the active mat3 prefix without exposing variable ndarray shapes."""
        dtype = np.float32 if src.dtype == ti.f32 else np.float64
        if n == 0:
            return np.empty((0, 3, 3), dtype=dtype)
        out = np.empty((n, 3, 3), dtype=dtype)
        buf = self._host_transfer_buffer("matrix", src, "download")
        for lo in range(0, n, _HOST_TRANSFER_CHUNK_SIZE):
            count = min(_HOST_TRANSFER_CHUNK_SIZE, n - lo)
            self._extract_mat3_field_prefix(src, buf, lo, count)
            ti.sync()
            out[lo : lo + count] = buf[:count]
        return out

    def _download_scalar_field(self, src: TaichiField, n: int) -> np.ndarray:
        """Download the active scalar prefix without transferring capacity padding."""
        dtype = np.float32 if src.dtype == ti.f32 else np.float64
        if n == 0:
            return np.empty((0,), dtype=dtype)
        out = np.empty((n,), dtype=dtype)
        buf = self._host_transfer_buffer("scalar", src, "download")
        for lo in range(0, n, _HOST_TRANSFER_CHUNK_SIZE):
            count = min(_HOST_TRANSFER_CHUNK_SIZE, n - lo)
            self._extract_scalar_field_prefix(src, buf, lo, count)
            ti.sync()
            out[lo : lo + count] = buf[:count]
        return out

    def extract_target_velocity(self, n: int) -> np.ndarray:
        """Return first n target velocity as a NumPy array (no full alloc transfer)."""
        return self.download_vector_field(self.target_velocity, n)

    def _resize_filtered_fields(self, N: int) -> None:
        """
        Validate filtered particle capacity for zone-aware field evaluation.

        Args:
            N: Required number of filtered particles
        """
        if self._filtered_field_size >= N:
            return

        raise ValueError(
            f"Filtered evaluation requires {N} particles, but max_n_particles="
            f"{self._filtered_field_size}. Increase Numerics.max_n_particles "
            "before constructing the solver."
        )

    # TREECODE MANAGEMENT (for hierarchical methods)

    def _get_or_create_treecode(self, required_size: int, theta: float = 0.5) -> "TaichiTreecode":
        """Reuse tree scratch sized to the active source count.

        The source ceiling remains a hard limit. Scratch grows geometrically
        and releases its previous Taichi allocation before replacement.

        Args:
            required_size: Required max_n_particles capacity
            theta: Opening angle for MAC (updates cached tree's theta)

        Returns:
            TaichiTreecode: Cached or newly created treecode instance
        """
        from .induction.treecode.lbvh import TaichiTreecode

        required_size = int(required_size)
        if required_size < 0 or required_size > self.max_n_particles:
            raise ValueError(
                f"Treecode source count {required_size} exceeds declared "
                f"max_n_particles={self.max_n_particles}"
            )
        if self._treecode is None or required_size > self._treecode_max_particles:
            alloc_size = min(
                self.max_n_particles,
                max(_TREECODE_MIN_CAPACITY, required_size, 2 * self._treecode_max_particles),
            )
            if self._treecode is not None:
                ti.sync()
                previous = self._treecode
                self._treecode = None
                self._treecode_max_particles = 0
                self._target_tree_key = None
                previous.destroy()
            self._treecode = TaichiTreecode(
                max_n_particles=alloc_size,
                max_nodes=2 * alloc_size,
                theta=theta,
                kernel_type=self.particle_kernel,
                multipole_order=self.treecode_multipole_order,
                sort_particle_targets=self.treecode_sort_particle_targets,
                traversal_block_dim=self.treecode_traversal_block_dim,
                max_evaluation_points=self.max_evaluation_points,
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

    def _define_taichi_kernels(self) -> None:
        """
        Import and bind Taichi kernels based on selected particle kernel type.

        Dynamically loads kernel functions from the appropriate module
        (gaussian, super_gaussian, or winckelmans).
        """
        # Import the shared numerical and radial-kernel factories.
        from ..kernels.base import make_device_vortex_kernels
        from ..numerics.kernels_common import create_kernels

        kernel_functions = make_device_vortex_kernels(
            self.particle_kernel,
            self.accumulator_dtype,
        )

        # Retain the device radial functions for induction
        # backends that assemble additional device-resident passes (notably the
        # production FMM).  The public physics surface remains the assembled
        # kernels below; this private registry avoids a second q/zeta selector.
        self._kernel_functions = kernel_functions

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
            method: "direct" (O(N²) GPU summation) or "treecode" (Barnes-Hut O(N log N)).
            theta:  Barnes-Hut opening angle (treecode only; smaller = more accurate).
        """
        self.velocity_method = method.lower()
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
    def _copy_vec3(self, src: ti.template(), dst: ti.template(), N: ti.i32):  # type: ignore[invalid-annotation]
        """Copy the first N entries of one vec3 field into another."""
        for i in range(N):
            dst[i] = src[i]

    @ti.kernel
    def _copy_vec3_offset(self, src: ti.template(), dst: ti.template(), start: ti.i32, N: ti.i32):  # type: ignore[invalid-annotation]
        """Copy one bounded target batch into its field destination."""
        for i in range(N):
            dst[start + i] = src[i]

    @ti.kernel
    def _fill_vec3_from_background(
        self,
        dst: ti.template(),  # type: ignore[invalid-annotation]
        background: ti.template(),  # type: ignore[invalid-annotation]
        start: ti.i32,
        N: ti.i32,
    ):
        """Fill a field destination when a query has no contributing sources."""
        for i in range(N):
            dst[start + i] = background[None]

    @ti.kernel
    def _zero_vec3_field(self, field: ti.template(), N: ti.i32):  # type: ignore[invalid-annotation]
        """Zero the first N entries of a vector field."""
        for i in range(N):
            field[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def _copy_mat3(self, src: ti.template(), dst: ti.template(), N: ti.i32):  # type: ignore[invalid-annotation]
        """Copy the first N entries of one 3×3 matrix field into another."""
        for i in range(N):
            dst[i] = src[i]

    def _ensure_target_tree_current(
        self, particles: ParticleSources, theta: float
    ) -> "TaichiTreecode":
        """Return a tree built for the current particle source revision.

        ``Particles.state_revision`` changes only when a Biot--Savart source
        field (position, vortex strength, radius, or particle count) changes.  A
        missing revision deliberately disables reuse so lightweight external
        particle adapters cannot accidentally observe a stale hierarchy.
        """
        N = len(particles)
        tree = self._get_or_create_treecode(N, theta)
        revision = getattr(particles, "state_revision", None)
        key = None if revision is None else (id(tree), id(particles), int(revision), int(N))
        if key is None or self._target_tree_key != key:
            tree.build(particles.position, particles.vortex_strength, particles.core_radius, N)
            self._target_tree_key = key
        return tree

    # FIELD EVALUATION METHODS

    @overload
    def compute_target_velocity(
        self,
        particles: ParticleSources,
        target_position: np.ndarray | TaichiField,
        target_velocity: None = None,
        include_freestream: bool = True,
        zone_mask: np.ndarray | None = None,
    ) -> np.ndarray: ...

    @overload
    def compute_target_velocity(
        self,
        particles: ParticleSources,
        target_position: np.ndarray | TaichiField,
        target_velocity: TaichiField,
        include_freestream: bool = True,
        zone_mask: np.ndarray | None = None,
    ) -> np.ndarray | None: ...

    def compute_target_velocity(
        self,
        particles: ParticleSources,
        target_position: np.ndarray | TaichiField,
        target_velocity: TaichiField | None = None,
        include_freestream: bool = True,
        zone_mask: np.ndarray | None = None,
    ) -> np.ndarray | None:
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
            if not isinstance(particles, HostParticleSources):
                raise TypeError("a particle zone mask requires host copies of the particle fields")
            return self._compute_target_velocity_filtered(
                particles, target_position, zone_mask, include_freestream, target_velocity
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
            return None if target_velocity is not None else np.zeros((0, 3), dtype=self.np_dtype)

        if N == 0:
            if target_velocity is not None:
                self._fill_vec3_from_background(target_velocity, background_velocity, 0, M)
                return None
            result: np.ndarray | None = np.zeros((M, 3), dtype=self.np_dtype)
            if include_freestream:
                bg = particles.velocity_background_cpu()
                result += bg
            return result

        target_backend = getattr(self, "induction", None)
        from .induction.treecode.evaluator import TreecodeInduction

        if isinstance(target_backend, TreecodeInduction) or (
            not hasattr(target_backend, "evaluate_targets") and self.velocity_method == "treecode"
        ):
            if target_velocity is None:
                return self.compute_target_velocity_hierarchical(
                    particles,
                    target_position,
                    theta=self.velocity_theta,
                    include_freestream=include_freestream,
                )
            tree = self._ensure_target_tree_current(particles, self.velocity_theta)
            background = None
            if include_freestream:
                bg = particles.velocity_background
                background = np.array([bg[None][0], bg[None][1], bg[None][2]], dtype=np.float32)
            for start, stop in self._target_batch_slices(M):
                values = tree.compute_target_velocity(target_position[start:stop], background)
                self.upload_vector_array(values, self.target_velocity)
                self._copy_vec3_offset(self.target_velocity, target_velocity, start, stop - start)
            return None

        # Route arbitrary target fields through the selected induction backend.
        if target_backend is not None and hasattr(target_backend, "evaluate_targets"):
            result = np.empty((M, 3), dtype=self.np_dtype) if target_velocity is None else None
            for start, stop in self._target_batch_slices(M):
                count = stop - start
                self.upload_vector_array(target_position[start:stop], self.target_position, count)
                target_backend.evaluate_targets(
                    target_position=self.target_position,
                    source_position=particles.position,
                    source_vortex_strength=particles.vortex_strength,
                    source_core_radius=particles.core_radius,
                    target_velocity=self.target_velocity,
                    target_velocity_gradient=None,
                    target_count=count,
                    source_count=N,
                    include_freestream=include_freestream,
                    background_velocity=background_velocity,
                )
                if target_velocity is not None:
                    self._copy_vec3_offset(self.target_velocity, target_velocity, start, count)
                elif result is not None:
                    result[start:stop] = self.extract_target_velocity(count)
            return result

        result = np.empty((M, 3), dtype=self.np_dtype) if target_velocity is None else None
        for start, stop in self._target_batch_slices(M):
            count = stop - start
            self.upload_vector_array(target_position[start:stop], self.target_position, count)
            self.compute_target_velocity_kernel(
                self.target_position,
                particles.position,
                particles.vortex_strength,
                particles.core_radius,
                self.target_velocity,
                background_velocity,
                count,
                N,
            )
            if target_velocity is not None:
                self._copy_vec3_offset(self.target_velocity, target_velocity, start, count)
            elif result is not None:
                result[start:stop] = self.extract_target_velocity(count)
        return result

    def compute_transport_target_velocity(
        self,
        particles: ParticleSources,
        target_position: np.ndarray,
        target_core_radius: np.ndarray | float,
        *,
        include_freestream: bool = True,
        use_induction_backend: bool = False,
    ) -> np.ndarray:
        """Evaluate finite-target probe velocities with the exact Taichi pair sum.

        Uses the configured vortex kernel and arithmetic mean of target/source
        radii, as in particle transport. Target workspaces are reused and no
        M-by-N host arrays are built. This remains a direct reference operator
        even when the advancing solver selects a hierarchical approximation.
        Inputs are positions (M, 3) in m and positive radii (M,) or scalar in m;
        the returned velocity (M, 3) is in m/s and includes optional freestream.

        For a scalar target radius, use_induction_backend=True replaces each
        source core by its arithmetic mean with that target core and evaluates
        through the configured target backend. This retains the pair filter
        with the backend's approximation. Nonuniform target cores and the
        default use the direct reference operator.
        """
        targets = np.asarray(target_position, dtype=self.np_dtype).reshape(-1, 3)
        radii = np.broadcast_to(
            np.asarray(target_core_radius, dtype=self.np_dtype), (len(targets),)
        )
        if not np.isfinite(targets).all() or not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError(
                "target positions must be finite and target_core_radius finite and positive"
            )
        if not len(targets):
            return np.zeros((0, 3), dtype=self.np_dtype)
        if len(targets) > self._target_field_size:
            return np.concatenate(
                [
                    self.compute_transport_target_velocity(
                        particles,
                        targets[start:stop],
                        target_core_radius
                        if np.ndim(target_core_radius) == 0
                        else radii[start:stop],
                        include_freestream=include_freestream,
                        use_induction_backend=use_induction_backend,
                    )
                    for start, stop in self._target_batch_slices(len(targets))
                ]
            )
        self._resize_target_fields(len(targets))
        if not hasattr(self, "_transport_target_core"):
            self._transport_target_core = ti.field(
                dtype=self.accumulator_dtype, shape=self.max_evaluation_points
            )
        self.upload_vector_array(targets, self.target_position, len(targets))
        backend = getattr(self, "induction", None)
        if use_induction_backend and np.ndim(target_core_radius) == 0 and backend is not None:
            if not hasattr(self, "_transport_source_core"):
                self._transport_source_core = ti.field(
                    dtype=self.accumulator_dtype, shape=self.max_n_particles
                )
            count = len(particles)
            cores = self._download_scalar_field(particles.core_radius, count)
            self._upload_scalar_array(
                0.5 * (cores + float(target_core_radius)), self._transport_source_core, count
            )
            backend.evaluate_targets(
                target_position=self.target_position,
                source_position=particles.position,
                source_vortex_strength=particles.vortex_strength,
                source_core_radius=self._transport_source_core,
                target_velocity=self.target_velocity,
                target_velocity_gradient=None,
                target_count=len(targets),
                source_count=count,
                include_freestream=include_freestream,
                background_velocity=particles.velocity_background
                if include_freestream
                else self._zero_velocity,
            )
            return self.download_vector_field(self.target_velocity, len(targets))
        self._upload_scalar_array(
            np.ascontiguousarray(radii), self._transport_target_core, len(targets)
        )
        self.compute_transport_target_velocity_kernel(
            self.target_position,
            self._transport_target_core,
            particles.position,
            particles.vortex_strength,
            particles.core_radius,
            self.target_velocity,
            particles.velocity_background if include_freestream else self._zero_velocity,
            len(targets),
            len(particles),
        )
        return self.download_vector_field(self.target_velocity, len(targets))

    def _compute_target_velocity_taichi(
        self,
        particles: ParticleSources,
        target_position: TaichiField,
        target_velocity: TaichiField | None,
        include_freestream: bool,
        background_velocity: TaichiField,
        N: int,
    ) -> np.ndarray | None:
        """Handle velocity computation when target_position is a Taichi field."""
        M = target_position.shape[0]
        if self._target_field_size < M:
            positions = self.download_vector_field(target_position, M)
            result = self.compute_target_velocity(
                particles,
                positions,
                include_freestream=include_freestream,
            )
            if target_velocity is not None:
                self.upload_vector_array(result, target_velocity, M)
                return None
            return result
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

        target_backend = getattr(self, "induction", None)
        from .induction.treecode.evaluator import TreecodeInduction

        if isinstance(target_backend, TreecodeInduction) or (
            not hasattr(target_backend, "evaluate_targets") and self.velocity_method == "treecode"
        ):
            positions = self.download_vector_field(target_position, M)
            result = self.compute_target_velocity(
                particles, positions, include_freestream=include_freestream
            )
            if target_velocity is not None:
                self.upload_vector_array(result, target_velocity, M)
                return None
            return result
        if target_backend is not None and hasattr(target_backend, "evaluate_targets"):
            self._resize_target_fields(M)
            out_field = target_velocity if target_velocity is not None else self.target_velocity
            target_backend.evaluate_targets(
                target_position=target_position,
                source_position=particles.position,
                source_vortex_strength=particles.vortex_strength,
                source_core_radius=particles.core_radius,
                target_velocity=out_field,
                target_velocity_gradient=None,
                target_count=M,
                source_count=N,
                include_freestream=include_freestream,
                background_velocity=background_velocity,
            )
            if target_velocity is None:
                return self.download_vector_field(out_field, M)
            return None

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
            return self.download_vector_field(out_field, M)
        return None

    def _compute_target_velocity_filtered(
        self,
        particles: HostParticleSources,
        target_position: np.ndarray | TaichiField,
        zone_mask: np.ndarray,
        include_freestream: bool = True,
        target_velocity: TaichiField | None = None,
    ) -> np.ndarray | None:
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
            return None if target_velocity is not None else np.zeros((0, 3), dtype=self.np_dtype)

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
            if target_velocity is not None:
                self._fill_vec3_from_background(
                    target_velocity,
                    particles.velocity_background if include_freestream else self._zero_velocity,
                    0,
                    M,
                )
                return None
            result: np.ndarray | None
            if include_freestream:
                bg = particles.velocity_background_cpu()
                result = np.tile(bg, (M, 1))
            else:
                result = np.zeros((M, 3), dtype=self.np_dtype)
            return result

        # Resize target and filtered fields (cached to prevent memory leak)
        self._resize_filtered_fields(N_filtered)

        # Copy filtered data through fixed-shape external buffers.
        self.upload_vector_array(filtered_pos, self._filtered_pos, N_filtered)
        self.upload_vector_array(
            filtered_vortex_strength, self._filtered_vortex_strength, N_filtered
        )
        self._upload_scalar_array(filtered_rad, self._filtered_rad, N_filtered)

        # Select background velocity
        background_velocity = (
            particles.velocity_background if include_freestream else self._zero_velocity
        )

        # Compute with filtered particles in the fixed target workspace.
        result = np.empty((M, 3), dtype=self.np_dtype) if target_velocity is None else None
        for start, stop in self._target_batch_slices(M):
            count = stop - start
            self.upload_vector_array(target_position[start:stop], self.target_position, count)
            self.compute_target_velocity_kernel(
                self.target_position,
                self._filtered_pos,
                self._filtered_vortex_strength,
                self._filtered_rad,
                self.target_velocity,
                background_velocity,
                count,
                N_filtered,
            )
            if target_velocity is not None:
                self._copy_vec3_offset(self.target_velocity, target_velocity, start, count)
            elif result is not None:
                result[start:stop] = self.extract_target_velocity(count)
        return result

    def compute_vorticities(self, particles: "Particles") -> None:
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

        slab = getattr(self, "_slip_slab_bounds", None)
        if slab is not None:
            self._compute_slab_vorticities(particles, slab)
            return
        if self.particle_kernel == "gaussian" and self.accumulator_dtype == ti.f32 and N >= 2048:
            self._compute_gaussian_vorticities(particles)
            return
        self.compute_vorticities_kernel(
            particles.position,
            particles.vortex_strength,
            particles.core_radius,
            particles.vorticity,
            N,
        )

    def _compute_slab_vorticities(self, particles: "Particles", slab: tuple[float, float]) -> None:
        """Vorticity at the particles from themselves and their slip-slab images.

        Images whose centres lie within the kernel reach of the slab are summed
        directly with the particles; beyond that reach a core encloses all but
        ``_VORTICITY_TAIL`` of its circulation.
        """
        N = len(particles)
        position = np.asarray(particles.position_cpu()[:N], dtype=np.float64)
        strength = np.asarray(particles.vortex_strength_cpu()[:N], dtype=np.float64)
        core = np.asarray(particles.core_radius_cpu()[:N], dtype=np.float64)
        reach = _kernel_reach(self.particle_kernel) * float(core.max())
        image_position, image_strength, owner = slab_image_particles(
            position, strength, slab, reach
        )
        source_position = np.concatenate((position, image_position))
        source_strength = np.concatenate((strength, image_strength))
        source_core = np.concatenate((core, core[owner]))

        size = self._target_field_size
        if getattr(self, "_slab_source_size", 0) != size:
            self._slab_source_position = ti.Vector.field(3, self.accumulator_dtype, shape=size)
            self._slab_source_strength = ti.Vector.field(3, self.accumulator_dtype, shape=size)
            self._slab_source_core = ti.field(self.accumulator_dtype, shape=size)
            self._slab_source_size = size
        vorticity = np.zeros((N, 3), dtype=np.float64)
        for start, stop in self._target_batch_slices(N):
            count = stop - start
            self.upload_vector_array(position[start:stop], self.target_position, count)
            for first, last in self._target_batch_slices(len(source_position)):
                sources = last - first
                self.upload_vector_array(
                    source_position[first:last], self._slab_source_position, sources
                )
                self.upload_vector_array(
                    source_strength[first:last], self._slab_source_strength, sources
                )
                self._upload_scalar_array(source_core[first:last], self._slab_source_core, sources)
                self.compute_target_vorticity_kernel(
                    self.target_position,
                    self._slab_source_position,
                    self._slab_source_strength,
                    self._slab_source_core,
                    self.target_vorticity,
                    count,
                    sources,
                )
                vorticity[start:stop] += self.download_vector_field(self.target_vorticity, count)
        self.upload_vector_array(vorticity, particles.vorticity, N)

    def _compute_gaussian_vorticities(self, particles: "Particles") -> None:
        """Reuse induction hierarchy storage, or allocate bounded diagnostic scratch."""
        from .induction.treecode.lbvh import TaichiTreecode

        count = len(particles)
        backend = getattr(self, "induction", None)
        evaluate = getattr(backend, "evaluate_gaussian_source_vorticity", None)
        if callable(evaluate):
            evaluate(
                position=particles.position,
                vortex_strength=particles.vortex_strength,
                core_radius=particles.core_radius,
                count=count,
                vorticity_out=particles.vorticity,
            )
            return
        tree: TaichiTreecode | None
        if self.velocity_method == "treecode":
            tree = self._ensure_target_tree_current(particles, self.velocity_theta)
        else:
            tree = self._vorticity_tree
            if tree is None or count > tree.max_n_particles:
                capacity = min(
                    self.max_n_particles, max(count, 2 * tree.max_n_particles if tree else 2048)
                )
                if tree is not None:
                    ti.sync()
                    tree.destroy()
                    self._vorticity_tree = None
                tree = TaichiTreecode(
                    max_n_particles=capacity,
                    max_nodes=2 * capacity,
                    kernel_type="gaussian",
                    hierarchy_only=True,
                    max_evaluation_points=min(4096, self.max_evaluation_points),
                )
                self._vorticity_tree = tree
            tree.build(particles.position, particles.vortex_strength, particles.core_radius, count)
        tree.compute_gaussian_particle_vorticity(particles.vorticity, count)

    def compute_target_velocity_gradient(
        self, particles: ParticleSources, target_position: np.ndarray
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

        target_backend = getattr(self, "induction", None)
        from .induction.treecode.evaluator import TreecodeInduction

        if isinstance(target_backend, TreecodeInduction) or (
            not hasattr(target_backend, "evaluate_targets") and self.velocity_method == "treecode"
        ):
            return self.compute_target_velocity_gradient_hierarchical(
                particles, target_position, theta=self.velocity_theta
            )
        if target_backend is not None and hasattr(target_backend, "evaluate_targets"):
            result = np.empty((M, 9), dtype=self.np_dtype)
            for start, stop in self._target_batch_slices(M):
                count = stop - start
                self.upload_vector_array(target_position[start:stop], self.target_position, count)
                target_backend.evaluate_targets(
                    target_position=self.target_position,
                    source_position=particles.position,
                    source_vortex_strength=particles.vortex_strength,
                    source_core_radius=particles.core_radius,
                    target_velocity=None,
                    target_velocity_gradient=self.target_velocity_gradient,
                    target_count=count,
                    source_count=N,
                    include_freestream=False,
                    background_velocity=self._zero_velocity,
                )
                result[start:stop] = self._download_matrix_field(
                    self.target_velocity_gradient, count
                ).reshape(-1, 9)
            return result

        result = np.empty((M, 9), dtype=self.np_dtype)
        for start, stop in self._target_batch_slices(M):
            count = stop - start
            self.upload_vector_array(target_position[start:stop], self.target_position, count)
            self.compute_target_velocity_gradient_kernel(
                self.target_position,
                particles.position,
                particles.vortex_strength,
                particles.core_radius,
                self.target_velocity_gradient,
                count,
                N,
            )
            result[start:stop] = self._download_matrix_field(
                self.target_velocity_gradient, count
            ).reshape(-1, 9)
        return result

    def compute_target_velocity_hierarchical(
        self,
        particles: ParticleSources,
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

        tree = self._ensure_target_tree_current(particles, theta)

        background_vel = None
        if include_freestream:
            bg = particles.velocity_background
            background_vel = np.array([bg[None][0], bg[None][1], bg[None][2]], dtype=np.float32)

        return np.concatenate(
            [
                tree.compute_target_velocity(target_position[start:stop], background_vel)
                for start, stop in self._target_batch_slices(M)
            ]
        )

    def compute_target_velocity_gradient_hierarchical(
        self, particles: ParticleSources, target_position: np.ndarray, theta: float = 0.5
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

        tree = self._ensure_target_tree_current(particles, theta)

        return np.concatenate(
            [
                tree.compute_target_velocity_gradient(target_position[start:stop]).reshape(-1, 9)
                for start, stop in self._target_batch_slices(M)
            ]
        )

    def compute_target_velocity_and_gradients_hierarchical(
        self,
        particles: ParticleSources,
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

        tree = self._ensure_target_tree_current(particles, theta)
        background = None
        if include_freestream:
            bg = particles.velocity_background
            background = np.array([bg[None][0], bg[None][1], bg[None][2]], dtype=np.float32)
        chunks = [
            tree.compute_target_velocity_and_gradients(target_position[start:stop], background)
            for start, stop in self._target_batch_slices(M)
        ]
        return (
            np.concatenate([velocity for velocity, _gradient in chunks]),
            np.concatenate([gradient.reshape(-1, 9) for _velocity, gradient in chunks]),
        )

    def compute_target_velocity_gradient_consistent(
        self, particles: ParticleSources, target_position: np.ndarray
    ) -> np.ndarray:
        """Evaluate a target Jacobian with the configured particle backend."""
        if self.velocity_method == "treecode":
            return self.compute_target_velocity_gradient_hierarchical(
                particles, target_position, theta=self.velocity_theta
            )
        return self.compute_target_velocity_gradient(particles, target_position)

    def compute_target_velocity_and_gradients_consistent(
        self,
        particles: ParticleSources,
        target_position: np.ndarray,
        *,
        include_freestream: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate velocity and Jacobian in one backend pass per target batch.

        Both fields use the same current source state and source-only target
        core convention. No fields or source validity survive this call: a
        following query may have changed particle values in the same objects.
        """
        if self.velocity_method == "treecode":
            return self.compute_target_velocity_and_gradients_hierarchical(
                particles,
                target_position,
                theta=self.velocity_theta,
                include_freestream=include_freestream,
            )
        backend = getattr(self, "induction", None)
        if backend is not None and hasattr(backend, "evaluate_targets"):
            source_count = len(particles)
            target_count = len(target_position)
            velocity = np.zeros((target_count, 3), dtype=self.np_dtype)
            gradient = np.zeros((target_count, 9), dtype=self.np_dtype)
            if target_count == 0:
                return velocity, gradient
            if source_count == 0:
                if include_freestream:
                    velocity += particles.velocity_background_cpu()
                return velocity, gradient
            background = (
                particles.velocity_background if include_freestream else self._zero_velocity
            )
            for start, stop in self._target_batch_slices(target_count):
                count = stop - start
                self.upload_vector_array(target_position[start:stop], self.target_position, count)
                backend.evaluate_targets(
                    target_position=self.target_position,
                    source_position=particles.position,
                    source_vortex_strength=particles.vortex_strength,
                    source_core_radius=particles.core_radius,
                    target_velocity=self.target_velocity,
                    target_velocity_gradient=self.target_velocity_gradient,
                    target_count=count,
                    source_count=source_count,
                    include_freestream=include_freestream,
                    background_velocity=background,
                )
                velocity[start:stop] = self.extract_target_velocity(count)
                gradient[start:stop] = self._download_matrix_field(
                    self.target_velocity_gradient, count
                ).reshape(-1, 9)
            return velocity, gradient
        return (
            self.compute_target_velocity(
                particles,
                target_position,
                include_freestream=include_freestream,
                zone_mask=None,
            ),
            self.compute_target_velocity_gradient(particles, target_position),
        )
