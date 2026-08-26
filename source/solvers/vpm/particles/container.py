"""Particle storage for the VPM solver."""

import numpy as np

try:
    import pyvista as pv

    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None
import taichi as ti

# Import VPM constants
from source import log_style
from source.vtk_output import write_vtk_dataset
from source.write_precision import DEFAULT_WRITE_PRECISION, cast_for_write

from ..config.constants import MAX_N_PARTICLES
from ..config.state import cached_particle_property
from ..io.logging import Logging


def _validate_finite_array(arr, name: str) -> None:
    """Check that an array contains only finite (non-NaN, non-Inf) values.

    Parameters
    ----------
    arr : np.ndarray
        Array to validate.
    name : str
        Human-readable name of the array, used in error messages.

    Raises
    ------
    ValueError
        If any element is NaN or Inf, with a count of each.
    """
    if not np.all(np.isfinite(arr)):
        nan_count = np.sum(np.isnan(arr))
        inf_count = np.sum(np.isinf(arr))
        raise ValueError(
            f"Invalid particle data in '{name}': "
            f"detected {nan_count} NaN and {inf_count} Inf values. "
            f"Cannot add particles with non-finite values."
        )


def _coerce_int_id_array(arr, N: int) -> np.ndarray:
    """Coerce an integer ID array to a contiguous ``np.int32`` array.

    Parameters
    ----------
    arr : np.ndarray | int | None
        Input array, a scalar int, or ``None``.
    N : int
        Desired length when ``arr`` is ``None`` or a scalar.

    Returns
    -------
    np.ndarray
        C-contiguous ``np.int32`` array of length ``N``.

    Examples
    --------
    >>> _coerce_int_id_array(None, 5)
    array([0, 0, 0, 0, 0], dtype=int32)
    >>> _coerce_int_id_array(3, 5)
    array([3, 3, 3, 3, 3], dtype=int32)
    """
    if arr is None:
        return np.zeros(N, dtype=np.int32)
    if isinstance(arr, int):
        return np.full(N, arr, dtype=np.int32)
    return np.ascontiguousarray(arr, dtype=np.int32)


@ti.data_oriented
class Particles:
    """
    A class to manage and manipulate a collection of particles for vortex particle methods (VPM).

    The class uses Taichi fields for efficient GPU/CPU computation with all data kept on GPU by default.
    CPU access methods are provided separately (e.g., position_cpu, velocity_cpu).

    Attributes:
        position (ti.Vector.field): Taichi field of particle position, shape (N, 3).
        velocity (ti.Vector.field): Taichi field of particle velocity, shape (N, 3).
        vortex_strength (ti.Vector.field): Particle alpha = omega*V [m³/s], shape (N, 3).
        core_radius (ti.field): Taichi field of particle core radius, shape (N,).
        particle_volume (ti.field): Taichi field of particle volume, shape (N,).
        kinematic_viscosity (ti.field): Taichi field of molecular kinematic viscosity, shape (N,).
        eddy_viscosity (ti.field): Taichi field of turbulent kinematic viscosity, shape (N,).
        effective_viscosity (ti.field): Taichi field of effective viscosity, shape (N,).
        strain_rate (ti.Matrix.field): Taichi field of strain-rate tensors, shape (N, 3, 3).
        vorticity (ti.Vector.field): Taichi field of particle vorticity, shape (N, 3).
        zone_id (ti.field): Taichi field of zone IDs (spatial zones), shape (N,).
    """

    _COPY_CHUNK_SIZE = 65_536

    def __init__(self, max_n_particles=MAX_N_PARTICLES, float_dtype: str = "f32"):
        """
        Initialize the Particles class with Taichi fields.

        Args:
            max_n_particles (int): Fixed particle capacity allocated at startup.
            float_dtype (str): 'f32' (default) or 'f64' - precision for particle data
        """
        self._max_particles = max_n_particles
        self.float_dtype = float_dtype or "f32"
        self._taichi_dtype = ti.f32 if self.float_dtype == "f32" else ti.f64
        self.n_particles_total = 0
        self.step = 0  # For cache invalidation
        self._cache_step = -1  # Track when cache was last updated
        # Monotone source-state version for consumers that cache spatial
        # acceleration structures.  Velocity/diagnostic writes do not affect
        # Biot-Savart sources; position, vortex strength, core_radius, and population do.
        self._state_revision = 0
        # NumPy dtype matching Taichi float precision (avoids repeated branching)
        self._np_float_dtype = np.float32 if self.float_dtype == "f32" else np.float64
        # External ndarray bindings are cached by Taichi's Vulkan/Metal
        # backends.  Reusing one ndarray for kernels specialised on different
        # template fields can eventually alias those bindings in long runs
        # (for example a vortex strength download returning position).  Give each
        # field and transfer direction its own fixed-shape staging array.
        self._host_vector_chunks = {}
        self._host_scalar_chunks = {}
        self._host_matrix_chunks = {}
        self._host_int_chunks = {}
        self._native_vector_uploads = {}
        self._native_scalar_uploads = {}
        self._native_matrix_uploads = {}
        self._native_int_uploads = {}
        # Initialize Taichi fields for particle properties
        self._init_taichi_fields()

    @property
    def capacity(self) -> int:
        """Allocated particle capacity, i.e. the real ceiling for regeneration."""
        return int(self._max_particles)

    @property
    def state_revision(self) -> int:
        """Version of the particle fields that define induced velocity."""
        return self._state_revision

    def touch_state(self) -> None:
        """Invalidate acceleration structures after a source-state mutation."""
        self._state_revision += 1
        self._cache_step = -1

    def _init_taichi_fields(self):
        """Initialize all Taichi fields for particle data storage with configurable dtype."""
        dtype = self._taichi_dtype
        # Vector fields for 3D properties
        self.position = ti.Vector.field(3, dtype=dtype, shape=self._max_particles)
        self.velocity = ti.Vector.field(3, dtype=dtype, shape=self._max_particles)
        self.vortex_strength = ti.Vector.field(3, dtype=dtype, shape=self._max_particles)
        self.vorticity = ti.Vector.field(3, dtype=dtype, shape=self._max_particles)
        # Scalar fields
        self.core_radius = ti.field(dtype=dtype, shape=self._max_particles)
        self.particle_volume = ti.field(dtype=dtype, shape=self._max_particles)
        self.kinematic_viscosity = ti.field(dtype=dtype, shape=self._max_particles)
        self.eddy_viscosity = ti.field(dtype=dtype, shape=self._max_particles)
        self.effective_viscosity = ti.field(dtype=dtype, shape=self._max_particles)
        self.group_id = ti.field(dtype=ti.i32, shape=self._max_particles)
        # Matrix fields
        self.velocity_gradient = ti.Matrix.field(3, 3, dtype=dtype, shape=self._max_particles)
        self.strain_rate = ti.Matrix.field(3, 3, dtype=dtype, shape=self._max_particles)
        self.zone_id = ti.field(dtype=ti.i32, shape=self._max_particles)

        # Device-side counter for atomic operations
        self.device_n_particles = ti.field(dtype=ti.i32, shape=())

        # Removal tag field for GPU-based particle filtering (1 = remove, 0 = keep)
        self._removal_tags = ti.field(dtype=ti.i32, shape=self._max_particles)

        # Device-side accumulators for subset moment reductions (e.g. vortex strength
        # and linear impulse of removed particles) — kept on device to avoid a
        # full position/vortex strength download just to sum a handful of indices.
        self._subset_vortex_strength = ti.Vector.field(3, dtype=dtype, shape=())
        self._subset_impulse = ti.Vector.field(3, dtype=dtype, shape=())

        # Global background velocity (single 3D vector shared by all particles)
        self.velocity_background = ti.Vector.field(3, dtype=dtype, shape=())
        self.velocity_background[None] = [0.0, 0.0, 0.0]

    def sync_device_counter(self):
        """Sync host particle count to device field."""
        self.device_n_particles[None] = self.n_particles_total

    def sync_host_counter(self):
        """Sync device particle count to host."""
        self.n_particles_total = self.device_n_particles[None]

    def _grow_capacity(self, needed: int) -> None:
        """Validate that a particle insertion fits the startup allocation."""
        if needed <= self._max_particles:
            return
        raise ValueError(
            f"Particle insertion requires capacity {needed}, but max_n_particles="
            f"{self._max_particles}. Increase VPMSetup.max_n_particles before "
            "constructing the solver; runtime Taichi field resizing is disabled "
            "because replaced fields retain device memory."
        )

    def resize(self, new_capacity: int) -> None:
        """
        Validate a requested particle capacity without reallocating fields.

        Args:
            new_capacity: New maximum number of particles.
        """
        if new_capacity != self._max_particles:
            raise ValueError(
                f"Particle capacity is fixed at {self._max_particles}; create a new "
                f"solver with max_n_particles={new_capacity} instead of resizing it."
            )

    # ---- Prefix-extraction helpers (GPU → CPU, only active prefix) ----

    def _host_transfer_buffer(self, family: str, field, direction: str) -> np.ndarray:
        """Return a field-specific fixed-shape external-array staging buffer.

        Taichi's Vulkan/Metal external-array staging can cache one device buffer
        per distinct ndarray shape.  Long runs with particle counts changing at
        every remeshing/regeneration step therefore leak cached staging buffers
        if we pass arrays of shape ``(N, ...)`` or tail slices of varying size.
        Keep all solver-loop transfers at one fixed chunk shape and pass the
        live count as a scalar kernel argument instead.
        """
        key = (direction, id(field))
        chunk = self._COPY_CHUNK_SIZE
        if family == "vector":
            buffers = self._host_vector_chunks
            shape = (chunk, 3)
            dtype = self._np_float_dtype
        elif family == "scalar":
            buffers = self._host_scalar_chunks
            shape = (chunk,)
            dtype = self._np_float_dtype
        elif family == "matrix":
            buffers = self._host_matrix_chunks
            shape = (chunk, 3, 3)
            dtype = self._np_float_dtype
        elif family == "int":
            buffers = self._host_int_chunks
            shape = (chunk,)
            dtype = np.int32
        else:
            raise ValueError(f"Unknown host transfer buffer family {family!r}")
        if key not in buffers:
            buffers[key] = np.empty(shape, dtype=dtype)
        return buffers[key]

    def _extract_scalar(self, field, n):
        """Return a detached NumPy copy of the first ``n`` scalar entries.

        Use bounded, field-specific transfers instead of ``field.to_numpy()``.
        Full-allocation Vulkan readbacks have returned corrupted data after a
        long mixed treecode/grid workload even though the device field itself
        remained valid.  The fixed staging identity also prevents Taichi from
        accumulating one external allocation for every changing active count.
        """
        if n == 0:
            return np.empty((0,), dtype=self._np_float_dtype)
        out = np.empty((n,), dtype=self._np_float_dtype)
        buf = self._host_transfer_buffer("scalar", field, "download")
        for lo in range(0, n, self._COPY_CHUNK_SIZE):
            count = min(self._COPY_CHUNK_SIZE, n - lo)
            self._extract_scalar_prefix(field, buf, lo, count)
            ti.sync()
            out[lo : lo + count] = buf[:count]
        return out

    def _extract_vector(self, field, n):
        """Return a detached NumPy copy of the first ``n`` vector entries."""
        if n == 0:
            return np.empty((0, 3), dtype=self._np_float_dtype)
        out = np.empty((n, 3), dtype=self._np_float_dtype)
        buf = self._host_transfer_buffer("vector", field, "download")
        for lo in range(0, n, self._COPY_CHUNK_SIZE):
            count = min(self._COPY_CHUNK_SIZE, n - lo)
            self._extract_vector_prefix(field, buf, lo, count)
            ti.sync()
            out[lo : lo + count] = buf[:count]
        return out

    def _extract_matrix(self, field, n):
        """Return a detached NumPy copy of the first ``n`` matrix entries."""
        if n == 0:
            return np.empty((0, 3, 3), dtype=self._np_float_dtype)
        out = np.empty((n, 3, 3), dtype=self._np_float_dtype)
        buf = self._host_transfer_buffer("matrix", field, "download")
        for lo in range(0, n, self._COPY_CHUNK_SIZE):
            count = min(self._COPY_CHUNK_SIZE, n - lo)
            self._extract_matrix_prefix(field, buf, lo, count)
            ti.sync()
            out[lo : lo + count] = buf[:count]
        return out

    def _extract_int(self, field, n):
        """Return a detached NumPy copy of the first ``n`` integer entries."""
        if n == 0:
            return np.empty((0,), dtype=np.int32)
        out = np.empty((n,), dtype=np.int32)
        buf = self._host_transfer_buffer("int", field, "download")
        for lo in range(0, n, self._COPY_CHUNK_SIZE):
            count = min(self._COPY_CHUNK_SIZE, n - lo)
            self._extract_int_prefix(field, buf, lo, count)
            ti.sync()
            out[lo : lo + count] = buf[:count]
        return out

    def _replace_field_native(self, family: str, field, values: np.ndarray, count: int) -> None:
        """Replace a field through Taichi's native fixed-shape ndarray path.

        The custom templated prefix kernels are efficient for incremental
        uploads, but long Vulkan runs have shown cross-field external-array
        binding corruption during full cloud replacement.  Native
        ``from_numpy`` is the backend-supported path; persistent per-field
        arrays keep its external allocation shape and identity fixed.
        """
        key = id(field)
        if family == "vector":
            buffers = self._native_vector_uploads
            shape = (self._max_particles, 3)
            dtype = self._np_float_dtype
        elif family == "scalar":
            buffers = self._native_scalar_uploads
            shape = (self._max_particles,)
            dtype = self._np_float_dtype
        elif family == "matrix":
            buffers = self._native_matrix_uploads
            shape = (self._max_particles, 3, 3)
            dtype = self._np_float_dtype
        elif family == "int":
            buffers = self._native_int_uploads
            shape = (self._max_particles,)
            dtype = np.int32
        else:
            raise ValueError(f"Unknown native replacement buffer family {family!r}")
        if key not in buffers:
            buffers[key] = np.empty(shape, dtype=dtype)
        buffer = buffers[key]
        buffer[:count] = values[:count]
        field.from_numpy(buffer)
        ti.sync()

    def _extract_cpu_data(self, n_particles_total):
        """Extract current data as NumPy arrays (only active prefix)."""
        n = n_particles_total
        return {
            "position": self._extract_vector(self.position, n),
            "velocity": self._extract_vector(self.velocity, n),
            "vortex_strength": self._extract_vector(self.vortex_strength, n),
            "core_radius": self._extract_scalar(self.core_radius, n),
            "particle_volume": self._extract_scalar(self.particle_volume, n),
            "kinematic_viscosity": self._extract_scalar(self.kinematic_viscosity, n),
            "eddy_viscosity": self._extract_scalar(self.eddy_viscosity, n),
            "effective_viscosity": self._extract_scalar(self.effective_viscosity, n),
            "group_id": self._extract_int(self.group_id, n),
            "velocity_gradient": self._extract_matrix(self.velocity_gradient, n),
            "strain_rate": self._extract_matrix(self.strain_rate, n),
            "vorticity": self._extract_vector(self.vorticity, n),
            "zone_id": self._extract_int(self.zone_id, n),
        }

    @ti.kernel
    def _compute_strain_rate_tensor(
        self, velocity_gradient: ti.template(), strain_rate: ti.template()
    ):  # type: ignore
        """Compute strain rate tensor from velocity gradient tensor."""
        for i in range(velocity_gradient.shape[0]):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    strain_rate[i][j, k] = 0.5 * (
                        velocity_gradient[i][j, k] + velocity_gradient[i][k, j]
                    )

    @ti.kernel
    def _copy_to_taichi_vectors(
        self, src: ti.types.ndarray(), dest: ti.template(), start_idx: ti.i32, count: ti.i32
    ):  # type: ignore
        """Copy NumPy array data to Taichi vector field."""
        for i in range(count):
            for k in ti.static(range(3)):
                dest[start_idx + i][k] = src[i, k]

    @ti.kernel
    def _copy_to_taichi_scalars(
        self, src: ti.types.ndarray(), dest: ti.template(), start_idx: ti.i32, count: ti.i32
    ):  # type: ignore
        """Copy NumPy array data to Taichi scalar field."""
        for i in range(count):
            dest[start_idx + i] = src[i]

    @ti.kernel
    def _copy_to_taichi_matrices(
        self, src: ti.types.ndarray(), dest: ti.template(), start_idx: ti.i32, count: ti.i32
    ):  # type: ignore
        """Copy NumPy array data to Taichi matrix field."""
        for i in range(count):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dest[start_idx + i][j, k] = src[i, j, k]

    @ti.kernel
    def _copy_to_taichi_ints(
        self, src: ti.types.ndarray(), dest: ti.template(), start_idx: ti.i32, count: ti.i32
    ):  # type: ignore
        """Copy NumPy integer array data to Taichi field."""
        for i in range(count):
            dest[start_idx + i] = src[i]

    # ---- Prefix extraction kernels (avoid full MAX_N_PARTICLES to_numpy()) ----

    @ti.kernel
    def _extract_scalar_prefix(
        self, src: ti.template(), dst: ti.types.ndarray(), start_idx: ti.i32, n: ti.i32
    ):  # type: ignore
        """Copy first n scalar entries from Taichi field to NumPy array."""
        for i in range(n):
            dst[i] = src[start_idx + i]

    @ti.kernel
    def _extract_vector_prefix(
        self, src: ti.template(), dst: ti.types.ndarray(), start_idx: ti.i32, n: ti.i32
    ):  # type: ignore
        """Copy first n vector entries from Taichi field to NumPy array."""
        for i in range(n):
            for k in ti.static(range(3)):
                dst[i, k] = src[start_idx + i][k]

    @ti.kernel
    def _extract_matrix_prefix(
        self, src: ti.template(), dst: ti.types.ndarray(), start_idx: ti.i32, n: ti.i32
    ):  # type: ignore
        """Copy first n matrix entries from Taichi field to NumPy array."""
        for i in range(n):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dst[i, j, k] = src[start_idx + i][j, k]

    @ti.kernel
    def _extract_int_prefix(
        self, src: ti.template(), dst: ti.types.ndarray(), start_idx: ti.i32, n: ti.i32
    ):  # type: ignore
        """Copy first n integer entries from Taichi field to NumPy array."""
        for i in range(n):
            dst[i] = src[start_idx + i]

    @ti.kernel
    def _accumulate_subset_moments(self, indices: ti.types.ndarray(), n_idx: ti.i32):  # type: ignore
        """Sum vortex strength Σalpha and impulse 0.5*Σ(r×alpha) for selected particles."""
        self._subset_vortex_strength[None] = ti.Vector.zero(self._taichi_dtype, 3)
        self._subset_impulse[None] = ti.Vector.zero(self._taichi_dtype, 3)
        for m in range(n_idx):
            i = indices[m]
            p = self.position[i]
            c = self.vortex_strength[i]
            self._subset_vortex_strength[None] += c
            self._subset_impulse[None] += 0.5 * p.cross(c)

    def subset_moments(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute (Σalpha, 0.5*Σ r×alpha) for selected particles entirely on device.

        Only the index list is uploaded and two 3-vectors are downloaded, avoiding
        a full download of every particle's position and vortex strength.

        Args:
            indices: Particle indices to reduce over [k].

        Returns:
            (vortex_strength_sum, linear_impulse): two NumPy arrays of shape (3,).
        """
        idx = np.ascontiguousarray(indices, dtype=np.int32)
        if idx.size == 0:
            zero = np.zeros(3, dtype=self._np_float_dtype)
            return zero, zero.copy()
        self._accumulate_subset_moments(idx, idx.size)
        vortex_strength = self._subset_vortex_strength[None].to_numpy()
        impulse = self._subset_impulse[None].to_numpy()
        return vortex_strength, impulse

    @ti.kernel
    def _accumulate_prefix_vortex_strength(self, n: ti.i32):  # type: ignore
        """Sum Σalpha over the first n live particles on the device."""
        self._subset_vortex_strength[None] = ti.Vector.zero(self._taichi_dtype, 3)
        for i in range(n):
            self._subset_vortex_strength[None] += self.vortex_strength[i]

    def net_vortex_strength(self) -> np.ndarray:
        """Sum Σalpha over all live particles, returning a shape-(3,) array."""
        n = self.n_particles_total
        if n == 0:
            return np.zeros(3, dtype=self._np_float_dtype)
        self._accumulate_prefix_vortex_strength(n)
        return self._subset_vortex_strength[None].to_numpy()

    @ti.kernel
    def _tag_particles_in_bounds_kernel(
        self,
        position: ti.template(),
        tags: ti.template(),
        xmin: ti.f32,
        xmax: ti.f32,
        ymin: ti.f32,
        ymax: ti.f32,
        zmin: ti.f32,
        zmax: ti.f32,
        n: ti.i32,
    ):  # type: ignore
        """
        Tag particles inside a bounding box for removal (GPU kernel).

        Sets tags[i] = 1 if particle i is inside bounds, 0 otherwise.
        This avoids transferring all position to CPU for simple bound checks.
        """
        for i in range(n):
            p = position[i]
            in_bounds = (
                (xmin <= p[0])
                and (p[0] <= xmax)
                and (ymin <= p[1])
                and (p[1] <= ymax)
                and (zmin <= p[2])
                and (p[2] <= zmax)
            )
            tags[i] = 1 if in_bounds else 0

    @ti.kernel
    def _count_tagged(self, tags: ti.template(), n: ti.i32) -> ti.i32:  # type: ignore
        """Count particles tagged for removal."""
        count = 0
        for i in range(n):
            if tags[i] == 1:
                count += 1
        return count

    @ti.kernel
    def _compact_particles_kernel(
        self,
        position: ti.template(),
        velocity: ti.template(),
        vortex_strength: ti.template(),
        vorticity: ti.template(),
        core_radius: ti.template(),
        particle_volume: ti.template(),
        kinematic_viscosity: ti.template(),
        viscosities_t: ti.template(),
        viscosities_eff: ti.template(),
        group_id: ti.template(),
        zone_id: ti.template(),
        velocity_gradient: ti.template(),
        strain_rate: ti.template(),
        tags: ti.template(),
        n: ti.i32,
        new_count: ti.template(),
    ):  # type: ignore
        """
        Compact particles by removing tagged ones (GPU kernel).

        Uses serial compaction (sequential write) which is correct but not
        optimal. For large N, consider a parallel prefix sum approach.
        """
        write_idx = 0
        for i in range(n):
            if tags[i] == 0:  # Keep this particle
                if write_idx != i:  # Only copy if indices differ
                    position[write_idx] = position[i]
                    velocity[write_idx] = velocity[i]
                    vortex_strength[write_idx] = vortex_strength[i]
                    vorticity[write_idx] = vorticity[i]
                    core_radius[write_idx] = core_radius[i]
                    particle_volume[write_idx] = particle_volume[i]
                    kinematic_viscosity[write_idx] = kinematic_viscosity[i]
                    viscosities_t[write_idx] = viscosities_t[i]
                    viscosities_eff[write_idx] = viscosities_eff[i]
                    group_id[write_idx] = group_id[i]
                    zone_id[write_idx] = zone_id[i]
                    for j in ti.static(range(3)):
                        for k in ti.static(range(3)):
                            velocity_gradient[write_idx][j, k] = velocity_gradient[i][j, k]
                            strain_rate[write_idx][j, k] = strain_rate[i][j, k]
                write_idx += 1
        new_count[None] = write_idx

    def remove_particles_by_bounds(self, bounds: list, invert_selection: bool = False) -> int:
        """
        Remove particles based on their position relative to a bounding box.

        Uses GPU kernel for tagging (fast for large N), then CPU-based numpy
        operations for safe compaction (avoids data race in GPU compaction).

        Args:
            bounds: [xmin, xmax, ymin, ymax, zmin, zmax] defining the reference box.
                   Use -inf/inf for unbounded dimensions.
            invert_selection: If False (default), remove particles INSIDE the box.
                        If True, remove particles OUTSIDE the box (keep those inside).

        Returns:
            Number of particles removed.

        Examples:
            >>> # Remove particles in far wake (x > 10)
            >>> particles.remove_particles_by_bounds([10, np.inf, -np.inf, np.inf, -np.inf, np.inf])

            >>> # Keep only particles inside domain (remove those outside)
            >>> particles.remove_particles_by_bounds([-5, 10, -5, 5, -5, 5], invert_selection=True)
        """
        if len(bounds) != 6:
            raise ValueError("bounds must be [xmin, xmax, ymin, ymax, zmin, zmax]")

        n = self.n_particles_total
        if n == 0:
            return 0

        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        # Classify on the host from the same position array used for compaction.
        #
        # This used to run through ``_tag_particles_in_bounds_kernel`` and then
        # download an integer tag field.  On long Vulkan runs that tag dispatch
        # could sporadically return an all-zero field after GBD replacement,
        # causing every in-domain particle to be deleted.  Removal already has
        # to download all retained fields for race-free compaction, so the GPU
        # tag pass saved no material transfer when anything was removed.  The
        # host mask is deterministic and also lets us reuse the downloaded
        # position below.
        position = self._extract_vector(self.position, n)
        inside = (
            (float(xmin) <= position[:, 0])
            & (position[:, 0] <= float(xmax))
            & (float(ymin) <= position[:, 1])
            & (position[:, 1] <= float(ymax))
            & (float(zmin) <= position[:, 2])
            & (position[:, 2] <= float(zmax))
        )
        keep_mask = inside if invert_selection else ~inside

        n_remove = n - keep_mask.sum()

        if n_remove == 0:
            return 0

        new_position = position[keep_mask]
        new_velocity = self._extract_vector(self.velocity, n)[keep_mask]
        new_vortex_strength = self._extract_vector(self.vortex_strength, n)[keep_mask]
        new_radius = self._extract_scalar(self.core_radius, n)[keep_mask]
        new_particle_volume = self._extract_scalar(self.particle_volume, n)[keep_mask]
        new_kinematic_viscosity = self._extract_scalar(self.kinematic_viscosity, n)[keep_mask]
        new_eddy_viscosity = self._extract_scalar(self.eddy_viscosity, n)[keep_mask]
        new_group_id = self._extract_int(self.group_id, n)[keep_mask]
        new_zone_id = self._extract_int(self.zone_id, n)[keep_mask]
        new_velocity_gradient = self._extract_matrix(self.velocity_gradient, n)[keep_mask]
        new_strain_rate = self._extract_matrix(self.strain_rate, n)[keep_mask]

        self.replace_from_numpy(
            position=new_position,
            velocity=new_velocity,
            vortex_strength=new_vortex_strength,
            core_radius=new_radius,
            particle_volume=new_particle_volume,
            kinematic_viscosity=new_kinematic_viscosity,
            eddy_viscosity=new_eddy_viscosity,
            group_id=new_group_id,
            zone_id=new_zone_id,
            velocity_gradient=new_velocity_gradient,
            strain_rate=new_strain_rate,
        )

        return n_remove

    def _fast_batch_add(
        self,
        position,
        velocity,
        vortex_strength,
        vorticity,
        core_radius,
        particle_volume,
        kinematic_viscosity,
        eddy_viscosity,
        effective_viscosity,
        group_id,
        zone_id,
        velocity_gradient,
        strain_rate,
    ):
        """Fast batch addition using direct NumPy-to-Taichi copy kernels.

        This method uses the pre-compiled _copy_to_taichi_* kernels to copy
        NumPy arrays directly to Taichi fields at a specified offset.

        IMPORTANT: Does NOT create temporary Taichi fields - this is critical
        because Taichi fields cannot be garbage collected, so creating temp
        fields on every call would cause GPU memory exhaustion.
        """
        N = position.shape[0]
        start_idx = self.n_particles_total

        # Check if we have enough space
        if start_idx + N > self._max_particles:
            return False

        # Copy vectors directly using pre-compiled kernels (no temp fields!)
        self._copy_vectors_chunked(position, self.position, start_idx, N)
        self._copy_vectors_chunked(velocity, self.velocity, start_idx, N)
        self._copy_vectors_chunked(vortex_strength, self.vortex_strength, start_idx, N)
        self._copy_vectors_chunked(vorticity, self.vorticity, start_idx, N)

        # Copy scalars directly
        self._copy_scalars_chunked(core_radius, self.core_radius, start_idx, N)
        self._copy_scalars_chunked(particle_volume, self.particle_volume, start_idx, N)
        self._copy_scalars_chunked(kinematic_viscosity, self.kinematic_viscosity, start_idx, N)
        self._copy_scalars_chunked(eddy_viscosity, self.eddy_viscosity, start_idx, N)
        self._copy_scalars_chunked(effective_viscosity, self.effective_viscosity, start_idx, N)

        # Copy integer fields
        self._copy_ints_chunked(group_id, self.group_id, start_idx, N)
        self._copy_ints_chunked(zone_id, self.zone_id, start_idx, N)

        # Copy matrices directly
        self._copy_matrices_chunked(velocity_gradient, self.velocity_gradient, start_idx, N)
        self._copy_matrices_chunked(strain_rate, self.strain_rate, start_idx, N)

        return True

    def _copy_vectors_chunked(self, src, dest, start_idx: int, count: int) -> None:
        """Upload vector data in bounded external-array chunks."""
        buf = self._host_transfer_buffer("vector", dest, "upload")
        for lo in range(0, count, self._COPY_CHUNK_SIZE):
            hi = min(lo + self._COPY_CHUNK_SIZE, count)
            n_chunk = hi - lo
            buf[:n_chunk] = src[lo:hi]
            self._copy_to_taichi_vectors(buf, dest, start_idx + lo, n_chunk)
            ti.sync()

    def _copy_scalars_chunked(self, src, dest, start_idx: int, count: int) -> None:
        """Upload scalar data in bounded external-array chunks."""
        buf = self._host_transfer_buffer("scalar", dest, "upload")
        for lo in range(0, count, self._COPY_CHUNK_SIZE):
            hi = min(lo + self._COPY_CHUNK_SIZE, count)
            n_chunk = hi - lo
            buf[:n_chunk] = src[lo:hi]
            self._copy_to_taichi_scalars(buf, dest, start_idx + lo, n_chunk)
            ti.sync()

    def _copy_matrices_chunked(self, src, dest, start_idx: int, count: int) -> None:
        """Upload matrix data in bounded external-array chunks."""
        buf = self._host_transfer_buffer("matrix", dest, "upload")
        for lo in range(0, count, self._COPY_CHUNK_SIZE):
            hi = min(lo + self._COPY_CHUNK_SIZE, count)
            n_chunk = hi - lo
            buf[:n_chunk] = src[lo:hi]
            self._copy_to_taichi_matrices(buf, dest, start_idx + lo, n_chunk)
            ti.sync()

    def _copy_ints_chunked(self, src, dest, start_idx: int, count: int) -> None:
        """Upload integer data in bounded external-array chunks."""
        buf = self._host_transfer_buffer("int", dest, "upload")
        for lo in range(0, count, self._COPY_CHUNK_SIZE):
            hi = min(lo + self._COPY_CHUNK_SIZE, count)
            n_chunk = hi - lo
            buf[:n_chunk] = src[lo:hi]
            self._copy_to_taichi_ints(buf, dest, start_idx + lo, n_chunk)
            ti.sync()

    def _validate_numpy_input(self, arr, expected_shape_suffix, name):
        """Validate NumPy array input for Taichi kernels."""
        if not arr.flags["C_CONTIGUOUS"]:
            raise ValueError(f"{name} array must be C-contiguous")
        if len(expected_shape_suffix) > 0 and arr.shape[1:] != expected_shape_suffix:
            raise ValueError(
                f"{name} array must have shape (N, {', '.join(map(str, expected_shape_suffix))}), got {arr.shape}"
            )

        # Convert to appropriate dtype for Taichi compatibility
        if arr.dtype.kind == "f":  # floating point
            return np.ascontiguousarray(arr, dtype=self._np_float_dtype)
        elif arr.dtype.kind == "i":  # integer
            return np.ascontiguousarray(arr, dtype=np.int32)
        else:
            return arr

    def _populate_from_numpy(
        self,
        position,
        velocity,
        vortex_strength,
        core_radius,
        particle_volume,
        kinematic_viscosity,
        eddy_viscosity,
        effective_viscosity,
        group_id,
        velocity_gradient,
        strain_rate,
        vorticity,
        zone_id,
    ):
        """Populate Taichi fields from NumPy arrays."""
        count = position.shape[0]

        # Validate all inputs and convert to f32/i32
        position = self._validate_numpy_input(position, (3,), "position")
        velocity = self._validate_numpy_input(velocity, (3,), "velocity")
        vortex_strength = self._validate_numpy_input(vortex_strength, (3,), "vortex_strength")
        vorticity = self._validate_numpy_input(vorticity, (3,), "vorticity")
        core_radius = self._validate_numpy_input(core_radius, (), "core_radius")
        particle_volume = self._validate_numpy_input(particle_volume, (), "particle_volume")
        kinematic_viscosity = self._validate_numpy_input(
            kinematic_viscosity, (), "kinematic_viscosity"
        )
        eddy_viscosity = self._validate_numpy_input(eddy_viscosity, (), "eddy_viscosity")
        effective_viscosity = self._validate_numpy_input(
            effective_viscosity, (), "effective_viscosity"
        )
        group_id = self._validate_numpy_input(group_id, (), "group_id")
        zone_id = self._validate_numpy_input(zone_id, (), "zone_id")
        velocity_gradient = self._validate_numpy_input(
            velocity_gradient, (3, 3), "velocity_gradient"
        )
        strain_rate = self._validate_numpy_input(strain_rate, (3, 3), "strain_rate")

        # Full cloud replacement uses native fixed-shape transfers.  This is
        # intentionally separate from the chunked append path above.
        self._replace_field_native("vector", self.position, position, count)
        self._replace_field_native("vector", self.velocity, velocity, count)
        self._replace_field_native("vector", self.vortex_strength, vortex_strength, count)
        self._replace_field_native("vector", self.vorticity, vorticity, count)
        self._replace_field_native("scalar", self.core_radius, core_radius, count)
        self._replace_field_native("scalar", self.particle_volume, particle_volume, count)
        self._replace_field_native("scalar", self.kinematic_viscosity, kinematic_viscosity, count)
        self._replace_field_native("scalar", self.eddy_viscosity, eddy_viscosity, count)
        self._replace_field_native("scalar", self.effective_viscosity, effective_viscosity, count)
        self._replace_field_native("int", self.group_id, group_id, count)
        self._replace_field_native("int", self.zone_id, zone_id, count)
        self._replace_field_native("matrix", self.velocity_gradient, velocity_gradient, count)
        self._replace_field_native("matrix", self.strain_rate, strain_rate, count)

        self.n_particles_total = count

    # CPU access methods (return NumPy arrays) - now with caching
    @cached_particle_property
    def position_cpu(self):
        """Get position as NumPy array (CPU copy) - cached per time step."""
        return self._extract_vector(self.position, self.n_particles_total)

    @cached_particle_property
    def velocity_cpu(self):
        """Get velocity as NumPy array (CPU copy) - cached per time step."""
        return self._extract_vector(self.velocity, self.n_particles_total)

    @cached_particle_property
    def vortex_strength_cpu(self):
        """Get vortex_strength as NumPy array (CPU copy) - cached per time step."""
        return self._extract_vector(self.vortex_strength, self.n_particles_total)

    @cached_particle_property
    def core_radius_cpu(self):
        """Get core_radius as NumPy array (CPU copy) - cached per time step."""
        return self._extract_scalar(self.core_radius, self.n_particles_total)

    @cached_particle_property
    def particle_volume_cpu(self):
        """Get particle_volume as NumPy array (CPU copy) - cached per time step."""
        return self._extract_scalar(self.particle_volume, self.n_particles_total)

    @cached_particle_property
    def kinematic_viscosity_cpu(self):
        """Get kinematic_viscosity as NumPy array (CPU copy) - cached per time step."""
        return self._extract_scalar(self.kinematic_viscosity, self.n_particles_total)

    @cached_particle_property
    def eddy_viscosity_cpu(self):
        """Get eddy viscosity as a cached NumPy CPU copy."""
        return self._extract_scalar(self.eddy_viscosity, self.n_particles_total)

    @cached_particle_property
    def effective_viscosity_cpu(self):
        """Get effective kinematic_viscosity as NumPy array (CPU copy) - cached per time step."""
        return self._extract_scalar(self.effective_viscosity, self.n_particles_total)

    @cached_particle_property
    def group_id_cpu(self):
        """Get group IDs as NumPy array (CPU copy) - cached per time step."""
        return self._extract_int(self.group_id, self.n_particles_total)

    @cached_particle_property
    def velocity_gradient_cpu(self):
        """Get gradient of velocity field on CPU - cached per time step."""
        return self._extract_matrix(self.velocity_gradient, self.n_particles_total)

    @cached_particle_property
    def strain_rate_cpu(self):
        """Get strain rate tensors as NumPy array (CPU copy) - cached per time step."""
        return self._extract_matrix(self.strain_rate, self.n_particles_total)

    @cached_particle_property
    def vorticity_cpu(self):
        """Get vorticity as NumPy array (CPU copy) - cached per time step."""
        return self._extract_vector(self.vorticity, self.n_particles_total)

    @cached_particle_property
    def zone_id_cpu(self):
        """Get zone IDs as NumPy array (CPU copy) - cached per time step."""
        return self._extract_int(self.zone_id, self.n_particles_total)

    def velocity_background_cpu(self) -> np.ndarray:
        """Get background velocity as NumPy array (3,)."""
        v = self.velocity_background[None]
        return np.array([v[0], v[1], v[2]], dtype=np.float32)

    def set_freestream_velocity(self, velocity: np.ndarray) -> None:
        """
        Set the global background velocity for all particles.

        Args:
            velocity: 3D velocity vector [ux, uy, uz] in m/s
        """
        self.velocity_background[None] = [
            float(velocity[0]),
            float(velocity[1]),
            float(velocity[2]),
        ]

    def __len__(self):
        return int(self.n_particles_total)

    def report_rows(self) -> list:
        """Return particle-system statistics as log detail rows."""
        N = self.n_particles_total
        rows: list[log_style.Row] = [("particles", f"{N:,}")]

        if N == 0:
            rows.append(("status", "empty, no particles"))
            return rows

        position = self.position_cpu()
        core_radius = self.core_radius_cpu()
        particle_volume = self.particle_volume_cpu()
        vortex_strength = self.vortex_strength_cpu()
        velocity = self.velocity_cpu()

        vortex_strength_magnitude = np.linalg.norm(vortex_strength, axis=1)
        velocity_magnitude = np.linalg.norm(velocity, axis=1)

        bbox_min = np.min(position, axis=0)
        bbox_max = np.max(position, axis=0)
        domain_size = bbox_max - bbox_min

        rows.append(("spatial extent:", ""))
        for index, axis in enumerate("xyz"):
            rows.append((f"  {axis}", f"[{bbox_min[index]:.3e}, {bbox_max[index]:.3e}]", "m"))
            rows.append((f"  {axis}, span", f"{domain_size[index]:.3e}", "m"))

        rows.extend(
            (
                ("core radius:", ""),
                ("  min", f"{np.min(core_radius):.4e}", "m"),
                ("  mean", f"{np.mean(core_radius):.4e}", "m"),
                ("  max", f"{np.max(core_radius):.4e}", "m"),
                ("particle volume:", ""),
                ("  min", f"{np.min(particle_volume):.4e}", "m^3"),
                ("  mean", f"{np.mean(particle_volume):.4e}", "m^3"),
                ("  max", f"{np.max(particle_volume):.4e}", "m^3"),
                ("  total", f"{np.sum(particle_volume):.4e}", "m^3"),
                ("vortex strength magnitude:", ""),
                ("  min", f"{np.min(vortex_strength_magnitude):.4e}", "m^3/s"),
                ("  mean", f"{np.mean(vortex_strength_magnitude):.4e}", "m^3/s"),
                ("  max", f"{np.max(vortex_strength_magnitude):.4e}", "m^3/s"),
                ("velocity magnitude:", ""),
                ("  min", f"{np.min(velocity_magnitude):.4e}", "m/s"),
                ("  mean", f"{np.mean(velocity_magnitude):.4e}", "m/s"),
                ("  max", f"{np.max(velocity_magnitude):.4e}", "m/s"),
            )
        )

        group_id = self.group_id_cpu()
        unique_groups = np.unique(group_id)
        if len(unique_groups) > 1:
            rows.append(("particle groups", f"{len(unique_groups):,}"))
            for gid in unique_groups:
                count = int(np.sum(group_id == gid))
                rows.append((f"  group {gid}", f"{count:,}", f"particles, {100 * count / N:.1f}%"))
        return rows

    def __str__(self):
        """Return the particle-system statistics as one indented block."""
        return "\n".join(
            log_style.record("vpm", "particle system", *self.report_rows()).split("\n")[1:]
        )

    def __getitem__(self, index):
        """Return particle data at index (CPU copy)."""
        return {
            "position": self.position_cpu()[index],
            "velocity": self.velocity_cpu()[index],
            "vortex_strength": self.vortex_strength_cpu()[index],
            "core_radius": self.core_radius_cpu()[index],
            "particle_volume": self.particle_volume_cpu()[index],
            "kinematic_viscosity": self.kinematic_viscosity_cpu()[index],
            "eddy_viscosity": self.eddy_viscosity_cpu()[index],
            "effective_viscosity": self.effective_viscosity_cpu()[index],
            "group_id": self.group_id_cpu()[index],
            "velocity_gradient": self.velocity_gradient_cpu()[index],
            "strain_rate": self.strain_rate_cpu()[index],
            "vorticity": self.vorticity_cpu()[index],
            "zone_id": self.zone_id_cpu()[index],
        }

    def _log_population(self, change: log_style.Row) -> None:
        """Report the population left behind by an operation that changed it."""
        total = int(self.n_particles_total)
        capacity = self.capacity
        fraction = 100.0 * total / capacity if capacity else 0.0
        Logging.record(
            "particles",
            change,
            ("count", f"{total:,}"),
            ("capacity", f"{capacity:,}"),
            ("utilization", f"{fraction:.1f}", "%"),
        )

    def _log_particles_added(self, count: int) -> None:
        """Report the population after particles were appended."""
        self._log_population(("added", f"{int(count):,}"))

    def _log_particles_replaced(self, previous: int) -> None:
        """Report the population after the whole cloud was replaced."""
        self._log_population(("count, previous", f"{int(previous):,}"))

    def add_vortex_particle(
        self,
        position: np.ndarray = np.zeros(3),
        velocity: np.ndarray = np.zeros(3),
        vortex_strength: np.ndarray = np.zeros(3),
        core_radius: float = 1.0,
        particle_volume: float = 1.0,
        kinematic_viscosity: float = 0.0,
        eddy_viscosity: float = 0.0,
        group_id: int = 0,
        zone_id: int = 0,
        velocity_gradient: np.ndarray = np.zeros((3, 3), dtype=np.float32),
        vorticity: np.ndarray = np.zeros(3),
    ):
        # Ensure we have space for one more particle
        self._grow_capacity(self.n_particles_total + 1)

        # Prepare data arrays with a single particle (using float32 for Taichi compatibility)
        position = np.ascontiguousarray(position, dtype=np.float32).reshape(1, 3)
        velocity = np.ascontiguousarray(velocity, dtype=np.float32).reshape(1, 3)
        vortex_strength = np.ascontiguousarray(vortex_strength, dtype=np.float32).reshape(1, 3)
        vorticity = np.ascontiguousarray(vorticity, dtype=np.float32).reshape(1, 3)
        velocity_gradient = np.ascontiguousarray(velocity_gradient, dtype=np.float32).reshape(
            1, 3, 3
        )
        strain_rate = np.ascontiguousarray(np.zeros((1, 3, 3), dtype=np.float32))

        # Scalar values
        # Scalar values
        core_radius = np.array([core_radius], dtype=np.float32).reshape(1)
        particle_volume = np.array([particle_volume], dtype=np.float32).reshape(1)
        kinematic_viscosity = np.array([kinematic_viscosity], dtype=np.float32).reshape(1)
        eddy_viscosity = np.array([eddy_viscosity], dtype=np.float32).reshape(1)
        effective_viscosity = np.array(
            [kinematic_viscosity + eddy_viscosity], dtype=np.float32
        ).reshape(1)
        group_id = np.array([group_id], dtype=np.int32).reshape(1)
        zone_id = np.array([zone_id], dtype=np.int32).reshape(1)

        # Copy to Taichi fields at the current end position
        idx = self.n_particles_total
        self._copy_to_taichi_vectors(position, self.position, idx, 1)
        self._copy_to_taichi_vectors(velocity, self.velocity, idx, 1)
        self._copy_to_taichi_vectors(vortex_strength, self.vortex_strength, idx, 1)
        self._copy_to_taichi_vectors(vorticity, self.vorticity, idx, 1)
        self._copy_to_taichi_scalars(core_radius, self.core_radius, idx, 1)
        self._copy_to_taichi_scalars(particle_volume, self.particle_volume, idx, 1)
        self._copy_to_taichi_scalars(kinematic_viscosity, self.kinematic_viscosity, idx, 1)
        self._copy_to_taichi_scalars(eddy_viscosity, self.eddy_viscosity, idx, 1)
        self._copy_to_taichi_scalars(effective_viscosity, self.effective_viscosity, idx, 1)
        self._copy_to_taichi_ints(group_id, self.group_id, idx, 1)
        self._copy_to_taichi_ints(zone_id, self.zone_id, idx, 1)
        self._copy_to_taichi_matrices(velocity_gradient, self.velocity_gradient, idx, 1)
        self._copy_to_taichi_matrices(strain_rate, self.strain_rate, idx, 1)

        # Increment particle count
        self.n_particles_total += 1
        self.touch_state()
        self._log_particles_added(1)

    def add_vortex_particles(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        vortex_strength: np.ndarray,
        core_radius: np.ndarray,
        particle_volume: np.ndarray,
        kinematic_viscosity: np.ndarray,
        eddy_viscosity: np.ndarray = None,
        group_id: np.ndarray = None,
        zone_id: np.ndarray = None,
        velocity_gradient: np.ndarray = None,
    ):
        """
        Initialize particle system from user-provided numpy arrays.

        **Validates input for NaN/Inf values before adding particles.**

        Args:
            position: Particle position [N, 3] in meters
            velocity: Particle velocity [N, 3] in m/s
            vortex_strength: Particle strength (α = ω·V) [N, 3] in m³/s
            core_radius: Particle core radius [N] in meters
            particle_volume: Particle volume [N] in m³
            kinematic_viscosity: Molecular kinematic viscosity [N] in m²/s
            eddy_viscosity: Turbulent viscosity [N] in m²/s (optional)
            group_id: Particle group identifiers [N] (optional)
            zone_id: Spatial zone identifiers [N] (optional)
            velocity_gradient: Velocity gradient tensors [N, 3, 3] (optional)

        Raises:
            ValueError: If input contains NaN or Inf values
            ValueError: If array shapes are inconsistent
        """

        # ---- INPUT VALIDATION: NaN/Inf CHECKS ----
        _validate_finite_array(position, "position")
        _validate_finite_array(velocity, "velocity")
        _validate_finite_array(vortex_strength, "vortex_strength")
        _validate_finite_array(core_radius, "core_radius")
        _validate_finite_array(particle_volume, "particle_volume")
        _validate_finite_array(kinematic_viscosity, "kinematic_viscosity")

        if eddy_viscosity is not None:
            _validate_finite_array(eddy_viscosity, "eddy_viscosity")
        if velocity_gradient is not None:
            _validate_finite_array(velocity_gradient, "velocity_gradient")

        # ---- CONTINUE WITH NORMAL PROCESSING ----
        # Honor the configured float precision: the Taichi fields are created
        # with self._taichi_dtype, so feeding them self._np_float_dtype arrays
        # keeps the transfer exact (no f32←f64 / f64←f32 precision warnings)
        # and makes a precision='f64' VPM run end-to-end double-precision.
        float_dtype = self._np_float_dtype
        position = np.ascontiguousarray(position, dtype=float_dtype)
        velocity = np.ascontiguousarray(velocity, dtype=float_dtype)
        vortex_strength = np.ascontiguousarray(vortex_strength, dtype=float_dtype)
        core_radius = np.ascontiguousarray(core_radius, dtype=float_dtype)
        particle_volume = np.ascontiguousarray(particle_volume, dtype=float_dtype)
        kinematic_viscosity = np.ascontiguousarray(kinematic_viscosity, dtype=float_dtype)

        N = position.shape[0]
        if position.shape[1] != 3 or velocity.shape[1] != 3 or vortex_strength.shape[1] != 3:
            raise ValueError("position, velocity, and vortex_strength must have shape (N, 3)")
        if not (core_radius.shape[0] == N):
            raise ValueError("core_radius must match the number of position")

        # Ensure all arrays are contiguous and have the correct shape
        if eddy_viscosity is None:
            eddy_viscosity = np.zeros(N, dtype=float_dtype)
        else:
            eddy_viscosity = np.ascontiguousarray(eddy_viscosity, dtype=float_dtype)

        # Calculate effective viscosity
        effective_viscosity = kinematic_viscosity + eddy_viscosity

        # Prepare other fields
        group_id = _coerce_int_id_array(group_id, N)
        zone_id = _coerce_int_id_array(zone_id, N)

        # Ensure velocity_gradient and strain_rate are properly initialized
        if velocity_gradient is None:
            velocity_gradient = np.zeros((N, 3, 3), dtype=float_dtype)
        else:
            velocity_gradient = np.ascontiguousarray(velocity_gradient, dtype=float_dtype)

        # Initialize strain_rate tensor - will be computed from velocity_gradient later
        strain_rate = np.zeros((N, 3, 3), dtype=float_dtype)

        # Initialize vorticity field: vorticity = vortex strength / particle_volume
        vorticity = (vortex_strength / particle_volume[:, None]).astype(float_dtype)

        # Ensure we have enough space for all particles
        total_particles = self.n_particles_total + N
        self._grow_capacity(total_particles)

        # Copy all data to Taichi fields at once
        start_idx = self.n_particles_total

        # Try fast batch add first (for initial particle loading)
        if not self._fast_batch_add(
            position,
            velocity,
            vortex_strength,
            vorticity,
            core_radius,
            particle_volume,
            kinematic_viscosity,
            eddy_viscosity,
            effective_viscosity,
            group_id,
            zone_id,
            velocity_gradient,
            strain_rate,
        ):
            # Fall back to element-by-element copy for appending
            self._copy_vectors_chunked(position, self.position, start_idx, N)
            self._copy_vectors_chunked(velocity, self.velocity, start_idx, N)
            self._copy_vectors_chunked(vortex_strength, self.vortex_strength, start_idx, N)
            self._copy_vectors_chunked(vorticity, self.vorticity, start_idx, N)
            self._copy_scalars_chunked(core_radius, self.core_radius, start_idx, N)
            self._copy_scalars_chunked(particle_volume, self.particle_volume, start_idx, N)
            self._copy_scalars_chunked(kinematic_viscosity, self.kinematic_viscosity, start_idx, N)
            self._copy_scalars_chunked(eddy_viscosity, self.eddy_viscosity, start_idx, N)
            self._copy_scalars_chunked(effective_viscosity, self.effective_viscosity, start_idx, N)
            self._copy_ints_chunked(group_id, self.group_id, start_idx, N)
            self._copy_ints_chunked(zone_id, self.zone_id, start_idx, N)
            self._copy_matrices_chunked(velocity_gradient, self.velocity_gradient, start_idx, N)
            self._copy_matrices_chunked(strain_rate, self.strain_rate, start_idx, N)

        # Update particle count
        self.n_particles_total = total_particles

        self.touch_state()
        self._log_particles_added(N)

    def replace_from_numpy(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        vortex_strength: np.ndarray,
        core_radius: np.ndarray,
        particle_volume: np.ndarray,
        kinematic_viscosity: np.ndarray,
        eddy_viscosity: np.ndarray = None,
        group_id: np.ndarray = None,
        zone_id: np.ndarray = None,
        velocity_gradient: np.ndarray = None,
        strain_rate: np.ndarray = None,
    ) -> None:
        """Replace the active particle cloud with NumPy arrays."""
        previous = int(self.n_particles_total)
        _validate_finite_array(position, "position")
        _validate_finite_array(velocity, "velocity")
        _validate_finite_array(vortex_strength, "vortex_strength")
        _validate_finite_array(core_radius, "core_radius")
        _validate_finite_array(particle_volume, "particle_volume")
        _validate_finite_array(kinematic_viscosity, "kinematic_viscosity")

        if eddy_viscosity is not None:
            _validate_finite_array(eddy_viscosity, "eddy_viscosity")
        if velocity_gradient is not None:
            _validate_finite_array(velocity_gradient, "velocity_gradient")
        if strain_rate is not None:
            _validate_finite_array(strain_rate, "strain_rate")

        float_dtype = self._np_float_dtype
        position = np.ascontiguousarray(position, dtype=float_dtype)
        velocity = np.ascontiguousarray(velocity, dtype=float_dtype)
        vortex_strength = np.ascontiguousarray(vortex_strength, dtype=float_dtype)
        core_radius = np.ascontiguousarray(core_radius, dtype=float_dtype)
        particle_volume = np.ascontiguousarray(particle_volume, dtype=float_dtype)
        kinematic_viscosity = np.ascontiguousarray(kinematic_viscosity, dtype=float_dtype)

        N = position.shape[0]
        if N == 0:
            self.n_particles_total = 0
            self.sync_device_counter()
            self.touch_state()
            self._log_particles_replaced(previous)
            return
        if position.shape != (N, 3) or velocity.shape != (N, 3) or vortex_strength.shape != (N, 3):
            raise ValueError("position, velocity, and vortex_strength must have shape (N, 3)")
        if (
            core_radius.shape != (N,)
            or particle_volume.shape != (N,)
            or kinematic_viscosity.shape != (N,)
        ):
            raise ValueError("Radius, particle_volume, and viscosity must have shape (N,).")

        if eddy_viscosity is None:
            eddy_viscosity = np.zeros(N, dtype=float_dtype)
        else:
            eddy_viscosity = np.ascontiguousarray(eddy_viscosity, dtype=float_dtype)
            if eddy_viscosity.shape != (N,):
                raise ValueError("Turbulent viscosity must have shape (N,).")
        effective_viscosity = kinematic_viscosity + eddy_viscosity

        group_id = _coerce_int_id_array(group_id, N)
        zone_id = _coerce_int_id_array(zone_id, N)
        if group_id.shape != (N,) or zone_id.shape != (N,):
            raise ValueError("Group and zone IDs must have shape (N,).")

        if velocity_gradient is None:
            velocity_gradient = np.zeros((N, 3, 3), dtype=float_dtype)
        else:
            velocity_gradient = np.ascontiguousarray(velocity_gradient, dtype=float_dtype)
        if strain_rate is None:
            strain_rate = np.zeros((N, 3, 3), dtype=float_dtype)
        else:
            strain_rate = np.ascontiguousarray(strain_rate, dtype=float_dtype)
        if velocity_gradient.shape != (N, 3, 3) or strain_rate.shape != (N, 3, 3):
            raise ValueError("Velocity gradient and strain rate must have shape (N, 3, 3).")
        vorticity = (vortex_strength / particle_volume[:, None]).astype(float_dtype)

        self._grow_capacity(N)

        self._populate_from_numpy(
            position,
            velocity,
            vortex_strength,
            core_radius,
            particle_volume,
            kinematic_viscosity,
            eddy_viscosity,
            effective_viscosity,
            group_id,
            velocity_gradient,
            strain_rate,
            vorticity,
            zone_id,
        )
        self.sync_device_counter()
        self.touch_state()
        self._log_particles_replaced(previous)

    # ---- GPU-TO-GPU DATA TRANSFER ----

    def add_vortex_particles_from_fields(
        self,
        count: int,
        position: ti.template(),
        velocity: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        particle_volume: ti.template(),
        group_id: int = 0,
        kinematic_viscosity: float = 1.5e-5,
    ) -> bool:
        """
        Add particles directly from Taichi fields (GPU-to-GPU transfer).

        This avoids CPU round-trips when shedding wake particles from the VLM solver.

        Args:
            count: Number of particles to copy from input fields
            position: Source position field (Vector)
            velocity: Source velocity field (Vector)
            vortex_strength: Source vortex-strength field (Vector)
            core_radius: Source core-radius field (Scalar)
            particle_volume: Source particle-volume field (scalar)
            group_id: Group ID to assign to new particles
            kinematic_viscosity: Molecular kinematic viscosity to assign

        Returns:
            bool: True if successful, False if container is full
        """
        start_idx = self.n_particles_total

        # Check if we have space
        if start_idx + count > self._max_particles:
            return False

        # Kernel to copy data
        @ti.kernel
        def copy_particles_kernel(
            dest_offset: ti.i32,
            src_count: ti.i32,
            # Source fields
            src_pos: ti.template(),
            src_vel: ti.template(),
            src_str: ti.template(),
            src_rad: ti.template(),
            src_vol: ti.template(),
            # Dest fields (implicit self fields not passed as template to avoid complexity)
            # but we can't access self inside kernel easily if it's not ti.data_oriented struct
            # Since self IS ti.data_oriented, we can access self.position etc.
            p_group_id: ti.i32,
            p_viscosity: ti.f32,
        ):
            for i in range(src_count):
                dest_idx = dest_offset + i

                # Copy vectors
                self.position[dest_idx] = src_pos[i]
                self.velocity[dest_idx] = src_vel[i]
                self.vortex_strength[dest_idx] = src_str[i]

                # Copy scalars
                self.core_radius[dest_idx] = src_rad[i]
                self.particle_volume[dest_idx] = src_vol[i]

                # Set fixed properties
                self.kinematic_viscosity[dest_idx] = p_viscosity
                self.group_id[dest_idx] = p_group_id

                # Initialize others to zero
                self.eddy_viscosity[dest_idx] = 0.0
                self.effective_viscosity[dest_idx] = p_viscosity
                self.vorticity[dest_idx] = ti.Vector([0.0, 0.0, 0.0])
                self.zone_id[dest_idx] = 0

                # Zero matrices
                self.velocity_gradient[dest_idx].fill(0.0)
                self.strain_rate[dest_idx].fill(0.0)

        # Launch copy kernel
        copy_particles_kernel(
            start_idx,
            count,
            position,
            velocity,
            vortex_strength,
            core_radius,
            particle_volume,
            group_id,
            kinematic_viscosity,
        )

        # Update counter
        self.n_particles_total += count
        self.sync_device_counter()
        self.touch_state()
        self._log_particles_added(count)

        return True

    def add_particles_from_taichi(
        self,
        position,  # ti.Vector.field
        velocity,  # ti.Vector.field
        vortex_strength,  # ti.Vector.field
        core_radius,  # ti.field
        particle_volume,  # ti.field
        count: int,
        kinematic_viscosity: float,
    ):
        """
        Add particles directly from Taichi fields (GPU-to-GPU transfer).

        This method enables direct transfer from VLM wake buffers to VPM particles
        without numpy intermediates, providing significant performance improvement.

        Args:
            position: Taichi Vector.field (N x 3) source position
            velocity: Taichi Vector.field (N x 3) source velocity
            vortex_strength: Taichi Vector.field (N x 3) source vortex_strength
            core_radius: Taichi field (N,) source core_radius
            particle_volume: Taichi field (N,) containing source particle volume
            count: Number of particles to transfer (must be <= source field size)
            viscosity: Molecular viscosity to assign to all transferred particles
        """
        if count == 0:
            return

        # Ensure we have capacity
        total_particles = self.n_particles_total + count
        self._grow_capacity(total_particles)

        start_idx = self.n_particles_total

        # Direct Taichi-to-Taichi copy via kernel
        # Initializes all particle properties to match numpy version behavior
        self._copy_from_vlm_wake(
            position,
            velocity,
            vortex_strength,
            core_radius,
            particle_volume,
            self.position,
            self.velocity,
            self.vortex_strength,
            self.core_radius,
            self.particle_volume,
            self.kinematic_viscosity,
            self.eddy_viscosity,
            self.effective_viscosity,
            self.vorticity,
            self.group_id,
            self.zone_id,  # Pass self.zone_id
            self.velocity_gradient,
            self.strain_rate,
            start_idx,
            count,
            kinematic_viscosity,
        )

        # Update particle count
        self.n_particles_total = total_particles

        self.touch_state()
        self._log_particles_added(count)

    @ti.kernel
    def _copy_from_vlm_wake(
        self,
        src_pos: ti.template(),
        src_vel: ti.template(),
        src_str: ti.template(),
        src_rad: ti.template(),
        src_vol: ti.template(),
        dst_pos: ti.template(),
        dst_vel: ti.template(),
        dst_str: ti.template(),
        dst_rad: ti.template(),
        dst_vol: ti.template(),
        dst_visc: ti.template(),
        dst_visc_t: ti.template(),
        dst_visc_eff: ti.template(),
        dst_vort: ti.template(),
        dst_group: ti.template(),
        destination_velocity_gradient: ti.template(),
        destination_strain_rate: ti.template(),
        start_idx: ti.i32,
        count: ti.i32,
        kinematic_viscosity: ti.f32,
    ):
        """
        Taichi kernel for GPU-to-GPU particle copy from VLM wake buffer.

        Initializes all particle properties to match behavior of add_vortex_particles():
        - position, velocity, vortex_strength, core_radius, particle_volume: from source
        - vorticity: computed as strength / particle_volume
        - kinematic_viscosity: set to provided viscosity value
        - viscosities_t: set to 0 (no turbulent viscosity)
        - viscosities_eff: set to viscosity (molecular only)
        - group_id: set to 0 (default group)
        - velocity_gradient: set to zero (will be computed on next velocity update)
        - strain_rate: set to zero (will be computed from velocity_gradient)
        """
        for i in range(count):
            dst_idx = start_idx + i

            # Copy primary fields from source
            dst_pos[dst_idx] = src_pos[i]
            dst_vel[dst_idx] = src_vel[i]
            dst_str[dst_idx] = src_str[i]
            dst_rad[dst_idx] = src_rad[i]
            dst_vol[dst_idx] = src_vol[i]

            # Set viscosity fields (matching numpy version)
            dst_visc[dst_idx] = kinematic_viscosity
            dst_visc_t[dst_idx] = 0.0
            dst_visc_eff[dst_idx] = kinematic_viscosity  # eff = molecular + turbulent

            # Compute vorticity from strength and particle_volume (matching numpy version)
            # vorticity = strength / particle_volume
            vol = src_vol[i]
            if vol > 1e-15:
                dst_vort[dst_idx] = src_str[i] / vol
            else:
                dst_vort[dst_idx] = ti.Vector([0.0, 0.0, 0.0])

            # Assign to default group (matching numpy default)
            dst_group[dst_idx] = 0

            # Initialize velocity_gradient to zero (matching numpy version)
            # Will be computed during velocity gradient update
            for row in ti.static(range(3)):
                for col in ti.static(range(3)):
                    destination_velocity_gradient[dst_idx][row, col] = 0.0
                    destination_strain_rate[dst_idx][row, col] = 0.0

    def save_vortex_particles(
        self,
        particle_file_name: str,
        write_precision: str = DEFAULT_WRITE_PRECISION,
    ) -> None:
        """Export the particle cloud to a VTP point cloud (field names match ``load_vortex_particles``).

        Point coordinates already carry the positions, and strain rate is the
        symmetric part of the velocity gradient, so neither is stored again;
        ParaView derives the magnitude of any vector on its own.
        """
        if not HAS_PYVISTA:
            Logging.warning(
                f"[Output] status=skipped format=vtk reason=pyvista_unavailable "
                f"path={particle_file_name}"
            )
            return

        n = int(self.n_particles_total)
        point_cloud = pv.PolyData(cast_for_write(self.position_cpu(), write_precision))
        fields = {
            "velocity": self.velocity_cpu(),
            "vortex_strength": self.vortex_strength_cpu(),
            "core_radius": self.core_radius_cpu(),
            "particle_volume": self.particle_volume_cpu(),
            "kinematic_viscosity": self.kinematic_viscosity_cpu(),
            "eddy_viscosity": self.eddy_viscosity_cpu(),
            "effective_viscosity": self.effective_viscosity_cpu(),
            "group_id": self.group_id_cpu(),
            "zone_id": self.zone_id_cpu(),
            "vorticity": self.vorticity_cpu(),
            "velocity_gradient": self.velocity_gradient_cpu().reshape(n, 9),
        }
        for name, values in fields.items():
            point_cloud.point_data[name] = cast_for_write(values, write_precision)
        write_vtk_dataset(point_cloud, particle_file_name)

        Logging.record(
            "particle output written",
            ("format", "vtk"),
            ("particles", f"{n:,}"),
            ("path", str(particle_file_name)),
        )

    def load_vortex_particles(self, particle_file_name: str, remove_current_particles: bool = True):
        """
        Import particle data from a VTP file and repopulate the particle list.
        """
        if not HAS_PYVISTA:
            raise ImportError("pyvista is required for VTP file operations")

        if remove_current_particles:
            self.n_particles_total = 0

        point_cloud = pv.read(particle_file_name)
        position = np.array(point_cloud.points, dtype=np.float32)
        point_data = point_cloud.point_data

        def _read(name: str):
            if name in point_data:
                return point_data[name]
            raise KeyError(f"particle field {name!r} is missing from {particle_file_name!r}")

        velocity = np.array(_read("velocity"), dtype=np.float32)
        vortex_strength = np.array(_read("vortex_strength"), dtype=np.float32)
        core_radius = np.array(_read("core_radius"), dtype=np.float32)
        particle_volume = np.array(_read("particle_volume"), dtype=np.float32)
        kinematic_viscosity = np.array(_read("kinematic_viscosity"), dtype=np.float32)
        viscosities_t = np.array(_read("eddy_viscosity"), dtype=np.float32)
        group_id = np.array(_read("group_id"), dtype=np.int32)
        velocity_gradient = np.array(_read("velocity_gradient"), dtype=np.float32)
        velocity_gradient = velocity_gradient.reshape(len(velocity_gradient), 3, 3)

        # Use the class's add_particle_field method for consistency
        self.add_vortex_particles(
            position=position,
            velocity=velocity,
            vortex_strength=vortex_strength,
            core_radius=core_radius,
            particle_volume=particle_volume,
            kinematic_viscosity=kinematic_viscosity,
            eddy_viscosity=viscosities_t,
            group_id=group_id,
            velocity_gradient=velocity_gradient,
        )

        Logging.record(
            "particle field loaded",
            ("format", "vtk"),
            ("particles", f"{len(self):,}"),
            ("path", str(particle_file_name)),
        )

    def _remove_weak_particles(self, percent: float = 0.0) -> np.ndarray:
        """Remove particles below a fraction of the global maximum strength.

        Args:
            percent: Percentage threshold relative to the cloud-wide maximum
                vortex-strength magnitude, in the range 0-100.
        """
        N = self.n_particles_total

        # Early return if no particles or no removal requested
        if N == 0 or percent <= 0.0:
            return np.empty(0, dtype=np.int64)

        vortex_strength = self.vortex_strength_cpu()
        vortex_strength_magnitudes = np.linalg.norm(vortex_strength, axis=1)

        max_strength_global = np.max(vortex_strength_magnitudes)
        if max_strength_global == 0:
            Logging.warning("component=particle_pruning status=skipped reason=zero_strength_field")
            return np.empty(0, dtype=np.int64)
        else:
            cutoff = (percent / 100.0) * max_strength_global
            remove_mask = vortex_strength_magnitudes < cutoff

        indices_to_remove = np.where(remove_mask)[0]

        if len(indices_to_remove) > 0:
            # Safety cap: never remove ALL particles via weak-removal (keep at least 1)
            if len(indices_to_remove) >= N:
                Logging.warning(
                    "component=particle_pruning status=skipped "
                    f"reason=all_particles_selected count={N}"
                )
                return np.empty(0, dtype=np.int64)

            self.remove_vortex_particles(indices=indices_to_remove, remove_all=False)
        return indices_to_remove

    def update_vortex_strength_masked(
        self, mask: np.ndarray, vortex_strength_increment: np.ndarray
    ) -> None:
        """Apply an in-place vortex strength delta to a masked subset of particles.

        The operation is: Γ_i ← Γ_i + ΔΓ_i  for all i where mask[i] is True.

        Args:
            mask: Boolean array of shape (N,) selecting particles to update.
            vortex_strength_increment: Array of shape (M, 3), where M = mask.sum().
        """
        N = self.n_particles_total
        if N == 0 or int(mask.sum()) == 0:
            return
        vortex_strength = self._extract_vector(self.vortex_strength, N)
        vortex_strength[mask] += vortex_strength_increment.astype(vortex_strength.dtype)
        self._copy_vectors_chunked(vortex_strength, self.vortex_strength, 0, N)
        self.touch_state()

    def remove_vortex_particles(self, indices, remove_all: bool = False):
        if remove_all:
            self.n_particles_total = 0
        else:
            # Get current data
            current_data = self._extract_cpu_data(self.n_particles_total)

            # Create mask for particles to keep
            mask = np.ones(self.n_particles_total, dtype=bool)
            mask[indices] = False

            # Filter all arrays
            filtered_data = {
                "position": current_data["position"][mask],
                "velocity": current_data["velocity"][mask],
                "vortex_strength": current_data["vortex_strength"][mask],
                "core_radius": current_data["core_radius"][mask],
                "particle_volume": current_data["particle_volume"][mask],
                "kinematic_viscosity": current_data["kinematic_viscosity"][mask],
                "eddy_viscosity": current_data["eddy_viscosity"][mask],
                "effective_viscosity": current_data["effective_viscosity"][mask],
                "group_id": current_data["group_id"][mask],
                "velocity_gradient": current_data["velocity_gradient"][mask],
                "strain_rate": current_data["strain_rate"][mask],
                "vorticity": current_data["vorticity"][mask],
                "zone_id": current_data["zone_id"][mask],
            }

            # Repopulate fields with filtered data
            self.n_particles_total = 0  # Reset count
            if filtered_data["position"].shape[0] > 0:
                self._populate_from_numpy(**filtered_data)
        self.touch_state()

    def set_field(self, field_name: str, values: np.ndarray):
        """
        Set a specific field (e.g., 'kinematic_viscosity', 'core_radius', 'vortex_strength', etc.) with new values.
        Handles scalar, vector, matrix, and int fields.
        """
        if not hasattr(self, field_name):
            raise ValueError(f"Field '{field_name}' does not exist in Particles class.")

        field = getattr(self, field_name)
        count = self.n_particles_total
        if values.shape[0] != count:
            raise ValueError(
                f"Values for field '{field_name}' must have the same number of particles ({count})."
            )

        # Determine field type and expected shape
        # Scalar fields
        scalar_fields = [
            "core_radius",
            "particle_volume",
            "kinematic_viscosity",
            "eddy_viscosity",
            "effective_viscosity",
        ]
        int_fields = ["group_id", "zone_id"]
        vector_fields = ["position", "velocity", "vortex_strength", "vorticity"]
        matrix_fields = ["velocity_gradient", "strain_rate"]

        if field_name in scalar_fields:
            values = self._validate_numpy_input(values, (), field_name)
            self._copy_scalars_chunked(values, field, 0, count)
        elif field_name in int_fields:
            values = self._validate_numpy_input(values, (), field_name)
            self._copy_ints_chunked(values, field, 0, count)
        elif field_name in vector_fields:
            values = self._validate_numpy_input(values, (3,), field_name)
            self._copy_vectors_chunked(values, field, 0, count)
        elif field_name in matrix_fields:
            values = self._validate_numpy_input(values, (3, 3), field_name)
            self._copy_matrices_chunked(values, field, 0, count)
        else:
            raise ValueError(f"Field '{field_name}' type not recognized for set_field.")
        if field_name in {"position", "vortex_strength", "core_radius"}:
            self.touch_state()

    def add_vortex_particles_from_fields_grouped(
        self,
        count: int,
        position: ti.template(),
        velocity: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        particle_volume: ti.template(),
        group_id: ti.template(),
        kinematic_viscosity: float = 1.5e-5,
    ) -> bool:
        """
        Add particles directly from Taichi fields with per-particle group IDs.
        """
        start_idx = self.n_particles_total

        # Check if we have space
        if start_idx + count > self._max_particles:
            return False

        # Kernel to copy data
        @ti.kernel
        def copy_particles_grouped_kernel(
            dest_offset: ti.i32,
            src_count: ti.i32,
            src_pos: ti.template(),
            src_vel: ti.template(),
            src_str: ti.template(),
            src_rad: ti.template(),
            src_vol: ti.template(),
            src_gid: ti.template(),
            p_viscosity: ti.f32,
        ):
            for i in range(src_count):
                dest_idx = dest_offset + i
                self.position[dest_idx] = src_pos[i]
                self.velocity[dest_idx] = src_vel[i]
                self.vortex_strength[dest_idx] = src_str[i]
                self.core_radius[dest_idx] = src_rad[i]
                self.particle_volume[dest_idx] = src_vol[i]
                self.group_id[dest_idx] = src_gid[i]
                self.kinematic_viscosity[dest_idx] = p_viscosity
                self.eddy_viscosity[dest_idx] = 0.0
                self.effective_viscosity[dest_idx] = p_viscosity

                vol = src_vol[i]
                if vol > 1e-15:
                    self.vorticity[dest_idx] = src_str[i] / vol
                else:
                    self.vorticity[dest_idx] = ti.Vector([0.0, 0.0, 0.0])

                self.zone_id[dest_idx] = 0
                self.velocity_gradient[dest_idx].fill(0.0)
                self.strain_rate[dest_idx].fill(0.0)

        copy_particles_grouped_kernel(
            start_idx,
            count,
            position,
            velocity,
            vortex_strength,
            core_radius,
            particle_volume,
            group_id,
            kinematic_viscosity,
        )

        self.n_particles_total += count
        self.sync_device_counter()
        self.touch_state()
        self._log_particles_added(count)
        return True
