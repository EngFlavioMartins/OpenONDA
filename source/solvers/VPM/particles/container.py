"""Particle storage for the VPM solver."""

import numpy as np

try:
    import pyvista as pv

    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None
import h5py
import taichi as ti

# Import VPM constants
from ..config.constants import MAX_PARTICLES
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
        position (ti.Vector.field): Taichi field of particle positions, shape (N, 3).
        velocity (ti.Vector.field): Taichi field of particle velocities, shape (N, 3).
        vortex_strength (ti.Vector.field): Particle alpha = omega*V [m³/s], shape (N, 3).
        radius (ti.field): Taichi field of particle core radii, shape (N,).
        volume (ti.field): Taichi field of particle volumes, shape (N,).
        viscosity (ti.field): Taichi field of molecular viscosities, shape (N,).
        viscosity_turbulent (ti.field): Taichi field of turbulent viscosities, shape (N,).
        viscosity_effective (ti.field): Taichi field of effective viscosities (nu + nut), shape (N,).
        strain_rate (ti.Matrix.field): Taichi field of strain rate tensors Sij, shape (N, 3, 3).
        vorticity (ti.Vector.field): Taichi field of particle vorticities, shape (N, 3).
        zone_id (ti.field): Taichi field of zone IDs (spatial zones), shape (N,).
    """

    _COPY_CHUNK_SIZE = 65_536

    def __init__(self, max_particles=MAX_PARTICLES, float_dtype: str = "f32"):
        """
        Initialize the Particles class with Taichi fields.

        Args:
            max_particles (int): Fixed particle capacity allocated at startup.
            float_dtype (str): 'f32' (default) or 'f64' - precision for particle data
        """
        self._max_particles = max_particles
        self.float_dtype = float_dtype or "f32"
        self._taichi_dtype = ti.f32 if self.float_dtype == "f32" else ti.f64
        self.n_particles = 0
        self.step = 0  # For cache invalidation
        self._cache_step = -1  # Track when cache was last updated
        # Monotone source-state version for consumers that cache spatial
        # acceleration structures.  Velocity/diagnostic writes do not affect
        # Biot-Savart sources; positions, circulation, radii, and population do.
        self._state_revision = 0
        # NumPy dtype matching Taichi float precision (avoids repeated branching)
        self._np_float_dtype = np.float32 if self.float_dtype == "f32" else np.float64
        # External ndarray bindings are cached by Taichi's Vulkan/Metal
        # backends.  Reusing one ndarray for kernels specialised on different
        # template fields can eventually alias those bindings in long runs
        # (for example a circulation download returning positions).  Give each
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
        self.volume = ti.field(dtype=dtype, shape=self._max_particles)
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

        # Device-side accumulators for subset moment reductions (e.g. circulation
        # and linear impulse of removed particles) — kept on device to avoid a
        # full position/circulation download just to sum a handful of indices.
        self._subset_vortex_strength = ti.Vector.field(3, dtype=dtype, shape=())
        self._subset_impulse = ti.Vector.field(3, dtype=dtype, shape=())

        # Global background velocity (single 3D vector shared by all particles)
        self.velocity_background = ti.Vector.field(3, dtype=dtype, shape=())
        self.velocity_background[None] = [0.0, 0.0, 0.0]

    def sync_device_counter(self):
        """Sync host particle count to device field."""
        self.device_n_particles[None] = self.n_particles

    def sync_host_counter(self):
        """Sync device particle count to host."""
        self.n_particles = self.device_n_particles[None]

    def _grow_capacity(self, needed: int) -> None:
        """Validate that a particle insertion fits the startup allocation."""
        if needed <= self._max_particles:
            return
        raise ValueError(
            f"Particle insertion requires capacity {needed}, but max_particles="
            f"{self._max_particles}. Increase VPMSetup.max_particles before "
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
                f"solver with max_particles={new_capacity} instead of resizing it."
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

    def _extract_cpu_data(self, num_particles):
        """Extract current data as NumPy arrays (only active prefix)."""
        n = num_particles
        return {
            "position": self._extract_vector(self.position, n),
            "velocity": self._extract_vector(self.velocity, n),
            "vortex_strength": self._extract_vector(self.vortex_strength, n),
            "core_radius": self._extract_scalar(self.core_radius, n),
            "volume": self._extract_scalar(self.volume, n),
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
    def _compute_strain_rate_tensor(self, grad_u: ti.template(), Sij: ti.template()):  # type: ignore
        """Compute strain rate tensor from velocity gradient tensor."""
        for i in range(grad_u.shape[0]):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    Sij[i][j, k] = 0.5 * (grad_u[i][j, k] + grad_u[i][k, j])

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

    # ---- Prefix extraction kernels (avoid full MAX_PARTICLES to_numpy()) ----

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
        a full download of every particle's position and circulation.

        Args:
            indices: Particle indices to reduce over [k].

        Returns:
            (circulation_sum, linear_impulse): two NumPy arrays of shape (3,).
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

    def total_vortex_strength(self) -> np.ndarray:
        """Sum Σalpha over all live particles, returning a shape-(3,) array."""
        n = self.n_particles
        if n == 0:
            return np.zeros(3, dtype=self._np_float_dtype)
        self._accumulate_prefix_vortex_strength(n)
        return self._subset_vortex_strength[None].to_numpy()

    @ti.kernel
    def _tag_particles_in_bounds_kernel(
        self,
        positions: ti.template(),
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
        This avoids transferring all positions to CPU for simple bound checks.
        """
        for i in range(n):
            p = positions[i]
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
        positions: ti.template(),
        velocities: ti.template(),
        vortex_strength: ti.template(),
        vorticities: ti.template(),
        radii: ti.template(),
        volumes: ti.template(),
        viscosities: ti.template(),
        viscosities_t: ti.template(),
        viscosities_eff: ti.template(),
        group_ids: ti.template(),
        zone_ids: ti.template(),
        gradU: ti.template(),
        Sij: ti.template(),
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
                    positions[write_idx] = positions[i]
                    velocities[write_idx] = velocities[i]
                    vortex_strength[write_idx] = vortex_strength[i]
                    vorticities[write_idx] = vorticities[i]
                    radii[write_idx] = radii[i]
                    volumes[write_idx] = volumes[i]
                    viscosities[write_idx] = viscosities[i]
                    viscosities_t[write_idx] = viscosities_t[i]
                    viscosities_eff[write_idx] = viscosities_eff[i]
                    group_ids[write_idx] = group_ids[i]
                    zone_ids[write_idx] = zone_ids[i]
                    for j in ti.static(range(3)):
                        for k in ti.static(range(3)):
                            gradU[write_idx][j, k] = gradU[i][j, k]
                            Sij[write_idx][j, k] = Sij[i][j, k]
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

        n = self.n_particles
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
        # positions below.
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
        new_volume = self._extract_scalar(self.volume, n)[keep_mask]
        new_viscosity = self._extract_scalar(self.kinematic_viscosity, n)[keep_mask]
        new_viscosity_turbulent = self._extract_scalar(self.eddy_viscosity, n)[keep_mask]
        new_group_id = self._extract_int(self.group_id, n)[keep_mask]
        new_zone_id = self._extract_int(self.zone_id, n)[keep_mask]
        new_velocity_gradient = self._extract_matrix(self.velocity_gradient, n)[keep_mask]
        new_strain_rate = self._extract_matrix(self.strain_rate, n)[keep_mask]

        self.replace_from_numpy(
            position=new_position,
            velocity=new_velocity,
            vortex_strength=new_vortex_strength,
            core_radius=new_radius,
            volume=new_volume,
            kinematic_viscosity=new_viscosity,
            eddy_viscosity=new_viscosity_turbulent,
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
        volume,
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
        start_idx = self.n_particles

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
        self._copy_scalars_chunked(volume, self.volume, start_idx, N)
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
        volume,
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
        volume = self._validate_numpy_input(volume, (), "volume")
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
        self._replace_field_native("scalar", self.volume, volume, count)
        self._replace_field_native("scalar", self.kinematic_viscosity, kinematic_viscosity, count)
        self._replace_field_native("scalar", self.eddy_viscosity, eddy_viscosity, count)
        self._replace_field_native("scalar", self.effective_viscosity, effective_viscosity, count)
        self._replace_field_native("int", self.group_id, group_id, count)
        self._replace_field_native("int", self.zone_id, zone_id, count)
        self._replace_field_native("matrix", self.velocity_gradient, velocity_gradient, count)
        self._replace_field_native("matrix", self.strain_rate, strain_rate, count)

        self.n_particles = count

    # CPU access methods (return NumPy arrays) - now with caching
    @cached_particle_property
    def position_cpu(self):
        """Get positions as NumPy array (CPU copy) - cached per time step."""
        return self._extract_vector(self.position, self.n_particles)

    @cached_particle_property
    def velocity_cpu(self):
        """Get velocities as NumPy array (CPU copy) - cached per time step."""
        return self._extract_vector(self.velocity, self.n_particles)

    @cached_particle_property
    def vortex_strength_cpu(self):
        """Get strengths as NumPy array (CPU copy) - cached per time step."""
        return self._extract_vector(self.vortex_strength, self.n_particles)

    @cached_particle_property
    def core_radius_cpu(self):
        """Get radii as NumPy array (CPU copy) - cached per time step."""
        return self._extract_scalar(self.core_radius, self.n_particles)

    @cached_particle_property
    def volume_cpu(self):
        """Get volumes as NumPy array (CPU copy) - cached per time step."""
        return self._extract_scalar(self.volume, self.n_particles)

    @cached_particle_property
    def kinematic_viscosity_cpu(self):
        """Get viscosities as NumPy array (CPU copy) - cached per time step."""
        return self._extract_scalar(self.kinematic_viscosity, self.n_particles)

    @cached_particle_property
    def eddy_viscosity_cpu(self):
        """Get turbulent viscosities as NumPy array (CPU copy) - cached per time step."""
        return self._extract_scalar(self.eddy_viscosity, self.n_particles)

    @cached_particle_property
    def effective_viscosity_cpu(self):
        """Get effective viscosities as NumPy array (CPU copy) - cached per time step."""
        return self._extract_scalar(self.effective_viscosity, self.n_particles)

    @cached_particle_property
    def group_id_cpu(self):
        """Get group IDs as NumPy array (CPU copy) - cached per time step."""
        return self._extract_int(self.group_id, self.n_particles)

    @cached_particle_property
    def velocity_gradient_cpu(self):
        """Get gradient of velocity field on CPU - cached per time step."""
        return self._extract_matrix(self.velocity_gradient, self.n_particles)

    @cached_particle_property
    def strain_rate_cpu(self):
        """Get strain rate tensors as NumPy array (CPU copy) - cached per time step."""
        return self._extract_matrix(self.strain_rate, self.n_particles)

    @cached_particle_property
    def vorticity_cpu(self):
        """Get vorticities as NumPy array (CPU copy) - cached per time step."""
        return self._extract_vector(self.vorticity, self.n_particles)

    @cached_particle_property
    def zone_id_cpu(self):
        """Get zone IDs as NumPy array (CPU copy) - cached per time step."""
        return self._extract_int(self.zone_id, self.n_particles)

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
        return int(self.n_particles)

    def __str__(self):
        """Return formatted string representation of particle system statistics."""
        lines = []

        N = self.n_particles
        lines.append(f"  Number of Particles      : {N:,}")

        if N > 0:
            # Get particle data
            positions = self.position_cpu()
            radii = self.core_radius_cpu()
            volumes = self.volume_cpu()
            vortex_strength = self.vortex_strength_cpu()
            velocities = self.velocity_cpu()

            # Compute statistics
            vortex_strength_magnitude = np.linalg.norm(vortex_strength, axis=1)
            velocity_mag = np.linalg.norm(velocities, axis=1)

            # Spatial extent
            bbox_min = np.min(positions, axis=0)
            bbox_max = np.max(positions, axis=0)
            domain_size = bbox_max - bbox_min

            lines.append("  Spatial Extent:")
            lines.append(
                f"    X: [{bbox_min[0]:>10.3e}, {bbox_max[0]:>10.3e}] m  (Δx = {domain_size[0]:.3e} m)"
            )
            lines.append(
                f"    Y: [{bbox_min[1]:>10.3e}, {bbox_max[1]:>10.3e}] m  (Δy = {domain_size[1]:.3e} m)"
            )
            lines.append(
                f"    Z: [{bbox_min[2]:>10.3e}, {bbox_max[2]:>10.3e}] m  (Δz = {domain_size[2]:.3e} m)"
            )

            lines.append("  Particle Radii:")
            lines.append(f"    Min                    : {np.min(radii):.4e} m")
            lines.append(f"    Max                    : {np.max(radii):.4e} m")
            lines.append(f"    Mean                   : {np.mean(radii):.4e} m")

            lines.append("  Particle Volumes:")
            lines.append(f"    Min                    : {np.min(volumes):.4e} m³")
            lines.append(f"    Max                    : {np.max(volumes):.4e} m³")
            lines.append(f"    Mean                   : {np.mean(volumes):.4e} m³")
            lines.append(f"    Total                  : {np.sum(volumes):.4e} m³")

            lines.append("  Vortex Strength Magnitude:")
            lines.append(
                f"    Min                    : {np.min(vortex_strength_magnitude):.4e} m³/s"
            )
            lines.append(
                f"    Max                    : {np.max(vortex_strength_magnitude):.4e} m³/s"
            )
            lines.append(
                f"    Mean                   : {np.mean(vortex_strength_magnitude):.4e} m³/s"
            )

            lines.append("  Velocity Magnitude:")
            lines.append(f"    Min                    : {np.min(velocity_mag):.4e} m/s")
            lines.append(f"    Max                    : {np.max(velocity_mag):.4e} m/s")
            lines.append(f"    Mean                   : {np.mean(velocity_mag):.4e} m/s")

            # Group information
            group_ids = self.group_id_cpu()
            unique_groups = np.unique(group_ids)
            if len(unique_groups) > 1:
                lines.append(f"  Particle Groups          : {len(unique_groups)} groups")
                for gid in unique_groups:
                    count = np.sum(group_ids == gid)
                    lines.append(
                        f"    Group {gid:<3}            : {count:,} particles ({100 * count / N:.1f}%)"
                    )
        else:
            lines.append("  Status: Empty (no particles)")

        return "\n".join(lines)

    def __getitem__(self, index):
        """Return particle data at index (CPU copy)."""
        return {
            "position": self.position_cpu()[index],
            "velocity": self.velocity_cpu()[index],
            "strength": self.vortex_strength_cpu()[index],
            "core_radius": self.core_radius_cpu()[index],
            "volume": self.volume_cpu()[index],
            "kinematic_viscosity": self.kinematic_viscosity_cpu()[index],
            "viscosity_t": self.eddy_viscosity_cpu()[index],
            "viscosity_eff": self.effective_viscosity_cpu()[index],
            "group_id": self.group_id_cpu()[index],
            "velocity_gradient": self.velocity_gradient_cpu()[index],
            "strain_rate": self.strain_rate_cpu()[index],
            "vorticity": self.vorticity_cpu()[index],
            "zone_id": self.zone_id_cpu()[index],
        }

    def _log_population(self, change: str, source: str) -> None:
        """Report the population left behind by an operation that changed it."""
        total = int(self.n_particles)
        capacity = self.capacity
        fraction = 100.0 * total / capacity if capacity else 0.0
        Logging.message(
            f"   [Particles] {change} ({source}) -> {total} total, {fraction:.1f}% of {capacity} capacity"
        )

    def _log_particles_added(self, count: int, source: str) -> None:
        """Report the population after particles were appended."""
        self._log_population(f"+{int(count)}", source)

    def _log_particles_replaced(self, previous: int, source: str) -> None:
        """Report the population after the whole cloud was replaced."""
        self._log_population(f"replaced {int(previous)}", source)

    def add_vortex_particle(
        self,
        position: np.ndarray = np.zeros(3),
        velocity: np.ndarray = np.zeros(3),
        vortex_strength: np.ndarray = np.zeros(3),
        core_radius: float = 1.0,
        volume: float = 1.0,
        kinematic_viscosity: float = 0.0,
        viscosity_t: float = 0.0,
        group_id: int = 0,
        zone_id: int = 0,
        grad_u: np.ndarray = np.zeros((3, 3), dtype=np.float32),
        vorticity: np.ndarray = np.zeros(3),
    ):
        # Ensure we have space for one more particle
        self._grow_capacity(self.n_particles + 1)

        # Prepare data arrays with a single particle (using float32 for Taichi compatibility)
        position = np.ascontiguousarray(position, dtype=np.float32).reshape(1, 3)
        velocity = np.ascontiguousarray(velocity, dtype=np.float32).reshape(1, 3)
        vortex_strength = np.ascontiguousarray(vortex_strength, dtype=np.float32).reshape(1, 3)
        vorticity = np.ascontiguousarray(vorticity, dtype=np.float32).reshape(1, 3)
        grad_u = np.ascontiguousarray(grad_u, dtype=np.float32).reshape(1, 3, 3)
        Sij = np.ascontiguousarray(np.zeros((1, 3, 3), dtype=np.float32))

        # Scalar values
        # Scalar values
        core_radius = np.array([core_radius], dtype=np.float32).reshape(1)
        volume = np.array([volume], dtype=np.float32).reshape(1)
        kinematic_viscosity = np.array([kinematic_viscosity], dtype=np.float32).reshape(1)
        viscosity_t = np.array([viscosity_t], dtype=np.float32).reshape(1)
        viscosity_eff = np.array([kinematic_viscosity + viscosity_t], dtype=np.float32).reshape(1)
        group_id = np.array([group_id], dtype=np.int32).reshape(1)
        zone_id = np.array([zone_id], dtype=np.int32).reshape(1)

        # Copy to Taichi fields at the current end position
        idx = self.n_particles
        self._copy_to_taichi_vectors(position, self.position, idx, 1)
        self._copy_to_taichi_vectors(velocity, self.velocity, idx, 1)
        self._copy_to_taichi_vectors(vortex_strength, self.vortex_strength, idx, 1)
        self._copy_to_taichi_vectors(vorticity, self.vorticity, idx, 1)
        self._copy_to_taichi_scalars(core_radius, self.core_radius, idx, 1)
        self._copy_to_taichi_scalars(volume, self.volume, idx, 1)
        self._copy_to_taichi_scalars(kinematic_viscosity, self.kinematic_viscosity, idx, 1)
        self._copy_to_taichi_scalars(viscosity_t, self.eddy_viscosity, idx, 1)
        self._copy_to_taichi_scalars(viscosity_eff, self.effective_viscosity, idx, 1)
        self._copy_to_taichi_ints(group_id, self.group_id, idx, 1)
        self._copy_to_taichi_ints(zone_id, self.zone_id, idx, 1)
        self._copy_to_taichi_matrices(grad_u, self.velocity_gradient, idx, 1)
        self._copy_to_taichi_matrices(Sij, self.strain_rate, idx, 1)

        # Increment particle count
        self.n_particles += 1
        self.touch_state()
        self._log_particles_added(1, "single")

    def add_vortex_particles(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        vortex_strength: np.ndarray,
        core_radius: np.ndarray,
        volume: np.ndarray,
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
            position: Particle positions [N, 3] in meters
            velocity: Particle velocities [N, 3] in m/s
            vortex_strength: Particle strength (α = ω·V) [N, 3] in m³/s
            core_radius: Particle core radii [N] in meters
            volume: Particle volumes [N] in m³
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
        _validate_finite_array(volume, "volume")
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
        time_step_size = self._np_float_dtype
        position = np.ascontiguousarray(position, dtype=time_step_size)
        velocity = np.ascontiguousarray(velocity, dtype=time_step_size)
        vortex_strength = np.ascontiguousarray(vortex_strength, dtype=time_step_size)
        core_radius = np.ascontiguousarray(core_radius, dtype=time_step_size)
        volume = np.ascontiguousarray(volume, dtype=time_step_size)
        kinematic_viscosity = np.ascontiguousarray(kinematic_viscosity, dtype=time_step_size)

        N = position.shape[0]
        if position.shape[1] != 3 or velocity.shape[1] != 3 or vortex_strength.shape[1] != 3:
            raise ValueError("position, velocity, and vortex_strength must have shape (N, 3)")
        if not (core_radius.shape[0] == N):
            raise ValueError("core_radius must match the number of positions")

        # Ensure all arrays are contiguous and have the correct shape
        if eddy_viscosity is None:
            eddy_viscosity = np.zeros(N, dtype=time_step_size)
        else:
            eddy_viscosity = np.ascontiguousarray(eddy_viscosity, dtype=time_step_size)

        # Calculate effective viscosity
        effective_viscosity = kinematic_viscosity + eddy_viscosity

        # Prepare other fields
        group_id = _coerce_int_id_array(group_id, N)
        zone_id = _coerce_int_id_array(zone_id, N)

        # Ensure velocity_gradient and strain_rate are properly initialized
        if velocity_gradient is None:
            velocity_gradient = np.zeros((N, 3, 3), dtype=time_step_size)
        else:
            velocity_gradient = np.ascontiguousarray(velocity_gradient, dtype=time_step_size)

        # Initialize strain_rate tensor - will be computed from velocity_gradient later
        strain_rate = np.zeros((N, 3, 3), dtype=time_step_size)

        # Initialize vorticity field: vorticity = circulation / volume
        vorticity = (vortex_strength / volume[:, None]).astype(time_step_size)

        # Ensure we have enough space for all particles
        total_particles = self.n_particles + N
        self._grow_capacity(total_particles)

        # Copy all data to Taichi fields at once
        start_idx = self.n_particles

        # Try fast batch add first (for initial particle loading)
        if not self._fast_batch_add(
            position,
            velocity,
            vortex_strength,
            vorticity,
            core_radius,
            volume,
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
            self._copy_scalars_chunked(volume, self.volume, start_idx, N)
            self._copy_scalars_chunked(kinematic_viscosity, self.kinematic_viscosity, start_idx, N)
            self._copy_scalars_chunked(eddy_viscosity, self.eddy_viscosity, start_idx, N)
            self._copy_scalars_chunked(effective_viscosity, self.effective_viscosity, start_idx, N)
            self._copy_ints_chunked(group_id, self.group_id, start_idx, N)
            self._copy_ints_chunked(zone_id, self.zone_id, start_idx, N)
            self._copy_matrices_chunked(velocity_gradient, self.velocity_gradient, start_idx, N)
            self._copy_matrices_chunked(strain_rate, self.strain_rate, start_idx, N)

        # Update particle count
        self.n_particles = total_particles

        self.touch_state()
        self._log_particles_added(N, "numpy arrays")

    def replace_from_numpy(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        vortex_strength: np.ndarray,
        core_radius: np.ndarray,
        volume: np.ndarray,
        kinematic_viscosity: np.ndarray,
        eddy_viscosity: np.ndarray = None,
        group_id: np.ndarray = None,
        zone_id: np.ndarray = None,
        velocity_gradient: np.ndarray = None,
        strain_rate: np.ndarray = None,
    ) -> None:
        """Replace the active particle cloud with NumPy arrays."""
        previous = int(self.n_particles)
        _validate_finite_array(position, "position")
        _validate_finite_array(velocity, "velocity")
        _validate_finite_array(vortex_strength, "vortex_strength")
        _validate_finite_array(core_radius, "core_radius")
        _validate_finite_array(volume, "volume")
        _validate_finite_array(kinematic_viscosity, "kinematic_viscosity")

        if eddy_viscosity is not None:
            _validate_finite_array(eddy_viscosity, "eddy_viscosity")
        if velocity_gradient is not None:
            _validate_finite_array(velocity_gradient, "velocity_gradient")
        if strain_rate is not None:
            _validate_finite_array(strain_rate, "strain_rate")

        time_step_size = self._np_float_dtype
        position = np.ascontiguousarray(position, dtype=time_step_size)
        velocity = np.ascontiguousarray(velocity, dtype=time_step_size)
        vortex_strength = np.ascontiguousarray(vortex_strength, dtype=time_step_size)
        core_radius = np.ascontiguousarray(core_radius, dtype=time_step_size)
        volume = np.ascontiguousarray(volume, dtype=time_step_size)
        kinematic_viscosity = np.ascontiguousarray(kinematic_viscosity, dtype=time_step_size)

        N = position.shape[0]
        if N == 0:
            self.n_particles = 0
            self.sync_device_counter()
            self.touch_state()
            self._log_particles_replaced(previous, "numpy arrays, emptied")
            return
        if position.shape != (N, 3) or velocity.shape != (N, 3) or vortex_strength.shape != (N, 3):
            raise ValueError("Position, velocity, and circulation must have shape (N x 3).")
        if core_radius.shape != (N,) or volume.shape != (N,) or kinematic_viscosity.shape != (N,):
            raise ValueError("Radius, volume, and viscosity must have shape (N,).")

        if eddy_viscosity is None:
            eddy_viscosity = np.zeros(N, dtype=time_step_size)
        else:
            eddy_viscosity = np.ascontiguousarray(eddy_viscosity, dtype=time_step_size)
            if eddy_viscosity.shape != (N,):
                raise ValueError("Turbulent viscosity must have shape (N,).")
        effective_viscosity = kinematic_viscosity + eddy_viscosity

        group_id = _coerce_int_id_array(group_id, N)
        zone_id = _coerce_int_id_array(zone_id, N)
        if group_id.shape != (N,) or zone_id.shape != (N,):
            raise ValueError("Group and zone IDs must have shape (N,).")

        if velocity_gradient is None:
            velocity_gradient = np.zeros((N, 3, 3), dtype=time_step_size)
        else:
            velocity_gradient = np.ascontiguousarray(velocity_gradient, dtype=time_step_size)
        if strain_rate is None:
            strain_rate = np.zeros((N, 3, 3), dtype=time_step_size)
        else:
            strain_rate = np.ascontiguousarray(strain_rate, dtype=time_step_size)
        if velocity_gradient.shape != (N, 3, 3) or strain_rate.shape != (N, 3, 3):
            raise ValueError("Velocity gradient and strain rate must have shape (N, 3, 3).")
        vorticity = (vortex_strength / volume[:, None]).astype(time_step_size)

        self._grow_capacity(N)

        self._populate_from_numpy(
            position,
            velocity,
            vortex_strength,
            core_radius,
            volume,
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
        self._log_particles_replaced(previous, "numpy arrays")

    # ---- GPU-TO-GPU DATA TRANSFER ----

    def add_vortex_particles_from_fields(
        self,
        count: int,
        position: ti.template(),
        velocity: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        volume: ti.template(),
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
            strength: Source strength field (Vector)
            radius: Source radius field (Scalar)
            volume: Source volume field (Scalar)
            group_id: Group ID to assign to new particles
            viscosity: Molecular viscosity to assign

        Returns:
            bool: True if successful, False if container is full
        """
        start_idx = self.n_particles

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
                self.volume[dest_idx] = src_vol[i]

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
            volume,
            group_id,
            kinematic_viscosity,
        )

        # Update counter
        self.n_particles += count
        self.sync_device_counter()
        self.touch_state()
        self._log_particles_added(count, "VLM wake buffer")

        return True

    def add_particles_from_taichi(
        self,
        positions,  # ti.Vector.field
        velocities,  # ti.Vector.field
        vortex_strength,  # ti.Vector.field
        radii,  # ti.field
        volumes,  # ti.field
        count: int,
        kinematic_viscosity: float,
    ):
        """
        Add particles directly from Taichi fields (GPU-to-GPU transfer).

        This method enables direct transfer from VLM wake buffers to VPM particles
        without numpy intermediates, providing significant performance improvement.

        Args:
            positions: Taichi Vector.field (N x 3) source positions
            velocities: Taichi Vector.field (N x 3) source velocities
            strengths: Taichi Vector.field (N x 3) source strengths
            radii: Taichi field (N,) source radii
            volumes: Taichi field (N,) source volumes
            count: Number of particles to transfer (must be <= source field size)
            viscosity: Molecular viscosity to assign to all transferred particles
        """
        if count == 0:
            return

        # Ensure we have capacity
        total_particles = self.n_particles + count
        self._grow_capacity(total_particles)

        start_idx = self.n_particles

        # Direct Taichi-to-Taichi copy via kernel
        # Initializes all particle properties to match numpy version behavior
        self._copy_from_vlm_wake(
            positions,
            velocities,
            vortex_strength,
            radii,
            volumes,
            self.position,
            self.velocity,
            self.vortex_strength,
            self.core_radius,
            self.volume,
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
        self.n_particles = total_particles

        self.touch_state()
        self._log_particles_added(count, "GPU transfer")

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
        dst_gradU: ti.template(),
        dst_Sij: ti.template(),
        start_idx: ti.i32,
        count: ti.i32,
        kinematic_viscosity: ti.f32,
    ):
        """
        Taichi kernel for GPU-to-GPU particle copy from VLM wake buffer.

        Initializes all particle properties to match behavior of add_vortex_particles():
        - positions, velocities, strengths, radii, volumes: from source
        - vorticities: computed as strength / volume
        - viscosities: set to provided viscosity value
        - viscosities_t: set to 0 (no turbulent viscosity)
        - viscosities_eff: set to viscosity (molecular only)
        - group_ids: set to 0 (default group)
        - gradU: set to zero (will be computed on next velocity update)
        - Sij: set to zero (will be computed from gradU)
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

            # Compute vorticity from strength and volume (matching numpy version)
            # vorticity = strength / volume
            vol = src_vol[i]
            if vol > 1e-15:
                dst_vort[dst_idx] = src_str[i] / vol
            else:
                dst_vort[dst_idx] = ti.Vector([0.0, 0.0, 0.0])

            # Assign to default group (matching numpy default)
            dst_group[dst_idx] = 0

            # Initialize gradU to zero (matching numpy version)
            # Will be computed during velocity gradient update
            for row in ti.static(range(3)):
                for col in ti.static(range(3)):
                    dst_gradU[dst_idx][row, col] = 0.0
                    dst_Sij[dst_idx][row, col] = 0.0

    def save_vortex_particles(self, particle_file_name: str) -> None:
        """Export the particle cloud to a VTP point cloud (field names match ``load_vortex_particles``)."""
        if not HAS_PYVISTA:
            Logging.message(f"   [Particles] skipped {particle_file_name}: pyvista not available")
            return

        n = int(self.n_particles)
        point_cloud = pv.PolyData(self.position_cpu())
        point_cloud.point_data["Velocity"] = self.velocity_cpu()
        point_cloud.point_data["VortexStrength"] = self.vortex_strength_cpu()
        point_cloud.point_data["CoreRadius"] = self.core_radius_cpu()
        point_cloud.point_data["Volume"] = self.volume_cpu()
        point_cloud.point_data["KinematicViscosity"] = self.kinematic_viscosity_cpu()
        point_cloud.point_data["EddyViscosity"] = self.eddy_viscosity_cpu()
        point_cloud.point_data["GroupID"] = self.group_id_cpu()
        point_cloud.point_data["VelocityGradient"] = self.velocity_gradient_cpu().reshape(n, 9)
        point_cloud.save(particle_file_name)

        Logging.message(f"   [Particles] wrote {n} to {particle_file_name}")

    def load_vortex_particles(self, particle_file_name: str, remove_current_particles: bool = True):
        """
        Import particle data from a VTP file and repopulate the particle list.
        """
        if not HAS_PYVISTA:
            raise ImportError("pyvista is required for VTP file operations")

        if remove_current_particles:
            self.n_particles = 0

        point_cloud = pv.read(particle_file_name)

        positions = np.array(point_cloud.points, dtype=np.float32)
        velocities = np.array(point_cloud.point_data["Velocity"], dtype=np.float32)
        vortex_strength = np.array(point_cloud.point_data["VortexStrength"], dtype=np.float32)
        radii = np.array(point_cloud.point_data["CoreRadius"], dtype=np.float32)
        volumes = np.array(point_cloud.point_data["Volume"], dtype=np.float32)
        viscosities = np.array(point_cloud.point_data["KinematicViscosity"], dtype=np.float32)
        viscosities_t = np.array(point_cloud.point_data["EddyViscosity"], dtype=np.float32)
        group_id = np.array(point_cloud.point_data["GroupID"], dtype=np.int32)
        grad_u = np.array(point_cloud.point_data["VelocityGradient"], dtype=np.float32)
        grad_u = grad_u.reshape(len(grad_u), 3, 3)

        # Use the class's add_particle_field method for consistency
        self.add_vortex_particles(
            position=positions,
            velocity=velocities,
            vortex_strength=vortex_strength,
            core_radius=radii,
            volume=volumes,
            kinematic_viscosity=viscosities,
            eddy_viscosity=viscosities_t,
            group_id=group_id,
            velocity_gradient=grad_u,
        )

        print(f"Loaded {len(self)} particles from {particle_file_name}")

    @staticmethod
    def _per_group_removal_mask(
        group_ids: np.ndarray, vortex_strength_magnitudes: np.ndarray, percent: float
    ) -> np.ndarray:
        N = len(vortex_strength_magnitudes)
        unique_groups = np.unique(group_ids)
        remove_mask = np.zeros(N, dtype=bool)
        for gid in unique_groups:
            group_mask = group_ids == gid
            group_strengths = vortex_strength_magnitudes[group_mask]
            if len(group_strengths) == 0:
                continue
            max_s = np.max(group_strengths)
            if max_s == 0:
                remove_mask[group_mask] = True
            else:
                cutoff = (percent / 100.0) * max_s
                group_indices = np.where(group_mask)[0]
                remove_mask[group_indices[group_strengths < cutoff]] = True
        return remove_mask

    # Optional: filter out weak particles by percentile
    def _remove_weak_particles(self, percent: float = 0.0, per_group: bool = True):
        """
        Remove particles based on their strength magnitude and shrink Taichi field sizes.

        Args:
            percent: Percentage threshold relative to maximum strength (0-100)
            per_group: If True, apply threshold independently to each group to preserve
                      relative distribution. If False, use global threshold across all particles.

        Note:
            - per_group=True (default): Ensures each group loses the same percentage of
              particles based on their own maximum strength. This preserves the relative
              structure of each vortex system.
            - per_group=False: Uses global maximum across all particles. This can cause
              uneven removal if groups have different strength scales.
        """
        N = self.n_particles

        # Early return if no particles or no removal requested
        if N == 0 or percent <= 0.0:
            return np.empty(0, dtype=np.int64)

        vortex_strength = self.vortex_strength_cpu()
        vortex_strength_magnitudes = np.linalg.norm(vortex_strength, axis=1)

        if per_group:
            group_ids = self.group_id_cpu()
            remove_mask = self._per_group_removal_mask(
                group_ids, vortex_strength_magnitudes, percent
            )
        else:
            # Use global threshold (original behavior - can cause uneven removal)
            max_strength_global = np.max(vortex_strength_magnitudes)
            if max_strength_global == 0:
                print(
                    "(Warning) _remove_weak_particles: all particle strengths are zero — skipping removal to avoid emptying the system."
                )
                return np.empty(0, dtype=np.int64)
            else:
                cutoff = (percent / 100.0) * max_strength_global
                remove_mask = vortex_strength_magnitudes < cutoff

        indices_to_remove = np.where(remove_mask)[0]

        if len(indices_to_remove) > 0:
            # Safety cap: never remove ALL particles via weak-removal (keep at least 1)
            if len(indices_to_remove) >= N:
                print(
                    "(Warning) _remove_weak_particles would remove all particles — skipping to preserve at least one particle."
                )
                return np.empty(0, dtype=np.int64)

            self.remove_vortex_particles(indices=indices_to_remove, remove_all=False)
        return indices_to_remove

    def update_vortex_strength_masked(
        self, mask: np.ndarray, vortex_strength_increment: np.ndarray
    ) -> None:
        """Apply an in-place circulation delta to a masked subset of particles.

        The operation is: Γ_i ← Γ_i + ΔΓ_i  for all i where mask[i] is True.

        Args:
            mask:       Boolean array of shape (N,) selecting which particles to update.
            delta_circ: Float array of shape (M, 3) where M = mask.sum().
        """
        N = self.n_particles
        if N == 0 or int(mask.sum()) == 0:
            return
        vortex_strength = self._extract_vector(self.vortex_strength, N)
        vortex_strength[mask] += vortex_strength_increment.astype(vortex_strength.dtype)
        self._copy_vectors_chunked(vortex_strength, self.vortex_strength, 0, N)
        self.touch_state()

    def remove_vortex_particles(self, indices, remove_all: bool = False):
        if remove_all:
            self.n_particles = 0
        else:
            # Get current data
            current_data = self._extract_cpu_data(self.n_particles)

            # Create mask for particles to keep
            mask = np.ones(self.n_particles, dtype=bool)
            mask[indices] = False

            # Filter all arrays
            filtered_data = {
                "position": current_data["position"][mask],
                "velocity": current_data["velocity"][mask],
                "vortex_strength": current_data["vortex_strength"][mask],
                "core_radius": current_data["core_radius"][mask],
                "volume": current_data["volume"][mask],
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
            self.n_particles = 0  # Reset count
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
        count = self.n_particles
        if values.shape[0] != count:
            raise ValueError(
                f"Values for field '{field_name}' must have the same number of particles ({count})."
            )

        # Determine field type and expected shape
        # Scalar fields
        scalar_fields = [
            "core_radius",
            "volume",
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

    # ---- Checkpoint methods ----

    def _create_xdmf_file(self, xmf_filename, h5_filename, step, num_particles, precision_bytes=4):
        """Create XDMF metadata file for ParaView compatibility with HDF5 data."""
        import os

        h5_basename = os.path.basename(h5_filename)

        xmf_content = f'''<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="Particles" GridType="Uniform">
      <Topology TopologyType="Polyvertex" NumberOfElements="{num_particles}"/>
      <Geometry GeometryType="XYZ">
        <DataItem Dimensions="{num_particles} 3" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
          {h5_basename}:/Geometry/Points
        </DataItem>
      </Geometry>

      <Attribute Name="Velocity" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="{num_particles} 3" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
          {h5_basename}:/Fields/Velocity
        </DataItem>
      </Attribute>

      <Attribute Name="Strength" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="{num_particles} 3" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
          {h5_basename}:/Fields/Strength
        </DataItem>
      </Attribute>

      <Attribute Name="Vorticity" AttributeType="Vector" Center="Node">
        <DataItem Dimensions="{num_particles} 3" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
          {h5_basename}:/Fields/Vorticity
        </DataItem>
      </Attribute>

      <Attribute Name="Radius" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{num_particles}" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
          {h5_basename}:/Fields/Radius
        </DataItem>
      </Attribute>

      <Attribute Name="Volumes" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{num_particles}" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
          {h5_basename}:/Fields/Volumes
        </DataItem>
      </Attribute>

      <Attribute Name="Group_ID" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{num_particles}" NumberType="Int" Format="HDF">
          {h5_basename}:/Fields/Group_ID
        </DataItem>
      </Attribute>

      <Attribute Name="Viscosity" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{num_particles}" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
          {h5_basename}:/Fields/Viscosity
        </DataItem>
      </Attribute>

      <Attribute Name="Viscosity_t" AttributeType="Scalar" Center="Node">
        <DataItem Dimensions="{num_particles}" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
          {h5_basename}:/Fields/Viscosity_t
        </DataItem>
      </Attribute>

            <Attribute Name="Grad_U_xx" AttributeType="Scalar" Center="Node">
                <DataItem Dimensions="{num_particles}" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
                    {h5_basename}:/Fields/Grad_U_xx
                </DataItem>
            </Attribute>

            <Attribute Name="Grad_U_yy" AttributeType="Scalar" Center="Node">
                <DataItem Dimensions="{num_particles}" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
                    {h5_basename}:/Fields/Grad_U_yy
                </DataItem>
            </Attribute>

            <Attribute Name="Grad_U_zz" AttributeType="Scalar" Center="Node">
                <DataItem Dimensions="{num_particles}" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
                    {h5_basename}:/Fields/Grad_U_zz
                </DataItem>
            </Attribute>

            <Attribute Name="Grad_U_xy" AttributeType="Scalar" Center="Node">
                <DataItem Dimensions="{num_particles}" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
                    {h5_basename}:/Fields/Grad_U_xy
                </DataItem>
            </Attribute>

            <Attribute Name="Grad_U_xz" AttributeType="Scalar" Center="Node">
                <DataItem Dimensions="{num_particles}" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
                    {h5_basename}:/Fields/Grad_U_xz
                </DataItem>
            </Attribute>

            <Attribute Name="Grad_U_yz" AttributeType="Scalar" Center="Node">
                <DataItem Dimensions="{num_particles}" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
                    {h5_basename}:/Fields/Grad_U_yz
                </DataItem>
            </Attribute>

            <Attribute Name="Sij_xx" AttributeType="Scalar" Center="Node">
                <DataItem Dimensions="{num_particles}" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
                    {h5_basename}:/Fields/Sij_xx
                </DataItem>
            </Attribute>

            <Attribute Name="Sij_yy" AttributeType="Scalar" Center="Node">
                <DataItem Dimensions="{num_particles}" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
                    {h5_basename}:/Fields/Sij_yy
                </DataItem>
            </Attribute>

            <Attribute Name="Sij_zz" AttributeType="Scalar" Center="Node">
                <DataItem Dimensions="{num_particles}" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
                    {h5_basename}:/Fields/Sij_zz
                </DataItem>
            </Attribute>

            <Attribute Name="Sij_xy" AttributeType="Scalar" Center="Node">
                <DataItem Dimensions="{num_particles}" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
                    {h5_basename}:/Fields/Sij_xy
                </DataItem>
            </Attribute>

            <Attribute Name="Sij_xz" AttributeType="Scalar" Center="Node">
                <DataItem Dimensions="{num_particles}" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
                    {h5_basename}:/Fields/Sij_xz
                </DataItem>
            </Attribute>

            <Attribute Name="Sij_yz" AttributeType="Scalar" Center="Node">
                <DataItem Dimensions="{num_particles}" NumberType="Float" Precision="{precision_bytes}" Format="HDF">
                    {h5_basename}:/Fields/Sij_yz
                </DataItem>
            </Attribute>

      <Information Name="TimeValue" Value="{step}"/>
    </Grid>
  </Domain>
</Xdmf>'''

        with open(xmf_filename, "w") as f:
            f.write(xmf_content)

    def _load_hdf5_particles(self, f) -> None:
        """Populate particle system from an open HDF5 file object."""
        positions = f["Geometry/Points"][:]
        fields = f["Fields"]
        velocities = fields["Velocity"][:]
        vortex_strength = fields["Strength"][:]
        vorticities = fields["Vorticity"][:]
        radii = fields["Radius"][:]
        volumes = fields["Volumes"][:]
        group_id = fields["Group_ID"][:]
        viscosities = fields["Viscosity"][:]
        viscosities_t = fields["Viscosity_t"][:]
        grad_u = fields["Grad_U"][:] if "Grad_U" in fields else np.array([])
        Sij = fields["strain_rate"][:] if "strain_rate" in fields else np.array([])
        metadata = f["Metadata"]
        step = metadata.attrs["time_step"]
        num_particles = metadata.attrs["num_particles"]
        print(f"Loaded {num_particles} particles from {f.filename} (time step {step})")
        if num_particles > 0:
            if len(grad_u) > 0:
                grad_u = grad_u.reshape(num_particles, 3, 3)
            if len(Sij) > 0:
                Sij = Sij.reshape(num_particles, 3, 3)
            self.n_particles = 0
            self._populate_from_numpy(
                position=positions,
                velocity=velocities,
                vortex_strength=vortex_strength,
                vorticity=vorticities,
                core_radius=radii,
                volume=volumes,
                group_id=group_id,
                kinematic_viscosity=viscosities,
                eddy_viscosity=viscosities_t,
                velocity_gradient=grad_u,
                strain_rate=Sij,
            )

    def load_from_hdf5(self, h5_filename):
        """
        Load particle data from an HDF5 file.

        Args:
            h5_filename: Path to the HDF5 file to load
        """
        try:
            with h5py.File(h5_filename, "r") as f:
                self._load_hdf5_particles(f)
        except Exception as e:
            print(f"(Error) Failed to load HDF5 file {h5_filename}: {e}")
            raise

    def add_vortex_particles_from_fields_grouped(
        self,
        count: int,
        position: ti.template(),
        velocity: ti.template(),
        vortex_strength: ti.template(),
        core_radius: ti.template(),
        volume: ti.template(),
        group_ids: ti.template(),
        kinematic_viscosity: float = 1.5e-5,
    ) -> bool:
        """
        Add particles directly from Taichi fields with per-particle group IDs.
        """
        start_idx = self.n_particles

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
                self.volume[dest_idx] = src_vol[i]
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
            volume,
            group_ids,
            kinematic_viscosity,
        )

        self.n_particles += count
        self.sync_device_counter()
        self.touch_state()
        self._log_particles_added(count, "VLM wake buffer, grouped")
        return True
