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
from ..config.types import CachedParticleProperty


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
        circulation (ti.Vector.field): Taichi field of particle circulation (α = ω·V), shape (N, 3).
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
        self.number_of_particles = 0
        self.time_step = 0  # For cache invalidation
        self._cached_step = -1  # Track when cache was last updated
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

    def _init_taichi_fields(self):
        """Initialize all Taichi fields for particle data storage with configurable dtype."""
        dtype = self._taichi_dtype
        # Vector fields for 3D properties
        self.position = ti.Vector.field(3, dtype=dtype, shape=self._max_particles)
        self.velocity = ti.Vector.field(3, dtype=dtype, shape=self._max_particles)
        self.circulation = ti.Vector.field(3, dtype=dtype, shape=self._max_particles)
        self.vorticity = ti.Vector.field(3, dtype=dtype, shape=self._max_particles)
        # Scalar fields
        self.radius = ti.field(dtype=dtype, shape=self._max_particles)
        self.volume = ti.field(dtype=dtype, shape=self._max_particles)
        self.viscosity = ti.field(dtype=dtype, shape=self._max_particles)
        self.viscosity_turbulent = ti.field(dtype=dtype, shape=self._max_particles)
        self.viscosity_effective = ti.field(dtype=dtype, shape=self._max_particles)
        self.group_id = ti.field(dtype=ti.i32, shape=self._max_particles)
        # Matrix fields
        self.velocity_gradient = ti.Matrix.field(3, 3, dtype=dtype, shape=self._max_particles)
        self.strain_rate = ti.Matrix.field(3, 3, dtype=dtype, shape=self._max_particles)
        self.zone_id = ti.field(dtype=ti.i32, shape=self._max_particles)

        # Device-side counter for atomic operations
        self.device_number_of_particles = ti.field(dtype=ti.i32, shape=())

        # Removal tag field for GPU-based particle filtering (1 = remove, 0 = keep)
        self._removal_tags = ti.field(dtype=ti.i32, shape=self._max_particles)

        # Device-side accumulators for subset moment reductions (e.g. circulation
        # and linear impulse of removed particles) — kept on device to avoid a
        # full position/circulation download just to sum a handful of indices.
        self._subset_circulation = ti.Vector.field(3, dtype=dtype, shape=())
        self._subset_impulse = ti.Vector.field(3, dtype=dtype, shape=())

        # Global background velocity (single 3D vector shared by all particles)
        self.velocity_background = ti.Vector.field(3, dtype=dtype, shape=())
        self.velocity_background[None] = [0.0, 0.0, 0.0]

    def sync_device_counter(self):
        """Sync host particle count to device field."""
        self.device_number_of_particles[None] = self.number_of_particles

    def sync_host_counter(self):
        """Sync device particle count to host."""
        self.number_of_particles = self.device_number_of_particles[None]

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
            "circulation": self._extract_vector(self.circulation, n),
            "radius": self._extract_scalar(self.radius, n),
            "volume": self._extract_scalar(self.volume, n),
            "viscosity": self._extract_scalar(self.viscosity, n),
            "viscosity_turbulent": self._extract_scalar(self.viscosity_turbulent, n),
            "viscosity_effective": self._extract_scalar(self.viscosity_effective, n),
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
        """Sum circulation ΣΓ and linear impulse 0.5·Σ(r×Γ) over a subset of indices (device-side)."""
        self._subset_circulation[None] = ti.Vector.zero(self._taichi_dtype, 3)
        self._subset_impulse[None] = ti.Vector.zero(self._taichi_dtype, 3)
        for m in range(n_idx):
            i = indices[m]
            p = self.position[i]
            c = self.circulation[i]
            self._subset_circulation[None] += c
            self._subset_impulse[None] += 0.5 * p.cross(c)

    def subset_moments(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute (ΣΓ, 0.5·Σ r×Γ) over a subset of particles entirely on device.

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
        circ = self._subset_circulation[None].to_numpy()
        impulse = self._subset_impulse[None].to_numpy()
        return circ, impulse

    @ti.kernel
    def _accumulate_prefix_circulation(self, n: ti.i32):  # type: ignore
        """Sum ΣΓ over the first n live particles (device-side)."""
        self._subset_circulation[None] = ti.Vector.zero(self._taichi_dtype, 3)
        for i in range(n):
            self._subset_circulation[None] += self.circulation[i]

    def total_circulation(self) -> np.ndarray:
        """Sum ΣΓ over all live particles on device, returning a shape-(3,) array."""
        n = self.number_of_particles
        if n == 0:
            return np.zeros(3, dtype=self._np_float_dtype)
        self._accumulate_prefix_circulation(n)
        return self._subset_circulation[None].to_numpy()

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
        strengths: ti.template(),
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
                    strengths[write_idx] = strengths[i]
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

        n = self.number_of_particles
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
        new_circulation = self._extract_vector(self.circulation, n)[keep_mask]
        new_radius = self._extract_scalar(self.radius, n)[keep_mask]
        new_volume = self._extract_scalar(self.volume, n)[keep_mask]
        new_viscosity = self._extract_scalar(self.viscosity, n)[keep_mask]
        new_viscosity_turbulent = self._extract_scalar(self.viscosity_turbulent, n)[keep_mask]
        new_group_id = self._extract_int(self.group_id, n)[keep_mask]
        new_zone_id = self._extract_int(self.zone_id, n)[keep_mask]
        new_velocity_gradient = self._extract_matrix(self.velocity_gradient, n)[keep_mask]
        new_strain_rate = self._extract_matrix(self.strain_rate, n)[keep_mask]

        self.replace_from_numpy(
            position=new_position,
            velocity=new_velocity,
            circulation=new_circulation,
            radius=new_radius,
            volume=new_volume,
            viscosity=new_viscosity,
            viscosity_turbulent=new_viscosity_turbulent,
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
        circulation,
        vorticity,
        radius,
        volume,
        viscosity,
        viscosity_turbulent,
        viscosity_effective,
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
        start_idx = self.number_of_particles

        # Check if we have enough space
        if start_idx + N > self._max_particles:
            return False

        # Copy vectors directly using pre-compiled kernels (no temp fields!)
        self._copy_vectors_chunked(position, self.position, start_idx, N)
        self._copy_vectors_chunked(velocity, self.velocity, start_idx, N)
        self._copy_vectors_chunked(circulation, self.circulation, start_idx, N)
        self._copy_vectors_chunked(vorticity, self.vorticity, start_idx, N)

        # Copy scalars directly
        self._copy_scalars_chunked(radius, self.radius, start_idx, N)
        self._copy_scalars_chunked(volume, self.volume, start_idx, N)
        self._copy_scalars_chunked(viscosity, self.viscosity, start_idx, N)
        self._copy_scalars_chunked(viscosity_turbulent, self.viscosity_turbulent, start_idx, N)
        self._copy_scalars_chunked(viscosity_effective, self.viscosity_effective, start_idx, N)

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
        circulation,
        radius,
        volume,
        viscosity,
        viscosity_turbulent,
        viscosity_effective,
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
        circulation = self._validate_numpy_input(circulation, (3,), "circulation")
        vorticity = self._validate_numpy_input(vorticity, (3,), "vorticity")
        radius = self._validate_numpy_input(radius, (), "radius")
        volume = self._validate_numpy_input(volume, (), "volume")
        viscosity = self._validate_numpy_input(viscosity, (), "viscosity")
        viscosity_turbulent = self._validate_numpy_input(
            viscosity_turbulent, (), "viscosity_turbulent"
        )
        viscosity_effective = self._validate_numpy_input(
            viscosity_effective, (), "viscosity_effective"
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
        self._replace_field_native("vector", self.circulation, circulation, count)
        self._replace_field_native("vector", self.vorticity, vorticity, count)
        self._replace_field_native("scalar", self.radius, radius, count)
        self._replace_field_native("scalar", self.volume, volume, count)
        self._replace_field_native("scalar", self.viscosity, viscosity, count)
        self._replace_field_native("scalar", self.viscosity_turbulent, viscosity_turbulent, count)
        self._replace_field_native("scalar", self.viscosity_effective, viscosity_effective, count)
        self._replace_field_native("int", self.group_id, group_id, count)
        self._replace_field_native("int", self.zone_id, zone_id, count)
        self._replace_field_native("matrix", self.velocity_gradient, velocity_gradient, count)
        self._replace_field_native("matrix", self.strain_rate, strain_rate, count)

        self.number_of_particles = count

    # CPU access methods (return NumPy arrays) - now with caching
    @CachedParticleProperty
    def position_cpu(self):
        """Get positions as NumPy array (CPU copy) - cached per time step."""
        return self._extract_vector(self.position, self.number_of_particles)

    @CachedParticleProperty
    def velocity_cpu(self):
        """Get velocities as NumPy array (CPU copy) - cached per time step."""
        return self._extract_vector(self.velocity, self.number_of_particles)

    @CachedParticleProperty
    def circulation_cpu(self):
        """Get strengths as NumPy array (CPU copy) - cached per time step."""
        return self._extract_vector(self.circulation, self.number_of_particles)

    @CachedParticleProperty
    def radius_cpu(self):
        """Get radii as NumPy array (CPU copy) - cached per time step."""
        return self._extract_scalar(self.radius, self.number_of_particles)

    @CachedParticleProperty
    def volume_cpu(self):
        """Get volumes as NumPy array (CPU copy) - cached per time step."""
        return self._extract_scalar(self.volume, self.number_of_particles)

    @CachedParticleProperty
    def viscosity_cpu(self):
        """Get viscosities as NumPy array (CPU copy) - cached per time step."""
        return self._extract_scalar(self.viscosity, self.number_of_particles)

    @CachedParticleProperty
    def viscosity_turbulent_cpu(self):
        """Get turbulent viscosities as NumPy array (CPU copy) - cached per time step."""
        return self._extract_scalar(self.viscosity_turbulent, self.number_of_particles)

    @CachedParticleProperty
    def viscosity_effective_cpu(self):
        """Get effective viscosities as NumPy array (CPU copy) - cached per time step."""
        return self._extract_scalar(self.viscosity_effective, self.number_of_particles)

    @CachedParticleProperty
    def group_id_cpu(self):
        """Get group IDs as NumPy array (CPU copy) - cached per time step."""
        return self._extract_int(self.group_id, self.number_of_particles)

    @CachedParticleProperty
    def velocity_gradient_cpu(self):
        """Get gradient of velocity field on CPU - cached per time step."""
        return self._extract_matrix(self.velocity_gradient, self.number_of_particles)

    @CachedParticleProperty
    def strain_rate_cpu(self):
        """Get strain rate tensors as NumPy array (CPU copy) - cached per time step."""
        return self._extract_matrix(self.strain_rate, self.number_of_particles)

    @CachedParticleProperty
    def vorticity_cpu(self):
        """Get vorticities as NumPy array (CPU copy) - cached per time step."""
        return self._extract_vector(self.vorticity, self.number_of_particles)

    @CachedParticleProperty
    def zone_id_cpu(self):
        """Get zone IDs as NumPy array (CPU copy) - cached per time step."""
        return self._extract_int(self.zone_id, self.number_of_particles)

    def velocity_background_cpu(self) -> np.ndarray:
        """Get background velocity as NumPy array (3,)."""
        v = self.velocity_background[None]
        return np.array([v[0], v[1], v[2]], dtype=np.float32)

    def set_background_velocity(self, velocity: np.ndarray) -> None:
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
        return int(self.number_of_particles)

    def __str__(self):
        """Return formatted string representation of particle system statistics."""
        lines = []

        N = self.number_of_particles
        lines.append(f"  Number of Particles      : {N:,}")

        if N > 0:
            # Get particle data
            positions = self.position_cpu()
            radii = self.radius_cpu()
            volumes = self.volume_cpu()
            strengths = self.circulation_cpu()
            velocities = self.velocity_cpu()

            # Compute statistics
            strength_mag = np.linalg.norm(strengths, axis=1)
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
            lines.append(f"    Min                    : {np.min(strength_mag):.4e} m³/s")
            lines.append(f"    Max                    : {np.max(strength_mag):.4e} m³/s")
            lines.append(f"    Mean                   : {np.mean(strength_mag):.4e} m³/s")

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
            "strength": self.circulation_cpu()[index],
            "radius": self.radius_cpu()[index],
            "volume": self.volume_cpu()[index],
            "viscosity": self.viscosity_cpu()[index],
            "viscosity_t": self.viscosity_turbulent_cpu()[index],
            "viscosity_eff": self.viscosity_effective_cpu()[index],
            "group_id": self.group_id_cpu()[index],
            "velocity_gradient": self.velocity_gradient_cpu()[index],
            "strain_rate": self.strain_rate_cpu()[index],
            "vorticity": self.vorticity_cpu()[index],
            "zone_id": self.zone_id_cpu()[index],
        }

    def add_vortex_particle(
        self,
        position: np.ndarray = np.zeros(3),
        velocity: np.ndarray = np.zeros(3),
        strength: np.ndarray = np.zeros(3),
        radius: float = 1.0,
        volume: float = 1.0,
        viscosity: float = 0.0,
        viscosity_t: float = 0.0,
        group_id: int = 0,
        zone_id: int = 0,
        grad_u: np.ndarray = np.zeros((3, 3), dtype=np.float32),
        vorticity: np.ndarray = np.zeros(3),
    ):
        # Ensure we have space for one more particle
        self._grow_capacity(self.number_of_particles + 1)

        # Prepare data arrays with a single particle (using float32 for Taichi compatibility)
        position = np.ascontiguousarray(position, dtype=np.float32).reshape(1, 3)
        velocity = np.ascontiguousarray(velocity, dtype=np.float32).reshape(1, 3)
        strength = np.ascontiguousarray(strength, dtype=np.float32).reshape(1, 3)
        vorticity = np.ascontiguousarray(vorticity, dtype=np.float32).reshape(1, 3)
        grad_u = np.ascontiguousarray(grad_u, dtype=np.float32).reshape(1, 3, 3)
        Sij = np.ascontiguousarray(np.zeros((1, 3, 3), dtype=np.float32))

        # Scalar values
        # Scalar values
        radius = np.array([radius], dtype=np.float32).reshape(1)
        volume = np.array([volume], dtype=np.float32).reshape(1)
        viscosity = np.array([viscosity], dtype=np.float32).reshape(1)
        viscosity_t = np.array([viscosity_t], dtype=np.float32).reshape(1)
        viscosity_eff = np.array([viscosity + viscosity_t], dtype=np.float32).reshape(1)
        group_id = np.array([group_id], dtype=np.int32).reshape(1)
        zone_id = np.array([zone_id], dtype=np.int32).reshape(1)

        # Copy to Taichi fields at the current end position
        idx = self.number_of_particles
        self._copy_to_taichi_vectors(position, self.position, idx, 1)
        self._copy_to_taichi_vectors(velocity, self.velocity, idx, 1)
        self._copy_to_taichi_vectors(strength, self.circulation, idx, 1)
        self._copy_to_taichi_vectors(vorticity, self.vorticity, idx, 1)
        self._copy_to_taichi_scalars(radius, self.radius, idx, 1)
        self._copy_to_taichi_scalars(volume, self.volume, idx, 1)
        self._copy_to_taichi_scalars(viscosity, self.viscosity, idx, 1)
        self._copy_to_taichi_scalars(viscosity_t, self.viscosity_turbulent, idx, 1)
        self._copy_to_taichi_scalars(viscosity_eff, self.viscosity_effective, idx, 1)
        self._copy_to_taichi_ints(group_id, self.group_id, idx, 1)
        self._copy_to_taichi_ints(zone_id, self.zone_id, idx, 1)
        self._copy_to_taichi_matrices(grad_u, self.velocity_gradient, idx, 1)
        self._copy_to_taichi_matrices(Sij, self.strain_rate, idx, 1)

        # Increment particle count
        self.number_of_particles += 1

    def add_vortex_particles(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        circulation: np.ndarray,
        radius: np.ndarray,
        volume: np.ndarray,
        viscosity: np.ndarray,
        viscosity_turbulent: np.ndarray = None,
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
            circulation: Particle circulation (α = ω·V) [N, 3] in m²/s
            radius: Particle core radii [N] in meters
            volume: Particle volumes [N] in m³
            viscosity: Molecular kinematic viscosity [N] in m²/s
            viscosity_turbulent: Turbulent viscosity [N] in m²/s (optional)
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
        _validate_finite_array(circulation, "circulation")
        _validate_finite_array(radius, "radius")
        _validate_finite_array(volume, "volume")
        _validate_finite_array(viscosity, "viscosity")

        if viscosity_turbulent is not None:
            _validate_finite_array(viscosity_turbulent, "viscosity_turbulent")
        if velocity_gradient is not None:
            _validate_finite_array(velocity_gradient, "velocity_gradient")

        # ---- CONTINUE WITH NORMAL PROCESSING ----
        # Honor the configured float precision: the Taichi fields are created
        # with self._taichi_dtype, so feeding them self._np_float_dtype arrays
        # keeps the transfer exact (no f32←f64 / f64←f32 precision warnings)
        # and makes a precision='f64' VPM run end-to-end double-precision.
        dt = self._np_float_dtype
        position = np.ascontiguousarray(position, dtype=dt)
        velocity = np.ascontiguousarray(velocity, dtype=dt)
        circulation = np.ascontiguousarray(circulation, dtype=dt)
        radius = np.ascontiguousarray(radius, dtype=dt)
        volume = np.ascontiguousarray(volume, dtype=dt)
        viscosity = np.ascontiguousarray(viscosity, dtype=dt)

        N = position.shape[0]
        if position.shape[1] != 3 or velocity.shape[1] != 3 or circulation.shape[1] != 3:
            raise ValueError("Position, velocity, and circulation must have shape (N x 3).")
        if not (radius.shape[0] == N):
            raise ValueError("Radius must match the number of positions.")

        # Ensure all arrays are contiguous and have the correct shape
        if viscosity_turbulent is None:
            viscosity_turbulent = np.zeros(N, dtype=dt)
        else:
            viscosity_turbulent = np.ascontiguousarray(viscosity_turbulent, dtype=dt)

        # Calculate effective viscosity
        viscosity_effective = viscosity + viscosity_turbulent

        # Prepare other fields
        group_id = _coerce_int_id_array(group_id, N)
        zone_id = _coerce_int_id_array(zone_id, N)

        # Ensure velocity_gradient and strain_rate are properly initialized
        if velocity_gradient is None:
            velocity_gradient = np.zeros((N, 3, 3), dtype=dt)
        else:
            velocity_gradient = np.ascontiguousarray(velocity_gradient, dtype=dt)

        # Initialize strain_rate tensor - will be computed from velocity_gradient later
        strain_rate = np.zeros((N, 3, 3), dtype=dt)

        # Initialize vorticity field: vorticity = circulation / volume
        vorticity = (circulation / volume[:, None]).astype(dt)

        # Ensure we have enough space for all particles
        total_particles = self.number_of_particles + N
        self._grow_capacity(total_particles)

        # Copy all data to Taichi fields at once
        start_idx = self.number_of_particles

        # Try fast batch add first (for initial particle loading)
        if not self._fast_batch_add(
            position,
            velocity,
            circulation,
            vorticity,
            radius,
            volume,
            viscosity,
            viscosity_turbulent,
            viscosity_effective,
            group_id,
            zone_id,
            velocity_gradient,
            strain_rate,
        ):
            # Fall back to element-by-element copy for appending
            self._copy_vectors_chunked(position, self.position, start_idx, N)
            self._copy_vectors_chunked(velocity, self.velocity, start_idx, N)
            self._copy_vectors_chunked(circulation, self.circulation, start_idx, N)
            self._copy_vectors_chunked(vorticity, self.vorticity, start_idx, N)
            self._copy_scalars_chunked(radius, self.radius, start_idx, N)
            self._copy_scalars_chunked(volume, self.volume, start_idx, N)
            self._copy_scalars_chunked(viscosity, self.viscosity, start_idx, N)
            self._copy_scalars_chunked(viscosity_turbulent, self.viscosity_turbulent, start_idx, N)
            self._copy_scalars_chunked(viscosity_effective, self.viscosity_effective, start_idx, N)
            self._copy_ints_chunked(group_id, self.group_id, start_idx, N)
            self._copy_ints_chunked(zone_id, self.zone_id, start_idx, N)
            self._copy_matrices_chunked(velocity_gradient, self.velocity_gradient, start_idx, N)
            self._copy_matrices_chunked(strain_rate, self.strain_rate, start_idx, N)

        # Update particle count
        self.number_of_particles = total_particles

        # Invalidate cache since particle data has changed
        self._cached_step = -1

    def replace_from_numpy(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        circulation: np.ndarray,
        radius: np.ndarray,
        volume: np.ndarray,
        viscosity: np.ndarray,
        viscosity_turbulent: np.ndarray = None,
        group_id: np.ndarray = None,
        zone_id: np.ndarray = None,
        velocity_gradient: np.ndarray = None,
        strain_rate: np.ndarray = None,
    ) -> None:
        """Replace the active particle cloud with NumPy arrays."""
        _validate_finite_array(position, "position")
        _validate_finite_array(velocity, "velocity")
        _validate_finite_array(circulation, "circulation")
        _validate_finite_array(radius, "radius")
        _validate_finite_array(volume, "volume")
        _validate_finite_array(viscosity, "viscosity")

        if viscosity_turbulent is not None:
            _validate_finite_array(viscosity_turbulent, "viscosity_turbulent")
        if velocity_gradient is not None:
            _validate_finite_array(velocity_gradient, "velocity_gradient")
        if strain_rate is not None:
            _validate_finite_array(strain_rate, "strain_rate")

        dt = self._np_float_dtype
        position = np.ascontiguousarray(position, dtype=dt)
        velocity = np.ascontiguousarray(velocity, dtype=dt)
        circulation = np.ascontiguousarray(circulation, dtype=dt)
        radius = np.ascontiguousarray(radius, dtype=dt)
        volume = np.ascontiguousarray(volume, dtype=dt)
        viscosity = np.ascontiguousarray(viscosity, dtype=dt)

        N = position.shape[0]
        if N == 0:
            self.number_of_particles = 0
            self.sync_device_counter()
            self._cached_step = -1
            return
        if position.shape != (N, 3) or velocity.shape != (N, 3) or circulation.shape != (N, 3):
            raise ValueError("Position, velocity, and circulation must have shape (N x 3).")
        if radius.shape != (N,) or volume.shape != (N,) or viscosity.shape != (N,):
            raise ValueError("Radius, volume, and viscosity must have shape (N,).")

        if viscosity_turbulent is None:
            viscosity_turbulent = np.zeros(N, dtype=dt)
        else:
            viscosity_turbulent = np.ascontiguousarray(viscosity_turbulent, dtype=dt)
            if viscosity_turbulent.shape != (N,):
                raise ValueError("Turbulent viscosity must have shape (N,).")
        viscosity_effective = viscosity + viscosity_turbulent

        group_id = _coerce_int_id_array(group_id, N)
        zone_id = _coerce_int_id_array(zone_id, N)
        if group_id.shape != (N,) or zone_id.shape != (N,):
            raise ValueError("Group and zone IDs must have shape (N,).")

        if velocity_gradient is None:
            velocity_gradient = np.zeros((N, 3, 3), dtype=dt)
        else:
            velocity_gradient = np.ascontiguousarray(velocity_gradient, dtype=dt)
        if strain_rate is None:
            strain_rate = np.zeros((N, 3, 3), dtype=dt)
        else:
            strain_rate = np.ascontiguousarray(strain_rate, dtype=dt)
        if velocity_gradient.shape != (N, 3, 3) or strain_rate.shape != (N, 3, 3):
            raise ValueError("Velocity gradient and strain rate must have shape (N, 3, 3).")
        vorticity = (circulation / volume[:, None]).astype(dt)

        self._grow_capacity(N)

        self._populate_from_numpy(
            position,
            velocity,
            circulation,
            radius,
            volume,
            viscosity,
            viscosity_turbulent,
            viscosity_effective,
            group_id,
            velocity_gradient,
            strain_rate,
            vorticity,
            zone_id,
        )
        self.sync_device_counter()
        self._cached_step = -1

    # ---- GPU-TO-GPU DATA TRANSFER ----

    def add_vortex_particles_from_fields(
        self,
        count: int,
        position: ti.template(),
        velocity: ti.template(),
        strength: ti.template(),
        radius: ti.template(),
        volume: ti.template(),
        group_id: int = 0,
        viscosity: float = 1.5e-5,
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
        start_idx = self.number_of_particles

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
                self.circulation[dest_idx] = src_str[i]

                # Copy scalars
                self.radius[dest_idx] = src_rad[i]
                self.volume[dest_idx] = src_vol[i]

                # Set fixed properties
                self.viscosity[dest_idx] = p_viscosity
                self.group_id[dest_idx] = p_group_id

                # Initialize others to zero
                self.viscosity_turbulent[dest_idx] = 0.0
                self.viscosity_effective[dest_idx] = p_viscosity
                self.vorticity[dest_idx] = ti.Vector([0.0, 0.0, 0.0])
                self.zone_id[dest_idx] = 0

                # Zero matrices
                self.velocity_gradient[dest_idx].fill(0.0)
                self.strain_rate[dest_idx].fill(0.0)

        # Launch copy kernel
        copy_particles_kernel(
            start_idx, count, position, velocity, strength, radius, volume, group_id, viscosity
        )

        # Update counter
        self.number_of_particles += count
        self.sync_device_counter()

        return True

    def add_particles_from_taichi(
        self,
        positions,  # ti.Vector.field
        velocities,  # ti.Vector.field
        strengths,  # ti.Vector.field
        radii,  # ti.field
        volumes,  # ti.field
        count: int,
        viscosity: float,
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
        total_particles = self.number_of_particles + count
        self._grow_capacity(total_particles)

        start_idx = self.number_of_particles

        # Direct Taichi-to-Taichi copy via kernel
        # Initializes all particle properties to match numpy version behavior
        self._copy_from_vlm_wake(
            positions,
            velocities,
            strengths,
            radii,
            volumes,
            self.position,
            self.velocity,
            self.circulation,
            self.radius,
            self.volume,
            self.viscosity,
            self.viscosity_turbulent,
            self.viscosity_effective,
            self.vorticity,
            self.group_id,
            self.zone_id,  # Pass self.zone_id
            self.velocity_gradient,
            self.strain_rate,
            start_idx,
            count,
            viscosity,
        )

        # Update particle count
        self.number_of_particles = total_particles

        # Invalidate cache since particle data has changed
        self._cached_step = -1

        print(
            f"   [INFO] Added {count} particles via GPU transfer. Particle system with {total_particles} particles."
        )

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
        viscosity: ti.f32,
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
            dst_visc[dst_idx] = viscosity
            dst_visc_t[dst_idx] = 0.0
            dst_visc_eff[dst_idx] = viscosity  # eff = molecular + turbulent

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

    def load_vortex_particles(self, particle_file_name: str, remove_current_particles: bool = True):
        """
        Import particle data from a VTP file and repopulate the particle list.
        """
        if not HAS_PYVISTA:
            raise ImportError("pyvista is required for VTP file operations")

        if remove_current_particles:
            self.number_of_particles = 0

        point_cloud = pv.read(particle_file_name)

        positions = np.array(point_cloud.points, dtype=np.float32)
        velocities = np.array(point_cloud.point_data["Velocity"], dtype=np.float32)
        strengths = np.array(point_cloud.point_data["Strength"], dtype=np.float32)
        radii = np.array(point_cloud.point_data["Radius"], dtype=np.float32)
        volumes = np.array(point_cloud.point_data["Volumes"], dtype=np.float32)
        viscosities = np.array(point_cloud.point_data["Viscosity"], dtype=np.float32)
        viscosities_t = np.array(point_cloud.point_data["Viscosity_t"], dtype=np.float32)
        group_id = np.array(point_cloud.point_data["Group_ID"], dtype=np.int32)
        grad_u = np.array(point_cloud.point_data["Grad_U"], dtype=np.float32)
        grad_u = grad_u.reshape(len(grad_u), 3, 3)

        # Use the class's add_particle_field method for consistency
        self.add_vortex_particles(
            positions=positions,
            velocities=velocities,
            strengths=strengths,
            radii=radii,
            volumes=volumes,
            viscosities=viscosities,
            viscosities_t=viscosities_t,
            group_id=group_id,
            grad_u=grad_u,
        )

        print(f"Loaded {len(self)} particles from {particle_file_name}")

    @staticmethod
    def _per_group_removal_mask(
        group_ids: np.ndarray, strength_mags: np.ndarray, percent: float
    ) -> np.ndarray:
        N = len(strength_mags)
        unique_groups = np.unique(group_ids)
        remove_mask = np.zeros(N, dtype=bool)
        for gid in unique_groups:
            group_mask = group_ids == gid
            group_strengths = strength_mags[group_mask]
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
        N = self.number_of_particles

        # Early return if no particles or no removal requested
        if N == 0 or percent <= 0.0:
            return np.empty(0, dtype=np.int64)

        strengths = self.circulation_cpu()
        strength_mags = np.linalg.norm(strengths, axis=1)

        if per_group:
            group_ids = self.group_id_cpu()
            remove_mask = self._per_group_removal_mask(group_ids, strength_mags, percent)
        else:
            # Use global threshold (original behavior - can cause uneven removal)
            max_strength_global = np.max(strength_mags)
            if max_strength_global == 0:
                print(
                    "(Warning) _remove_weak_particles: all particle strengths are zero — skipping removal to avoid emptying the system."
                )
                return np.empty(0, dtype=np.int64)
            else:
                cutoff = (percent / 100.0) * max_strength_global
                remove_mask = strength_mags < cutoff

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

    def update_circulations_masked(self, mask: np.ndarray, delta_circ: np.ndarray) -> None:
        """Apply an in-place circulation delta to a masked subset of particles.

        The operation is: Γ_i ← Γ_i + ΔΓ_i  for all i where mask[i] is True.

        Args:
            mask:       Boolean array of shape (N,) selecting which particles to update.
            delta_circ: Float array of shape (M, 3) where M = mask.sum().
        """
        N = self.number_of_particles
        if N == 0 or int(mask.sum()) == 0:
            return
        circ = self._extract_vector(self.circulation, N)
        circ[mask] += delta_circ.astype(circ.dtype)
        self._copy_vectors_chunked(circ, self.circulation, 0, N)
        self._cached_step = -1

    def remove_vortex_particles(self, indices, remove_all: bool = False):
        if remove_all:
            self.number_of_particles = 0
        else:
            # Get current data
            current_data = self._extract_cpu_data(self.number_of_particles)

            # Create mask for particles to keep
            mask = np.ones(self.number_of_particles, dtype=bool)
            mask[indices] = False

            # Filter all arrays
            filtered_data = {
                "position": current_data["position"][mask],
                "velocity": current_data["velocity"][mask],
                "circulation": current_data["circulation"][mask],
                "radius": current_data["radius"][mask],
                "volume": current_data["volume"][mask],
                "viscosity": current_data["viscosity"][mask],
                "viscosity_turbulent": current_data["viscosity_turbulent"][mask],
                "viscosity_effective": current_data["viscosity_effective"][mask],
                "group_id": current_data["group_id"][mask],
                "velocity_gradient": current_data["velocity_gradient"][mask],
                "strain_rate": current_data["strain_rate"][mask],
                "vorticity": current_data["vorticity"][mask],
                "zone_id": current_data["zone_id"][mask],
            }

            # Repopulate fields with filtered data
            self.number_of_particles = 0  # Reset count
            if filtered_data["position"].shape[0] > 0:
                self._populate_from_numpy(**filtered_data)

    def set_field(self, field_name: str, values: np.ndarray):
        """
        Set a specific field (e.g., 'viscosity', 'radius', 'circulation', etc.) with new values.
        Handles scalar, vector, matrix, and int fields.
        """
        if not hasattr(self, field_name):
            raise ValueError(f"Field '{field_name}' does not exist in Particles class.")

        field = getattr(self, field_name)
        count = self.number_of_particles
        if values.shape[0] != count:
            raise ValueError(
                f"Values for field '{field_name}' must have the same number of particles ({count})."
            )

        # Determine field type and expected shape
        # Scalar fields
        scalar_fields = [
            "radius",
            "volume",
            "viscosity",
            "viscosity_turbulent",
            "viscosity_effective",
        ]
        int_fields = ["group_id", "zone_id"]
        vector_fields = ["position", "velocity", "circulation", "vorticity"]
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

    # ---- Backup methods ----
    def backup_solution(self, backup_file_name, time_step):
        """
        Export particle data to an HDF5 file (.h5).

        This is a faster alternative to VTK files that is still compatible with ParaView.
        ParaView can read HDF5 files using the 'XDMF Reader' with an accompanying .xmf file.
        """
        # Format step number as 6-digit sequential id (ParaView-friendly)
        step_str = str(time_step).zfill(6)

        # Get NumPy arrays from Taichi fields (CPU copies)
        points = self.position_cpu()
        velocities = self.velocity_cpu()
        strengths = self.circulation_cpu()
        vorticities = self.vorticity_cpu()
        radii = self.radius_cpu()
        volumes = self.volume_cpu()
        group_id = self.group_id_cpu()
        viscosities = self.viscosity_cpu()
        viscosities_t = self.viscosity_turbulent_cpu()
        grad_u = self.velocity_gradient_cpu()
        Sij = self.strain_rate_cpu()

        # File names with underscore + 6-digit sequential id: <name>_XXXXXX.ext
        h5_filename = f"{backup_file_name}_{step_str}.h5"
        xmf_filename = f"{backup_file_name}_{step_str}.xmf"

        try:
            # Save data to HDF5 file
            with h5py.File(h5_filename, "w") as f:
                # Create geometry group
                geometry = f.create_group("Geometry")
                geometry.create_dataset(
                    "Points", data=points, compression="gzip", compression_opts=6
                )

                # Create fields group
                fields = f.create_group("Fields")
                fields.create_dataset(
                    "Velocity", data=velocities, compression="gzip", compression_opts=6
                )
                fields.create_dataset(
                    "Strength", data=strengths, compression="gzip", compression_opts=6
                )
                fields.create_dataset(
                    "Vorticity", data=vorticities, compression="gzip", compression_opts=6
                )
                fields.create_dataset("Radius", data=radii, compression="gzip", compression_opts=6)
                fields.create_dataset(
                    "Volumes", data=volumes, compression="gzip", compression_opts=6
                )
                fields.create_dataset(
                    "Group_ID", data=group_id, compression="gzip", compression_opts=6
                )
                fields.create_dataset(
                    "Viscosity", data=viscosities, compression="gzip", compression_opts=6
                )
                fields.create_dataset(
                    "Viscosity_t", data=viscosities_t, compression="gzip", compression_opts=6
                )

                # Handle tensor fields
                fields.create_dataset(
                    "Grad_U",
                    data=grad_u.reshape(len(grad_u), -1),
                    compression="gzip",
                    compression_opts=6,
                )
                fields.create_dataset(
                    "strain_rate",
                    data=Sij.reshape(len(Sij), -1),
                    compression="gzip",
                    compression_opts=6,
                )
                fields.create_dataset(
                    "Grad_U_xx", data=grad_u[:, 0, 0], compression="gzip", compression_opts=6
                )
                fields.create_dataset(
                    "Grad_U_yy", data=grad_u[:, 1, 1], compression="gzip", compression_opts=6
                )
                fields.create_dataset(
                    "Grad_U_zz", data=grad_u[:, 2, 2], compression="gzip", compression_opts=6
                )
                fields.create_dataset(
                    "Grad_U_xy", data=grad_u[:, 0, 1], compression="gzip", compression_opts=6
                )
                fields.create_dataset(
                    "Grad_U_xz", data=grad_u[:, 0, 2], compression="gzip", compression_opts=6
                )
                fields.create_dataset(
                    "Grad_U_yz", data=grad_u[:, 1, 2], compression="gzip", compression_opts=6
                )
                fields.create_dataset(
                    "Sij_xx", data=Sij[:, 0, 0], compression="gzip", compression_opts=6
                )
                fields.create_dataset(
                    "Sij_yy", data=Sij[:, 1, 1], compression="gzip", compression_opts=6
                )
                fields.create_dataset(
                    "Sij_zz", data=Sij[:, 2, 2], compression="gzip", compression_opts=6
                )
                fields.create_dataset(
                    "Sij_xy", data=Sij[:, 0, 1], compression="gzip", compression_opts=6
                )
                fields.create_dataset(
                    "Sij_xz", data=Sij[:, 0, 2], compression="gzip", compression_opts=6
                )
                fields.create_dataset(
                    "Sij_yz", data=Sij[:, 1, 2], compression="gzip", compression_opts=6
                )

                # Store metadata
                metadata = f.create_group("Metadata")
                metadata.attrs["time_step"] = time_step
                metadata.attrs["num_particles"] = len(points)
                metadata.attrs["format_version"] = "1.0"

            # Create XDMF file for ParaView compatibility
            precision_bytes = 4 if self.float_dtype == "f32" else 8
            self._create_xdmf_file(
                xmf_filename, h5_filename, time_step, len(points), precision_bytes
            )

            print(f"\u2022 Particle data exported to {h5_filename} (HDF5 format)")
            print(f"\u2022 ParaView metadata written to {xmf_filename}")

        except Exception as e:
            print(f"(Error) Failed to save HDF5 file {h5_filename}: {e}")

    def _create_xdmf_file(
        self, xmf_filename, h5_filename, time_step, num_particles, precision_bytes=4
    ):
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

      <Information Name="TimeValue" Value="{time_step}"/>
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
        strengths = fields["Strength"][:]
        vorticities = fields["Vorticity"][:]
        radii = fields["Radius"][:]
        volumes = fields["Volumes"][:]
        group_id = fields["Group_ID"][:]
        viscosities = fields["Viscosity"][:]
        viscosities_t = fields["Viscosity_t"][:]
        grad_u = fields["Grad_U"][:] if "Grad_U" in fields else np.array([])
        Sij = fields["strain_rate"][:] if "strain_rate" in fields else np.array([])
        metadata = f["Metadata"]
        time_step = metadata.attrs["time_step"]
        num_particles = metadata.attrs["num_particles"]
        print(f"Loaded {num_particles} particles from {f.filename} (time step {time_step})")
        if num_particles > 0:
            if len(grad_u) > 0:
                grad_u = grad_u.reshape(num_particles, 3, 3)
            if len(Sij) > 0:
                Sij = Sij.reshape(num_particles, 3, 3)
            self.number_of_particles = 0
            self._populate_from_numpy(
                position=positions,
                velocity=velocities,
                circulation=strengths,
                vorticity=vorticities,
                radius=radii,
                volume=volumes,
                group_id=group_id,
                viscosity=viscosities,
                viscosity_turbulent=viscosities_t,
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
        strength: ti.template(),
        radius: ti.template(),
        volume: ti.template(),
        group_ids: ti.template(),
        viscosity: float = 1.5e-5,
    ) -> bool:
        """
        Add particles directly from Taichi fields with per-particle group IDs.
        """
        start_idx = self.number_of_particles

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
                self.circulation[dest_idx] = src_str[i]
                self.radius[dest_idx] = src_rad[i]
                self.volume[dest_idx] = src_vol[i]
                self.group_id[dest_idx] = src_gid[i]
                self.viscosity[dest_idx] = p_viscosity
                self.viscosity_turbulent[dest_idx] = 0.0
                self.viscosity_effective[dest_idx] = p_viscosity

                vol = src_vol[i]
                if vol > 1e-15:
                    self.vorticity[dest_idx] = src_str[i] / vol
                else:
                    self.vorticity[dest_idx] = ti.Vector([0.0, 0.0, 0.0])

                self.zone_id[dest_idx] = 0
                self.velocity_gradient[dest_idx].fill(0.0)
                self.strain_rate[dest_idx].fill(0.0)

        copy_particles_grouped_kernel(
            start_idx, count, position, velocity, strength, radius, volume, group_ids, viscosity
        )

        self.number_of_particles += count
        self.sync_device_counter()
        self._cached_step = -1
        return True
