"""The stateful grid of the DVH and GBD viscous-diffusion schemes.

:class:`DiffusionLattice` owns the Taichi scratch grids (the vortex-strength ping-pong pair,
the body mask and wall links, the node viscosity), the fixed lattice origin and maximum
extent, the active box of one step, the explicit Laplacian kernels, and the chunked
transfers between the grids and host arrays. The schemes themselves are in ``dvh.py`` and
``gbd.py``; the solid body is :class:`~openonda.vpm.physics.diffusion.wall.BodyWall`, held
as ``lattice.wall``.
"""

from collections.abc import Sequence
import math

import numpy as np
import taichi as ti

from openonda.core.log_style import StepLog
from openonda.core.profiling import Profiler

from ...runtime.host_staging import HostStagingBuffers
from ..induction.base import TaichiField
from .wall import BodyWall

# Radius assigned to freshly regenerated particles: σ = _REGEN_RADIUS_RATIO * particle_spacing
_REGEN_RADIUS_RATIO = 2.5


def empty_gbd_moment_correction() -> dict[str, bool | int | float]:
    """Return the moment-correction record of a GBD step that removed no node."""
    return {
        "applied": False,
        "nonzero_node_count": 0,
        "retained_node_count": 0,
        "removed_node_count": 0,
        "support_augmented_node_count": 0,
        "correction_fraction": 0.0,
        "normalized_vortex_strength_residual": 0.0,
        "normalized_linear_impulse_residual": 0.0,
        "normalized_angular_impulse_residual": 0.0,
    }


def empty_gbd_wall_transfer() -> dict[str, int | float]:
    """Return the near-wall M4 scatter budget of a GBD step without a wall."""
    return {
        "wall_adjacent_particles": 0,
        "excluded_signed_weight_l1": 0.0,
        "fluid_correction_l1": 0.0,
    }


@ti.data_oriented
class DiffusionLattice:
    """Grid scratch, lattice origin and kernels shared by the DVH and GBD steps."""

    # Nodes per chunk of a transfer between a grid and a host array.
    TRANSFER_CHUNK = 65536

    # Geometric growth amortizes grid allocation and kernel recompilation.
    _ALLOC_HEADROOM = 1.5
    _REALLOC_HEADROOM = 2.0

    @property
    def _has_body_links(self) -> bool:
        return self.body_link_grid is not None

    def __init__(self) -> None:
        """Create an empty lattice; the grids are allocated by the first step."""
        # Record of the latest GBD step, read by the coupler and the tests.
        self.last_gbd_diffusion_substeps: int = 1
        self.last_gbd_moment_correction = empty_gbd_moment_correction()
        self.last_gbd_wall_transfer = empty_gbd_wall_transfer()

        # Core radius assigned to regenerated particles (σ = ratio·particle_spacing).
        self.core_radius_ratio: float = _REGEN_RADIUS_RATIO

        # Slip-slab z bounds, set only by a bound slip-slab induction.
        self.slip_slab_bounds: tuple[float, float] | None = None

        # Maximum grid dimensions from VPM domain (set by configure_max_grid_extent).
        self.max_grid_dims: tuple[int, int, int] | None = None
        self._grid_domain_bounds: np.ndarray | None = None

        # Fixed grid origin when the domain is pre-allocated.
        self.fixed_grid_min: np.ndarray | None = None

        # Grid scratch owns one replaceable Taichi allocation.
        # Taichi SNode tree and fields, None until the first allocation.
        self._grid_tree: TaichiField = None
        self._grid_a: TaichiField = None
        self._grid_b: TaichiField = None
        self.body_mask_grid: TaichiField = None  # 1=inside solid, 0=fluid
        self.body_link_grid = None  # Three bits: positive x/y/z links that cross the wall.
        self.effective_viscosity_grid: TaichiField = None
        self._grid_shape: tuple[int, ...] | None = None

        # Ping-pong flag: True → _grid_a is the current/source field.
        self.ping: bool = True

        self.wall = BodyWall(self)

        self._require_fixed_grid_allocation: bool = False
        self._grid_staging = HostStagingBuffers(self.TRANSFER_CHUNK)

        # Step profiler of the owning solver; it times the GBD phases.
        self.profiler = Profiler(enabled=False)
        # Log of the owning solver; it receives the diffusion records and warnings.
        self.logger = StepLog.disabled()

    def require_fixed_grid_allocation(self, enabled: bool = True) -> None:
        """Require grid-based diffusion to pre-allocate one fixed grid.

        GPU backends use a fixed lattice to avoid allocation and kernel
        recompilation as the cloud grows.
        """
        self._require_fixed_grid_allocation = bool(enabled)

    @property
    def current_grid(self) -> TaichiField:
        """Field holding the current vorticity (source)."""
        return self._grid_a if self.ping else self._grid_b

    @property
    def other_grid(self) -> TaichiField:
        """Field that will receive the updated vorticity (destination)."""
        return self._grid_b if self.ping else self._grid_a

    def compute_grid_bounds(
        self,
        pos: np.ndarray,
        particle_spacing: float,
        padding: float,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Return an origin and dimensions covering every finite source and its halo."""
        pos = np.asarray(pos)
        if not np.isfinite(pos).all():
            raise ValueError("Diffusion requires finite particle positions")
        if not np.isfinite(particle_spacing) or particle_spacing <= 0:
            raise ValueError("Diffusion grid spacing must be finite and positive")
        if not np.isfinite(padding) or padding < 0:
            raise ValueError("Diffusion grid padding must be finite and nonnegative")
        if len(pos) == 0:
            return np.zeros(3, dtype=np.float32), (5, 5, 5)
        margin = padding * particle_spacing
        lo = pos.min(axis=0) - margin
        hi = pos.max(axis=0) + margin

        nx = max(5, int(np.ceil((hi[0] - lo[0]) / particle_spacing)) + 1)
        ny = max(5, int(np.ceil((hi[1] - lo[1]) / particle_spacing)) + 1)
        nz = max(5, int(np.ceil((hi[2] - lo[2]) / particle_spacing)) + 1)

        return lo.astype(np.float32), (nx, ny, nz)

    def lattice_aligned_bounds(
        self,
        pos: np.ndarray,
        particle_spacing: float,
        padding: float,
        *,
        include_halo_sources: bool = False,
        required_z_bounds: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Active sub-box covering the cloud, with the origin on the fixed lattice.

        Returns ``(grid_min, (nx, ny, nz))``.  ``grid_min`` is offset from
        ``fixed_grid_min`` by a whole number of cells, so every node keeps the
        position it would have on the full pre-allocated grid; only the extent
        shrinks to the occupied region.  Clamped to stay inside the allocation.
        DVH includes sources in the allocation's padding halo: regenerated
        particles there still require their complete heat-kernel support.
        ``required_z_bounds`` keeps the physical slip planes and their mirror
        support inside the active box even when the particle cloud retreats.
        """
        anchor = np.asarray(self.fixed_grid_min, dtype=np.float64).reshape(3)
        cap = np.asarray(self.max_grid_dims, dtype=np.int64)
        finite = np.isfinite(pos).all(axis=1)
        if self._grid_domain_bounds is not None and not include_halo_sources:
            bounds = self._grid_domain_bounds
            finite &= (
                (pos[:, 0] >= bounds[0])
                & (pos[:, 0] <= bounds[1])
                & (pos[:, 1] >= bounds[2])
                & (pos[:, 1] <= bounds[3])
                & (pos[:, 2] >= bounds[4])
                & (pos[:, 2] <= bounds[5])
            )
        pts = pos[finite]
        if len(pts) == 0:
            return anchor.astype(np.float32), (5, 5, 5)

        margin = float(padding) * float(particle_spacing)
        lo = pts.min(axis=0) - margin
        hi = pts.max(axis=0) + margin
        if required_z_bounds is not None:
            lo[2] = min(lo[2], required_z_bounds[0])
            hi[2] = max(hi[2], required_z_bounds[1])

        first = np.floor((lo - anchor) / particle_spacing).astype(np.int64)
        last = np.ceil((hi - anchor) / particle_spacing).astype(np.int64)
        first = np.clip(first, 0, np.maximum(cap - 5, 0))
        last = np.clip(last, first + 4, cap - 1)
        grid_min = anchor + first * particle_spacing
        ext = last - first + 1
        if required_z_bounds is not None:
            active_hi = grid_min[2] + (int(ext[2]) - 1) * particle_spacing
            tolerance = 2.0e-5 * particle_spacing
            if (
                grid_min[2] > required_z_bounds[0] + tolerance
                or active_hi < required_z_bounds[1] - tolerance
            ):
                raise ValueError(
                    "configured GBD z extent cannot retain the slip-plane mirror/diffusion halo "
                    f"[{required_z_bounds[0]:.6g}, {required_z_bounds[1]:.6g}]; "
                    "increase GBD domain padding or reduce the diffusion interval"
                )
        return grid_min.astype(np.float32), (int(ext[0]), int(ext[1]), int(ext[2]))

    def _allocate_grid(self, shape: tuple[int, ...]) -> None:
        """Replace disposable diffusion scratch, releasing its previous storage."""
        if self._grid_tree is not None:
            ti.sync()
            self._grid_tree.destroy()
            self._grid_tree = None
        self._grid_a = self._grid_b = None
        self.body_mask_grid = self.body_link_grid = None
        self.effective_viscosity_grid = None
        self._grid_shape = None
        self.wall.mask_cache_key = None
        self.wall.mask_host = self.wall.links_host = None
        self._grid_staging = HostStagingBuffers(self.TRANSFER_CHUNK)

        fields = (
            ti.Vector.field(3, dtype=ti.f32),
            ti.Vector.field(3, dtype=ti.f32),
            ti.field(dtype=ti.i32),
            ti.field(dtype=ti.i32),
            ti.field(dtype=ti.f32),
        )
        builder = ti.FieldsBuilder()
        builder.dense(ti.ijk, shape).place(*fields)
        self._grid_tree = builder.finalize()
        (
            self._grid_a,
            self._grid_b,
            self.body_mask_grid,
            self.body_link_grid,
            self.effective_viscosity_grid,
        ) = fields
        self._grid_shape = shape
        self.ping = True

    def ensure_grid_capacity(self, nx: int, ny: int, nz: int) -> tuple[int, int, int]:
        """Ensure the complete active grid fits without discarding source support."""
        requested = (int(nx), int(ny), int(nz))
        if min(requested) < 1:
            raise ValueError("Diffusion grid dimensions must be positive")
        cap = self.max_grid_dims
        if cap is not None and any(n > c for n, c in zip(requested, cap, strict=True)):
            raise ValueError(f"Diffusion grid {requested} exceeds configured domain grid {cap}")
        if cap is None and self._require_fixed_grid_allocation:
            raise RuntimeError("Fixed diffusion allocation requires domain_bounds")
        previous = self._grid_shape
        if previous is not None and all(
            n <= old for n, old in zip(requested, previous, strict=True)
        ):
            return requested
        shape: tuple[int, ...]
        if cap is not None:
            shape = cap
        elif previous is None:
            shape = tuple(int(np.ceil(n * self._ALLOC_HEADROOM)) for n in requested)
        else:
            shape = tuple(
                int(np.ceil(n * self._REALLOC_HEADROOM)) if n > old else old
                for n, old in zip(requested, previous, strict=True)
            )
        self._allocate_grid(shape)
        self.logger.section(
            "diffusion grid allocation",
            ("requested shape", requested),
            ("allocated shape", shape),
        )
        return requested

    def configure_max_grid_extent(
        self,
        domain_bounds: Sequence[float],
        particle_spacing: float,
        padding: float = 3.0,
    ) -> None:
        """Set maximum grid-diffusion dimensions from VPM domain bounds.

        A configured domain fixes the lattice and allocates its full padded
        extent once. The physical domain and spacing determine its storage.

        Parameters
        ----------
        domain_bounds : list[float]
            ``[xmin, xmax, ymin, ymax, zmin, zmax]`` of the VPM domain.
        particle_spacing : float
            Particle / grid spacing [m].
        padding : float
            Margin in multiples of *particle_spacing* added to each side.  Default 3.0.
        """
        slab = self.slip_slab_bounds
        if slab is not None:
            if not np.allclose(np.asarray(domain_bounds)[4:6], slab, rtol=0.0, atol=1e-7):
                raise ValueError(
                    "slip-slab grid diffusion needs domain_bounds z equal to the slip planes: "
                    f"domain z=({domain_bounds[4]:.6g}, {domain_bounds[5]:.6g}), "
                    f"planes=({slab[0]:.6g}, {slab[1]:.6g})"
                )
            if padding < 3.0:
                raise ValueError("slip-slab grid diffusion requires at least three cells of z halo")

        margin = padding * particle_spacing
        nx = max(
            5, math.ceil((domain_bounds[1] - domain_bounds[0] + 2 * margin) / particle_spacing) + 1
        )
        ny = max(
            5, math.ceil((domain_bounds[3] - domain_bounds[2] + 2 * margin) / particle_spacing) + 1
        )
        nz = max(
            5, math.ceil((domain_bounds[5] - domain_bounds[4] + 2 * margin) / particle_spacing) + 1
        )
        origin_z = domain_bounds[4] - margin
        if slab is not None:
            # Slip planes lie on half nodes, so particle layers sit at cell
            # centres and a slab one spacing wide holds a single layer. A
            # coupled lattice anchor may move the origin down by under one
            # cell; the extra node keeps the full upper halo.
            origin_z = slab[0] - (math.ceil(padding) + 0.5) * particle_spacing
            nz = math.ceil((slab[1] + margin - origin_z) / particle_spacing) + 2
        self.max_grid_dims = (nx, ny, nz)
        self._grid_domain_bounds = np.asarray(domain_bounds, dtype=np.float64)

        # Store a fixed grid origin derived from the domain bounds.
        self.fixed_grid_min = np.array(
            [domain_bounds[0] - margin, domain_bounds[2] - margin, origin_z],
            dtype=np.float32,
        )

        if self._grid_shape != (nx, ny, nz):
            self._allocate_grid((nx, ny, nz))
            self.logger.section(
                "diffusion grid preallocated",
                ("shape", f"{nx}x{ny}x{nz}"),
            )

    def configure_grid_lattice_anchor(
        self, anchor: tuple[float, float, float], particle_spacing: float
    ) -> None:
        """Align the fixed diffusion-grid phase with a coupled FVM lattice.

        Only the origin phase changes; the pre-allocated dimensions remain
        unchanged.  This avoids alternating between half-cell-shifted GBD and
        transfer lattices every coupling interval.
        """
        if self.fixed_grid_min is None:
            return
        a = np.asarray(anchor, dtype=np.float64).reshape(3)
        particle_spacing = float(particle_spacing)
        if (
            not np.all(np.isfinite(a))
            or not np.isfinite(particle_spacing)
            or particle_spacing <= 0.0
        ):
            raise ValueError("grid lattice anchor and spacing must be finite")
        origin = np.asarray(self.fixed_grid_min, dtype=np.float64)
        origin = a + np.floor((origin - a) / particle_spacing) * particle_spacing
        self.fixed_grid_min = origin.astype(np.float32)

    def require_fixed_slab_extent(self, scheme: str) -> None:
        """A slip-slab diffusion step needs the pre-allocated lattice that fixes its z phase."""
        if self.fixed_grid_min is None or self.max_grid_dims is None:
            raise RuntimeError(f"slip-slab {scheme} requires a configured fixed grid extent")

    def fit_active_box(
        self,
        position: np.ndarray,
        particle_spacing: float,
        padding: float,
        *,
        include_halo_sources: bool = False,
        required_z_bounds: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Return the origin and extent of the active box for ``position``, allocating to fit it.

        A configured domain fixes the lattice origin; otherwise the box follows the cloud.
        """
        if self.fixed_grid_min is not None and self.max_grid_dims is not None:
            grid_min_np, (nx, ny, nz) = self.lattice_aligned_bounds(
                position,
                particle_spacing,
                padding,
                include_halo_sources=include_halo_sources,
                required_z_bounds=required_z_bounds,
            )
        else:
            grid_min_np, (nx, ny, nz) = self.compute_grid_bounds(
                position, particle_spacing, padding
            )
        return grid_min_np, self.ensure_grid_capacity(nx, ny, nz)

    def apply_body_mask(self, nx: int, ny: int, nz: int) -> None:
        """Zero vorticity inside masked (solid) cells on the active grid."""
        if not self.wall.active or self.body_mask_grid is None:
            return
        self._apply_body_mask_kernel(self.current_grid, self.body_mask_grid, nx, ny, nz)

    @ti.func
    def _fluid_link_open(self, body_mask: ti.template(), i, j, k, ni, nj, nk):
        fluid_link = body_mask[ni, nj, nk] == 0
        if ti.static(self._has_body_links):
            axis = 0
            if nj != j:
                axis = 1
            elif nk != k:
                axis = 2
            bits = self.body_link_grid[ti.min(i, ni), ti.min(j, nj), ti.min(k, nk)]
            fluid_link = fluid_link and ((bits & (1 << axis)) == 0)
        return fluid_link

    @ti.kernel
    def laplacian_step_gpu_kernel(
        self,
        src: ti.template(),
        dst: ti.template(),
        body_mask: ti.template(),
        alpha: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """GPU 7-point explicit Laplacian diffusion step with Neumann BC.

        Computes dst = src + α·∇²src where α = kinematic_viscosity·dt/particle_spacing².  Boundary nodes use
        index clamping (Neumann / zero-gradient), which is equivalent to the
        ``np.pad(mode='edge')`` used in the CPU path.

        Only the active sub-particle_volume ``[0, nx) × [0, ny) × [0, nz)`` is updated;
        cells beyond (due to headroom allocation) are left at zero.
        """
        for i, j, k in ti.ndrange(nx, ny, nz):
            if body_mask[i, j, k] != 0:
                dst[i, j, k] = ti.Vector.zero(ti.f32, 3)
                continue
            # Neumann BC: clamp neighbour indices to the active domain
            im = ti.max(i - 1, 0)
            ip = ti.min(i + 1, nx - 1)
            jm = ti.max(j - 1, 0)
            jp = ti.min(j + 1, ny - 1)
            km = ti.max(k - 1, 0)
            kp = ti.min(k + 1, nz - 1)

            centre = src[i, j, k]
            xp = src[ip, j, k] if self._fluid_link_open(body_mask, i, j, k, ip, j, k) else centre
            xm = src[im, j, k] if self._fluid_link_open(body_mask, i, j, k, im, j, k) else centre
            yp = src[i, jp, k] if self._fluid_link_open(body_mask, i, j, k, i, jp, k) else centre
            ym = src[i, jm, k] if self._fluid_link_open(body_mask, i, j, k, i, jm, k) else centre
            zp = src[i, j, kp] if self._fluid_link_open(body_mask, i, j, k, i, j, kp) else centre
            zm = src[i, j, km] if self._fluid_link_open(body_mask, i, j, k, i, j, km) else centre
            laplacian = xp + xm + yp + ym + zp + zm - 6.0 * centre
            dst[i, j, k] = centre + alpha * laplacian

    @ti.func
    def _variable_diffusion_face_flux(
        self,
        src: ti.template(),
        effective_viscosity_grid: ti.template(),
        body_mask: ti.template(),
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        ni: ti.i32,
        nj: ti.i32,
        nk: ti.i32,
    ) -> ti.math.vec3:
        """Return the conservative flux through one grid face."""
        flux = ti.Vector.zero(ti.f32, 3)
        if (ni != i or nj != j or nk != k) and self._fluid_link_open(
            body_mask, i, j, k, ni, nj, nk
        ):
            nu_face = 0.5 * (
                effective_viscosity_grid[i, j, k] + effective_viscosity_grid[ni, nj, nk]
            )
            flux = nu_face * (src[ni, nj, nk] - src[i, j, k])
        return flux

    @ti.kernel
    def laplacian_step_variable_gpu_kernel(
        self,
        src: ti.template(),
        dst: ti.template(),
        effective_viscosity_grid: ti.template(),
        body_mask: ti.template(),
        time_step_size: ti.f32,
        particle_spacing: ti.f32,
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """Advance the conservative variable-viscosity heat operator.

        Face-centered arithmetic averages discretize
        ``div(nu_eff * grad(omega))``.  The same face flux is used with the
        opposite sign by the neighboring cell, so total vortex strength is
        conserved under the zero-flux outer/body boundary conditions.
        """
        h_sq = particle_spacing * particle_spacing
        for i, j, k in ti.ndrange(nx, ny, nz):
            if body_mask[i, j, k] != 0:
                dst[i, j, k] = ti.Vector.zero(ti.f32, 3)
                continue
            im = ti.max(i - 1, 0)
            ip = ti.min(i + 1, nx - 1)
            jm = ti.max(j - 1, 0)
            jp = ti.min(j + 1, ny - 1)
            km = ti.max(k - 1, 0)
            kp = ti.min(k + 1, nz - 1)

            flux = (
                self._variable_diffusion_face_flux(
                    src, effective_viscosity_grid, body_mask, i, j, k, ip, j, k
                )
                + self._variable_diffusion_face_flux(
                    src, effective_viscosity_grid, body_mask, i, j, k, im, j, k
                )
                + self._variable_diffusion_face_flux(
                    src, effective_viscosity_grid, body_mask, i, j, k, i, jp, k
                )
                + self._variable_diffusion_face_flux(
                    src, effective_viscosity_grid, body_mask, i, j, k, i, jm, k
                )
                + self._variable_diffusion_face_flux(
                    src, effective_viscosity_grid, body_mask, i, j, k, i, j, kp
                )
                + self._variable_diffusion_face_flux(
                    src, effective_viscosity_grid, body_mask, i, j, k, i, j, km
                )
            )
            dst[i, j, k] = src[i, j, k] + time_step_size * flux / h_sq

    @ti.kernel
    def zero_grid_kernel(
        self,
        field: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """Zero the active sub-particle_volume of a vec3 grid field."""
        for i, j, k in ti.ndrange(nx, ny, nz):
            field[i, j, k] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def _download_vec_chunk_kernel(
        self,
        src: ti.template(),
        dst: ti.types.ndarray(),
        offset: ti.i32,
        count: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """Copy active-box nodes [offset, offset+count) into a flat host chunk."""
        for t in range(count):
            idx = offset + t
            i = idx // (ny * nz)
            rem = idx - i * ny * nz
            j = rem // nz
            k = rem - j * nz
            v = src[i, j, k]
            dst[t, 0] = v[0]
            dst[t, 1] = v[1]
            dst[t, 2] = v[2]

    @ti.kernel
    def _upload_vec_chunk_kernel(
        self,
        dst: ti.template(),
        src: ti.types.ndarray(),
        offset: ti.i32,
        count: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """Write a flat host chunk into active-box nodes [offset, offset+count)."""
        for t in range(count):
            idx = offset + t
            i = idx // (ny * nz)
            rem = idx - i * ny * nz
            j = rem // nz
            k = rem - j * nz
            dst[i, j, k] = ti.Vector([src[t, 0], src[t, 1], src[t, 2]])

    @ti.kernel
    def upload_scalar_chunk_kernel(
        self,
        dst: ti.template(),
        src: ti.types.ndarray(),
        offset: ti.i32,
        count: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """Write a flat host chunk into active-box nodes of a scalar grid."""
        for t in range(count):
            idx = offset + t
            i = idx // (ny * nz)
            rem = idx - i * ny * nz
            j = rem // nz
            k = rem - j * nz
            dst[i, j, k] = src[t]

    def grid_transfer_buffer(self, family: str, field: TaichiField, direction: str) -> np.ndarray:
        """Return a fixed staging array unique to a grid field and direction."""
        dtype = np.int32 if family == "scalar" and field.dtype == ti.i32 else np.float32
        return self._grid_staging.get(family, field, direction, dtype)

    def download_active_vec_grid(self, src: TaichiField, nx: int, ny: int, nz: int) -> np.ndarray:
        """Return the active box of a vec3 grid as an (nx, ny, nz, 3) host array."""
        buf = self.grid_transfer_buffer("vector", src, "download")
        total = int(nx) * int(ny) * int(nz)
        out = np.empty((total, 3), dtype=np.float32)
        for offset in range(0, total, self.TRANSFER_CHUNK):
            count = min(self.TRANSFER_CHUNK, total - offset)
            self._download_vec_chunk_kernel(src, buf, offset, count, ny, nz)
            ti.sync()
            out[offset : offset + count] = buf[:count]
        return out.reshape(nx, ny, nz, 3)

    def upload_active_vec_grid(
        self, dst: TaichiField, values: np.ndarray, nx: int, ny: int, nz: int
    ) -> None:
        """Write an (nx, ny, nz, 3) host array into the active box of a vec3 grid."""
        buf = self.grid_transfer_buffer("vector", dst, "upload")
        flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1, 3)
        total = int(nx) * int(ny) * int(nz)
        for offset in range(0, total, self.TRANSFER_CHUNK):
            count = min(self.TRANSFER_CHUNK, total - offset)
            buf[:count] = flat[offset : offset + count]
            self._upload_vec_chunk_kernel(dst, buf, offset, count, ny, nz)
            ti.sync()

    def upload_active_scalar_grid(
        self, dst: TaichiField, values: np.ndarray, nx: int, ny: int, nz: int
    ) -> None:
        """Write an (nx, ny, nz) host array into the active box of a scalar grid."""
        buf = self.grid_transfer_buffer("scalar", dst, "upload")
        flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
        total = int(nx) * int(ny) * int(nz)
        for offset in range(0, total, self.TRANSFER_CHUNK):
            count = min(self.TRANSFER_CHUNK, total - offset)
            buf[:count] = flat[offset : offset + count]
            self.upload_scalar_chunk_kernel(dst, buf, offset, count, ny, nz)
            ti.sync()

    @ti.kernel
    def _apply_body_mask_kernel(
        self,
        grid: ti.template(),
        body_mask: ti.template(),
        nx: ti.i32,
        ny: ti.i32,
        nz: ti.i32,
    ):
        """Zero vorticity in solid cells for active domain extents."""
        for i, j, k in ti.ndrange(nx, ny, nz):
            if body_mask[i, j, k] != 0:
                grid[i, j, k] = ti.Vector.zero(ti.f32, 3)
