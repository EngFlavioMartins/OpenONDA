"""Line and surface field samplers for the FVM solver.

``LineSampler`` probes a line segment into a growing time-aware CSV;
``SurfaceSampler`` probes an axis-aligned plane into per-event ``.vts``
snapshots plus a PVD index. Both reconstruct solver cell fields at fixed probe
points with inverse-distance weighting by default; a linear-exact local affine
reconstruction is available for anisotropic probe layouts.

Sampling is partition-aware: in a partitioned run the owned cell cell_centre and
the required owned fields are gathered to root whenever a sampler is due, so
probes see the *global* field — never just one rank's subdomain.

These samplers are configured through ``FVMSetup(samplers=[...])`` and their
output always lands in ``<case_root>/samples/``.

Examples
--------
>>> line = LineSampler(start=[0, 0, 0], end=[1, 0, 0], n_points=5, file_name="centreline")
>>> plane = SurfaceSampler(
...     point=[0, 0, 0.5], normal=[0, 0, 1],
...     bounds=[0, 1, 0, 1], spacing=0.25, file_name="slice_z0",
... )
>>> from source.solvers.fvm.config.types import FVMSetup
>>> setup = FVMSetup(case_name="case", samplers=(line, plane))
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree  # type: ignore

from .base import SAMPLER_CSV_COLUMNS, Sampler, _register_sampler, append_csv_rows

if TYPE_CHECKING:
    from ..config.scheduling import RunSchedule
    from ..core.solver import FVMSolver

MAX_EXACT_DISTANCE = 1e-12


def _global_owned_view(context) -> tuple[np.ndarray, ...] | None:
    """Return ``(cell_centres, velocity, kinematic_pressure, vorticity)``.

    In serial (or offline post-processing) this is the whole mesh.  In a
    partitioned run the owned slice of each rank is gathered to root, so the
    returned arrays are global regardless of the caller's rank.
    """
    parallel = context.parallel
    if parallel.is_partitioned:
        n_owned = parallel.n_owned
        if n_owned is None:
            raise RuntimeError("Partitioned context has no owned-cell count")
        local = (
            np.asarray(context.geo_data["cell_centre"][:n_owned], dtype=np.float64),
            np.asarray(context.velocity[:n_owned], dtype=np.float64),
            np.asarray(context.kinematic_pressure[:n_owned], dtype=np.float64),
            np.asarray(context._vorticity_field()[:n_owned], dtype=np.float64),
        )
        gathered = parallel.comm.gather(local, root=0)
        if not parallel.is_root:
            return None
        return tuple(np.concatenate([part[k] for part in gathered]) for k in range(4))
    n_cells = context.mesh_data["n_cells"]
    return (
        np.asarray(context.geo_data["cell_centre"], dtype=np.float64),
        np.asarray(context.velocity[:n_cells], dtype=np.float64),
        np.asarray(context.kinematic_pressure[:n_cells], dtype=np.float64),
        np.asarray(context._vorticity_field()[:n_cells], dtype=np.float64),
    )


class _PointProbe(Sampler):
    """Interpolates solver fields at a fixed cloud of probe points."""

    def __init__(
        self,
        points: np.ndarray,
        k: int = 5,
        inverse_distance_power: float = 2.0,
        reconstruction: str = "idw",
        file_name: str | None = None,
        schedule=None,
    ):
        super().__init__(file_name=file_name, schedule=schedule)
        self.points = np.asarray(points, dtype=float)
        self.k = int(k)
        self.power = float(inverse_distance_power)
        self.reconstruction = str(reconstruction).strip().lower()
        if self.reconstruction not in {"idw", "affine"}:
            raise ValueError("reconstruction must be 'idw' or 'affine'")
        self._tree = None
        self._tree_key = None
        self._stencil = None

    def _interpolation_stencil(self, cell_centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return cached neighbour indices and linear-exact MLS weights."""
        cell_centre = np.asarray(cell_centre, dtype=float)
        key = (
            len(cell_centre),
            tuple(cell_centre[0]),
            tuple(cell_centre[-1]),
        )
        if self._tree is None or self._tree_key != key:
            self._tree = cKDTree(cell_centre)
            self._tree_key = key
            self._stencil = None
        if self._stencil is not None:
            return self._stencil

        k = min(self.k, len(cell_centre))
        dists, indices = self._tree.query(self.points, k=k)
        if k == 1:
            dists = dists[:, np.newaxis]
            indices = indices[:, np.newaxis]
        idw = 1.0 / (dists + 1.0e-12) ** self.power
        idw /= np.sum(idw, axis=1, keepdims=True)
        if self.reconstruction == "idw":
            self._stencil = (indices, idw)
            return self._stencil

        offsets = cell_centre[indices] - self.points[:, np.newaxis, :]
        coordinate_scale = np.max(np.abs(offsets), axis=1)
        coordinate_scale[coordinate_scale < 1.0e-12] = 1.0
        design = np.concatenate(
            (
                np.ones((*offsets.shape[:2], 1), dtype=float),
                offsets / coordinate_scale[:, np.newaxis, :],
            ),
            axis=2,
        )
        normal = np.einsum("nki,nk,nkj->nij", design, idw, design)
        inverse = np.linalg.pinv(normal, rcond=1.0e-12)
        intercept_column = inverse[:, :, 0]
        weights = idw * np.einsum("nki,ni->nk", design, intercept_column)

        # Constant fields must be reproduced exactly. A very ill-conditioned
        # local cloud can produce large signed extrapolation weights; retain
        # the bounded IDW result in that rare case.
        weight_sum = np.sum(weights, axis=1, keepdims=True)
        valid = (
            np.all(np.isfinite(weights), axis=1)
            & (np.abs(weight_sum[:, 0]) > 1.0e-12)
            & (np.sum(np.abs(weights), axis=1) < 10.0)
        )
        weights[valid] /= weight_sum[valid]
        weights[~valid] = idw[~valid]

        exact_rows = np.any(dists <= MAX_EXACT_DISTANCE, axis=1)
        if np.any(exact_rows):
            weights[exact_rows] = 0.0
            nearest = np.argmin(dists[exact_rows], axis=1)
            weights[np.flatnonzero(exact_rows), nearest] = 1.0
        self._stencil = (indices, weights)
        return self._stencil

    def _interpolate(self, field, cell_centre) -> np.ndarray:
        indices, weights = self._interpolation_stencil(cell_centre)
        values = np.asarray(field)[indices]
        if np.asarray(field).ndim == 1:
            return np.sum(weights * values, axis=1)
        return np.sum(weights[:, :, np.newaxis] * values, axis=1)

    def sample(self, context) -> dict[str, np.ndarray] | None:
        """Interpolate velocity, vorticity and pressure at the probe points.

        In a partitioned run this is collective (owned cells are gathered to
        root); non-root ranks return ``None``.
        """
        basis = _global_owned_view(context)
        if basis is None:
            return None
        cell_centre, velocity, kinematic_pressure, vorticity = basis
        data = {
            "position_x": self.points[:, 0],
            "position_y": self.points[:, 1],
            "position_z": self.points[:, 2],
            "velocity_x": self._interpolate(velocity[:, 0], cell_centre),
            "velocity_y": self._interpolate(velocity[:, 1], cell_centre),
            "velocity_z": self._interpolate(velocity[:, 2], cell_centre),
            "vorticity_x": self._interpolate(vorticity[:, 0], cell_centre),
            "vorticity_y": self._interpolate(vorticity[:, 1], cell_centre),
            "vorticity_z": self._interpolate(vorticity[:, 2], cell_centre),
            "kinematic_pressure": self._interpolate(kinematic_pressure, cell_centre),
        }
        return data


class LineSampler(_PointProbe):
    """Sample fields at uniformly spaced points along a line segment.

    Every sampling event appends one row per probe point to a growing
    ``<name>.csv`` tagged with ``time``/``step``.

    Examples
    --------
    >>> sampler = LineSampler(
    ...     start=[0, 0, 0], end=[1, 0, 0],
    ...     n_points=5, file_name="centreline",
    ... )
    >>> sampler.name
    'centreline'
    """

    def __init__(
        self,
        start: Sequence[float] | np.ndarray,
        end: Sequence[float] | np.ndarray,
        n_points: int | None = None,
        spacing: float | None = None,
        k: int = 5,
        inverse_distance_power: float = 2.0,
        reconstruction: str = "idw",
        file_name: str | None = None,
        schedule: RunSchedule | None = None,
    ) -> None:
        """Initialize the line sampler.

        Args:
            start: Start point of the line, array-like of shape (3,).
            end: End point of the line, array-like of shape (3,).
            n_points: Number of uniformly spaced sample points.
            spacing: Point spacing; alternative to ``n_points``.
            k: Number of nearest neighbours used for interpolation.
            inverse_distance_power: Exponent used for inverse-distance weighting.
            reconstruction: ``"idw"`` or linear-exact ``"affine"``.
            file_name: Base name for the output CSV.
            schedule: Optional :class:`~source.solvers.fvm.config.RunSchedule`.
        """
        if (n_points is None) == (spacing is None):
            raise ValueError("Provide exactly one of n_points or spacing")
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        if n_points is None:
            assert spacing is not None
            n_points = int(np.linalg.norm(end - start) / spacing) + 1
        t = np.linspace(0.0, 1.0, n_points)
        super().__init__(
            start + np.outer(t, end - start),
            k=k,
            inverse_distance_power=inverse_distance_power,
            reconstruction=reconstruction,
            file_name=file_name,
            schedule=schedule,
        )
        self.start = start
        self.end = end
        self.n_points = n_points

    def config_dict(self) -> dict:
        spec = super().config_dict()
        spec.update(
            {
                "start": self.start.tolist(),
                "end": self.end.tolist(),
                "n_points": self.n_points,
                "k": self.k,
                "inverse_distance_power": self.power,
                "reconstruction": self.reconstruction,
            }
        )
        return spec

    def write_csv(
        self,
        context: FVMSolver,
        samples_dir: str,
    ) -> dict[str, np.ndarray] | None:
        """Append the current step's samples to ``<samples_dir>/<name>.csv``."""
        data = self.sample(context)
        if data is None:
            return None
        rows = [
            [context.time, context.step, *[data[name][i] for name in SAMPLER_CSV_COLUMNS]]
            for i in range(len(data["position_x"]))
        ]
        append_csv_rows(
            f"{samples_dir}/{self.name}.csv",
            ["time", "step", *SAMPLER_CSV_COLUMNS],
            rows,
        )
        return data


class SurfaceSampler(_PointProbe):
    """Sample fields on an axis-aligned planar grid.

    Writes one ``.vts`` structured grid per sampling event and a ``<name>.pvd``
    index, using the same point-array names as the VPM ``SurfaceSampler``.
    Cadence is owned entirely by :class:`~source.solvers.fvm.config.RunSchedule` — there
    is no separate ``stride`` counter, so live and offline runs select the
    same physical states.

    Examples
    --------
    >>> sampler = SurfaceSampler(
    ...     point=[0, 0, 0.5], normal=[0, 0, 1],
    ...     bounds=[0, 1, 0, 1], spacing=0.25,
    ...     file_name="slice_z0",
    ...     schedule=RunSchedule(every_n_steps=2),
    ... )
    """

    sampler_kind = "SurfaceSampler"

    def __init__(
        self,
        point: Sequence[float] | np.ndarray,
        normal: Sequence[float] | np.ndarray,
        bounds: Sequence[float] | np.ndarray,
        spacing: float,
        k: int = 5,
        inverse_distance_power: float = 2.0,
        reconstruction: str = "idw",
        file_name: str | None = None,
        schedule: RunSchedule | None = None,
        body_bounds: Sequence[float] | np.ndarray | None = None,
        body_geometry: str = "box",
    ) -> None:
        """Initialize the surface sampler.

        Args:
            point: A point on the plane [x, y, z].
            normal: Plane normal; only axis-aligned normals are supported.
            bounds: Grid bounds [min1, max1, min2, max2] for the two in-plane
                axes, ordered as for the VPM sampler (z-plane -> x,y;
                y-plane -> x,z; x-plane -> y,z).
            spacing: Grid point spacing.
            k: Number of nearest neighbours used for interpolation.
            inverse_distance_power: Exponent used for inverse-distance weighting.
            reconstruction: ``"idw"`` or linear-exact ``"affine"``.
            file_name: Base name for the output files.
            schedule: Optional :class:`~source.solvers.fvm.config.RunSchedule`.
            body_bounds: Optional axis-aligned solid bounds ``(xmin, xmax,
                ymin, ymax, zmin, zmax)``.  Probe points geometrically inside
                the body are masked in the output (``vtkValidPointMask`` zero,
                NaN field values) so the slice never shows a flow field inside
                the solid.
            body_geometry: ``"box"`` masks the complete bounding box;
                ``"cylinder_z"`` interprets equal x/y half-widths as a circular
                cylinder radius and retains the supplied z bounds.
        """
        point = np.asarray(point, dtype=float)
        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        bounds = np.asarray(bounds, dtype=float)

        axis = int(np.argmax(np.abs(normal)))
        offset = point[axis]
        c1 = np.arange(bounds[0], bounds[1] + spacing / 2, spacing)
        c2 = np.arange(bounds[2], bounds[3] + spacing / 2, spacing)
        C1, C2 = np.meshgrid(c1, c2, indexing="ij")

        points = np.zeros((C1.size, 3), dtype=float)
        if axis == 2:
            points[:, 0], points[:, 1], points[:, 2] = C1.ravel(), C2.ravel(), offset
        elif axis == 1:
            points[:, 0], points[:, 1], points[:, 2] = C1.ravel(), offset, C2.ravel()
        else:
            points[:, 0], points[:, 1], points[:, 2] = offset, C1.ravel(), C2.ravel()

        super().__init__(
            points,
            k=k,
            inverse_distance_power=inverse_distance_power,
            reconstruction=reconstruction,
            file_name=file_name,
            schedule=schedule,
        )
        self.point = point
        self.normal = normal
        self.bounds = bounds
        self.spacing = float(spacing)
        self.grid_shape = C1.shape
        self._pvd_entries: list[tuple[float, str]] = []
        body_geometry = str(body_geometry).strip().lower()
        if body_geometry not in {"box", "cylinder_z"}:
            raise ValueError("body_geometry must be 'box' or 'cylinder_z'")
        self._body_geometry = body_geometry
        if body_bounds is not None:
            body_bounds = np.asarray(body_bounds, dtype=float)
            if body_bounds.shape != (6,):
                raise ValueError("body_bounds must contain six coordinates")
            inside_z = (points[:, 2] > body_bounds[4]) & (points[:, 2] < body_bounds[5])
            if body_geometry == "cylinder_z":
                centre_x = 0.5 * (body_bounds[0] + body_bounds[1])
                centre_y = 0.5 * (body_bounds[2] + body_bounds[3])
                radius_x = 0.5 * (body_bounds[1] - body_bounds[0])
                radius_y = 0.5 * (body_bounds[3] - body_bounds[2])
                if not np.isclose(radius_x, radius_y, rtol=1.0e-12, atol=1.0e-12):
                    raise ValueError("cylinder_z body bounds must have equal x/y radii")
                inside = (
                    (points[:, 0] - centre_x) ** 2 + (points[:, 1] - centre_y) ** 2 < radius_x**2
                ) & inside_z
            else:
                inside = (
                    (points[:, 0] > body_bounds[0])
                    & (points[:, 0] < body_bounds[1])
                    & (points[:, 1] > body_bounds[2])
                    & (points[:, 1] < body_bounds[3])
                    & inside_z
                )
            self._body_bounds = body_bounds
            self._inside_mask = inside
        else:
            self._body_bounds = None
            self._inside_mask = None

    def config_dict(self) -> dict:
        spec = super().config_dict()
        spec.update(
            {
                "point": self.point.tolist(),
                "normal": self.normal.tolist(),
                "bounds": self.bounds.tolist(),
                "spacing": self.spacing,
                "k": self.k,
                "inverse_distance_power": self.power,
                "reconstruction": self.reconstruction,
                "body_bounds": self._body_bounds.tolist()
                if self._body_bounds is not None
                else None,
                "body_geometry": self._body_geometry,
            }
        )
        return spec

    def save_vts(
        self,
        context: FVMSolver,
        filepath: str,
    ) -> dict[str, np.ndarray] | None:
        """Write one structured-grid snapshot of the sampled plane."""
        import pyvista as pv

        data = self.sample(context)
        if data is None:
            return None
        ni, nj = self.grid_shape
        shape = (ni, nj, 1)

        def _grid(values):
            return np.asarray(values, dtype=np.float32).reshape(shape)

        def _field(values):
            return np.asarray(values, dtype=np.float32).reshape(self.grid_shape).ravel(order="F")

        grid = pv.StructuredGrid(
            _grid(data["position_x"]), _grid(data["position_y"]), _grid(data["position_z"])
        )
        velocity = np.column_stack(
            [_field(data["velocity_x"]), _field(data["velocity_y"]), _field(data["velocity_z"])]
        )
        vorticity = np.column_stack(
            [
                _field(data["vorticity_x"]),
                _field(data["vorticity_y"]),
                _field(data["vorticity_z"]),
            ]
        )
        pressure = _field(data["kinematic_pressure"])
        valid_mask = np.ones(len(self.points), dtype=np.uint8)
        if self._inside_mask is not None:
            # Probe points inside the body have no physical fluid value; mark
            # them invalid and blank their fields so the slice never appears to
            # contain a flow inside the solid.  Point fields have already been
            # converted from the sampler's C-order plane to VTK's Fortran
            # ordering, so the geometric mask must undergo the same mapping.
            inside = _field(self._inside_mask).astype(bool)
            valid_mask[inside] = 0
            velocity[inside] = np.nan
            vorticity[inside] = np.nan
            pressure[inside] = np.nan
        grid.point_data["velocity"] = velocity
        grid.point_data["velocity_magnitude"] = np.linalg.norm(velocity, axis=1)
        grid.point_data["vorticity"] = vorticity
        grid.point_data["vorticity_magnitude"] = np.linalg.norm(vorticity, axis=1)
        grid.point_data["kinematic_pressure"] = pressure
        grid.point_data["vtkValidPointMask"] = valid_mask
        grid.field_data["surface_ordering"] = np.asarray([1], dtype=np.uint8)
        grid.save(str(filepath), binary=True)
        return data


_register_sampler(LineSampler)
_register_sampler(SurfaceSampler)
