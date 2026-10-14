"""Line and surface field samplers of the FVM solver.

:class:`LineSampler` appends the fields at points on a line to
``samples/<name>.csv``; :class:`SurfaceSampler` writes the fields on an
axis-aligned plane to ``samples/<name>_{step:06d}.vts`` frames listed in
``samples/<name>.pvd``. Both interpolate the cell-centred fields with inverse
distance weights or a linear-exact local affine reconstruction.

In a partitioned run the owned cells are gathered to rank 0, so a sampler
sees the global field.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree  # type: ignore

from openonda.core.file_output import append_csv_rows
from openonda.core.sampling import FIELD_SAMPLE_COLUMNS, Sampler, line_points, surface_points
from openonda.core.schedules import EverySteps, Schedule

if TYPE_CHECKING:
    from ..core.solver import FVMSolver

MAX_EXACT_DISTANCE = 1e-12
LINE_SAMPLE_COLUMNS = (*FIELD_SAMPLE_COLUMNS, "kinematic_pressure")


def _global_owned_view(context: FVMSolver) -> tuple[np.ndarray, ...] | None:
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
    """Interpolate the cell-centred FVM fields at fixed probe points.

    ``idw`` uses positive inverse-distance weights; ``affine`` a locally
    linear least-squares reconstruction that falls back to the inverse
    distance weights on ill-conditioned stencils. The stencil is cached.
    """

    def __init__(
        self,
        points: np.ndarray,
        n_neighbours: int = 5,
        inverse_distance_power: float = 2.0,
        reconstruction: str = "idw",
        *,
        file_name: str | None = None,
        schedule: Schedule = EverySteps(1),
        initial: bool = True,
    ) -> None:
        super().__init__(file_name=file_name, schedule=schedule, initial=initial)
        self.points = np.asarray(points, dtype=float)
        self.n_neighbours = int(n_neighbours)
        self.inverse_distance_power = float(inverse_distance_power)
        if reconstruction not in {"idw", "affine"}:
            raise ValueError("reconstruction must be 'idw' or 'affine'")
        self.reconstruction = reconstruction
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

        k = min(self.n_neighbours, len(cell_centre))
        dists, indices = self._tree.query(self.points, k=k)
        if k == 1:
            dists = dists[:, np.newaxis]
            indices = indices[:, np.newaxis]
        idw = 1.0 / (dists + 1.0e-12) ** self.inverse_distance_power
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

    def _interpolate(self, field: np.ndarray, cell_centre: np.ndarray) -> np.ndarray:
        """Interpolate one scalar or vector field using the cached stencil."""
        indices, weights = self._interpolation_stencil(cell_centre)
        values = np.asarray(field)[indices]
        if np.asarray(field).ndim == 1:
            return np.sum(weights * values, axis=1)
        return np.sum(weights[:, :, np.newaxis] * values, axis=1)

    def sample(self, context: FVMSolver) -> dict[str, np.ndarray] | None:
        """Interpolate velocity, vorticity and pressure at the probe points.

        In a partitioned run this is collective (owned cells are gathered to
        root); non-root ranks return ``None``.

        Parameters
        ----------
        context : FVMSolver
            Solver at the accepted state.

        Returns
        -------
        dict[str, numpy.ndarray] or None
            Standard columns on root, or ``None`` on non-root MPI ranks.
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


def interpolate_cell_field(
    field: np.ndarray,
    cell_centre: np.ndarray,
    points: np.ndarray,
    *,
    n_neighbours: int = 5,
    inverse_distance_power: float = 2.0,
    reconstruction: str = "idw",
) -> np.ndarray:
    """Interpolate a cell-centred field at points with the sampler stencil.

    Parameters
    ----------
    field : numpy.ndarray
        Cell values, shape ``(n_cells,)`` or ``(n_cells, n_components)``.
    cell_centre : numpy.ndarray
        Cell-centre coordinates in m, shape ``(n_cells, 3)``.
    points : numpy.ndarray
        Probe points in m, shape ``(n_points, 3)``.
    n_neighbours, inverse_distance_power, reconstruction
        Stencil of :class:`LineSampler` and :class:`SurfaceSampler`:
        ``"idw"`` inverse-distance weights or ``"affine"`` linear-exact
        local reconstruction.

    Returns
    -------
    numpy.ndarray
        Interpolated values, shape ``(n_points,)`` or
        ``(n_points, n_components)``.
    """
    probe = _PointProbe(points, n_neighbours, inverse_distance_power, reconstruction)
    return probe._interpolate(field, cell_centre)


class LineSampler(_PointProbe):
    """Sample the fields at equally spaced points on a line segment.

    Every sampled state appends one row per point to ``samples/<name>.csv``
    with the columns :data:`LINE_SAMPLE_COLUMNS`.

    Parameters
    ----------
    start, end : sequence of float
        End points in m, shape ``(3,)``; both are sample points.
    n_points : int or None
        Number of points. Give exactly one of ``n_points`` and ``spacing``.
    spacing : float or None
        Largest point spacing in m.
    n_neighbours : int, default=5
        Nearest cell centres of each interpolation stencil.
    inverse_distance_power : float, default=2.0
        Exponent of the inverse distance weights.
    reconstruction : {'idw', 'affine'}, default='idw'
        Interpolation model.
    file_name : str or None, default=None
        Output file stem; ``None`` uses ``line``.
    schedule : EverySteps, EveryTime or FinalOnly, default=EverySteps(1)
        Accepted states that are sampled.
    initial : bool, default=True
        Sample the initial state.
    """

    def __init__(
        self,
        start: Sequence[float] | np.ndarray,
        end: Sequence[float] | np.ndarray,
        n_points: int | None = None,
        spacing: float | None = None,
        n_neighbours: int = 5,
        inverse_distance_power: float = 2.0,
        reconstruction: str = "idw",
        *,
        file_name: str | None = None,
        schedule: Schedule = EverySteps(1),
        initial: bool = True,
    ) -> None:
        super().__init__(
            line_points(start, end, n_points=n_points, spacing=spacing),
            n_neighbours,
            inverse_distance_power,
            reconstruction,
            file_name=file_name,
            schedule=schedule,
            initial=initial,
        )
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.n_points = None if n_points is None else int(n_points)
        self.spacing = None if spacing is None else float(spacing)

    def config_dict(self) -> dict:
        """Return the JSON-compatible constructor arguments."""
        return {
            **super().config_dict(),
            "start": self.start.tolist(),
            "end": self.end.tolist(),
            "n_points": self.n_points,
            "spacing": self.spacing,
            "n_neighbours": self.n_neighbours,
            "inverse_distance_power": self.inverse_distance_power,
            "reconstruction": self.reconstruction,
        }

    def write_csv(self, context: FVMSolver, samples_dir: str | Path) -> None:
        """Append the sample of the accepted state to ``<samples_dir>/<name>.csv``.

        Collective in a partitioned run; rank 0 writes.
        """
        data = self.sample(context)
        if data is None:
            return
        n_points = len(self.points)
        columns = {"time": np.full(n_points, float(context.time)), **data}
        rows = zip(*(columns[name] for name in LINE_SAMPLE_COLUMNS), strict=True)
        append_csv_rows(Path(samples_dir) / f"{self.name}.csv", LINE_SAMPLE_COLUMNS, rows)


class SurfaceSampler(_PointProbe):
    """Sample the fields on a uniform grid on an axis-aligned plane.

    Every sampled state writes the frame ``samples/<name>_{step:06d}.vts``,
    listed in ``samples/<name>.pvd``, with the point data ``velocity``,
    ``vorticity``, ``kinematic_pressure`` and ``vtkValidPointMask`` and the
    field data ``time``.

    Parameters
    ----------
    point : sequence of float
        A point on the plane in m.
    normal : sequence of float
        Plane normal; the plane is normal to the axis of its largest
        component.
    bounds : sequence of float
        ``(lower_1, upper_1, lower_2, upper_2)`` in m on the two in-plane
        axes in coordinate order; both bounds are grid points.
    spacing : float
        Largest grid spacing in m.
    n_neighbours : int, default=5
        Nearest cell centres of each interpolation stencil.
    inverse_distance_power : float, default=2.0
        Exponent of the inverse distance weights.
    reconstruction : {'idw', 'affine'}, default='idw'
        Interpolation model.
    body_bounds : sequence of float or None, default=None
        Solid box ``(xmin, xmax, ymin, ymax, zmin, zmax)`` in m whose interior
        points are marked invalid and blanked.
    file_name : str or None, default=None
        Output file stem; ``None`` uses ``surface``.
    schedule : EverySteps, EveryTime or FinalOnly, default=EverySteps(1)
        Accepted states that are sampled.
    initial : bool, default=True
        Sample the initial state.
    """

    def __init__(
        self,
        point: Sequence[float] | np.ndarray,
        normal: Sequence[float] | np.ndarray,
        bounds: Sequence[float] | np.ndarray,
        spacing: float,
        n_neighbours: int = 5,
        inverse_distance_power: float = 2.0,
        reconstruction: str = "idw",
        body_bounds: Sequence[float] | np.ndarray | None = None,
        *,
        file_name: str | None = None,
        schedule: Schedule = EverySteps(1),
        initial: bool = True,
    ) -> None:
        points, grid_shape, _ = surface_points(point, normal, bounds, spacing)
        super().__init__(
            points,
            n_neighbours,
            inverse_distance_power,
            reconstruction,
            file_name=file_name,
            schedule=schedule,
            initial=initial,
        )
        self.point = np.asarray(point, dtype=float)
        self.normal = np.asarray(normal, dtype=float)
        self.bounds = np.asarray(bounds, dtype=float)
        self.spacing = float(spacing)
        self.grid_shape = grid_shape
        self.body_bounds = None
        self._inside_mask = None
        if body_bounds is not None:
            self.body_bounds = np.asarray(body_bounds, dtype=float)
            if self.body_bounds.shape != (6,):
                raise ValueError("body_bounds must contain six coordinates")
            self._inside_mask = np.all(
                (points > self.body_bounds[::2]) & (points < self.body_bounds[1::2]), axis=1
            )

    def config_dict(self) -> dict:
        """Return the JSON-compatible constructor arguments."""
        return {
            **super().config_dict(),
            "point": self.point.tolist(),
            "normal": self.normal.tolist(),
            "bounds": self.bounds.tolist(),
            "spacing": self.spacing,
            "n_neighbours": self.n_neighbours,
            "inverse_distance_power": self.inverse_distance_power,
            "reconstruction": self.reconstruction,
            "body_bounds": None if self.body_bounds is None else self.body_bounds.tolist(),
        }

    def write_vtk(self, context: FVMSolver, path: str | Path) -> bool:
        """Write the sample of the accepted state as a structured-grid frame.

        Collective in a partitioned run; rank 0 writes and returns ``True``.
        The values use the precision and compression of the case output.
        """
        import pyvista as pv

        from openonda.core.vtk_output import write_vtk_dataset
        from openonda.core.write_precision import cast_for_write

        data = self.sample(context)
        if data is None:
            return False
        output = context.case.output
        n_first, n_second = self.grid_shape

        def grid_coordinates(values: np.ndarray) -> np.ndarray:
            return np.asarray(values, dtype=np.float64).reshape((n_first, n_second, 1))

        def point_values(values: np.ndarray) -> np.ndarray:
            # Sampler points are in C order; VTK structured points in Fortran order.
            return np.asarray(values).reshape(self.grid_shape).ravel(order="F")

        grid = pv.StructuredGrid(
            grid_coordinates(data["position_x"]),
            grid_coordinates(data["position_y"]),
            grid_coordinates(data["position_z"]),
        )
        velocity = np.column_stack([point_values(data[f"velocity_{axis}"]) for axis in "xyz"])
        vorticity = np.column_stack([point_values(data[f"vorticity_{axis}"]) for axis in "xyz"])
        pressure = point_values(data["kinematic_pressure"]).astype(np.float64)
        valid = np.ones(len(self.points), dtype=np.uint8)
        if self._inside_mask is not None:
            inside = point_values(self._inside_mask).astype(bool)
            valid[inside] = 0
            velocity[inside] = np.nan
            vorticity[inside] = np.nan
            pressure[inside] = np.nan
        grid.point_data["velocity"] = cast_for_write(velocity, output.precision)
        grid.point_data["vorticity"] = cast_for_write(vorticity, output.precision)
        grid.point_data["kinematic_pressure"] = cast_for_write(pressure, output.precision)
        grid.point_data["vtkValidPointMask"] = valid
        grid.field_data["time"] = np.asarray([context.time], dtype=np.float64)
        write_vtk_dataset(grid, path, compression=output.compression)
        return True
