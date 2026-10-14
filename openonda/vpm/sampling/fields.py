"""Velocity and vorticity of the VPM flow sampled on lines and planes.

Both samplers evaluate the complete velocity (particles, freestream, sources
and an attached vortex lattice) and its gradient at fixed points. Vorticity
is the curl of the sampled velocity gradient. The CSV columns start with the
shared :data:`~openonda.core.sampling.FIELD_SAMPLE_COLUMNS`, so a VPM sample
compares row by row with an FVM sample.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from openonda.core.file_output import append_csv_rows
from openonda.core.sampling import FIELD_SAMPLE_COLUMNS, Sampler, line_points, surface_points
from openonda.core.schedules import EverySteps, Schedule
from openonda.core.vtk_output import write_vtk_dataset
from openonda.core.write_precision import cast_for_write

if TYPE_CHECKING:
    from ..core.solver import VPMSolver
    from .manager import SamplingContext

__all__ = ["DERIVATIVE_SAMPLE_COLUMNS", "LineSampler", "SurfaceSampler", "field_sample_columns"]

DERIVATIVE_SAMPLE_COLUMNS = (
    "strain_rate_xx",
    "strain_rate_xy",
    "strain_rate_xz",
    "strain_rate_yy",
    "strain_rate_yz",
    "strain_rate_zz",
    "velocity_gradient_xx",
    "velocity_gradient_xy",
    "velocity_gradient_xz",
    "velocity_gradient_yx",
    "velocity_gradient_yy",
    "velocity_gradient_yz",
    "velocity_gradient_zx",
    "velocity_gradient_zy",
    "velocity_gradient_zz",
)
"""VPM columns after the shared field-sample columns: strain rate and
velocity gradient ``J[i, j] = du_i/dx_j``, both in 1/s."""


def field_sample_columns(include_derivatives: bool) -> tuple[str, ...]:
    """Return the CSV columns of a VPM field sample."""
    if include_derivatives:
        return (*FIELD_SAMPLE_COLUMNS, *DERIVATIVE_SAMPLE_COLUMNS)
    return FIELD_SAMPLE_COLUMNS


def _sample_fields(
    solver: VPMSolver, points: np.ndarray, spacing: float, include_derivatives: bool
) -> dict[str, np.ndarray]:
    """Evaluate velocity, vorticity and velocity gradient at ``points``."""
    velocity, gradient = solver.compute_velocity_and_gradient_at_points(
        points, particle_spacing=spacing
    )
    velocity = np.asarray(velocity).reshape(-1, 3)
    gradient = np.asarray(gradient).reshape(-1, 3, 3)
    vorticity = np.column_stack(
        (
            gradient[:, 2, 1] - gradient[:, 1, 2],
            gradient[:, 0, 2] - gradient[:, 2, 0],
            gradient[:, 1, 0] - gradient[:, 0, 1],
        )
    )
    data = {
        "position_x": points[:, 0],
        "position_y": points[:, 1],
        "position_z": points[:, 2],
        "velocity_x": velocity[:, 0],
        "velocity_y": velocity[:, 1],
        "velocity_z": velocity[:, 2],
        "vorticity_x": vorticity[:, 0],
        "vorticity_y": vorticity[:, 1],
        "vorticity_z": vorticity[:, 2],
    }
    if include_derivatives:
        flat = gradient.reshape(-1, 9)
        data.update(
            {
                "strain_rate_xx": flat[:, 0],
                "strain_rate_xy": 0.5 * (flat[:, 1] + flat[:, 3]),
                "strain_rate_xz": 0.5 * (flat[:, 2] + flat[:, 6]),
                "strain_rate_yy": flat[:, 4],
                "strain_rate_yz": 0.5 * (flat[:, 5] + flat[:, 7]),
                "strain_rate_zz": flat[:, 8],
            }
        )
        for index, component in enumerate(("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")):
            data[f"velocity_gradient_{component}"] = flat[:, index]
    return data


def _csv_rows(data: dict[str, np.ndarray], columns: tuple[str, ...], time: float) -> list[list]:
    """Return one CSV row per sample point with the time in the first column."""
    values = [np.asarray(data[name]) for name in columns[1:]]
    return [[time, *(float(column[index]) for column in values)] for index in range(len(values[0]))]


class SurfaceSampler(Sampler):
    """Sample the flow on a uniform grid on an axis-aligned plane.

    Each accepted state is written as a VTK structured grid
    ``<name>_{step:06d}.vts`` listed in ``<name>.pvd``.

    Parameters
    ----------
    point : array-like
        A point on the plane in m, shape ``(3,)``.
    normal : array-like
        Plane normal, shape ``(3,)``; the plane is normal to the coordinate
        axis of its largest component.
    bounds : array-like
        ``(lower_1, upper_1, lower_2, upper_2)`` in m on the in-plane axes:
        (y, z) for an x-normal, (x, z) for a y-normal and (x, y) for a
        z-normal plane.
    spacing : float
        Largest grid spacing in m; both bounds are grid points.
    include_derivatives : bool, default=True
        Also write the velocity gradient.
    file_name, schedule, initial
        Output arguments of :class:`~openonda.core.sampling.Sampler`.
    """

    vtk_extension = ".vts"

    def __init__(
        self,
        point: np.ndarray | list,
        normal: np.ndarray | list,
        bounds: np.ndarray | list,
        spacing: float,
        include_derivatives: bool = True,
        *,
        file_name: str | None = None,
        schedule: Schedule = EverySteps(1),
        initial: bool = True,
    ) -> None:
        super().__init__(file_name=file_name, schedule=schedule, initial=initial)
        self.grid_points, self._grid_shape, axes = surface_points(point, normal, bounds, spacing)
        self.point = np.asarray(point, dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)
        self.normal = normal / np.linalg.norm(normal)
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.spacing = float(spacing)
        self.include_derivatives = bool(include_derivatives)
        self._axis1_name, self._axis2_name = axes

    def config_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible constructor arguments."""
        return {
            "point": self.point.tolist(),
            "normal": self.normal.tolist(),
            "bounds": self.bounds.tolist(),
            "spacing": self.spacing,
            "include_derivatives": self.include_derivatives,
            **super().config_dict(),
        }

    def _vtk_points(self) -> np.ndarray:
        shape = self._grid_shape
        return np.column_stack(
            [self.grid_points[:, i].reshape(shape).ravel(order="F") for i in range(3)]
        )

    def _grid_metadata(self) -> dict[str, object]:
        points = np.ascontiguousarray(self._vtk_points(), dtype="<f8")
        return {
            "configuration": {
                "point": self.point.tolist(),
                "normal": self.normal.tolist(),
                "bounds": self.bounds.tolist(),
                "spacing": self.spacing,
            },
            "dimensions": [*self._grid_shape, 1],
            "coordinates_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
        }

    def validate_existing_vtk(self, path: Path) -> None:
        """Reject continuing a frame series written on a different grid."""
        import pyvista as pv

        previous = pv.read(path)
        key = "openonda_sampling_grid"
        if (
            key not in previous.field_data
            or json.loads(str(np.asarray(previous.field_data[key])[0])) != self._grid_metadata()
        ):
            raise ValueError(
                f"sampling grid differs from existing frame {path}; "
                "use a new case variant or the original sampling geometry"
            )

    def sample(self, solver: VPMSolver) -> dict[str, np.ndarray]:
        """Return the sampled columns at every grid point (SI units)."""
        return _sample_fields(solver, self.grid_points, self.spacing, self.include_derivatives)

    def write_csv(self, solver: VPMSolver, path: str | Path, time: float) -> Path:
        """Write one sample of the current state to a CSV file.

        Parameters
        ----------
        solver : VPMSolver
            Solver at the sampled state.
        path : str or pathlib.Path
            Destination file; it is replaced.
        time : float
            Solution time in s, written to the ``time`` column.
        """
        path = Path(path)
        path.unlink(missing_ok=True)
        columns = field_sample_columns(self.include_derivatives)
        return append_csv_rows(path, columns, _csv_rows(self.sample(solver), columns, time))

    def write_vtk(self, solver: VPMSolver, path: str | Path, time: float) -> Path:
        """Write one sample of the current state as a VTK structured grid.

        Parameters
        ----------
        solver : VPMSolver
            Solver at the sampled state.
        path : str or pathlib.Path
            Destination ``.vts`` file.
        time : float
            Solution time in s, stored in the ``time`` and ``TimeValue``
            field data.
        """
        import pyvista as pv

        data = self.sample(solver)
        shape = self._grid_shape

        def grid_order(name: str) -> np.ndarray:
            return np.asarray(data[name]).reshape(shape).ravel(order="F")

        x, y, z = (
            np.asarray(data[f"position_{axis}"]).reshape(shape)[:, :, np.newaxis] for axis in "xyz"
        )
        grid = pv.StructuredGrid(x, y, z)
        grid.field_data["openonda_sampling_grid"] = np.array(
            [json.dumps(self._grid_metadata(), sort_keys=True, separators=(",", ":"))]
        )
        precision = solver.write_precision
        for name in ("velocity", "vorticity"):
            vector = np.column_stack([grid_order(f"{name}_{axis}") for axis in "xyz"])
            grid.point_data[name] = cast_for_write(vector, precision)
        if self.include_derivatives:
            components = ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")
            tensor = np.column_stack(
                [grid_order(f"velocity_gradient_{component}") for component in components]
            )
            grid.point_data["velocity_gradient"] = cast_for_write(tensor, precision)
        for name in ("time", "TimeValue"):
            grid.field_data[name] = np.array([float(time)], dtype=np.float64)
        return write_vtk_dataset(grid, Path(path), compression=solver.case.output.compression)


class LineSampler(Sampler):
    """Sample the flow at equally spaced points on a line.

    Every accepted state appends one row per point to ``<name>.csv``.

    Parameters
    ----------
    start, end : array-like
        End points of the line in m, shape ``(3,)``; both are sample points.
    n_points : int or None, default=None
        Number of points, at least two.
    spacing : float or None, default=None
        Largest point spacing in m. Give exactly one of ``n_points`` and
        ``spacing``.
    include_derivatives : bool, default=True
        Also write the strain rate and velocity gradient columns.
    file_name, schedule, initial
        Output arguments of :class:`~openonda.core.sampling.Sampler`.
    """

    def __init__(
        self,
        start: np.ndarray | list,
        end: np.ndarray | list,
        n_points: int | None = None,
        spacing: float | None = None,
        include_derivatives: bool = True,
        *,
        file_name: str | None = None,
        schedule: Schedule = EverySteps(1),
        initial: bool = True,
    ) -> None:
        super().__init__(file_name=file_name, schedule=schedule, initial=initial)
        self.line_points = line_points(start, end, n_points=n_points, spacing=spacing)
        self.start = np.asarray(start, dtype=np.float64)
        self.end = np.asarray(end, dtype=np.float64)
        self.n_points = len(self.line_points)
        self.spacing = spacing
        self.include_derivatives = bool(include_derivatives)

    def config_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible constructor arguments."""
        return {
            "start": self.start.tolist(),
            "end": self.end.tolist(),
            "n_points": None if self.spacing is not None else self.n_points,
            "spacing": self.spacing,
            "include_derivatives": self.include_derivatives,
            **super().config_dict(),
        }

    def sample(self, solver: VPMSolver) -> dict[str, np.ndarray]:
        """Return the sampled columns at every line point (SI units)."""
        point_spacing = float(np.linalg.norm(self.end - self.start)) / (self.n_points - 1)
        return _sample_fields(solver, self.line_points, point_spacing, self.include_derivatives)

    def write_csv(self, solver: VPMSolver, path: str | Path, time: float) -> Path:
        """Write one sample of the current state to a CSV file that is replaced."""
        path = Path(path)
        path.unlink(missing_ok=True)
        columns = field_sample_columns(self.include_derivatives)
        return append_csv_rows(path, columns, _csv_rows(self.sample(solver), columns, time))

    def write(self, context: SamplingContext) -> None:
        """Append the current state to the time series ``<name>.csv``."""
        path = context.output_directory / f"{self.name}.csv"
        if not context.continuing_output:
            path.unlink(missing_ok=True)
        columns = field_sample_columns(self.include_derivatives)
        append_csv_rows(
            path, columns, _csv_rows(self.sample(context.solver), columns, context.time)
        )
