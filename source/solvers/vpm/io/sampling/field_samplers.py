"""
Field Samplers module for VPM solver.
==================
Provides SurfaceSampler and LineSampler classes for computing VPM-induced
fields (velocity, vorticity, etc.) at fixed grid points.

These samplers create regular grids or line points and use the solver's
compute_*_at() methods to evaluate the induced fields from all particles.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from source.vtk_output import write_vtk_dataset
from source.write_precision import DEFAULT_WRITE_PRECISION, cast_for_write

from .schedule import OutputSchedule

if TYPE_CHECKING:
    from ...core.solver import VPMSolver

# Canonical CSV column order for SurfaceSampler / LineSampler output.  Single
# source of truth: the header row and every data row are built from this list,
# so the written header always matches the data (no magic column indices on the
# reader side — see ``_read_sampler_csv`` in the tutorials' post-processing).
SAMPLER_BASE_CSV_COLUMNS = [
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "vorticity_x",
    "vorticity_y",
    "vorticity_z",
]

SAMPLER_DERIVATIVE_CSV_COLUMNS = [
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
]

SAMPLER_CSV_COLUMNS = [*SAMPLER_BASE_CSV_COLUMNS, *SAMPLER_DERIVATIVE_CSV_COLUMNS]


def sampler_csv_columns(sampler) -> list[str]:
    """Return the persisted CSV schema for one sampler."""
    if getattr(sampler, "include_derivatives", True):
        return SAMPLER_CSV_COLUMNS
    return SAMPLER_BASE_CSV_COLUMNS


def _validated_sample_data(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Require samplers to construct the canonical output schema directly."""
    allowed = {*SAMPLER_CSV_COLUMNS, "line_parameter"}
    unexpected = sorted(set(data) - allowed)
    if unexpected:
        raise ValueError("Sampler produced non-canonical fields: " + ", ".join(unexpected))
    return data


def _sample_velocity_vorticity_gradient(solver, points, spacing):
    """Evaluate velocity and define vorticity by the compatible velocity curl."""
    if hasattr(solver, "compute_velocity_and_gradient_at_points"):
        velocity, gradient = solver.compute_velocity_and_gradient_at_points(
            points, particle_spacing=spacing
        )
    else:
        velocity = solver.compute_velocity_at_points(points, include_freestream=True)
        gradient = solver.compute_velocity_gradient_at_points(points)
    gradient = np.asarray(gradient).reshape(-1, 3, 3)
    vorticity = np.column_stack(
        (
            gradient[:, 2, 1] - gradient[:, 1, 2],
            gradient[:, 0, 2] - gradient[:, 2, 0],
            gradient[:, 1, 0] - gradient[:, 0, 1],
        )
    )
    return np.asarray(velocity).reshape(-1, 3), vorticity, gradient


def resolve_samples_dir(case_directory, sample_directory: str | None = None) -> Path:
    """Return the mandatory ``<case>/samples`` directory."""
    samples_dir = Path(case_directory).resolve() / "samples"
    return samples_dir / sample_directory if sample_directory else samples_dir


class SurfaceSampler:
    """Sample a body-complete VPM field on a regular planar grid.

    The sampler creates a flattened ``(n_points, 3)`` point list on the plane
    through ``point``. For a z-normal the two bounds axes are x/y; for a
    y-normal they are x/z; for an x-normal they are y/z. ``sample`` returns
    canonical scalar CSV columns, while ``save_vtp`` preserves the structured
    grid topology in a VTS file.

    Attributes
    ----------
    point : numpy.ndarray
        Plane origin/offset, shape ``(3,)``, in m.
    normal : numpy.ndarray
        Normalized plane normal, shape ``(3,)``, dimensionless.
    bounds : numpy.ndarray
        Two in-plane intervals ``(min1, max1, min2, max2)`` in m.
    spacing : float
        Uniform point spacing in m.
    grid_points : numpy.ndarray
        Flattened generated points, shape ``(n_points, 3)``, in m.

    Examples
    --------
    >>> sampler = SurfaceSampler([0, 0, 0], [0, 0, 1], [-1, 1, -1, 1], 0.5)
    >>> sampler.grid_points.shape
    (25, 3)
    """

    def __init__(
        self,
        point: np.ndarray | list,
        normal: np.ndarray | list,
        bounds: np.ndarray | list,
        spacing: float,
        file_name: str | None = None,
        include_derivatives: bool = True,
        schedule: OutputSchedule | None = None,
        initial: bool | None = None,
    ):
        """Create and validate a planar sampling grid.

        Parameters
        ----------
        point : array-like
            Plane point, shape ``(3,)``, in m.
        normal : array-like
            Plane normal, shape ``(3,)``. It is normalized and must be
            non-zero; the dominant coordinate selects the in-plane axes.
        bounds : array-like
            ``[min1, max1, min2, max2]`` in m, ordered along the two in-plane
            axes.
        spacing : float
            Positive grid spacing in m. Endpoints are included when reached by
            the spacing sequence.
        file_name : str or None, default=None
            Base output name. The output manager supplies a class-based name
            when omitted.
        include_derivatives : bool, default=True
            Include gradient/strain columns and VTK arrays. Vorticity is always
            reconstructed from the velocity curl.
        schedule : OutputSchedule or None, default=None
            Accepted-step/time schedule used by :class:`OutputManager`.
        initial : bool or None, default=None
            Include the initial state. ``None`` retains a subclass's policy.

        Raises
        ------
        ValueError
            If point/normal shapes, bounds length, normal magnitude, or
            spacing are invalid.
        """
        self.point = np.asarray(point, dtype=np.float32)
        normal = np.asarray(normal, dtype=np.float32)
        self.normal = normal / np.linalg.norm(normal)
        self.bounds = np.asarray(bounds, dtype=np.float32)
        self.spacing = float(spacing)
        self.file_name = file_name
        self.include_derivatives = bool(include_derivatives)
        self.schedule = schedule
        if initial is not None:
            self.initial = initial

        if self.point.shape != (3,):
            raise ValueError(f"point must be 3D, got shape {self.point.shape}")
        if self.normal.shape != (3,):
            raise ValueError(f"normal must be 3D, got shape {self.normal.shape}")
        if len(self.bounds) != 4:
            raise ValueError(f"bounds must have 4 elements, got {len(self.bounds)}")
        if self.spacing <= 0:
            raise ValueError(f"spacing must be positive, got {self.spacing}")

        # Build grid points
        self._build_grid()

    def _build_grid(self):
        """Build the 2D grid of sample points on the plane."""
        # Determine which axis is the normal direction
        abs_normal = np.abs(self.normal)
        normal_axis = np.argmax(abs_normal)

        # Extract plane offset value
        plane_offset = self.point[normal_axis]

        # Create coordinate arrays for the two in-plane axes
        c1 = np.arange(self.bounds[0], self.bounds[1] + self.spacing / 2, self.spacing)
        c2 = np.arange(self.bounds[2], self.bounds[3] + self.spacing / 2, self.spacing)
        C1, C2 = np.meshgrid(c1, c2, indexing="ij")

        n_points = C1.size
        self.grid_points = np.zeros((n_points, 3), dtype=np.float32)

        if normal_axis == 2:  # z-plane (normal = [0,0,1])
            self.grid_points[:, 0] = C1.ravel()  # x
            self.grid_points[:, 1] = C2.ravel()  # y
            self.grid_points[:, 2] = plane_offset  # z constant
            self._axis1_name = "x"
            self._axis2_name = "y"
        elif normal_axis == 1:  # y-plane (normal = [0,1,0])
            self.grid_points[:, 0] = C1.ravel()  # x
            self.grid_points[:, 1] = plane_offset  # y constant
            self.grid_points[:, 2] = C2.ravel()  # z
            self._axis1_name = "x"
            self._axis2_name = "z"
        else:  # x-plane (normal = [1,0,0])
            self.grid_points[:, 0] = plane_offset  # x constant
            self.grid_points[:, 1] = C1.ravel()  # y
            self.grid_points[:, 2] = C2.ravel()  # z
            self._axis1_name = "y"
            self._axis2_name = "z"

        self._grid_shape = C1.shape
        self._n_points = n_points

    def sample(self, solver: "VPMSolver") -> dict[str, np.ndarray]:
        """Evaluate the solver field at every generated grid point.

        Parameters
        ----------
        solver : VPMSolver
            Solver providing body-complete velocity and velocity-gradient
            queries at the current accepted time.

        Returns
        -------
        dict[str, numpy.ndarray]
            Equal-length flattened columns. Position columns are m, velocity
            columns m/s, vorticity/gradient/strain columns 1/s. Derivative
            columns are zero when ``include_derivatives`` is false. An empty
            wake still includes freestream and any solved bound/body field.
        """
        velocity, vorticity, physical_gradient = _sample_velocity_vorticity_gradient(
            solver, self.grid_points, self.spacing
        )
        if self.include_derivatives:
            grad_u_flat = physical_gradient.reshape(-1, 9)
        else:
            grad_u_flat = np.zeros((len(self.grid_points), 9), dtype=np.float64)
        strain_rate_xx = grad_u_flat[:, 0]
        strain_rate_xy = 0.5 * (grad_u_flat[:, 1] + grad_u_flat[:, 3])
        strain_rate_xz = 0.5 * (grad_u_flat[:, 2] + grad_u_flat[:, 6])
        strain_rate_yy = grad_u_flat[:, 4]
        strain_rate_yz = 0.5 * (grad_u_flat[:, 5] + grad_u_flat[:, 7])
        strain_rate_zz = grad_u_flat[:, 8]

        velocity_gradient_xx = grad_u_flat[:, 0]
        velocity_gradient_xy = grad_u_flat[:, 1]
        velocity_gradient_xz = grad_u_flat[:, 2]
        velocity_gradient_yx = grad_u_flat[:, 3]
        velocity_gradient_yy = grad_u_flat[:, 4]
        velocity_gradient_yz = grad_u_flat[:, 5]
        velocity_gradient_zx = grad_u_flat[:, 6]
        velocity_gradient_zy = grad_u_flat[:, 7]
        velocity_gradient_zz = grad_u_flat[:, 8]

        return _validated_sample_data(
            {
                "position_x": self.grid_points[:, 0],
                "position_y": self.grid_points[:, 1],
                "position_z": self.grid_points[:, 2],
                "velocity_x": velocity[:, 0],
                "velocity_y": velocity[:, 1],
                "velocity_z": velocity[:, 2],
                "vorticity_x": vorticity[:, 0],
                "vorticity_y": vorticity[:, 1],
                "vorticity_z": vorticity[:, 2],
                "strain_rate_xx": strain_rate_xx,
                "strain_rate_xy": strain_rate_xy,
                "strain_rate_xz": strain_rate_xz,
                "strain_rate_yy": strain_rate_yy,
                "strain_rate_yz": strain_rate_yz,
                "strain_rate_zz": strain_rate_zz,
                "velocity_gradient_xx": velocity_gradient_xx,
                "velocity_gradient_xy": velocity_gradient_xy,
                "velocity_gradient_xz": velocity_gradient_xz,
                "velocity_gradient_yx": velocity_gradient_yx,
                "velocity_gradient_yy": velocity_gradient_yy,
                "velocity_gradient_yz": velocity_gradient_yz,
                "velocity_gradient_zx": velocity_gradient_zx,
                "velocity_gradient_zy": velocity_gradient_zy,
                "velocity_gradient_zz": velocity_gradient_zz,
            }
        )

    def save_csv(
        self,
        solver: "VPMSolver",
        filepath: str | Path,
        time: float | None = None,
    ) -> Path:
        """Evaluate one snapshot and write canonical columns to CSV.

        Parameters
        ----------
        solver : VPMSolver
            Source solver at the state being sampled.
        filepath : str or pathlib.Path
            Destination CSV path; parent directories are created.
        time : float or None, default=None
            Optional physical time in seconds written as a comment line.

        Returns
        -------
        pathlib.Path
            Path of the created file.

        Side Effects
        ------------
        Evaluates the field (including CPU transfers as required) and
        overwrites the destination file.
        """
        data = self.sample(solver)
        filepath = Path(filepath)

        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="") as f:
            # Embed simulation time so post-processing never needs to reconstruct it
            if time is not None:
                f.write(f"# time={time}\n")

            writer = csv.writer(f, lineterminator="\n")

            # Header (single source of truth: SAMPLER_CSV_COLUMNS)
            columns = sampler_csv_columns(self)
            writer.writerow(columns)

            # Data rows — built from SAMPLER_CSV_COLUMNS so they always align
            for i in range(self._n_points):
                writer.writerow([data[col][i] for col in columns])

        return filepath

    def save_vtp(
        self,
        solver: "VPMSolver",
        filepath: str | Path,
        time: float | None = None,
    ) -> Path:
        """
        Compute and export field data to VTS (VTK StructuredGrid) file.

        VTS is an efficient binary format for structured grids that preserves
        the 2D grid topology. Can be directly opened in ParaView with proper
        mesh connectivity and vector/scalar field data.

        Note: Despite the method name, this exports .vts format (StructuredGrid)
        which is more appropriate for planar grid data than .vtp (PolyData).

        Parameters
        ----------
        solver : VPMSolver
            Source solver at the state being sampled.
        filepath : str or pathlib.Path
            Destination path. The suffix is replaced with ``.vts``.
        time : float or None, default=None
            Accepted physical time in seconds; retained for protocol
            compatibility and not embedded by this writer.

        Returns
        -------
        pathlib.Path
            Path of the created VTS file.

        Raises
        ------
        ImportError
            If PyVista is not installed.
        """
        try:
            import pyvista as pv
        except ImportError as e:
            raise ImportError(
                "pyvista is required for VTS export. Install with: pip install pyvista"
            ) from e

        data = self.sample(solver)
        filepath = Path(filepath)

        # Use .vts extension for StructuredGrid format
        filepath = filepath.with_suffix(".vts")

        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Get grid dimensions
        ni, nj = self._grid_shape  # 2D grid shape from meshgrid with indexing='ij'

        # Reshape flat arrays back to 2D grid shape
        x_2d = data["position_x"].reshape(self._grid_shape)
        y_2d = data["position_y"].reshape(self._grid_shape)
        z_2d = data["position_z"].reshape(self._grid_shape)

        # Expand to 3D for StructuredGrid (add third dimension of size 1)
        x_3d = x_2d[:, :, np.newaxis]
        y_3d = y_2d[:, :, np.newaxis]
        z_3d = z_2d[:, :, np.newaxis]

        # Create StructuredGrid with dimensions (ni, nj, 1)
        grid = pv.StructuredGrid(x_3d, y_3d, z_3d)

        # Reshape field data to match grid point ordering
        # PyVista uses Fortran-order for StructuredGrid point data
        Ux_2d = data["velocity_x"].reshape(self._grid_shape)
        Uy_2d = data["velocity_y"].reshape(self._grid_shape)
        Uz_2d = data["velocity_z"].reshape(self._grid_shape)

        vorticity_x_grid = data["vorticity_x"].reshape(self._grid_shape)
        vorticity_y_grid = data["vorticity_y"].reshape(self._grid_shape)
        vorticity_z_grid = data["vorticity_z"].reshape(self._grid_shape)

        # Flatten in Fortran order to match PyVista's expected point ordering
        velocity = np.column_stack(
            [Ux_2d.ravel(order="F"), Uy_2d.ravel(order="F"), Uz_2d.ravel(order="F")]
        )
        vorticity = np.column_stack(
            [
                vorticity_x_grid.ravel(order="F"),
                vorticity_y_grid.ravel(order="F"),
                vorticity_z_grid.ravel(order="F"),
            ]
        )

        # Reshape optional derivative fields
        def _reshape_scalar(key):
            return data[key].reshape(self._grid_shape).ravel(order="F")

        # Magnitudes are not stored: ParaView offers the magnitude of any
        # vector directly, and strain rate is the symmetric part of the
        # velocity gradient, so both are recovered without occupying the file.
        write_precision = getattr(solver, "write_precision", DEFAULT_WRITE_PRECISION)
        grid.point_data["velocity"] = cast_for_write(velocity, write_precision)
        grid.point_data["vorticity"] = cast_for_write(vorticity, write_precision)

        if self.include_derivatives:
            velocity_gradient = np.column_stack(
                [
                    _reshape_scalar("velocity_gradient_xx"),
                    _reshape_scalar("velocity_gradient_xy"),
                    _reshape_scalar("velocity_gradient_xz"),
                    _reshape_scalar("velocity_gradient_yx"),
                    _reshape_scalar("velocity_gradient_yy"),
                    _reshape_scalar("velocity_gradient_yz"),
                    _reshape_scalar("velocity_gradient_zx"),
                    _reshape_scalar("velocity_gradient_zy"),
                    _reshape_scalar("velocity_gradient_zz"),
                ]
            )
            grid.point_data["velocity_gradient"] = cast_for_write(
                velocity_gradient,
                write_precision,
            )

        write_vtk_dataset(grid, filepath)

        return filepath


class LineSampler:
    """Sample a body-complete VPM field along a uniformly spaced line.

    The generated ``line_points`` array has shape ``(n_points, 3)`` and the
    returned table includes a dimensionless ``line_parameter`` from 0 to 1.
    Position, velocity, vorticity, gradient, and strain columns use the same
    units and names as :class:`SurfaceSampler`; CSV output can therefore be
    consumed by the same post-processing code.

    Attributes
    ----------
    start, end : numpy.ndarray
        Endpoints, shape ``(3,)``, in m.
    length : float
        Segment length in m.
    spacing : float
        Requested point spacing in m.
    line_points : numpy.ndarray
        Generated points, shape ``(n_points, 3)``, in m.

    Examples
    --------
    >>> sampler = LineSampler([-1, 0, 0], [1, 0, 0], 0.5)
    >>> sampler.line_points.shape
    (5, 3)
    """

    # The sampler executor owns the persistent, time-aware CSV representation.
    # ``save_csv`` remains available for callers that explicitly want one snapshot.
    csv_time_series = True

    def __init__(
        self,
        start: np.ndarray | list,
        end: np.ndarray | list,
        spacing: float,
        file_name: str | None = None,
        include_derivatives: bool = True,
        schedule: OutputSchedule | None = None,
    ):
        """Create and validate a line sampling grid.

        Parameters
        ----------
        start, end : array-like
            Endpoints with shape ``(3,)`` in m.
        spacing : float
            Positive requested spacing in m. At least two points are generated,
            including both endpoints.
        file_name : str or None, default=None
            Base output name used by :class:`OutputManager` when provided.
        include_derivatives : bool, default=True
            Persist velocity-gradient and strain-rate columns.
        schedule : OutputSchedule or None, default=None
            Accepted-step/time schedule used by framework dispatch.

        Raises
        ------
        ValueError
            If endpoints are not three-vectors or spacing is non-positive.
        """
        self.start = np.asarray(start, dtype=np.float32)
        self.end = np.asarray(end, dtype=np.float32)
        self.spacing = float(spacing)
        self.file_name = file_name
        self.include_derivatives = bool(include_derivatives)
        self.schedule = schedule

        if self.start.shape != (3,):
            raise ValueError(f"start must be 3D, got shape {self.start.shape}")
        if self.end.shape != (3,):
            raise ValueError(f"end must be 3D, got shape {self.end.shape}")
        if self.spacing <= 0:
            raise ValueError(f"spacing must be positive, got {self.spacing}")

        # Build line points
        self._build_line()

    def _build_line(self):
        """Build the array of sample points along the line."""
        direction = self.end - self.start
        self.length = np.linalg.norm(direction)

        # Compute number of points from spacing
        self.n_points = max(2, int(np.ceil(self.length / self.spacing)) + 1)

        t = np.linspace(0, 1, self.n_points, dtype=np.float32)

        self.line_points = np.zeros((self.n_points, 3), dtype=np.float32)
        for i in range(3):
            self.line_points[:, i] = self.start[i] + t * direction[i]

        self.t_param = t  # Parametric coordinate [0, 1]

    def sample(self, solver: "VPMSolver") -> dict[str, np.ndarray]:
        """Evaluate the solver field at every generated line point.

        Parameters
        ----------
        solver : VPMSolver
            Solver providing field queries at the current accepted state.

        Returns
        -------
        dict[str, numpy.ndarray]
            Equal-length flattened columns. Positions are m, velocity m/s,
            vorticity/gradient/strain 1/s, and ``line_parameter`` is
            dimensionless in ``[0, 1]``.
        """
        velocity, vorticity, gradient = _sample_velocity_vorticity_gradient(
            solver, self.line_points, self.spacing
        )
        grad_u_flat = gradient.reshape(-1, 9)
        strain_rate_xx = grad_u_flat[:, 0]
        strain_rate_xy = 0.5 * (grad_u_flat[:, 1] + grad_u_flat[:, 3])
        strain_rate_xz = 0.5 * (grad_u_flat[:, 2] + grad_u_flat[:, 6])
        strain_rate_yy = grad_u_flat[:, 4]
        strain_rate_yz = 0.5 * (grad_u_flat[:, 5] + grad_u_flat[:, 7])
        strain_rate_zz = grad_u_flat[:, 8]

        velocity_gradient_xx = grad_u_flat[:, 0]
        velocity_gradient_xy = grad_u_flat[:, 1]
        velocity_gradient_xz = grad_u_flat[:, 2]
        velocity_gradient_yx = grad_u_flat[:, 3]
        velocity_gradient_yy = grad_u_flat[:, 4]
        velocity_gradient_yz = grad_u_flat[:, 5]
        velocity_gradient_zx = grad_u_flat[:, 6]
        velocity_gradient_zy = grad_u_flat[:, 7]
        velocity_gradient_zz = grad_u_flat[:, 8]

        return _validated_sample_data(
            {
                "position_x": self.line_points[:, 0],
                "position_y": self.line_points[:, 1],
                "position_z": self.line_points[:, 2],
                "line_parameter": self.t_param,
                "velocity_x": velocity[:, 0],
                "velocity_y": velocity[:, 1],
                "velocity_z": velocity[:, 2],
                "vorticity_x": vorticity[:, 0],
                "vorticity_y": vorticity[:, 1],
                "vorticity_z": vorticity[:, 2],
                "strain_rate_xx": strain_rate_xx,
                "strain_rate_xy": strain_rate_xy,
                "strain_rate_xz": strain_rate_xz,
                "strain_rate_yy": strain_rate_yy,
                "strain_rate_yz": strain_rate_yz,
                "strain_rate_zz": strain_rate_zz,
                "velocity_gradient_xx": velocity_gradient_xx,
                "velocity_gradient_xy": velocity_gradient_xy,
                "velocity_gradient_xz": velocity_gradient_xz,
                "velocity_gradient_yx": velocity_gradient_yx,
                "velocity_gradient_yy": velocity_gradient_yy,
                "velocity_gradient_yz": velocity_gradient_yz,
                "velocity_gradient_zx": velocity_gradient_zx,
                "velocity_gradient_zy": velocity_gradient_zy,
                "velocity_gradient_zz": velocity_gradient_zz,
            }
        )

    def save_csv(
        self,
        solver: "VPMSolver",
        filepath: str | Path,
        time: float | None = None,
    ) -> Path:
        """Evaluate one line snapshot and write canonical columns to CSV.

        Parameters
        ----------
        solver : VPMSolver
            Source solver at the state being sampled.
        filepath : str or pathlib.Path
            Destination CSV path; parent directories are created.
        time : float or None, default=None
            Optional accepted physical time in seconds written as a comment.

        Returns
        -------
        pathlib.Path
            Path of the created CSV file.
        """
        data = self.sample(solver)
        filepath = Path(filepath)

        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="") as f:
            # Embed simulation time so post-processing never needs to reconstruct it
            if time is not None:
                f.write(f"# time={time}\n")

            writer = csv.writer(f, lineterminator="\n")

            # Header (single source of truth: SAMPLER_CSV_COLUMNS)
            columns = sampler_csv_columns(self)
            writer.writerow(columns)

            # Data rows — built from SAMPLER_CSV_COLUMNS so they always align
            for i in range(self.n_points):
                writer.writerow([data[col][i] for col in columns])

        return filepath
