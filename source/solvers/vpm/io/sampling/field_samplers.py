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

import numpy as np

from .schedule import SamplingSchedule

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None  # noqa: N816

# Canonical CSV column order for SurfaceSampler / LineSampler output.  Single
# source of truth: the header row and every data row are built from this list,
# so the written header always matches the data (no magic column indices on the
# reader side — see ``_read_sampler_csv`` in the tutorials' post-processing).
SAMPLER_CSV_COLUMNS = [
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "vorticity_x",
    "vorticity_y",
    "vorticity_z",
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


def _validated_sample_data(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Require samplers to construct the canonical output schema directly."""
    allowed = {*SAMPLER_CSV_COLUMNS, "line_parameter"}
    unexpected = sorted(set(data) - allowed)
    if unexpected:
        raise ValueError("Sampler produced non-canonical fields: " + ", ".join(unexpected))
    return data


# =========================================================


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


def _extract_stl_config_from_solver(solver) -> tuple[str | None, Path | None]:
    """Extract body_stl path and case_dir from solver config (priority: FVM > VPM > direct)."""
    try:
        body_stl = None
        case_dir = None
        if hasattr(solver.setup, "fvm_solver") and hasattr(solver.setup.fvm_solver, "surface"):
            body_stl = solver.setup.fvm_solver.surface.body_stl
            if hasattr(solver.setup.fvm_solver, "case_dir"):
                case_dir = Path(solver.setup.fvm_solver.case_dir)
        if not body_stl and hasattr(solver.setup, "vpm_solver"):
            body_stl = solver.setup.vpm_solver.body_stl
        if not body_stl and hasattr(solver.setup, "body_stl"):
            body_stl = solver.setup.body_stl
    except Exception:
        body_stl = None
    return body_stl, case_dir


def resolve_samples_dir(case_directory, sample_subdirectory: str | None = None) -> Path:
    """Return the mandatory ``<case>/samples`` directory."""
    samples_dir = Path(case_directory).resolve() / "samples"
    return samples_dir / sample_subdirectory if sample_subdirectory else samples_dir


class SurfaceSampler:
    """
    Compute VPM-induced fields on a 2D planar grid.

    Creates a regular 2D grid of points on a plane and uses the solver's
    compute_*_at() methods to evaluate induced velocity and vorticity.

    Attributes:
        point: A point on the plane (3D array).
        normal: Unit normal vector to the plane (3D array).
        bounds: Extent of the sampling grid.
        spacing: Grid point spacing.

    Example:
        >>> sampler = SurfaceSampler(
        ...     point=[0, 0, 0],
        ...     normal=[0, 0, 1],  # z=0 plane
        ...     bounds=[-5, 10, -3, 3],  # x: [-5, 10], y: [-3, 3]
        ...     spacing=0.1
        ... )
        >>> data = sampler.sample(solver)
        >>> print(f"Computed velocity at {len(data['x'])} grid points")
    """

    def __init__(
        self,
        point: np.ndarray | list,
        normal: np.ndarray | list,
        bounds: np.ndarray | list,
        spacing: float,
        file_name: str | None = None,
        include_derivatives: bool = True,
        schedule: SamplingSchedule | None = None,
    ):
        """
        Initialize the surface sampler.

        Args:
            point: A point on the plane [x, y, z].
            normal: Normal vector to the plane [nx, ny, nz]. Will be normalized.
                   Only axis-aligned normals are supported: [1,0,0], [0,1,0], [0,0,1].
            bounds: Grid bounds [min1, max1, min2, max2] for the two in-plane axes.
                   For normal=[0,0,1] (z-plane): bounds=[xmin, xmax, ymin, ymax]
                   For normal=[0,1,0] (y-plane): bounds=[xmin, xmax, zmin, zmax]
                   For normal=[1,0,0] (x-plane): bounds=[ymin, ymax, zmin, zmax]
            spacing: Grid point spacing.
            file_name: Optional base name for output files. If None, uses
                      default naming based on sampler class name.
            include_derivatives: Export velocity-gradient and strain-rate
                      fields. Vorticity always comes from the velocity curl.
            schedule: Optional independent step- or flow-time output cadence.
        """
        self.point = np.asarray(point, dtype=np.float32)
        normal = np.asarray(normal, dtype=np.float32)
        self.normal = normal / np.linalg.norm(normal)
        self.bounds = np.asarray(bounds, dtype=np.float32)
        self.spacing = float(spacing)
        self.file_name = file_name
        self.include_derivatives = bool(include_derivatives)
        self.schedule = schedule

        # Body geometry cache for masking / wall projection
        self._body_mesh = None
        self._body_tree = None
        self._body_normal = None
        self._body_loaded = False

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

    def _resolve_body_stl_path(self, solver) -> Path | None:
        """Resolve body STL from coupler config (absolute path if possible)."""
        if not hasattr(solver, "setup"):
            return None
        body_stl, case_dir = _extract_stl_config_from_solver(solver)
        if not body_stl:
            return None
        body_stl_path = Path(body_stl)
        if body_stl_path.is_absolute():
            return body_stl_path if body_stl_path.exists() else None
        if case_dir is not None:
            case_body_stl_path = case_dir / body_stl
            if case_body_stl_path.exists():
                return case_body_stl_path
        return body_stl_path if body_stl_path.exists() else None

    def _ensure_body_geometry(self, solver) -> None:
        """Load/caches STL and surface KDTree once."""
        if self._body_loaded:
            return
        self._body_loaded = True

        try:
            import pyvista as pv
        except Exception:
            return

        stl_path = self._resolve_body_stl_path(solver)
        if stl_path is None:
            return

        try:
            self._body_mesh = pv.read(str(stl_path))
            if cKDTree is not None:
                surf = self._body_mesh.extract_surface().compute_normals(
                    point_normals=True, cell_normals=False
                )
                pts = np.asarray(surf.points)
                nrm = np.asarray(surf["Normals"])
                nn = np.linalg.norm(nrm, axis=1)
                nn[nn < 1e-16] = 1.0
                nrm = nrm / nn[:, None]
                self._body_tree = cKDTree(pts)
                self._body_normal = nrm
        except Exception:
            self._body_mesh = None
            self._body_tree = None
            self._body_normal = None

    def _get_exterior_mask(self, solver):
        """
        Determine which grid points are outside the body geometry.

        Returns boolean mask: True for exterior points, False for interior.
        Uses MeshHandler's body_tree if available (fast), otherwise assumes all exterior.
        """
        self._ensure_body_geometry(solver)
        if self._body_mesh is None:
            return np.ones(len(self.grid_points), dtype=bool)

        try:
            import pyvista as pv

            grid_pv = pv.PolyData(self.grid_points)
            selection = grid_pv.select_enclosed_points(
                self._body_mesh, tolerance=0.0, check_surface=True
            )
            inside_mask = selection["SelectedPoints"].astype(bool)
            return ~inside_mask
        except Exception:
            pass

        # Default: assume all points are exterior (no body geometry available)
        return np.ones(len(self.grid_points), dtype=bool)

    def _project_near_wall_tangent(
        self, solver, points: np.ndarray, velocity: np.ndarray, exterior_mask: np.ndarray
    ) -> None:
        """Project near-wall velocity tangentially so its normal component is zero."""
        self._ensure_body_geometry(solver)
        if self._body_tree is None or self._body_normal is None:
            return

        ext_idx = np.where(exterior_mask)[0]
        if len(ext_idx) == 0:
            return

        pts_ext = points[ext_idx]
        d_wall, idx_wall = self._body_tree.query(pts_ext)
        near = d_wall <= (0.5 * self.spacing)
        if not np.any(near):
            return

        global_index = ext_idx[near]
        wall_normal = self._body_normal[idx_wall[near]]
        wall_velocity = velocity[global_index]
        normal_velocity = np.einsum("ij,ij->i", wall_velocity, wall_normal)[:, None]
        velocity[global_index] = wall_velocity - normal_velocity * wall_normal

    def sample(self, solver) -> dict[str, np.ndarray]:
        """
        Compute induced fields at all grid points.

        Args:
            solver: VPM Solver instance.

        Returns:
            Dictionary with computed fields:
                - 'position_x', 'position_y', 'position_z': Grid point coordinates
                - 'velocity_x', 'velocity_y', 'velocity_z': Velocity components
                - 'vorticity_x', 'vorticity_y', 'vorticity_z': Vorticity components
                - 'strain_rate_xx' ... 'strain_rate_zz': Strain-rate components [1/s]
                - 'velocity_gradient_xx' ... 'velocity_gradient_zz': Gradient components [1/s]
        """
        # Guard: if no particles exist, return zero fields (avoid SVD crash)
        n_particles_total = solver.particles.n_particles_total
        if n_particles_total == 0:
            n_pts = len(self.grid_points)
            zeros_vec = np.zeros((n_pts, 3), dtype=np.float64)
            zeros_s = np.zeros((n_pts, 6), dtype=np.float64)
            zeros_g = np.zeros((n_pts, 9), dtype=np.float64)
            return _validated_sample_data(
                {
                    "position_x": self.grid_points[:, 0],
                    "position_y": self.grid_points[:, 1],
                    "position_z": self.grid_points[:, 2],
                    "velocity_x": zeros_vec[:, 0].copy(),
                    "velocity_y": zeros_vec[:, 1].copy(),
                    "velocity_z": zeros_vec[:, 2].copy(),
                    "vorticity_x": zeros_vec[:, 0].copy(),
                    "vorticity_y": zeros_vec[:, 1].copy(),
                    "vorticity_z": zeros_vec[:, 2].copy(),
                    "strain_rate_xx": zeros_s[:, 0].copy(),
                    "strain_rate_xy": zeros_s[:, 1].copy(),
                    "strain_rate_xz": zeros_s[:, 2].copy(),
                    "strain_rate_yy": zeros_s[:, 3].copy(),
                    "strain_rate_yz": zeros_s[:, 4].copy(),
                    "strain_rate_zz": zeros_s[:, 5].copy(),
                    "velocity_gradient_xx": zeros_g[:, 0].copy(),
                    "velocity_gradient_xy": zeros_g[:, 1].copy(),
                    "velocity_gradient_xz": zeros_g[:, 2].copy(),
                    "velocity_gradient_yx": zeros_g[:, 3].copy(),
                    "velocity_gradient_yy": zeros_g[:, 4].copy(),
                    "velocity_gradient_yz": zeros_g[:, 5].copy(),
                    "velocity_gradient_zx": zeros_g[:, 6].copy(),
                    "velocity_gradient_zy": zeros_g[:, 7].copy(),
                    "velocity_gradient_zz": zeros_g[:, 8].copy(),
                }
            )

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
        solver,
        filepath: str | Path,
        time: float | None = None,
    ) -> Path:
        """
        Compute and export field data to CSV file.

        Args:
            solver: VPM Solver instance.
            filepath: Output file path.
            time: Optional simulation time (for logging).

        Returns:
            Path to the created CSV file.
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
            writer.writerow(SAMPLER_CSV_COLUMNS)

            # Data rows — built from SAMPLER_CSV_COLUMNS so they always align
            for i in range(self._n_points):
                writer.writerow([data[col][i] for col in SAMPLER_CSV_COLUMNS])

        return filepath

    def save_vtp(
        self,
        solver,
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

        Args:
            solver: VPM Solver instance.
            filepath: Output file path (will use .vts extension).
            time: Optional simulation time (for logging).

        Returns:
            Path to the created VTS file.

        Raises:
            ImportError: If pyvista is not installed.
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

        # Add velocity as vector field
        grid.point_data["velocity"] = velocity

        # Add velocity magnitude as scalar
        grid.point_data["velocity_magnitude"] = np.linalg.norm(velocity, axis=1)

        # Add vorticity as vector field
        grid.point_data["vorticity"] = vorticity

        # Add vorticity magnitude as scalar
        grid.point_data["vorticity_magnitude"] = np.linalg.norm(vorticity, axis=1)

        if self.include_derivatives:
            strain_rate = np.column_stack(
                [
                    _reshape_scalar("strain_rate_xx"),
                    _reshape_scalar("strain_rate_xy"),
                    _reshape_scalar("strain_rate_xz"),
                    _reshape_scalar("strain_rate_yy"),
                    _reshape_scalar("strain_rate_yz"),
                    _reshape_scalar("strain_rate_zz"),
                ]
            )
            grid.point_data["strain_rate"] = strain_rate
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
            grid.point_data["velocity_gradient"] = velocity_gradient

        # Save as VTS (binary by default for efficiency)
        grid.save(str(filepath), binary=True)

        return filepath


class LineSampler:
    """
    Compute VPM-induced fields along a line segment.

    Creates regularly spaced points along a line segment and uses the solver's
    compute_*_at() methods to evaluate induced velocity and vorticity.

    Attributes:
        start: Start point of the line segment (3D array).
        end: End point of the line segment (3D array).
        spacing: Distance between sample points along the line.

    Example:
        >>> sampler = LineSampler(
        ...     start=[-5, 0, 0],
        ...     end=[10, 0, 0],
        ...     spacing=0.1
        ... )
        >>> data = sampler.sample(solver)
        >>> print(f"Computed velocity at {sampler.n_points} points along centreline")
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
        schedule: SamplingSchedule | None = None,
    ):
        """
        Initialize the line sampler.

        Args:
            start: Start point of line segment [x, y, z].
            end: End point of line segment [x, y, z].
            spacing: Distance between sample points along the line.
            file_name: Optional base name for output CSV files. If None, uses
                      default naming based on sampler class name.
            schedule: Optional independent step- or flow-time output cadence.
        """
        self.start = np.asarray(start, dtype=np.float32)
        self.end = np.asarray(end, dtype=np.float32)
        self.spacing = float(spacing)
        self.file_name = file_name
        self.schedule = schedule

        # Body geometry cache for masking / wall projection
        self._body_mesh = None
        self._body_tree = None
        self._body_normal = None
        self._body_loaded = False

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

    def _resolve_body_stl_path(self, solver) -> Path | None:
        """Resolve body STL from coupler config (absolute path if possible)."""
        if not hasattr(solver, "setup"):
            return None
        body_stl, case_dir = _extract_stl_config_from_solver(solver)
        if not body_stl:
            return None
        body_stl_path = Path(body_stl)
        if body_stl_path.is_absolute():
            return body_stl_path if body_stl_path.exists() else None
        if case_dir is not None:
            case_body_stl_path = case_dir / body_stl
            if case_body_stl_path.exists():
                return case_body_stl_path
        return body_stl_path if body_stl_path.exists() else None

    def _ensure_body_geometry(self, solver) -> None:
        if self._body_loaded:
            return
        self._body_loaded = True

        try:
            import pyvista as pv
        except Exception:
            return

        stl_path = self._resolve_body_stl_path(solver)
        if stl_path is None:
            return

        try:
            self._body_mesh = pv.read(str(stl_path))
            if cKDTree is not None:
                surf = self._body_mesh.extract_surface().compute_normals(
                    point_normals=True, cell_normals=False
                )
                pts = np.asarray(surf.points)
                nrm = np.asarray(surf["Normals"])
                nn = np.linalg.norm(nrm, axis=1)
                nn[nn < 1e-16] = 1.0
                nrm = nrm / nn[:, None]
                self._body_tree = cKDTree(pts)
                self._body_normal = nrm
        except Exception:
            self._body_mesh = None
            self._body_tree = None
            self._body_normal = None

    def _get_exterior_mask(self, solver) -> np.ndarray:
        self._ensure_body_geometry(solver)
        if self._body_mesh is None:
            return np.ones(self.n_points, dtype=bool)

        try:
            import pyvista as pv

            pts = pv.PolyData(self.line_points)
            sel = pts.select_enclosed_points(self._body_mesh, tolerance=0.0, check_surface=True)
            inside = sel["SelectedPoints"].astype(bool)
            return ~inside
        except Exception:
            return np.ones(self.n_points, dtype=bool)

    def _project_near_wall_tangent(
        self, solver, velocity: np.ndarray, exterior_mask: np.ndarray
    ) -> None:
        self._ensure_body_geometry(solver)
        if self._body_tree is None or self._body_normal is None:
            return

        ext_idx = np.where(exterior_mask)[0]
        if len(ext_idx) == 0:
            return

        pts_ext = self.line_points[ext_idx]
        d_wall, idx_wall = self._body_tree.query(pts_ext)
        near = d_wall <= (0.5 * self.spacing)
        if not np.any(near):
            return

        global_index = ext_idx[near]
        wall_normal = self._body_normal[idx_wall[near]]
        wall_velocity = velocity[global_index]
        normal_velocity = np.einsum("ij,ij->i", wall_velocity, wall_normal)[:, None]
        velocity[global_index] = wall_velocity - normal_velocity * wall_normal

    def sample(self, solver) -> dict[str, np.ndarray]:
        """
        Compute induced fields at all line points.

        Args:
            solver: VPM Solver instance.

        Returns:
            Dictionary with computed fields:
                - 'position_x', 'position_y', 'position_z': Line point coordinates
                - 'line_parameter': Parametric coordinate along line [0, 1]
                - 'velocity_x', 'velocity_y', 'velocity_z': Velocity components
                - 'vorticity_x', 'vorticity_y', 'vorticity_z': Vorticity components
                - 'strain_rate_xx' ... 'strain_rate_zz': Strain-rate components [1/s]
                - 'velocity_gradient_xx' ... 'velocity_gradient_zz': Gradient components [1/s]
        """
        # Guard: if no particles exist, return zero fields (avoid SVD crash)
        n_particles_total = solver.particles.n_particles_total
        if n_particles_total == 0:
            n_pts = self.n_points
            zeros_vec = np.zeros((n_pts, 3), dtype=np.float64)
            zeros_s = np.zeros((n_pts, 6), dtype=np.float64)
            zeros_g = np.zeros((n_pts, 9), dtype=np.float64)
            return _validated_sample_data(
                {
                    "position_x": self.line_points[:, 0],
                    "position_y": self.line_points[:, 1],
                    "position_z": self.line_points[:, 2],
                    "line_parameter": self.t_param,
                    "velocity_x": zeros_vec[:, 0].copy(),
                    "velocity_y": zeros_vec[:, 1].copy(),
                    "velocity_z": zeros_vec[:, 2].copy(),
                    "vorticity_x": zeros_vec[:, 0].copy(),
                    "vorticity_y": zeros_vec[:, 1].copy(),
                    "vorticity_z": zeros_vec[:, 2].copy(),
                    "strain_rate_xx": zeros_s[:, 0].copy(),
                    "strain_rate_xy": zeros_s[:, 1].copy(),
                    "strain_rate_xz": zeros_s[:, 2].copy(),
                    "strain_rate_yy": zeros_s[:, 3].copy(),
                    "strain_rate_yz": zeros_s[:, 4].copy(),
                    "strain_rate_zz": zeros_s[:, 5].copy(),
                    "velocity_gradient_xx": zeros_g[:, 0].copy(),
                    "velocity_gradient_xy": zeros_g[:, 1].copy(),
                    "velocity_gradient_xz": zeros_g[:, 2].copy(),
                    "velocity_gradient_yx": zeros_g[:, 3].copy(),
                    "velocity_gradient_yy": zeros_g[:, 4].copy(),
                    "velocity_gradient_yz": zeros_g[:, 5].copy(),
                    "velocity_gradient_zx": zeros_g[:, 6].copy(),
                    "velocity_gradient_zy": zeros_g[:, 7].copy(),
                    "velocity_gradient_zz": zeros_g[:, 8].copy(),
                }
            )

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
        solver,
        filepath: str | Path,
        time: float | None = None,
    ) -> Path:
        """
        Compute and export field data to CSV file.

        Args:
            solver: VPM Solver instance.
            filepath: Output file path.
            time: Optional simulation time (for logging).

        Returns:
            Path to the created CSV file.
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
            writer.writerow(SAMPLER_CSV_COLUMNS)

            # Data rows — built from SAMPLER_CSV_COLUMNS so they always align
            for i in range(self.n_points):
                writer.writerow([data[col][i] for col in SAMPLER_CSV_COLUMNS])

        return filepath
