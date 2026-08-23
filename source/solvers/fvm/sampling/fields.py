"""Line and surface field samplers for the FVM solver.

``LineSampler`` probes a line segment into a growing time-aware CSV;
``SurfaceSampler`` probes an axis-aligned plane into per-event ``.vts``
snapshots plus a PVD index.  Both interpolate solver cell fields at fixed
probe points by inverse distance weighting over the nearest cell centroids.

Sampling is partition-aware: in a partitioned run the owned cell centroids and
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

import numpy as np
from scipy.spatial import cKDTree  # type: ignore

from .base import SAMPLER_CSV_COLUMNS, Sampler, _register_sampler, append_csv_rows

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
        file_name: str | None = None,
        schedule=None,
    ):
        super().__init__(file_name=file_name, schedule=schedule)
        self.points = np.asarray(points, dtype=float)
        self.k = int(k)
        self.power = float(inverse_distance_power)
        self._tree = None
        self._tree_key = None

    def _interpolate(self, field, centroids) -> np.ndarray:
        # The key is derived from content so the tree survives across events
        # even when the (gathered) centroid array is rebuilt on every step.
        key = (len(centroids), tuple(np.asarray(centroids[0], dtype=float)))
        if self._tree is None or self._tree_key != key:
            self._tree = cKDTree(centroids)
            self._tree_key = key
        k = min(self.k, len(centroids))
        dists, indices = self._tree.query(self.points, k=k)
        if k == 1:
            dists = dists[:, np.newaxis]
            indices = indices[:, np.newaxis]
        weights = 1.0 / (dists + 1e-12) ** self.power
        weights /= weights.sum(axis=1, keepdims=True)
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
        centroids, velocity, kinematic_pressure, vorticity = basis
        data = {
            "position_x": self.points[:, 0],
            "position_y": self.points[:, 1],
            "position_z": self.points[:, 2],
            "velocity_x": self._interpolate(velocity[:, 0], centroids),
            "velocity_y": self._interpolate(velocity[:, 1], centroids),
            "velocity_z": self._interpolate(velocity[:, 2], centroids),
            "vorticity_x": self._interpolate(vorticity[:, 0], centroids),
            "vorticity_y": self._interpolate(vorticity[:, 1], centroids),
            "vorticity_z": self._interpolate(vorticity[:, 2], centroids),
            "kinematic_pressure": self._interpolate(kinematic_pressure, centroids),
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
        start,
        end,
        n_points: int | None = None,
        spacing: float | None = None,
        k: int = 5,
        inverse_distance_power: float = 2.0,
        file_name: str | None = None,
        schedule=None,
    ):
        """Initialize the line sampler.

        Args:
            start: Start point of the line, array-like of shape (3,).
            end: End point of the line, array-like of shape (3,).
            n_points: Number of uniformly spaced sample points.
            spacing: Point spacing; alternative to ``n_points``.
            k: Number of nearest neighbours used for interpolation.
            inverse_distance_power: Exponent used for inverse-distance weighting.
            file_name: Base name for the output CSV.
            schedule: Optional :class:`~.base.SamplingSchedule`.
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
                "power": self.power,
            }
        )
        return spec

    def write_csv(self, context, samples_dir: str) -> dict[str, np.ndarray] | None:
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
    Cadence is owned entirely by the :class:`~.base.SamplingSchedule` — there
    is no separate ``stride`` counter, so live and offline runs select the
    same physical states.

    Examples
    --------
    >>> sampler = SurfaceSampler(
    ...     point=[0, 0, 0.5], normal=[0, 0, 1],
    ...     bounds=[0, 1, 0, 1], spacing=0.25,
    ...     file_name="slice_z0",
    ...     schedule=SamplingSchedule(every_n_steps=2),
    ... )
    """

    sampler_kind = "SurfaceSampler"

    def __init__(
        self,
        point,
        normal,
        bounds,
        spacing: float,
        k: int = 5,
        inverse_distance_power: float = 2.0,
        file_name: str | None = None,
        schedule=None,
        body_bounds=None,
    ):
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
            file_name: Base name for the output files.
            schedule: Optional :class:`~.base.SamplingSchedule`.
            body_bounds: Optional axis-aligned solid bounds ``(xmin, xmax,
                ymin, ymax, zmin, zmax)``.  Probe points geometrically inside
                the body are masked in the output (``vtkValidPointMask`` zero,
                NaN field values) so the slice never shows a flow field inside
                the solid.
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
            file_name=file_name,
            schedule=schedule,
        )
        self.point = point
        self.normal = normal
        self.bounds = bounds
        self.spacing = float(spacing)
        self.grid_shape = C1.shape
        self._pvd_entries: list[tuple[float, str]] = []
        if body_bounds is not None:
            body_bounds = np.asarray(body_bounds, dtype=float)
            if body_bounds.shape != (6,):
                raise ValueError("body_bounds must contain six coordinates")
            inside = (
                (points[:, 0] > body_bounds[0])
                & (points[:, 0] < body_bounds[1])
                & (points[:, 1] > body_bounds[2])
                & (points[:, 1] < body_bounds[3])
                & (points[:, 2] > body_bounds[4])
                & (points[:, 2] < body_bounds[5])
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
                "power": self.power,
                "body_bounds": self._body_bounds.tolist()
                if self._body_bounds is not None
                else None,
            }
        )
        return spec

    def save_vts(self, context, filepath: str) -> dict[str, np.ndarray] | None:
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
            # contain a flow inside the solid.
            inside = self._inside_mask
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
