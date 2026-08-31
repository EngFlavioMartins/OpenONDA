"""Lagrangian marker representation of an immersed body.

An :class:`ImmersedBody` is a cloud of surface-marker positions ``position``
(``n_markers``, 3) with a
prescribed velocity ``prescribed_velocity`` (Ns, 3) — zero for a fixed body, the local body
velocity for moving bodies. Markers must be spaced
``marker_spacing_ratio * grid_spacing`` apart, with a ratio near one relative
to the local Eulerian grid spacing (Pinelli et
al. 2010; Constant et al. Table 1), which the factory methods enforce.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _polygon_contains_xy(
    points: np.ndarray, vertex_position: np.ndarray, include_boundary: bool
) -> np.ndarray:
    """Vectorized point-in-polygon test with an explicit boundary policy."""
    result = np.zeros(len(points), dtype=bool)
    if len(points) == 0:
        return result

    lo = vertex_position.min(axis=0)
    hi = vertex_position.max(axis=0)
    candidates = np.flatnonzero(np.all((points >= lo) & (points <= hi), axis=1))
    if len(candidates) == 0:
        return result

    query = points[candidates]
    x = query[:, 0]
    y = query[:, 1]
    inside = np.zeros(len(query), dtype=bool)
    boundary = np.zeros(len(query), dtype=bool)
    scale = max(
        float(np.ptp(vertex_position[:, 0])),
        float(np.ptp(vertex_position[:, 1])),
        1.0,
    )
    tolerance = 1e-10 * scale

    previous = vertex_position[-1]
    for current in vertex_position:
        x0, y0 = previous
        x1, y1 = current
        dy = y1 - y0
        crosses = (y0 > y) != (y1 > y)
        x_crossing = x0 + (y - y0) * (x1 - x0) / (dy if abs(dy) > 1e-30 else 1e-30)
        inside ^= crosses & (x < x_crossing)

        edge = current - previous
        edge_sq = float(np.dot(edge, edge))
        if edge_sq > 0.0:
            relative = query - previous
            fraction = np.clip((relative @ edge) / edge_sq, 0.0, 1.0)
            closest = previous + fraction[:, None] * edge
            boundary |= np.linalg.norm(query - closest, axis=1) <= tolerance
        previous = current

    result[candidates] = (inside | boundary) if include_boundary else (inside & ~boundary)
    return result


class ImmersedBody:
    """Marker cloud for one immersed obstacle.

    Attributes:
        name:     Identifier used in force logs.
        position: Marker positions ``(n_markers, 3)``.
        prescribed_velocity: Desired fluid velocity at the markers ``(Ns, 3)``.
    """

    def __init__(
        self,
        name: str,
        position: Sequence[Sequence[float]] | np.ndarray,
        prescribed_velocity: Sequence[float] | np.ndarray | None = None,
        *,
        geometry: dict | None = None,
    ):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Immersed body name must be a non-empty string")
        self.name = name
        self.position = np.atleast_2d(np.asarray(position, dtype=np.float64))
        if self.position.ndim != 2 or self.position.shape[1] != 3 or self.position.shape[0] == 0:
            raise ValueError(
                f"Marker position array must be (n_markers, 3), got {self.position.shape}"
            )
        if not np.all(np.isfinite(self.position)):
            raise ValueError("Immersed body markers must be finite")
        if prescribed_velocity is None:
            self.prescribed_velocity = np.zeros_like(self.position)
        else:
            self.prescribed_velocity = np.broadcast_to(
                np.asarray(prescribed_velocity, dtype=np.float64), self.position.shape
            ).copy()
        if not np.all(np.isfinite(self.prescribed_velocity)):
            raise ValueError("Immersed body target velocity must be finite")
        self._geometry = geometry

    @property
    def n_markers(self) -> int:
        return self.position.shape[0]

    @property
    def has_solid_geometry(self) -> bool:
        """Whether this marker cloud also carries an exact interior test."""
        return self._geometry is not None

    @property
    def solid_bounds(self) -> np.ndarray | None:
        """Axis-aligned bounds of the represented solid, when available."""
        if self._geometry is None:
            return None
        bounds = self._geometry.get("bounds")
        return None if bounds is None else np.asarray(bounds, dtype=np.float64).copy()

    def contains(
        self,
        points: Sequence[Sequence[float]] | np.ndarray,
        *,
        include_boundary: bool = False,
    ) -> np.ndarray:
        """Return which query points lie in the represented solid.

        The marker cloud remains the IBM forcing representation.  This exact
        geometry metadata is used by coupled particle handoff to prevent solid
        leakage; arbitrary ``from_points`` bodies can opt in via ``geometry``.
        """
        query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self._geometry is None:
            raise ValueError(f"Immersed body {self.name!r} has no solid geometry metadata")

        geometry_type = self._geometry["type"]
        z_bounds = self._geometry.get("z_bounds")
        z_mask = np.ones(len(query), dtype=bool)
        if z_bounds is not None:
            z0, z1 = (float(value) for value in z_bounds)
            if include_boundary:
                z_mask = (query[:, 2] >= z0) & (query[:, 2] <= z1)
            else:
                z_mask = (query[:, 2] > z0) & (query[:, 2] < z1)

        if geometry_type == "sphere":
            centre = np.asarray(self._geometry["centre"], dtype=np.float64)
            radius = float(self._geometry["radius"])
            distance_sq = np.sum((query - centre) ** 2, axis=1)
            return distance_sq <= radius**2 if include_boundary else distance_sq < radius**2
        if geometry_type == "cylinder_z":
            centre = np.asarray(self._geometry["centre"], dtype=np.float64)
            radius = float(self._geometry["radius"])
            radial_sq = np.sum((query[:, :2] - centre[:2]) ** 2, axis=1)
            radial = radial_sq <= radius**2 if include_boundary else radial_sq < radius**2
            return radial & z_mask
        if geometry_type == "rectangle_z":
            centre = np.asarray(self._geometry["centre"], dtype=np.float64)
            half_size = 0.5 * np.asarray(self._geometry["size"], dtype=np.float64)
            delta = np.abs(query[:, :2] - centre[:2])
            planar = (
                np.all(delta <= half_size, axis=1)
                if include_boundary
                else np.all(delta < half_size, axis=1)
            )
            return planar & z_mask
        if geometry_type == "polygon_z":
            planar = _polygon_contains_xy(
                query[:, :2],
                np.asarray(self._geometry["vertex_position"], dtype=np.float64),
                include_boundary,
            )
            return planar & z_mask
        raise ValueError(f"Unsupported immersed-body geometry type {geometry_type!r}")

    def signed_distance(
        self,
        points: Sequence[Sequence[float]] | np.ndarray,
    ) -> np.ndarray:
        """Signed distance to the solid surface: positive in the fluid.

        Feeds the C1 wall taper. Exact outside, a lower bound inside.
        """
        query = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self._geometry is None:
            raise ValueError(f"Immersed body {self.name!r} has no solid geometry metadata")

        geometry_type = self._geometry["type"]
        if geometry_type == "sphere":
            centre = np.asarray(self._geometry["centre"], dtype=np.float64)
            return np.linalg.norm(query - centre, axis=1) - float(self._geometry["radius"])

        if geometry_type == "cylinder_z":
            centre = np.asarray(self._geometry["centre"], dtype=np.float64)
            planar = np.linalg.norm(query[:, :2] - centre[:2], axis=1) - float(
                self._geometry["radius"]
            )
        elif geometry_type == "rectangle_z":
            centre = np.asarray(self._geometry["centre"], dtype=np.float64)
            half = 0.5 * np.asarray(self._geometry["size"], dtype=np.float64)
            delta = np.abs(query[:, :2] - centre[:2]) - half
            planar = np.linalg.norm(np.maximum(delta, 0.0), axis=1) + np.minimum(
                np.max(delta, axis=1), 0.0
            )
        elif geometry_type == "polygon_z":
            vertex_position = np.asarray(self._geometry["vertex_position"], dtype=np.float64)
            edge = np.roll(vertex_position, -1, axis=0) - vertex_position
            rel = query[:, None, :2] - vertex_position[None, :, :]
            t = np.clip(
                np.einsum("pvi,vi->pv", rel, edge)
                / np.maximum(np.einsum("vi,vi->v", edge, edge), 1e-300),
                0.0,
                1.0,
            )
            closest = rel - t[..., None] * edge[None, :, :]
            planar = np.min(np.linalg.norm(closest, axis=2), axis=1)
            inside = _polygon_contains_xy(query[:, :2], vertex_position, False)
            planar = np.where(inside, -planar, planar)
        else:
            raise ValueError(f"Unsupported immersed-body geometry type {geometry_type!r}")

        z_bounds = self._geometry.get("z_bounds")
        if z_bounds is None:
            return planar
        z0, z1 = (float(value) for value in z_bounds)
        axial = np.maximum(z0 - query[:, 2], query[:, 2] - z1)
        outside = np.linalg.norm(
            np.stack([np.maximum(planar, 0.0), np.maximum(axial, 0.0)], axis=1), axis=1
        )
        inside = np.minimum(np.maximum(planar, axial), 0.0)
        return outside + inside

    # ------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------ #

    @classmethod
    def cylinder_z(
        cls,
        centre: Sequence[float] | np.ndarray,
        diameter: float,
        grid_spacing: float,
        marker_spacing_ratio: float = 1.0,
        name: str = "cylinder",
    ) -> ImmersedBody:
        """Circle of markers in the (x, y) plane (a z-extruded 2D cylinder).

        Markers are placed on the circle of the given diameter around
        ``centre`` (the z-coordinate of ``centre`` should be the mid-plane of
        the single-cell-thick 2D mesh), spaced approximately
        ``marker_spacing_ratio * grid_spacing`` along the arc.

        Args:
            centre:   Cylinder centre ``[x, y, z]``.
            diameter: Cylinder diameter.
            grid_spacing: Local Eulerian grid spacing near the body.
            marker_spacing_ratio: Marker spacing / grid spacing ratio.
            name:     Body name for force logs.
        """
        centre = np.asarray(centre, dtype=np.float64)
        if centre.shape != (3,) or not np.all(np.isfinite(centre)):
            raise ValueError("Cylinder centre must be a finite 3-vector")
        if not np.isfinite(diameter) or diameter <= 0.0:
            raise ValueError("Cylinder diameter must be finite and positive")
        if (
            not np.isfinite(grid_spacing)
            or grid_spacing <= 0.0
            or not np.isfinite(marker_spacing_ratio)
            or marker_spacing_ratio <= 0.0
        ):
            raise ValueError(
                "Cylinder grid_spacing and marker_spacing_ratio must be finite and positive"
            )
        n = max(int(round(np.pi * diameter / (marker_spacing_ratio * grid_spacing))), 4)
        theta = 2.0 * np.pi * np.arange(n) / n
        position = np.empty((n, 3))
        position[:, 0] = centre[0] + 0.5 * diameter * np.cos(theta)
        position[:, 1] = centre[1] + 0.5 * diameter * np.sin(theta)
        position[:, 2] = centre[2]
        geometry = {
            "type": "cylinder_z",
            "centre": centre.copy(),
            "radius": 0.5 * float(diameter),
            "z_bounds": None,
            "bounds": [
                centre[0] - 0.5 * diameter,
                centre[0] + 0.5 * diameter,
                centre[1] - 0.5 * diameter,
                centre[1] + 0.5 * diameter,
                -np.inf,
                np.inf,
            ],
        }
        return cls(name, position, geometry=geometry)

    @classmethod
    def extruded_cylinder_z(
        cls,
        centre: Sequence[float] | np.ndarray,
        diameter: float,
        z_bounds: Sequence[float] | np.ndarray,
        grid_spacing: float,
        marker_spacing_ratio: float = 1.0,
        name: str = "cylinder",
        caps: bool = True,
    ) -> ImmersedBody:
        """Surface markers for a circular cylinder extruded along the z-axis.

        The curved surface is sampled at approximately
        ``marker_spacing_ratio * grid_spacing`` in both
        directions. With ``caps=False``, the marker surface still spans
        ``z_bounds`` but the represented solid is treated as infinite in z,
        as needed for a cylinder passing through the FVM domain.
        """
        centre = np.asarray(centre, dtype=np.float64)
        z = np.asarray(z_bounds, dtype=np.float64).reshape(-1)
        parameters = np.asarray([diameter, grid_spacing, marker_spacing_ratio], dtype=np.float64)
        if centre.shape != (3,) or not np.all(np.isfinite(centre)):
            raise ValueError("Cylinder centre must be a finite 3-vector")
        if z.shape != (2,) or not np.all(np.isfinite(z)) or z[1] <= z[0]:
            raise ValueError("z_bounds must contain two increasing finite values")
        if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
            raise ValueError(
                "Cylinder diameter, grid_spacing, and marker_spacing_ratio "
                "must be finite and positive"
            )

        spacing = float(marker_spacing_ratio * grid_spacing)
        radius = 0.5 * float(diameter)
        n_theta = max(int(np.ceil(np.pi * diameter / spacing)), 4)
        n_z_cells = max(int(np.ceil((z[1] - z[0]) / spacing)), 1)
        theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
        if caps:
            z_levels = np.linspace(z[0], z[1], n_z_cells + 1)
        else:
            dz = (z[1] - z[0]) / n_z_cells
            z_levels = z[0] + (np.arange(n_z_cells) + 0.5) * dz
        theta_grid, z_grid = np.meshgrid(theta, z_levels, indexing="ij")
        marker_parts = [
            np.column_stack(
                (
                    centre[0] + radius * np.cos(theta_grid.ravel()),
                    centre[1] + radius * np.sin(theta_grid.ravel()),
                    z_grid.ravel(),
                )
            )
        ]

        if caps:
            offsets = np.arange(-radius + 0.5 * spacing, radius, spacing)
            if len(offsets):
                xx, yy = np.meshgrid(offsets, offsets, indexing="ij")
                disk = np.column_stack((xx.ravel(), yy.ravel()))
                disk = disk[np.sum(disk**2, axis=1) < radius**2]
                if len(disk):
                    disk[:, 0] += centre[0]
                    disk[:, 1] += centre[1]
                    marker_parts.extend(
                        [
                            np.column_stack((disk, np.full(len(disk), z[0]))),
                            np.column_stack((disk, np.full(len(disk), z[1]))),
                        ]
                    )

        position = np.vstack(marker_parts)
        stored_z_bounds = z.tolist() if caps else None
        geometry = {
            "type": "cylinder_z",
            "centre": centre.copy(),
            "radius": radius,
            "z_bounds": stored_z_bounds,
            "bounds": [
                centre[0] - radius,
                centre[0] + radius,
                centre[1] - radius,
                centre[1] + radius,
                z[0] if caps else -np.inf,
                z[1] if caps else np.inf,
            ],
        }
        return cls(name, position, geometry=geometry)

    @classmethod
    def sphere(
        cls,
        centre: Sequence[float] | np.ndarray,
        diameter: float,
        grid_spacing: float,
        marker_spacing_ratio: float = 1.0,
        name: str = "sphere",
    ) -> ImmersedBody:
        """Fibonacci sphere with one marker per squared target marker spacing."""
        centre = np.asarray(centre, dtype=np.float64)
        if centre.shape != (3,) or not np.all(np.isfinite(centre)):
            raise ValueError("Sphere centre must be a finite 3-vector")
        if not np.isfinite(diameter) or diameter <= 0.0:
            raise ValueError("Sphere diameter must be finite and positive")
        if (
            not np.isfinite(grid_spacing)
            or grid_spacing <= 0.0
            or not np.isfinite(marker_spacing_ratio)
            or marker_spacing_ratio <= 0.0
        ):
            raise ValueError(
                "Sphere grid_spacing and marker_spacing_ratio must be finite and positive"
            )
        n = max(
            int(round(np.pi * diameter**2 / (marker_spacing_ratio * grid_spacing) ** 2)),
            8,
        )
        k = np.arange(n) + 0.5
        polar_angle = np.arccos(1.0 - 2.0 * k / n)
        golden = np.pi * (1.0 + np.sqrt(5.0))
        azimuthal_angle = golden * k
        r = 0.5 * diameter
        position = np.empty((n, 3))
        position[:, 0] = centre[0] + r * np.cos(azimuthal_angle) * np.sin(polar_angle)
        position[:, 1] = centre[1] + r * np.sin(azimuthal_angle) * np.sin(polar_angle)
        position[:, 2] = centre[2] + r * np.cos(polar_angle)
        geometry = {
            "type": "sphere",
            "centre": centre.copy(),
            "radius": r,
            "z_bounds": None,
            "bounds": [
                centre[0] - r,
                centre[0] + r,
                centre[1] - r,
                centre[1] + r,
                centre[2] - r,
                centre[2] + r,
            ],
        }
        return cls(name, position, geometry=geometry)

    @classmethod
    def rectangle_z(
        cls,
        centre: Sequence[float] | np.ndarray,
        width: float,
        height: float,
        grid_spacing: float,
        marker_spacing_ratio: float = 1.0,
        name: str = "rectangle",
    ) -> ImmersedBody:
        """Markers on an axis-aligned rectangle extruded along z."""
        centre = np.asarray(centre, dtype=np.float64)
        parameters = np.asarray(
            [width, height, grid_spacing, marker_spacing_ratio], dtype=np.float64
        )
        if centre.shape != (3,) or not np.all(np.isfinite(centre)):
            raise ValueError("Rectangle centre must be a finite 3-vector")
        if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
            raise ValueError(
                "Rectangle dimensions, grid_spacing, and marker_spacing_ratio "
                "must be finite and positive"
            )
        spacing = marker_spacing_ratio * grid_spacing
        nx = max(int(round(width / spacing)), 1)
        ny = max(int(round(height / spacing)), 1)
        x = np.linspace(-0.5 * width, 0.5 * width, nx, endpoint=False)
        y = np.linspace(-0.5 * height, 0.5 * height, ny, endpoint=False)
        points = np.concatenate(
            (
                np.column_stack((x, np.full(nx, -0.5 * height))),
                np.column_stack((np.full(ny, 0.5 * width), y)),
                np.column_stack((-x, np.full(nx, 0.5 * height))),
                np.column_stack((np.full(ny, -0.5 * width), -y)),
            )
        )
        position = np.column_stack(
            (
                points[:, 0] + centre[0],
                points[:, 1] + centre[1],
                np.full(len(points), centre[2]),
            )
        )
        geometry = {
            "type": "rectangle_z",
            "centre": centre.copy(),
            "size": [float(width), float(height)],
            "z_bounds": None,
            "bounds": [
                centre[0] - 0.5 * width,
                centre[0] + 0.5 * width,
                centre[1] - 0.5 * height,
                centre[1] + 0.5 * height,
                -np.inf,
                np.inf,
            ],
        }
        return cls(name, position, geometry=geometry)

    @classmethod
    def extruded_polygon_z(
        cls,
        vertex_position: Sequence[Sequence[float]] | np.ndarray,
        z_bounds: Sequence[float] | np.ndarray,
        grid_spacing: float,
        marker_spacing_ratio: float = 1.0,
        name: str = "polygon",
        caps: bool = True,
    ) -> ImmersedBody:
        """Surface markers for a polygon extruded along the z-axis.

        Side markers follow the polygon at approximately
        ``marker_spacing_ratio * grid_spacing``.
        With ``caps=True``, Cartesian interior markers close both end faces;
        otherwise the solid is treated as extending through the domain in z.
        """
        polygon = np.asarray(vertex_position, dtype=np.float64)
        z = np.asarray(z_bounds, dtype=np.float64).reshape(-1)
        parameters = np.asarray([grid_spacing, marker_spacing_ratio], dtype=np.float64)
        if (
            polygon.ndim != 2
            or polygon.shape[1] != 2
            or polygon.shape[0] < 3
            or not np.all(np.isfinite(polygon))
        ):
            raise ValueError("Polygon vertex_position must be a finite (N, 2) array with N >= 3")
        if z.shape != (2,) or not np.all(np.isfinite(z)) or z[1] <= z[0]:
            raise ValueError("z_bounds must contain two increasing finite values")
        if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
            raise ValueError(
                "Polygon grid_spacing and marker_spacing_ratio must be finite and positive"
            )

        spacing = float(marker_spacing_ratio * grid_spacing)
        closed = np.vstack((polygon, polygon[0]))
        edges = np.diff(closed, axis=0)
        edge_lengths = np.linalg.norm(edges, axis=1)
        perimeter_length = float(edge_lengths.sum())
        if perimeter_length <= 0.0:
            raise ValueError("Polygon perimeter must be positive")
        n_perimeter = max(int(np.ceil(perimeter_length / spacing)), 3)
        arclength = np.concatenate(([0.0], np.cumsum(edge_lengths)))
        targets = perimeter_length * np.arange(n_perimeter) / n_perimeter
        edge_ids = np.searchsorted(arclength, targets, side="right") - 1
        edge_ids = np.minimum(edge_ids, len(edges) - 1)
        fractions = (targets - arclength[edge_ids]) / edge_lengths[edge_ids]
        perimeter = polygon[edge_ids] + fractions[:, None] * edges[edge_ids]

        n_z = max(int(np.ceil((z[1] - z[0]) / spacing)), 1) + 1
        z_levels = np.linspace(z[0], z[1], n_z)
        side_xy = np.repeat(perimeter, n_z, axis=0)
        side_z = np.tile(z_levels, len(perimeter))
        marker_parts = [np.column_stack((side_xy, side_z))]

        if caps:
            lo = polygon.min(axis=0)
            hi = polygon.max(axis=0)
            xs = np.arange(lo[0] + 0.5 * spacing, hi[0], spacing)
            ys = np.arange(lo[1] + 0.5 * spacing, hi[1], spacing)
            if len(xs) and len(ys):
                xx, yy = np.meshgrid(xs, ys, indexing="ij")
                candidates = np.column_stack((xx.ravel(), yy.ravel()))
                cap_xy = candidates[_polygon_contains_xy(candidates, polygon, False)]
                if len(cap_xy):
                    marker_parts.extend(
                        [
                            np.column_stack((cap_xy, np.full(len(cap_xy), z[0]))),
                            np.column_stack((cap_xy, np.full(len(cap_xy), z[1]))),
                        ]
                    )

        position = np.vstack(marker_parts)
        stored_z_bounds = z.tolist() if caps else None
        geometry = {
            "type": "polygon_z",
            "vertex_position": polygon.copy(),
            "z_bounds": stored_z_bounds,
            "bounds": [
                polygon[:, 0].min(),
                polygon[:, 0].max(),
                polygon[:, 1].min(),
                polygon[:, 1].max(),
                z[0] if caps else -np.inf,
                z[1] if caps else np.inf,
            ],
        }
        return cls(name, position, geometry=geometry)

    @classmethod
    def from_points(
        cls,
        position: Sequence[Sequence[float]] | np.ndarray,
        prescribed_velocity: Sequence[float] | np.ndarray | None = None,
        name: str = "body",
        *,
        geometry: dict | None = None,
    ) -> ImmersedBody:
        """Arbitrary marker cloud (e.g. sampled from an STL surface)."""
        return cls(name, position, prescribed_velocity, geometry=geometry)
