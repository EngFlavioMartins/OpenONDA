"""Body-fitted boundary layers for the adaptive Cartesian mesher.

The first supported geometry is a straight, z-aligned circular cylinder.  A
layered O-grid follows the STL wall and ends on a Cartesian-lattice square so
it can be joined to the existing octree mesh without hanging interface faces.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .surface_classification import SurfaceIndex
from .triangulated_surface import TriangulatedSurface


@dataclass(frozen=True)
class BoundaryLayerSpec:
    """Wall-normal layers and the smooth transition to the volume mesh.

    ``interface_half_width`` and ``spanwise_cell_size`` are only needed by
    the specialised extruded-cylinder O-grid.  General closed STL bodies use
    the same first-height/layer/growth controls without either value. For a
    cylinder O-grid, ``transition_layers`` is a minimum: the adaptive mesher
    increases it when necessary to keep every transition step no larger than
    the adjoining finest Cartesian cell.
    """

    first_cell_height: float
    layers: int
    growth_ratio: float
    transition_layers: int
    interface_half_width: float | None = None
    spanwise_cell_size: float | None = None

    def validate(self) -> None:
        """Reject non-physical layer counts, sizes, and growth controls."""
        if not math.isfinite(self.first_cell_height) or self.first_cell_height <= 0.0:
            raise ValueError("first_cell_height must be finite and positive")
        if self.layers < 1:
            raise ValueError("layers must be at least one")
        if not math.isfinite(self.growth_ratio) or self.growth_ratio < 1.0:
            raise ValueError("growth_ratio must be finite and at least one")
        if self.transition_layers < 1:
            raise ValueError("transition_layers must be at least one")
        if self.interface_half_width is not None and (
            not math.isfinite(self.interface_half_width) or self.interface_half_width <= 0.0
        ):
            raise ValueError("interface_half_width must be finite and positive")
        if self.spanwise_cell_size is not None and (
            not math.isfinite(self.spanwise_cell_size) or self.spanwise_cell_size <= 0.0
        ):
            raise ValueError("spanwise_cell_size must be finite and positive")

    @property
    def thickness(self) -> float:
        """Total requested thickness of the constant-growth wall layers."""
        if math.isclose(self.growth_ratio, 1.0):
            return self.layers * self.first_cell_height
        return (
            self.first_cell_height
            * (self.growth_ratio**self.layers - 1.0)
            / (self.growth_ratio - 1.0)
        )

    @property
    def layer_heights(self) -> tuple[float, ...]:
        """Individual wall-normal cell heights, from the wall outwards."""
        return tuple(
            self.first_cell_height * self.growth_ratio**layer for layer in range(self.layers)
        )

    @property
    def cumulative_heights(self) -> tuple[float, ...]:
        """Cumulative extrusion distances expected by advancing-layer meshers."""
        return tuple(np.cumsum(np.asarray(self.layer_heights, dtype=np.float64)).tolist())


def _cylinder_geometry(
    surface: TriangulatedSurface,
    domain: tuple[float, float, float, float, float, float],
) -> tuple[np.ndarray, float]:
    """Return centre and radius after checking the supported STL geometry."""
    lower = np.asarray(surface.bounds[::2], dtype=np.float64)
    upper = np.asarray(surface.bounds[1::2], dtype=np.float64)
    extent = upper - lower
    scale = max(float(np.max(extent)), 1.0)
    tolerance = 1.0e-7 * scale
    if not math.isclose(float(extent[0]), float(extent[1]), rel_tol=1.0e-6):
        raise ValueError("Boundary layers currently require a circular cylinder STL")
    if extent[2] <= 2.0 * extent[0]:
        raise ValueError("Boundary layers currently require a z-aligned cylinder STL")
    if surface.bounds[4] > domain[4] + tolerance or surface.bounds[5] < domain[5] - tolerance:
        raise ValueError("The cylinder STL must span the complete mesh in z")

    centre = 0.5 * (lower[:2] + upper[:2])
    radius = 0.25 * float(extent[0] + extent[1])
    vertices = surface.triangles.reshape(-1, 3)
    radial = np.linalg.norm(vertices[:, :2] - centre, axis=1)
    side = radial > 0.5 * radius
    if not np.any(side) or np.max(np.abs(radial[side] - radius)) > tolerance:
        raise ValueError("STL side vertices do not describe a straight circular cylinder")
    return centre, radius


def cylinder_interface_bounds(
    surface: TriangulatedSurface,
    domain: tuple[float, float, float, float, float, float],
    spec: BoundaryLayerSpec,
) -> tuple[float, float, float, float, float, float]:
    """Cartesian square occupied by the body-fitted cylinder block."""
    spec.validate()
    centre, radius = _cylinder_geometry(surface, domain)
    if spec.interface_half_width is None:
        raise ValueError("Cylinder boundary layers require interface_half_width")
    half_width = spec.interface_half_width
    if half_width <= radius + spec.thickness:
        raise ValueError("interface_half_width must exceed cylinder radius plus layer thickness")
    bounds = (
        float(centre[0] - half_width),
        float(centre[0] + half_width),
        float(centre[1] - half_width),
        float(centre[1] + half_width),
        float(domain[4]),
        float(domain[5]),
    )
    if not (
        domain[0] < bounds[0] < bounds[1] < domain[1]
        and domain[2] < bounds[2] < bounds[3] < domain[3]
    ):
        raise ValueError("The boundary-layer interface must lie strictly inside the x-y domain")
    return bounds


def _square_points(centre: np.ndarray, half_width: float, cells_per_side: int) -> np.ndarray:
    """Equally spaced counter-clockwise points on a square, starting bottom-right."""
    count = 4 * cells_per_side
    points = np.empty((count, 2), dtype=np.float64)
    for index in range(count):
        side, local = divmod(index, cells_per_side)
        fraction = local / cells_per_side
        if side == 0:
            offset = (half_width, -half_width + 2.0 * half_width * fraction)
        elif side == 1:
            offset = (half_width - 2.0 * half_width * fraction, half_width)
        elif side == 2:
            offset = (-half_width, half_width - 2.0 * half_width * fraction)
        else:
            offset = (-half_width + 2.0 * half_width * fraction, -half_width)
        points[index] = centre + offset
    return points


def _wall_points(
    surface_index: SurfaceIndex,
    centre: np.ndarray,
    radius: float,
    z: float,
    count: int,
) -> np.ndarray:
    """Sample the circular side and map every point to the authoritative STL."""
    theta = -0.25 * np.pi + 2.0 * np.pi * np.arange(count) / count
    ideal = np.column_stack(
        (
            centre[0] + radius * np.cos(theta),
            centre[1] + radius * np.sin(theta),
            np.full(count, z),
        )
    )
    projected = np.empty_like(ideal)
    for index, point in enumerate(ideal):
        projected[index], _ = surface_index.nearest_point(point)
    return projected[:, :2]


def _geometric_transition_fractions(
    path_lengths: np.ndarray,
    first_height: float,
    layers: int,
) -> np.ndarray:
    """Return smooth per-path cumulative fractions for the O-grid transition.

    A linear blend makes the first transition cell arbitrarily larger or
    smaller than the last boundary-layer cell and is the direct cause of the
    conspicuous wedges in the old cylinder grid.  Each ray now uses a
    geometric series whose first term continues the wall-layer grading and
    whose sum lands exactly on the Cartesian square.  The ratio may be below
    one when the remaining gap is short, but every step stays positive.
    """
    fractions = np.empty((layers, len(path_lengths)), dtype=np.float64)
    for path_id, length in enumerate(path_lengths):
        if length <= 0.0:
            raise ValueError("Boundary-layer transition path has non-positive length")
        target_first = min(first_height, length / layers)
        if layers == 1:
            fractions[0, path_id] = 1.0
            continue

        def series_sum(ratio: float, first: float = target_first) -> float:
            if math.isclose(ratio, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                return layers * first
            return first * (ratio**layers - 1.0) / (ratio - 1.0)

        if math.isclose(series_sum(1.0), length, rel_tol=1.0e-10, abs_tol=1.0e-14):
            ratio = 1.0
        elif series_sum(1.0) < length:
            low, high = 1.0, 2.0
            while series_sum(high) < length:
                high *= 2.0
            for _ in range(80):
                middle = 0.5 * (low + high)
                if series_sum(middle) < length:
                    low = middle
                else:
                    high = middle
            ratio = 0.5 * (low + high)
        else:
            low, high = 0.0, 1.0
            for _ in range(80):
                middle = 0.5 * (low + high)
                if series_sum(middle) < length:
                    low = middle
                else:
                    high = middle
            ratio = 0.5 * (low + high)

        steps = target_first * np.power(ratio, np.arange(layers, dtype=np.float64))
        # Force the endpoint exactly onto the square despite the root-solve
        # tolerance; the correction is at roundoff scale.
        steps[-1] += length - float(np.sum(steps))
        fractions[:, path_id] = np.cumsum(steps) / length
    fractions[-1] = 1.0
    return fractions


def cylinder_transition_layer_count(
    surface: TriangulatedSurface,
    domain: tuple[float, float, float, float, float, float],
    lattice_size: float,
    spec: BoundaryLayerSpec,
    *,
    minimum: int = 1,
) -> int:
    """Minimum O-grid transition count whose wall-normal step is at most ``lattice_size``.

    The longest path is the corner ray from the outer wall-layer ring to the
    Cartesian square.  Solving against that path makes the guarantee apply to
    every circumferential location and every grid-study resolution.
    """
    spec.validate()
    if spec.interface_half_width is None:
        raise ValueError("Cylinder boundary layers require interface_half_width")
    if not math.isfinite(lattice_size) or lattice_size <= 0.0:
        raise ValueError("lattice_size must be finite and positive")
    if minimum < 1:
        raise ValueError("minimum transition layer count must be positive")
    _centre, radius = _cylinder_geometry(surface, domain)
    longest_path = math.sqrt(2.0) * spec.interface_half_width - (radius + spec.thickness)
    if longest_path <= 0.0:
        raise ValueError("Cylinder wall layers extend beyond the Cartesian interface")
    first_height = spec.layer_heights[-1] * spec.growth_ratio
    layers = minimum
    while layers < 100_000:
        fractions = _geometric_transition_fractions(
            np.asarray([longest_path]), first_height, layers
        )[:, 0]
        steps = np.diff(np.concatenate(([0.0], fractions))) * longest_path
        if float(np.max(steps)) <= lattice_size * (1.0 + 1.0e-12):
            return layers
        layers += 1
    raise RuntimeError("Could not resolve the cylinder transition within 100000 layers")


def _face_area_vectors(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    coordinates = points[faces]
    centres = coordinates.mean(axis=1)
    vectors = np.zeros((len(faces), 3), dtype=np.float64)
    for index in range(faces.shape[1]):
        vectors += 0.5 * np.cross(
            coordinates[:, index] - centres,
            coordinates[:, (index + 1) % faces.shape[1]] - centres,
        )
    return vectors


def _orient_faces(
    points: np.ndarray,
    faces: np.ndarray,
    cell_centres: np.ndarray,
    owners: np.ndarray,
    neighbours: np.ndarray | None = None,
) -> np.ndarray:
    """Orient faces from owner to neighbour, or out of the owner at a boundary."""
    face_centres = points[faces].mean(axis=1)
    if neighbours is None:
        direction = face_centres - cell_centres[owners]
    else:
        direction = cell_centres[neighbours] - cell_centres[owners]
    reverse = np.einsum("ij,ij->i", _face_area_vectors(points, faces), direction) < 0.0
    result = faces.copy()
    result[reverse] = result[reverse, ::-1]
    return result


def build_cylinder_layer_mesh(
    surface: TriangulatedSurface,
    surface_index: SurfaceIndex,
    domain: tuple[float, float, float, float, float, float],
    lattice_size: float,
    spec: BoundaryLayerSpec,
    wall_patch_name: str,
    interface_patch_name: str,
) -> dict:
    """Build the STL-conforming layered cylinder block as native hex cells."""
    spec.validate()
    centre, radius = _cylinder_geometry(surface, domain)
    interface = cylinder_interface_bounds(surface, domain, spec)
    if spec.interface_half_width is None:
        raise ValueError("Cylinder boundary layers require interface_half_width")
    side_value = 2.0 * spec.interface_half_width / lattice_size
    spanwise_cell_size = spec.spanwise_cell_size or lattice_size
    z_value = (domain[5] - domain[4]) / spanwise_cell_size
    cells_per_side = int(round(side_value))
    z_cells = int(round(z_value))
    if not math.isclose(side_value, cells_per_side, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("Boundary-layer square sides must align with the Cartesian lattice")
    if not math.isclose(z_value, z_cells, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("Boundary-layer span must be divisible by spanwise_cell_size")

    theta_cells = 4 * cells_per_side
    radial_cells = spec.layers + spec.transition_layers
    radial_points = radial_cells + 1
    z_points = z_cells + 1
    square = _square_points(centre, spec.interface_half_width, cells_per_side)
    wall = _wall_points(
        surface_index,
        centre,
        radius,
        0.5 * (domain[4] + domain[5]),
        theta_cells,
    )
    normals = wall - centre
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    rings = np.empty((radial_points, theta_cells, 2), dtype=np.float64)
    rings[0] = wall
    distance = 0.0
    layer_heights = list(spec.layer_heights)
    for layer, height in enumerate(layer_heights):
        distance += height
        rings[layer + 1] = wall + distance * normals
    transition_start = rings[spec.layers].copy()
    transition_vectors = square - transition_start
    transition_lengths = np.linalg.norm(transition_vectors, axis=1)
    transition_fractions = _geometric_transition_fractions(
        transition_lengths,
        layer_heights[-1] * spec.growth_ratio,
        spec.transition_layers,
    )
    transition_steps = np.diff(
        np.vstack((np.zeros((1, theta_cells)), transition_fractions)),
        axis=0,
    ) * transition_lengths[None, :]
    max_transition_height = float(np.max(transition_steps))
    if max_transition_height > lattice_size * (1.0 + 1.0e-10):
        raise ValueError(
            "Cylinder transition contains a wall-normal step larger than the "
            "finest Cartesian cell; increase transition_layers"
        )
    for layer in range(1, spec.transition_layers + 1):
        fraction = transition_fractions[layer - 1, :, None]
        rings[spec.layers + layer] = transition_start + fraction * transition_vectors

    z_coordinates = np.linspace(domain[4], domain[5], z_points)
    points = np.empty((z_points, radial_points, theta_cells, 3), dtype=np.float64)
    points[:, :, :, :2] = rings[None, :, :, :]
    points[:, :, :, 2] = z_coordinates[:, None, None]
    points = np.ascontiguousarray(points.reshape(-1, 3))

    def point_id(z_index, radial_index, theta_index):
        return (z_index * radial_points + radial_index) * theta_cells + theta_index

    def cell_id(z_index, radial_index, theta_index):
        return (z_index * radial_cells + radial_index) * theta_cells + theta_index

    z_index = np.repeat(np.arange(z_cells), radial_cells * theta_cells)
    radial_index = np.tile(np.repeat(np.arange(radial_cells), theta_cells), z_cells)
    theta_index = np.tile(np.arange(theta_cells), z_cells * radial_cells)
    next_theta = (theta_index + 1) % theta_cells
    cells = np.column_stack(
        (
            point_id(z_index, radial_index, theta_index),
            point_id(z_index, radial_index + 1, theta_index),
            point_id(z_index, radial_index + 1, next_theta),
            point_id(z_index, radial_index, next_theta),
            point_id(z_index + 1, radial_index, theta_index),
            point_id(z_index + 1, radial_index + 1, theta_index),
            point_id(z_index + 1, radial_index + 1, next_theta),
            point_id(z_index + 1, radial_index, next_theta),
        )
    ).astype(np.int32)
    cell_centres = points[cells].mean(axis=1)

    face_blocks = []
    owner_blocks = []
    neighbour_blocks = []

    # Faces between radial layers.
    zz = np.repeat(np.arange(z_cells), (radial_cells - 1) * theta_cells)
    rr = np.tile(np.repeat(np.arange(1, radial_cells), theta_cells), z_cells)
    tt = np.tile(np.arange(theta_cells), z_cells * (radial_cells - 1))
    nt = (tt + 1) % theta_cells
    faces = np.column_stack(
        (
            point_id(zz, rr, tt),
            point_id(zz, rr, nt),
            point_id(zz + 1, rr, nt),
            point_id(zz + 1, rr, tt),
        )
    ).astype(np.int32)
    owners = cell_id(zz, rr - 1, tt).astype(np.int32)
    neighbours = cell_id(zz, rr, tt).astype(np.int32)
    face_blocks.append(_orient_faces(points, faces, cell_centres, owners, neighbours))
    owner_blocks.append(owners)
    neighbour_blocks.append(neighbours)

    # Circumferential faces, including the periodic seam.
    zz = np.repeat(np.arange(z_cells), radial_cells * theta_cells)
    rr = np.tile(np.repeat(np.arange(radial_cells), theta_cells), z_cells)
    boundary_theta = np.tile(np.arange(theta_cells), z_cells * radial_cells)
    previous_theta = (boundary_theta - 1) % theta_cells
    faces = np.column_stack(
        (
            point_id(zz, rr, boundary_theta),
            point_id(zz, rr + 1, boundary_theta),
            point_id(zz + 1, rr + 1, boundary_theta),
            point_id(zz + 1, rr, boundary_theta),
        )
    ).astype(np.int32)
    owners = cell_id(zz, rr, previous_theta).astype(np.int32)
    neighbours = cell_id(zz, rr, boundary_theta).astype(np.int32)
    face_blocks.append(_orient_faces(points, faces, cell_centres, owners, neighbours))
    owner_blocks.append(owners)
    neighbour_blocks.append(neighbours)

    # Faces between spanwise cells.
    zz = np.repeat(np.arange(1, z_cells), radial_cells * theta_cells)
    rr = np.tile(np.repeat(np.arange(radial_cells), theta_cells), z_cells - 1)
    tt = np.tile(np.arange(theta_cells), (z_cells - 1) * radial_cells)
    nt = (tt + 1) % theta_cells
    faces = np.column_stack(
        (
            point_id(zz, rr, tt),
            point_id(zz, rr + 1, tt),
            point_id(zz, rr + 1, nt),
            point_id(zz, rr, nt),
        )
    ).astype(np.int32)
    owners = cell_id(zz - 1, rr, tt).astype(np.int32)
    neighbours = cell_id(zz, rr, tt).astype(np.int32)
    face_blocks.append(_orient_faces(points, faces, cell_centres, owners, neighbours))
    owner_blocks.append(owners)
    neighbour_blocks.append(neighbours)

    internal_faces = np.vstack(face_blocks)
    internal_owners = np.concatenate(owner_blocks)
    internal_neighbours = np.concatenate(neighbour_blocks)
    n_internal = len(internal_faces)

    boundaries = []
    boundary_faces = []
    boundary_owners = []

    def add_boundary(name: str, faces: np.ndarray, owners: np.ndarray) -> None:
        oriented = _orient_faces(points, faces, cell_centres, owners)
        boundaries.append(
            {
                "name": name,
                "start_face": n_internal + sum(len(block) for block in boundary_faces),
                "n_faces": len(oriented),
                "type": "wall" if name == wall_patch_name else "patch",
            }
        )
        boundary_faces.append(oriented)
        boundary_owners.append(owners.astype(np.int32))

    zz = np.repeat(np.arange(z_cells), theta_cells)
    tt = np.tile(np.arange(theta_cells), z_cells)
    nt = (tt + 1) % theta_cells
    add_boundary(
        wall_patch_name,
        np.column_stack(
            (
                point_id(zz, 0, tt),
                point_id(zz + 1, 0, tt),
                point_id(zz + 1, 0, nt),
                point_id(zz, 0, nt),
            )
        ).astype(np.int32),
        cell_id(zz, 0, tt),
    )
    add_boundary(
        interface_patch_name,
        np.column_stack(
            (
                point_id(zz, radial_cells, tt),
                point_id(zz, radial_cells, nt),
                point_id(zz + 1, radial_cells, nt),
                point_id(zz + 1, radial_cells, tt),
            )
        ).astype(np.int32),
        cell_id(zz, radial_cells - 1, tt),
    )

    rr = np.repeat(np.arange(radial_cells), theta_cells)
    tt = np.tile(np.arange(theta_cells), radial_cells)
    nt = (tt + 1) % theta_cells
    add_boundary(
        "zmin",
        np.column_stack(
            (
                point_id(0, rr, tt),
                point_id(0, rr, nt),
                point_id(0, rr + 1, nt),
                point_id(0, rr + 1, tt),
            )
        ).astype(np.int32),
        cell_id(0, rr, tt),
    )
    add_boundary(
        "zmax",
        np.column_stack(
            (
                point_id(z_cells, rr, tt),
                point_id(z_cells, rr + 1, tt),
                point_id(z_cells, rr + 1, nt),
                point_id(z_cells, rr, nt),
            )
        ).astype(np.int32),
        cell_id(z_cells - 1, rr, tt),
    )

    all_faces = np.vstack((internal_faces, *boundary_faces))
    all_owners = np.concatenate((internal_owners, *boundary_owners))
    interface_ring_ids = point_id(
        np.repeat(np.arange(z_points), theta_cells),
        radial_cells,
        np.tile(np.arange(theta_cells), z_points),
    ).astype(np.int32)
    radial_index_per_cell = np.tile(
        np.repeat(np.arange(radial_cells), theta_cells), z_cells
    ).astype(np.int16)
    return {
        "vertex_position": points,
        "faces": np.ascontiguousarray(all_faces, dtype=np.int32),
        "owners": np.ascontiguousarray(all_owners, dtype=np.int32),
        "neighbours": np.ascontiguousarray(internal_neighbours, dtype=np.int32),
        "boundary": boundaries,
        "n_cells": len(cells),
        "n_faces": len(all_faces),
        "n_interior_faces": n_internal,
        "n_points": len(points),
        "cell_vertex_indices": cells,
        "cell_type_code": np.full(len(cells), 5, dtype=np.int32),
        "boundary_layer_index": radial_index_per_cell,
        "interface_point_ids": interface_ring_ids,
        "mesh_generation": {
            "method": "cylinder_boundary_layer",
            "cylinder_centre": centre.tolist(),
            "cylinder_radius": radius,
            "interface_bounds": interface,
            "theta_cells": theta_cells,
            "z_cells": z_cells,
            "spanwise_cell_size": spanwise_cell_size,
            "radial_cells": radial_cells,
            "wall_layers": spec.layers,
            "transition_layers": spec.transition_layers,
            "first_cell_height": spec.first_cell_height,
            "growth_ratio": spec.growth_ratio,
            "layer_heights": layer_heights,
            "layer_thickness": distance,
            "interface_half_width": spec.interface_half_width,
            "transition_path_length_min": float(np.min(transition_lengths)),
            "transition_path_length_max": float(np.max(transition_lengths)),
            "transition_first_height_requested": layer_heights[-1] * spec.growth_ratio,
            "transition_cell_height_min": float(np.min(transition_steps)),
            "transition_cell_height_max": max_transition_height,
            "transition_to_lattice_ratio_max": max_transition_height / lattice_size,
        },
    }


def _patch_rows(mesh: dict, patch_name: str) -> tuple[np.ndarray, np.ndarray]:
    patch = next(item for item in mesh["boundary"] if item["name"] == patch_name)
    start = int(patch["start_face"])
    stop = start + int(patch["n_faces"])
    return np.asarray(mesh["faces"])[start:stop], np.asarray(mesh["owners"])[start:stop]


def _morton_order(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Spatial cell order used by the contiguous MPI partitioner."""
    centres = np.empty((len(cells), 3), dtype=np.float64)
    for start in range(0, len(cells), 50_000):
        stop = min(start + 50_000, len(cells))
        centres[start:stop] = points[cells[start:stop]].mean(axis=1)
    lower = centres.min(axis=0)
    extent = np.maximum(centres.max(axis=0) - lower, 1.0e-30)
    coordinates = np.clip(np.floor((centres - lower) / extent * 1023.0), 0, 1023).astype(np.uint64)

    def split(value: np.ndarray) -> np.ndarray:
        value = value & np.uint64(0x3FF)
        value = (value | value << np.uint64(16)) & np.uint64(0x30000FF)
        value = (value | value << np.uint64(8)) & np.uint64(0x300F00F)
        value = (value | value << np.uint64(4)) & np.uint64(0x30C30C3)
        value = (value | value << np.uint64(2)) & np.uint64(0x9249249)
        return value

    code = (
        split(coordinates[:, 0])
        | split(coordinates[:, 1]) << np.uint64(1)
        | split(coordinates[:, 2]) << np.uint64(2)
    )
    return np.argsort(code, kind="stable")


def _reorder_cells_for_partitioning(mesh: dict) -> None:
    """Renumber cells spatially while preserving face and patch ordering."""
    cells = np.asarray(mesh["cell_vertex_indices"], dtype=np.int32)
    order = _morton_order(np.asarray(mesh["vertex_position"]), cells)
    new_id = np.empty(len(order), dtype=np.int32)
    new_id[order] = np.arange(len(order), dtype=np.int32)
    mesh["owners"] = np.ascontiguousarray(new_id[np.asarray(mesh["owners"])], dtype=np.int32)
    mesh["neighbours"] = np.ascontiguousarray(
        new_id[np.asarray(mesh["neighbours"])], dtype=np.int32
    )
    for name in (
        "cell_vertex_indices",
        "cell_type_code",
        "cell_levels",
        "cell_sizes",
        "boundary_layer_index",
    ):
        mesh[name] = np.ascontiguousarray(np.asarray(mesh[name])[order])
    mesh["mesh_generation"]["cell_order"] = "morton"


def stitch_boundary_layer(outer: dict, layer: dict, interface_patch_name: str) -> dict:
    """Join a layer block to its exactly matching Cartesian hole."""
    outer_points = np.asarray(outer["vertex_position"], dtype=np.float64)
    layer_points = np.asarray(layer["vertex_position"], dtype=np.float64)
    scale = max(float(np.ptp(outer_points, axis=0).max()), 1.0)
    tolerance = 1.0e-10 * scale

    interface_ids = np.asarray(layer.pop("interface_point_ids"), dtype=np.int64)
    interface_set = np.zeros(len(layer_points), dtype=bool)
    interface_set[interface_ids] = True
    outer_key_to_id = {
        tuple(np.rint(point / tolerance).astype(np.int64)): index
        for index, point in enumerate(outer_points)
    }
    layer_point_map = np.empty(len(layer_points), dtype=np.int32)
    for point_id in interface_ids:
        key = tuple(np.rint(layer_points[point_id] / tolerance).astype(np.int64))
        if key not in outer_key_to_id:
            raise ValueError("Boundary-layer interface point is absent from Cartesian mesh")
        layer_point_map[point_id] = outer_key_to_id[key]
    new_ids = np.flatnonzero(~interface_set)
    layer_point_map[new_ids] = np.arange(
        len(outer_points), len(outer_points) + len(new_ids), dtype=np.int32
    )
    points = np.vstack((outer_points, layer_points[new_ids]))
    layer_faces = layer_point_map[np.asarray(layer["faces"], dtype=np.int32)]
    layer_cells = layer_point_map[np.asarray(layer["cell_vertex_indices"], dtype=np.int32)]

    outer_interface_faces, outer_interface_owners = _patch_rows(outer, interface_patch_name)
    layer_interface_faces, layer_interface_owners = _patch_rows(
        {**layer, "faces": layer_faces}, interface_patch_name
    )
    layer_by_signature = {
        tuple(sorted(map(int, face))): (face, int(owner))
        for face, owner in zip(layer_interface_faces, layer_interface_owners, strict=True)
    }
    if len(layer_by_signature) != len(layer_interface_faces):
        raise ValueError("Boundary-layer interface contains duplicate faces")
    matched_layer_owners = np.empty(len(outer_interface_faces), dtype=np.int32)
    for index, face in enumerate(outer_interface_faces):
        match = layer_by_signature.pop(tuple(sorted(map(int, face))), None)
        if match is None:
            raise ValueError("Cartesian and boundary-layer interface faces do not match")
        matched_layer_owners[index] = match[1]
    if layer_by_signature:
        raise ValueError("Boundary-layer interface has unmatched faces")

    outer_n_internal = int(outer["n_interior_faces"])
    layer_n_internal = int(layer["n_interior_faces"])
    cell_offset = int(outer["n_cells"])
    internal_faces = np.vstack(
        (
            np.asarray(outer["faces"][:outer_n_internal], dtype=np.int32),
            layer_faces[:layer_n_internal],
            np.asarray(outer_interface_faces, dtype=np.int32),
        )
    )
    internal_owners = np.concatenate(
        (
            np.asarray(outer["owners"][:outer_n_internal], dtype=np.int32),
            np.asarray(layer["owners"][:layer_n_internal], dtype=np.int32) + cell_offset,
            np.asarray(outer_interface_owners, dtype=np.int32),
        )
    )
    internal_neighbours = np.concatenate(
        (
            np.asarray(outer["neighbours"], dtype=np.int32),
            np.asarray(layer["neighbours"], dtype=np.int32) + cell_offset,
            matched_layer_owners + cell_offset,
        )
    )

    patch_order = []
    patch_data: dict[str, dict[str, list[np.ndarray] | str]] = {}

    def collect(mesh: dict, faces: np.ndarray, owner_offset: int) -> None:
        for patch in mesh["boundary"]:
            name = str(patch["name"])
            if name == interface_patch_name:
                continue
            if name not in patch_data:
                patch_order.append(name)
                patch_data[name] = {
                    "type": str(patch.get("type", "patch")),
                    "faces": [],
                    "owners": [],
                }
            start = int(patch["start_face"])
            stop = start + int(patch["n_faces"])
            face_blocks = patch_data[name]["faces"]
            owner_blocks = patch_data[name]["owners"]
            if not isinstance(face_blocks, list) or not isinstance(owner_blocks, list):
                raise RuntimeError("Boundary patch storage is internally inconsistent")
            face_blocks.append(np.asarray(faces[start:stop], dtype=np.int32))
            owner_blocks.append(
                np.asarray(mesh["owners"][start:stop], dtype=np.int32) + owner_offset
            )

    collect(outer, np.asarray(outer["faces"]), 0)
    collect(layer, layer_faces, cell_offset)
    face_blocks = [internal_faces]
    owner_blocks = [internal_owners]
    boundary = []
    start_face = len(internal_faces)
    for name in patch_order:
        entry = patch_data[name]
        entry_faces = entry["faces"]
        entry_owners = entry["owners"]
        if not isinstance(entry_faces, list) or not isinstance(entry_owners, list):
            raise RuntimeError("Boundary patch storage is internally inconsistent")
        faces = np.vstack(entry_faces)
        owners = np.concatenate(entry_owners)
        face_blocks.append(faces)
        owner_blocks.append(owners)
        boundary.append(
            {
                "name": name,
                "start_face": start_face,
                "n_faces": len(faces),
                "type": entry["type"],
            }
        )
        start_face += len(faces)

    cells = np.vstack(
        (
            np.asarray(outer["cell_vertex_indices"], dtype=np.int32),
            layer_cells,
        )
    )
    n_cells = len(cells)
    generation = dict(outer["mesh_generation"])
    generation["method"] = "adaptive_cartesian_with_boundary_layer"
    generation["boundary_layer"] = layer["mesh_generation"]
    result = {
        "vertex_position": np.ascontiguousarray(points),
        "faces": np.ascontiguousarray(np.vstack(face_blocks), dtype=np.int32),
        "owners": np.ascontiguousarray(np.concatenate(owner_blocks), dtype=np.int32),
        "neighbours": np.ascontiguousarray(internal_neighbours, dtype=np.int32),
        "boundary": boundary,
        "n_cells": n_cells,
        "n_faces": start_face,
        "n_interior_faces": len(internal_faces),
        "n_points": len(points),
        "cell_vertex_indices": np.ascontiguousarray(cells),
        "cell_type_code": np.full(n_cells, 5, dtype=np.int32),
        "cell_levels": np.concatenate(
            (
                np.asarray(outer["cell_levels"], dtype=np.int8),
                np.full(int(layer["n_cells"]), np.max(outer["cell_levels"]), dtype=np.int8),
            )
        ),
        "cell_sizes": np.concatenate(
            (
                np.asarray(outer["cell_sizes"], dtype=np.float32),
                np.full(int(layer["n_cells"]), generation["finest_cell_size"], dtype=np.float32),
            )
        ),
        "boundary_layer_index": np.concatenate(
            (
                np.full(int(outer["n_cells"]), -1, dtype=np.int16),
                np.asarray(layer["boundary_layer_index"], dtype=np.int16),
            )
        ),
        "mesh_generation": generation,
    }
    _reorder_cells_for_partitioning(result)
    return result


__all__ = [
    "BoundaryLayerSpec",
    "build_cylinder_layer_mesh",
    "cylinder_interface_bounds",
    "cylinder_transition_layer_count",
    "stitch_boundary_layer",
]
