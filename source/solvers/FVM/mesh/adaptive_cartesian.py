"""Native adaptive Cartesian meshes for the finite-volume solver.

This module implements the part of the cfMesh ``cartesianMesh`` workflow that
OpenONDA's axis-aligned verification cases need: triangulated-surface input,
dyadic local refinement, 2:1 transition bands, removal of an axis-aligned
solid, and direct construction of the solver's face-based ``mesh_data``
dictionary. It does not read or write an external solver case and does not
require an external mesher.

Algorithmic lineage and credit
------------------------------
The workflow is inspired by the open-source cfMesh Cartesian mesher by Dr.
Franjo Juretic and Creative Fields, Ltd.  cfMesh creates an octree template,
extracts a predominantly hexahedral/polyhedral mesh, maps it to the input
surface, and optionally creates boundary layers.  This is an independent
Python implementation of the axis-aligned template/extraction stages; it does
not copy cfMesh source code and does not yet implement arbitrary surface
mapping or boundary-layer extrusion.

Both cfMesh and OpenONDA are distributed under GPL-3.0-or-later.  See
``CFMESH_ATTRIBUTION.md`` beside this file for upstream links and the precise
scope of the adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from math import gcd, lcm
from pathlib import Path
from typing import cast
import warnings

import numpy as np

from .triangulated_surface import SurfaceBounds as Bounds
from .triangulated_surface import TriangulatedSurface

OUTER_PATCH_NAMES = ("inlet", "outlet", "ymin", "ymax", "zmin", "zmax")


@dataclass(frozen=True)
class BoxRefinement:
    """Request a cell size inside an axis-aligned box.

    The effective size is the first dyadic level not larger than
    ``cell_size``.  For example, a 0.04 background and a 0.0125 request yield
    0.01 cells, matching cfMesh's octree-level behaviour.
    """

    bounds: Bounds
    cell_size: float
    name: str = "refinement"


@dataclass(frozen=True)
class _IntegerBox:
    x0: int
    x1: int
    y0: int
    y1: int
    z0: int
    z1: int

    def expanded(self, amount: int, limits: tuple[int, int, int]) -> _IntegerBox:
        return _IntegerBox(
            max(0, self.x0 - amount),
            min(limits[0], self.x1 + amount),
            max(0, self.y0 - amount),
            min(limits[1], self.y1 + amount),
            max(0, self.z0 - amount),
            min(limits[2], self.z1 + amount),
        )

    def overlaps(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> bool:
        return (
            x0 < self.x1
            and x1 > self.x0
            and y0 < self.y1
            and y1 > self.y0
            and z0 < self.z1
            and z1 > self.z0
        )

    def contains(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> bool:
        return (
            x0 >= self.x0
            and x1 <= self.x1
            and y0 >= self.y0
            and y1 <= self.y1
            and z0 >= self.z0
            and z1 <= self.z1
        )


def _validate_bounds(bounds: Bounds, name: str) -> None:
    if len(bounds) != 6 or not all(math.isfinite(float(value)) for value in bounds):
        raise ValueError(f"{name} must contain six finite coordinates")
    if not all(bounds[2 * axis] < bounds[2 * axis + 1] for axis in range(3)):
        raise ValueError(f"{name} must have positive extent along every axis: {bounds}")


def _dyadic_level(background: float, requested: float) -> int:
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError(f"Requested cell size must be finite and positive, got {requested}")
    if requested >= background:
        return 0
    ratio = background / requested
    return max(0, int(math.ceil(math.log2(ratio) - 1.0e-12)))


def _fitted_background_size(domain: Bounds, requested: float) -> float:
    """Return the largest isotropic spacing no larger than ``requested`` that tiles ``domain``."""
    extents = [
        Fraction(str(domain[2 * axis + 1] - domain[2 * axis])).limit_denominator(1_000_000)
        for axis in range(3)
    ]
    denominator = 1
    for extent in extents:
        denominator = lcm(denominator, extent.denominator)
    common_numerator = 0
    for extent in extents:
        scaled = extent.numerator * (denominator // extent.denominator)
        common_numerator = gcd(common_numerator, scaled)
    common_extent = Fraction(common_numerator, denominator)
    requested_fraction = Fraction(str(requested)).limit_denominator(1_000_000)
    count = math.ceil(common_extent / requested_fraction)
    return float(common_extent / max(1, count))


class AdaptiveCartesianMesher:
    """Build an adaptive, body-fitted Cartesian mesh directly in memory.

    Parameters
    ----------
    domain:
        Axis-aligned fluid-domain bounds ``(xmin, xmax, ymin, ymax, zmin,
        zmax)``.
    max_cell_size:
        Background cell size.  Every domain extent must be an integer multiple
        of this value, which keeps the outer coupling lattice exact.
    surface_file:
        Location of the watertight STL surface that defines the solid. The
        current Cartesian subset supports closed axis-aligned solids.
    wall_patch_name:
        Boundary-patch name assigned to faces extracted from ``surface_file``.
        It must be supplied together with the surface location.
    surface_cell_size:
        Requested size in the first fluid-cell layer around the STL surface.
        Refinement coarsens outwards in automatically generated 2:1 bands.
    refinements:
        Additional box refinements, such as a resolved wake region.
    merge_outer_patch:
        Merge the six outer sides into one patch (the coupled FVM--VPM case),
        or leave the conventional inlet/outlet/ymin/ymax/zmin/zmax patches.

    Notes
    -----
    Coarse cells next to finer cells are represented as valid polyhedra with
    multiple coplanar subfaces.  Geometrically all cells remain cuboids.  This
    is the same useful finite-volume topology as the nine-face transition
    body-fitted polyhedra, without an external intermediary.
    """

    def __init__(
        self,
        domain: Bounds,
        max_cell_size: float,
        *,
        surface_file: str | Path | None = None,
        wall_patch_name: str | None = None,
        surface_cell_size: float | None = None,
        refinements: tuple[BoxRefinement, ...] = (),
        merge_outer_patch: str | None = None,
        include_cell_vertices: bool = True,
    ) -> None:
        _validate_bounds(domain, "domain")
        if not math.isfinite(max_cell_size) or max_cell_size <= 0.0:
            raise ValueError("max_cell_size must be finite and positive")
        fitted_max_cell_size = _fitted_background_size(domain, max_cell_size)
        if (surface_file is None) != (wall_patch_name is None):
            raise ValueError("surface_file and wall_patch_name must be supplied together")
        if surface_file is None and surface_cell_size is not None:
            raise ValueError("surface_cell_size requires surface_file")
        if wall_patch_name is not None and not wall_patch_name.strip():
            raise ValueError("wall_patch_name must not be empty")

        surface = TriangulatedSurface.from_stl(surface_file) if surface_file is not None else None
        if surface is not None:
            surface_bounds = surface.bounds
            if not all(
                domain[2 * axis]
                < surface_bounds[2 * axis]
                < surface_bounds[2 * axis + 1]
                < domain[2 * axis + 1]
                for axis in range(3)
            ):
                raise ValueError("STL surface must lie strictly inside domain")
        if wall_patch_name is not None and merge_outer_patch == wall_patch_name:
            raise ValueError("Outer and wall patch names must differ")
        for refinement in refinements:
            _validate_bounds(refinement.bounds, f"{refinement.name}.bounds")
            if not all(
                domain[2 * axis]
                <= refinement.bounds[2 * axis]
                < refinement.bounds[2 * axis + 1]
                <= domain[2 * axis + 1]
                for axis in range(3)
            ):
                raise ValueError(f"{refinement.name} must lie inside domain")
            _dyadic_level(fitted_max_cell_size, refinement.cell_size)

        self.domain: Bounds = cast(Bounds, tuple(float(value) for value in domain))
        self.requested_max_cell_size = float(max_cell_size)
        self.max_cell_size = fitted_max_cell_size
        self.surface = surface
        self.surface_file = str(surface.path) if surface is not None else None
        self.surface_bounds = surface.bounds if surface is not None else None
        self.surface_cell_size = float(surface_cell_size) if surface_cell_size is not None else None
        self.refinements = tuple(refinements)
        self.wall_patch_name = wall_patch_name or ""
        self.merge_outer_patch = merge_outer_patch
        self.include_cell_vertices = include_cell_vertices

    def effective_cell_size(self, requested: float) -> float:
        """Return the dyadic size generated for a requested size."""
        return self.max_cell_size / (2 ** _dyadic_level(self.max_cell_size, requested))

    def _base_counts(self) -> tuple[int, int, int]:
        counts = []
        for axis in range(3):
            extent = self.domain[2 * axis + 1] - self.domain[2 * axis]
            count = int(round(extent / self.max_cell_size))
            if count < 1 or not math.isclose(
                count * self.max_cell_size,
                extent,
                rel_tol=1.0e-10,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError("Fitted Cartesian background size does not tile the domain")
            counts.append(count)
        return tuple(counts)  # type: ignore[return-value]

    def _integer_box(self, bounds: Bounds, h_min: float) -> _IntegerBox:
        indices = []
        for axis in range(3):
            origin = self.domain[2 * axis]
            for side in range(2):
                value = (bounds[2 * axis + side] - origin) / h_min
                rounded = int(round(value))
                if math.isclose(value, rounded, rel_tol=1.0e-9, abs_tol=1.0e-8):
                    index = rounded
                elif side == 0:
                    index = math.floor(value)
                else:
                    index = math.ceil(value)
                indices.append(index)
        return _IntegerBox(*indices)

    def _warn_if_body_inflated(self, solid: _IntegerBox, h_min: float) -> None:
        """Warn when snapping the STL outward resizes the body.

        ``_integer_box`` floors the lower bound and ceils the upper one, so a
        surface that does not land on a finest-level cell boundary is absorbed
        outward.  The solver then sees a body larger than the STL while force
        coefficients are still divided by the nominal reference area, which
        biases every reported Cd.  Choose sizes that put the surface on a cell
        boundary instead.
        """
        bounds = self.surface_bounds
        if bounds is None:
            return
        corners = ((solid.x0, solid.x1), (solid.y0, solid.y1), (solid.z0, solid.z1))
        exact = [bounds[2 * a + 1] - bounds[2 * a] for a in range(3)]
        snapped = [(hi - lo) * h_min for lo, hi in corners]
        if any(span <= 0.0 for span in exact):
            return
        # Frontal area sets the pressure-drag scale, so report the worst of the
        # three projections rather than the volume.
        frontal = max(
            (snapped[(a + 1) % 3] * snapped[(a + 2) % 3])
            / (exact[(a + 1) % 3] * exact[(a + 2) % 3])
            for a in range(3)
        )
        if frontal - 1.0 <= 0.01:
            return
        warnings.warn(
            f"STL snapped outward to the finest cell ({h_min:.6g}): body spans "
            f"{snapped[0]:.6g} x {snapped[1]:.6g} x {snapped[2]:.6g} instead of "
            f"{exact[0]:.6g} x {exact[1]:.6g} x {exact[2]:.6g}, inflating frontal "
            f"area by {100.0 * (frontal - 1.0):.2f}%. Force coefficients divided by "
            f"the nominal area are biased by the same factor.",
            stacklevel=2,
        )

    def _refinement_regions(
        self,
        h_min: float,
        max_level: int,
        limits: tuple[int, int, int],
        solid: _IntegerBox | None,
    ) -> tuple[tuple[_IntegerBox, int], ...]:
        regions: list[tuple[_IntegerBox, int]] = []

        def add_balanced(box: _IntegerBox, target_level: int, first_padding: int = 0) -> None:
            current = box.expanded(first_padding, limits) if first_padding else box
            regions.append((current, target_level))
            for level in range(target_level - 1, 0, -1):
                # One cell at the next-coarser level gives an explicit 2:1
                # buffer before the following level is allowed.
                current = current.expanded(2 ** (max_level - level), limits)
                regions.append((current, level))

        if solid is not None and self.surface_cell_size is not None:
            surface_level = _dyadic_level(self.max_cell_size, self.surface_cell_size)
            if surface_level:
                fine_cell = 2 ** (max_level - surface_level)
                add_balanced(solid, surface_level, first_padding=fine_cell)

        for refinement in self.refinements:
            level = _dyadic_level(self.max_cell_size, refinement.cell_size)
            if level:
                add_balanced(self._integer_box(refinement.bounds, h_min), level)

        # Higher levels take precedence in overlap regions.
        regions.sort(key=lambda item: item[1], reverse=True)
        return tuple(regions)

    def _build_leaves(
        self,
        base_counts: tuple[int, int, int],
        max_level: int,
        solid: _IntegerBox | None,
        regions: tuple[tuple[_IntegerBox, int], ...],
    ) -> np.ndarray:
        base_width = 2**max_level
        # A Python tuple/list representation costs several times the 20 bytes
        # of actual octree data per cell.  Grow one packed numeric buffer so a
        # multi-million-cell wake mesh does not spend gigabytes on objects
        # before topology construction even begins.
        capacity = max(1024, math.prod(base_counts))
        leaves = np.empty((capacity, 5), dtype=np.int32)
        n_leaves = 0

        def append_leaf(x0: int, y0: int, z0: int, width: int, level: int) -> None:
            nonlocal capacity, leaves, n_leaves
            if n_leaves == capacity:
                new_capacity = int(capacity * 1.5) + 1
                grown = np.empty((new_capacity, 5), dtype=np.int32)
                grown[:n_leaves] = leaves[:n_leaves]
                leaves = grown
                capacity = new_capacity
            leaves[n_leaves] = (x0, y0, z0, width, level)
            n_leaves += 1

        def visit(x0: int, y0: int, z0: int, width: int, level: int) -> None:
            x1, y1, z1 = x0 + width, y0 + width, z0 + width
            if solid is not None and solid.contains(x0, x1, y0, y1, z0, z1):
                return

            target = level
            for region, region_level in regions:
                if region_level <= target:
                    break
                if region.overlaps(x0, x1, y0, y1, z0, z1):
                    target = region_level
                    break

            if level < target:
                child = width // 2
                for dz in (0, child):
                    for dy in (0, child):
                        for dx in (0, child):
                            visit(x0 + dx, y0 + dy, z0 + dz, child, level + 1)
                return

            if solid is not None and solid.overlaps(x0, x1, y0, y1, z0, z1):
                # Arbitrary cut cells require the surface-mapping stage that is
                # outside this Cartesian subset. Axis-aligned STL planes must
                # therefore coincide with the finest template level.
                centre_inside = (
                    2 * solid.x0 < 2 * x0 + width < 2 * solid.x1
                    and 2 * solid.y0 < 2 * y0 + width < 2 * solid.y1
                    and 2 * solid.z0 < 2 * z0 + width < 2 * solid.z1
                )
                if centre_inside:
                    return
                raise ValueError(
                    "STL surface cuts a leaf cell; request a surface_cell_size that "
                    "aligns the surface with the finest Cartesian level"
                )
            append_leaf(x0, y0, z0, width, level)

        nx, ny, nz = base_counts
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    visit(i * base_width, j * base_width, k * base_width, base_width, 0)

        if not n_leaves:
            raise ValueError("Meshing configuration removed every fluid cell")
        return leaves[:n_leaves].copy()

    @staticmethod
    def _face_codes(
        axis: int,
        coordinate: int,
        a0: int,
        a1: int,
        b0: int,
        b1: int,
        strides: tuple[int, int],
        positive: bool,
    ) -> tuple[int, int, int, int]:
        """Encoded point ring with normal along ``axis`` and sign."""
        sx, sy = strides

        def code(x: int, y: int, z: int) -> int:
            return x + sx * (y + sy * z)

        if axis == 0:
            ring = (
                code(coordinate, a0, b0),
                code(coordinate, a1, b0),
                code(coordinate, a1, b1),
                code(coordinate, a0, b1),
            )
        elif axis == 1:
            ring = (
                code(a0, coordinate, b0),
                code(a0, coordinate, b1),
                code(a1, coordinate, b1),
                code(a1, coordinate, b0),
            )
        else:
            ring = (
                code(a0, b0, coordinate),
                code(a1, b0, coordinate),
                code(a1, b1, coordinate),
                code(a0, b1, coordinate),
            )
        return ring if positive else (ring[3], ring[2], ring[1], ring[0])

    def _extract_topology(
        self,
        leaves: np.ndarray,
        max_level: int,
        limits: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], np.ndarray]:
        n_cells = len(leaves)
        nx, ny, nz = limits
        max_point_code = (nx + 1) * (ny + 1) * (nz + 1) - 1
        code_dtype = np.int32 if max_point_code <= np.iinfo(np.int32).max else np.int64
        point_strides = (nx + 1, ny + 1)
        cell_stride_x, cell_stride_y = nx, ny

        level_maps: list[dict[int, int]] = [{} for _ in range(max_level + 1)]
        for cell_id, (x0, y0, z0, _width, level) in enumerate(leaves):
            key = int(x0) + cell_stride_x * (int(y0) + cell_stride_y * int(z0))
            level_maps[int(level)][key] = cell_id

        levels = np.asarray(leaves[:, 4], dtype=np.int8)

        def find_cell(x: int, y: int, z: int) -> int:
            if x < 0 or y < 0 or z < 0 or x >= nx or y >= ny or z >= nz:
                return -1
            for level in range(max_level, -1, -1):
                width = 2 ** (max_level - level)
                ox, oy, oz = (x // width) * width, (y // width) * width, (z // width) * width
                key = ox + cell_stride_x * (oy + cell_stride_y * oz)
                found = level_maps[level].get(key)
                if found is not None:
                    return found
            return -1

        # A conforming Cartesian volume has about three internal faces per
        # cell.  The allowance covers 2:1 transition subfaces without building
        # millions of Python tuples.
        capacity = max(16, int(math.ceil(3.35 * n_cells)))
        interior_codes = np.empty((capacity, 4), dtype=code_dtype)
        interior_owners = np.empty(capacity, dtype=np.int32)
        interior_neighbours = np.empty(capacity, dtype=np.int32)
        n_interior = 0

        boundary_names = (self.merge_outer_patch,) if self.merge_outer_patch else OUTER_PATCH_NAMES
        patch_codes: dict[str, list[tuple[int, int, int, int]]] = {
            name: [] for name in boundary_names
        }
        patch_owners: dict[str, list[int]] = {name: [] for name in boundary_names}
        if self.surface is not None:
            patch_codes[self.wall_patch_name] = []
            patch_owners[self.wall_patch_name] = []

        def grow() -> None:
            nonlocal capacity, interior_codes, interior_owners, interior_neighbours
            new_capacity = int(capacity * 1.35) + 1
            new_codes = np.empty((new_capacity, 4), dtype=code_dtype)
            new_codes[:capacity] = interior_codes
            interior_codes = new_codes
            new_owners = np.empty(new_capacity, dtype=np.int32)
            new_owners[:capacity] = interior_owners
            interior_owners = new_owners
            new_neighbours = np.empty(new_capacity, dtype=np.int32)
            new_neighbours[:capacity] = interior_neighbours
            interior_neighbours = new_neighbours
            capacity = new_capacity

        def emit_interior(
            owner: int,
            neighbour: int,
            axis: int,
            coordinate: int,
            a0: int,
            a1: int,
            b0: int,
            b1: int,
        ) -> None:
            nonlocal n_interior
            if n_interior == capacity:
                grow()
            interior_codes[n_interior] = self._face_codes(
                axis, coordinate, a0, a1, b0, b1, point_strides, True
            )
            interior_owners[n_interior] = owner
            interior_neighbours[n_interior] = neighbour
            n_interior += 1

        def patch_for(axis: int, positive: bool, coordinate: int) -> str:
            limit = limits[axis] if positive else 0
            if coordinate == limit:
                if self.merge_outer_patch:
                    return self.merge_outer_patch
                return OUTER_PATCH_NAMES[2 * axis + int(positive)]
            if self.surface is None:
                raise RuntimeError("Adaptive mesh contains an unexplained internal boundary")
            return self.wall_patch_name

        def samples(lo: int, width: int) -> tuple[int, ...]:
            if width == 1:
                return (lo,)
            return (lo + width // 4, lo + (3 * width) // 4)

        for owner, (x0v, y0v, z0v, widthv, levelv) in enumerate(leaves):
            x0, y0, z0, width, level = map(int, (x0v, y0v, z0v, widthv, levelv))
            x1, y1, z1 = x0 + width, y0 + width, z0 + width
            origins = (x0, y0, z0)
            ends = (x1, y1, z1)
            tangential = ((y0, y1, z0, z1), (x0, x1, z0, z1), (x0, x1, y0, y1))

            for axis in range(3):
                a0, a1, b0, b1 = tangential[axis]
                query = [x0, y0, z0]
                query[axis] = ends[axis]
                neighbour_ids: set[int] = set()
                for a in samples(a0, width):
                    for b in samples(b0, width):
                        if axis == 0:
                            candidate = find_cell(query[0], a, b)
                        elif axis == 1:
                            candidate = find_cell(a, query[1], b)
                        else:
                            candidate = find_cell(a, b, query[2])
                        if candidate >= 0:
                            neighbour_ids.add(candidate)

                if neighbour_ids:
                    for neighbour in sorted(neighbour_ids):
                        nx0, ny0, nz0, nw, neighbour_level = map(int, leaves[neighbour])
                        if abs(level - neighbour_level) > 1:
                            raise RuntimeError(
                                "Refinement transition is not 2:1 balanced between cells "
                                f"{owner} (level {level}) and {neighbour} "
                                f"(level {neighbour_level})"
                            )
                        if axis == 0:
                            emit_interior(
                                owner,
                                neighbour,
                                axis,
                                x1,
                                max(y0, ny0),
                                min(y1, ny0 + nw),
                                max(z0, nz0),
                                min(z1, nz0 + nw),
                            )
                        elif axis == 1:
                            emit_interior(
                                owner,
                                neighbour,
                                axis,
                                y1,
                                max(x0, nx0),
                                min(x1, nx0 + nw),
                                max(z0, nz0),
                                min(z1, nz0 + nw),
                            )
                        else:
                            emit_interior(
                                owner,
                                neighbour,
                                axis,
                                z1,
                                max(x0, nx0),
                                min(x1, nx0 + nw),
                                max(y0, ny0),
                                min(y1, ny0 + nw),
                            )
                else:
                    name = patch_for(axis, True, ends[axis])
                    patch_codes[name].append(
                        self._face_codes(axis, ends[axis], a0, a1, b0, b1, point_strides, True)
                    )
                    patch_owners[name].append(owner)

                # Negative faces are needed only at a physical boundary;
                # interior faces are emitted once by the cell on the low side.
                query[axis] = origins[axis] - 1
                negative_neighbours: set[int] = set()
                for a in samples(a0, width):
                    for b in samples(b0, width):
                        if axis == 0:
                            candidate = find_cell(query[0], a, b)
                        elif axis == 1:
                            candidate = find_cell(a, query[1], b)
                        else:
                            candidate = find_cell(a, b, query[2])
                        if candidate >= 0:
                            negative_neighbours.add(candidate)
                if not negative_neighbours:
                    name = patch_for(axis, False, origins[axis])
                    patch_codes[name].append(
                        self._face_codes(axis, origins[axis], a0, a1, b0, b1, point_strides, False)
                    )
                    patch_owners[name].append(owner)

        face_blocks = [interior_codes[:n_interior]]
        owner_blocks = [interior_owners[:n_interior]]
        boundary: list[dict] = []
        start = n_interior
        ordered_names = list(boundary_names)
        if self.surface is not None:
            ordered_names.append(self.wall_patch_name)
        for name in ordered_names:
            codes = np.asarray(patch_codes[name], dtype=code_dtype).reshape(-1, 4)
            owners = np.asarray(patch_owners[name], dtype=np.int32)
            face_blocks.append(codes)
            owner_blocks.append(owners)
            boundary.append(
                {
                    "name": name,
                    "startFace": start,
                    "nFaces": len(codes),
                    "type": "wall" if name == self.wall_patch_name else "patch",
                }
            )
            start += len(codes)

        encoded_faces = np.ascontiguousarray(np.vstack(face_blocks), dtype=code_dtype)
        owners = np.ascontiguousarray(np.concatenate(owner_blocks), dtype=np.int32)
        neighbour_array = np.ascontiguousarray(interior_neighbours[:n_interior], dtype=np.int32)
        return encoded_faces, owners, neighbour_array, boundary, levels

    def build(self) -> dict:
        """Generate and return a solver-native ``mesh_data`` dictionary."""
        base_counts = self._base_counts()
        requested_sizes = [refinement.cell_size for refinement in self.refinements]
        if self.surface_cell_size is not None:
            requested_sizes.append(self.surface_cell_size)
        max_level = max(
            (_dyadic_level(self.max_cell_size, size) for size in requested_sizes),
            default=0,
        )
        h_min = self.max_cell_size / (2**max_level)
        base_width = 2**max_level
        limits = tuple(count * base_width for count in base_counts)
        solid = (
            self._integer_box(self.surface_bounds, h_min)
            if self.surface_bounds is not None
            else None
        )
        if solid is not None:
            self._warn_if_body_inflated(solid, h_min)
        regions = self._refinement_regions(h_min, max_level, limits, solid)
        leaves = self._build_leaves(base_counts, max_level, solid, regions)
        encoded_faces, owners, neighbours, boundary, levels = self._extract_topology(
            leaves, max_level, limits
        )

        point_codes = np.unique(encoded_faces)
        if len(point_codes) > np.iinfo(np.int32).max:
            raise MemoryError("Adaptive mesh has too many points for int32 face connectivity")
        faces = np.empty(encoded_faces.shape, dtype=np.int32)
        for start in range(0, len(encoded_faces), 250_000):
            stop = min(start + 250_000, len(encoded_faces))
            faces[start:stop] = np.searchsorted(point_codes, encoded_faces[start:stop]).astype(
                np.int32
            )
        del encoded_faces

        nx, ny, _nz = limits
        sx, sy = nx + 1, ny + 1
        px = point_codes % sx
        yz = point_codes // sx
        py = yz % sy
        pz = yz // sy
        points = np.column_stack((px, py, pz)).astype(np.float64)
        points *= h_min
        points += np.asarray(self.domain[::2], dtype=np.float64)

        mesh_data = {
            "points": np.ascontiguousarray(points),
            "faces": faces,
            "owners": owners,
            "neighbours": neighbours,
            "boundary": boundary,
            "n_elements": len(leaves),
            "n_faces": len(faces),
            "n_interior_faces": len(neighbours),
            "n_points": len(points),
            "cell_levels": levels,
            "cell_sizes": np.asarray(self.max_cell_size / np.power(2.0, levels), dtype=np.float32),
            "mesh_generation": {
                "method": "adaptive_cartesian",
                "max_cell_size": self.max_cell_size,
                "requested_max_cell_size": self.requested_max_cell_size,
                "finest_cell_size": h_min,
                "max_level": max_level,
                "base_counts": base_counts,
                "surface_file": self.surface_file,
                "surface_sha256": self.surface.sha256 if self.surface is not None else None,
                "surface_bounds": self.surface_bounds,
                "surface_triangle_count": (
                    len(self.surface.triangles) if self.surface is not None else 0
                ),
                "wall_patch_name": self.wall_patch_name or None,
                "attribution": "Inspired by cfMesh cartesianMesh (Franjo Juretic / Creative Fields)",
            },
        }

        if self.include_cell_vertices:
            # All geometrical cells are cuboids even where a coarse face is
            # topologically split.  Native hex vertices keep VTK output compact.
            cell_vertices = np.empty((len(leaves), 8), dtype=np.int32)
            for start in range(0, len(leaves), 250_000):
                stop = min(start + 250_000, len(leaves))
                block = leaves[start:stop]
                x0, y0, z0, width = (block[:, index].astype(np.int64) for index in range(4))
                x1, y1, z1 = x0 + width, y0 + width, z0 + width

                def encode(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
                    return x + sx * (y + sy * z)

                codes = np.column_stack(
                    (
                        encode(x0, y0, z0),
                        encode(x1, y0, z0),
                        encode(x1, y1, z0),
                        encode(x0, y1, z0),
                        encode(x0, y0, z1),
                        encode(x1, y0, z1),
                        encode(x1, y1, z1),
                        encode(x0, y1, z1),
                    )
                )
                indices = np.searchsorted(point_codes, codes)
                if np.any(indices >= len(point_codes)) or not np.array_equal(
                    point_codes[indices], codes
                ):
                    raise RuntimeError("A Cartesian cell corner is absent from the face points")
                cell_vertices[start:stop] = indices.astype(np.int32)
            mesh_data["cell_vertices"] = cell_vertices
            mesh_data["cell_type_codes"] = np.full(len(leaves), 5, dtype=np.int32)

        return mesh_data

    def __call__(self) -> dict:
        """Allow the mesher itself to be passed to ``setup_fvm_solver``."""
        return self.build()


def adaptive_cartesian_mesh(
    domain: Bounds,
    max_cell_size: float,
    **kwargs,
) -> dict:
    """Functional wrapper around :class:`AdaptiveCartesianMesher`."""
    return AdaptiveCartesianMesher(domain, max_cell_size, **kwargs).build()


__all__ = ["AdaptiveCartesianMesher", "Bounds", "BoxRefinement", "adaptive_cartesian_mesh"]
