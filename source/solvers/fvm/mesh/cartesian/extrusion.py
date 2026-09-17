"""Conforming prism meshes from a planar section of a Cartesian volume mesh."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import compute_mesh_geometry
from ..progress import mesh_stage
from ..surface_classification import SurfaceIndex
from ..validation import (
    validate_cell_area_closure,
    validate_geometry,
    validate_single_fluid_component,
    validate_topology,
    validate_wall_vertex_conformance,
)
from .config import BoundaryLayers, BoxDomain
from .mesher import CartesianMesher


def _section_edges(polygons):
    """Return oriented polygon adjacencies keyed by undirected section edge."""
    edges: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for cell, ids in enumerate(polygons):
        for a, b in zip(ids, np.roll(ids, -1), strict=True):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append((cell, int(a), int(b)))
    for adjacent in edges.values():
        if len(adjacent) == 2:
            if adjacent[0][1:] != adjacent[1][1:][::-1]:
                raise ValueError("Section has overlapping polygon windings")
        elif len(adjacent) != 1:
            raise ValueError("Section has a non-manifold edge")
    return edges


def _merge_projected_points(points, polygons, source_cells, *, tolerance=1.0e-10):
    """Merge wall vertices coincident after projection and remove collapsed cells."""
    key_to_point: dict[tuple[int, int], int] = {}
    remap = np.empty(len(points), dtype=np.int32)
    merged_points: list[np.ndarray] = []
    for index, point in enumerate(points):
        key = tuple(np.rint(point[:2] / tolerance).astype(np.int64))
        target = key_to_point.get(key)
        if target is None:
            target = len(merged_points)
            key_to_point[key] = target
            merged_points.append(point)
        remap[index] = target

    retained_polygons = []
    retained_sources = []
    for ids, source_cell in zip(polygons, source_cells, strict=True):
        mapped = remap[ids]
        compact = [int(mapped[0])]
        compact.extend(int(value) for value in mapped[1:] if int(value) != compact[-1])
        if len(compact) > 1 and compact[0] == compact[-1]:
            compact.pop()
        if len(compact) < 3:
            continue
        compact_array = np.asarray(compact, dtype=np.int32)
        xy = np.asarray(merged_points)[compact_array, :2]
        signed_area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))
        if abs(signed_area) <= 1.0e-14:
            continue
        retained_polygons.append(compact_array if signed_area > 0.0 else compact_array[::-1])
        retained_sources.append(source_cell)
    if not retained_polygons:
        raise ValueError("Surface projection collapsed the complete mesh section")
    return (
        np.asarray(merged_points, dtype=float),
        retained_polygons,
        np.asarray(retained_sources, dtype=int),
    )


def _section_boundary_name(xy, names, bounds, indices):
    """Classify a section edge by proximity to a box side or immersed surface."""
    box_distances = [np.max(np.abs(xy[:, side // 2] - bounds[side])) for side in range(4)]
    box_side = int(np.argmin(box_distances))
    if not indices:
        return names[box_side]
    centre = xy.mean(axis=0)
    surface_name, surface_distance = min(
        ((name, float(index.nearest_point(centre)[1])) for name, index in indices.items()),
        key=lambda item: item[1],
    )
    return names[box_side] if box_distances[box_side] <= surface_distance else surface_name


def extrude_mesh_section(mesh, *, coordinate, levels, domain, surfaces=()):
    """Extrude a z-normal section, preserving shared edges and named patches.

    The caller explicitly selects a span-invariant geometry. Conformance is
    checked on every extruded wall vertex, including both end planes. No
    recognition of a particular body shape is performed.
    """
    from ...io.vtk_exporter import VTKExporter

    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or len(levels) < 2 or not np.all(np.isfinite(levels)):
        raise ValueError("Extrusion levels must be a finite increasing sequence")
    if np.any(np.diff(levels) <= 0):
        raise ValueError("Extrusion levels must be strictly increasing")
    if not np.allclose(levels[[0, -1]], domain.bounds[4:6], rtol=0, atol=1e-12):
        raise ValueError("Extrusion levels must span the requested domain")
    if not mesh["vertex_position"][:, 2].min() < coordinate < mesh["vertex_position"][:, 2].max():
        raise ValueError("Section coordinate must lie inside the source mesh")
    grid = VTKExporter(mesh)._grid
    grid.cell_data["source_cell"] = np.arange(mesh["n_cells"])
    section = grid.slice(normal="z", origin=(0, 0, coordinate)).clean(
        tolerance=1e-10, absolute=True
    )
    points = np.asarray(section.points, dtype=float).copy()
    source_cells = np.asarray(section.cell_data["source_cell"], dtype=int)
    polygons = []
    packed = section.faces
    cursor = 0
    while cursor < len(packed):
        count = int(packed[cursor])
        ids = packed[cursor + 1 : cursor + count + 1].astype(np.int32)
        xy = points[ids, :2]
        signed_area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))
        if count < 3 or abs(signed_area) <= 1e-14:
            raise ValueError("Section contains a degenerate polygon")
        polygons.append(ids if signed_area > 0 else ids[::-1])
        cursor += count + 1
    if not polygons or len(polygons) != len(source_cells):
        raise ValueError("Source section is empty or contains non-polygon cells")

    edges = _section_edges(polygons)
    names = domain.patches.as_tuple()
    indices = {surface.patch: SurfaceIndex.build(surface.triangles) for surface in surfaces}
    edge_patches = {}
    patch_points: dict[str, set[int]] = {}
    for key, adjacent in edges.items():
        if len(adjacent) == 2:
            continue
        xy = points[list(key)]
        name = _section_boundary_name(xy, names, domain.bounds, indices)
        edge_patches[key] = name
        patch_points.setdefault(name, set()).update(key)

    for name, ids_set in patch_points.items():
        ids = np.array(sorted(ids_set), dtype=int)
        if name in indices:
            mapped, _, _ = indices[name].nearest_points(points[ids])
            if np.max(np.abs(mapped[:, 2] - coordinate)) > 1e-8:
                raise ValueError("Surface projection leaves the selected section plane")
            points[ids, :2] = mapped[:, :2]
        else:
            candidate_sides = [side for side in range(4) if names[side] == name]
            distances = np.column_stack(
                [np.abs(points[ids, side // 2] - domain.bounds[side]) for side in candidate_sides]
            )
            nearest = distances.min(axis=1)
            for column, side in enumerate(candidate_sides):
                on_side = distances[:, column] <= nearest + 1.0e-8
                points[ids[on_side], side // 2] = domain.bounds[side]

    points, polygons, source_cells = _merge_projected_points(points, polygons, source_cells)
    edges = _section_edges(polygons)
    edge_patches = {}
    for key, adjacent in edges.items():
        if len(adjacent) == 2:
            continue
        xy = points[list(key)]
        edge_patches[key] = _section_boundary_name(xy, names, domain.bounds, indices)

    np2, nc2, nz = len(points), len(polygons), len(levels) - 1
    vertices = np.vstack([np.column_stack((points[:, :2], np.full(np2, z))) for z in levels])
    interior, boundary_faces = [], {name: [] for name in (*names, *indices)}
    for layer in range(nz):
        offset = layer * np2
        for key, adjacent in edges.items():
            cell, a, b = adjacent[0]
            face = np.array(
                (a + offset, b + offset, b + offset + np2, a + offset + np2), dtype=np.int32
            )
            owner = layer * nc2 + cell
            if len(adjacent) == 2:
                interior.append((face, owner, layer * nc2 + adjacent[1][0]))
            else:
                boundary_faces[edge_patches[key]].append((face, owner))
    for layer in range(1, nz):
        interior.extend(
            (ids + layer * np2, (layer - 1) * nc2 + cell, layer * nc2 + cell)
            for cell, ids in enumerate(polygons)
        )
    boundary_faces[names[4]].extend((ids[::-1], cell) for cell, ids in enumerate(polygons))
    boundary_faces[names[5]].extend(
        (ids + nz * np2, (nz - 1) * nc2 + cell) for cell, ids in enumerate(polygons)
    )
    faces = [item[0] for item in interior]
    owners = [item[1] for item in interior]
    neighbours = np.array([item[2] for item in interior], dtype=np.int32)
    patches = []
    for name, entries in boundary_faces.items():
        if not entries:
            continue
        patches.append(
            {
                "name": name,
                "type": "wall" if name in indices else "patch",
                "start_face": len(faces),
                "n_faces": len(entries),
            }
        )
        faces.extend(item[0] for item in entries)
        owners.extend(item[1] for item in entries)
    result = {
        "vertex_position": vertices,
        "faces": faces,
        "owners": np.asarray(owners, dtype=np.int32),
        "neighbours": neighbours,
        "boundary": patches,
        "n_cells": nc2 * nz,
        "n_faces": len(faces),
        "n_interior_faces": len(interior),
        "n_points": len(vertices),
    }
    for key in ("cell_sizes", "cell_levels", "boundary_layer_index"):
        if key in mesh:
            result[key] = np.tile(np.asarray(mesh[key])[source_cells], nz)
    validate_topology(result)
    geometry = compute_mesh_geometry(result, compute_lsq=False)
    validate_geometry(result, geometry)
    validate_cell_area_closure(result, geometry)
    validate_single_fluid_component(result)
    for surface in surfaces:
        validate_wall_vertex_conformance(result, surface.triangles, surface.patch)
    result["mesh_generation"] = {
        "method": "cartesian_cfmesh_extruded_section",
        "section_coordinate": float(coordinate),
        "section_cells": nc2,
        "extrusion_levels": levels.tolist(),
        "source_cells": int(mesh["n_cells"]),
        "source_workflow": mesh.get("mesh_generation", {}),
    }
    return result


@dataclass
class ExtrudedCartesianMesher:
    """Build a conforming quasi-2D mesh from a Cartesian volume-mesh section.

    Parameters
    ----------
    source : CartesianMesher
        Configured three-dimensional source workflow. ``build`` runs it through
        ``meshOptimisation`` and takes a z-normal planar section.
    domain : BoxDomain
        Output domain. Its z bounds must equal the first and last ``levels``;
        x/y bounds and patch names classify the section perimeter.
    levels : tuple[float, ...]
        Strictly increasing z coordinates in m. Consecutive values define the
        extrusion layers and need not be uniformly spaced.

    Notes
    -----
    Construction stores references only. :meth:`build` performs source mesh
    generation, sectioning, extrusion, validation, and optional callback
    invocation; it returns newly allocated native mesh data.
    """

    source: CartesianMesher
    domain: BoxDomain
    levels: tuple[float, ...]

    @property
    def max_cell_size(self) -> float:
        """Background maximum cell size inherited from ``source``, in m."""
        return self.source.max_cell_size

    @property
    def boundary_layers(self) -> tuple[BoundaryLayers, ...]:
        """Wall-normal layer requests inherited from the source mesher."""
        return self.source.boundary_layers

    def effective_cell_size(self, requested: float) -> float:
        """Return the source octree's realizable dyadic size for a request in m."""
        return self.source.effective_cell_size(requested)

    def build(self, *, on_generated=None) -> dict[str, object]:
        """Generate, section, extrude, validate, and return a native FVM mesh.

        Parameters
        ----------
        on_generated : callable or None, optional
            Callback invoked once with the completed mutable mesh mapping. The
            callback may inspect or persist it; its return value is ignored.

        Returns
        -------
        dict[str, object]
            Native mesh connectivity and provenance. Vertex coordinates have
            shape ``(n_vertices, 3)`` in m; owner/neighbour indices use the
            standard owner-to-neighbour face orientation.

        Raises
        ------
        ValueError
            If section selection, extrusion levels, resolved sizes, topology,
            geometry, or wall conformance is invalid.

        Notes
        -----
        This method performs the expensive source build and invokes
        ``on_generated`` after validation. It does not create solver fields.
        """
        # Select the planar interior before the unrelated end-rim correction.
        with mesh_stage("source mesh generation"):
            raw = self.source.build(stop_after="meshOptimisation")
        dx = min(item.cell_size for item in self.source.patch_refinements)
        # Stay inside a finest-size slab, clear of its central wrapper
        # transition. Near a transition, a planar cut can graze a sliver
        # even though the original three-dimensional cell is well shaped.
        coordinate = 0.75 * dx
        with mesh_stage("section extrusion") as progress:
            result = extrude_mesh_section(
                raw,
                coordinate=coordinate,
                levels=self.levels,
                domain=self.domain,
                surfaces=self.source.surfaces,
            )
            progress.details(
                cells=result.get("n_cells"),
                faces=result.get("n_faces"),
                points=result.get("n_points"),
            )
        generation = result["mesh_generation"]
        native = raw["mesh_generation"]
        root_size = native["root_box"][1] - native["root_box"][0]
        background = root_size / 2 ** native["global_refinement_level"]
        boundary = root_size / 2 ** native["boundary_refinement_level"]
        patch_sizes = {
            name: root_size / 2**level
            for name, level in native["surface_patch_refinement_levels"].items()
        }
        if not np.isclose(background, self.max_cell_size, rtol=1e-12, atol=0):
            raise ValueError("Source mesh changed the requested background spacing")
        sizes = [("background", self.max_cell_size), ("boundary", self.source.boundary_cell_size)]
        sizes += [(item.name, item.cell_size) for item in self.source.refinements]
        sizes += [("patch:" + item.patch, item.cell_size) for item in self.source.patch_refinements]
        generation.update(
            resolved_background_cell_size=background,
            resolved_boundary_cell_size=boundary,
            resolved_surface_patch_sizes=patch_sizes,
            requested_sizes=[
                {"name": name, "requested": size, "effective": self.effective_cell_size(size)}
                for name, size in sizes
            ],
            source_domain=list(self.source.domain.bounds),
            domain=list(self.domain.bounds),
        )
        if on_generated is not None:
            on_generated(result)
        return result

    __call__ = build
