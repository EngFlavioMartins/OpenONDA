"""Surface-conforming volume meshes with boundary layers for general STL bodies.

This module completes the general-body stage missing from OpenONDA's
cfMesh-inspired Cartesian extractor.  It follows the same high-level workflow
as cfMesh: classify sharp surface features, advance a graded prismatic layer
along surface normals, transition the requested surface size into the volume,
fill the remaining domain, and convert the result directly to the FVM solver's
face-based mesh representation.

The advancing-layer and core fill are executed through the Gmsh library API
that is already a required OpenONDA dependency.  No external executable,
temporary solver case, or pre-generated ``.msh`` file is involved.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING
import uuid

import numpy as np

from .boundary_layer import BoundaryLayerSpec
from .gmsh_importer import GmshImporter
from .triangulated_surface import SurfaceBounds as Bounds
from .triangulated_surface import TriangulatedSurface
from .validation import validate_curved_wall_conformance, validate_topology

if TYPE_CHECKING:
    from .adaptive_cartesian import BoxRefinement

try:
    import gmsh
except ImportError:  # pragma: no cover - covered by the public dependency error
    gmsh = None


_OUTER_SURFACES = (
    "zmin",
    "zmax",
    "ymin",
    "outlet",
    "ymax",
    "inlet",
)


def _validate_bounds(bounds: Bounds, name: str) -> None:
    if len(bounds) != 6 or not all(math.isfinite(float(value)) for value in bounds):
        raise ValueError(f"{name} must contain six finite coordinates")
    if not all(bounds[2 * axis] < bounds[2 * axis + 1] for axis in range(3)):
        raise ValueError(f"{name} must have positive extent along every axis: {bounds}")


def _signed_surface_volume(triangles: np.ndarray) -> float:
    """Signed volume enclosed by a consistently wound triangle surface."""
    return float(
        np.einsum(
            "ij,ij->i",
            triangles[:, 0],
            np.cross(triangles[:, 1], triangles[:, 2]),
        ).sum()
        / 6.0
    )


def _add_outer_box(model, domain: Bounds, mesh_size: float) -> list[int]:
    """Create the six independently named outer-domain surfaces."""
    geo = model.geo
    xmin, xmax, ymin, ymax, zmin, zmax = domain
    coordinates = (
        (xmin, ymin, zmin),
        (xmax, ymin, zmin),
        (xmax, ymax, zmin),
        (xmin, ymax, zmin),
        (xmin, ymin, zmax),
        (xmax, ymin, zmax),
        (xmax, ymax, zmax),
        (xmin, ymax, zmax),
    )
    points = [geo.addPoint(*coordinate, mesh_size) for coordinate in coordinates]
    edges = [
        geo.addLine(points[start], points[stop])
        for start, stop in (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
    ]
    loops = [
        geo.addCurveLoop(tags)
        for tags in (
            (edges[0], edges[1], edges[2], edges[3]),
            (edges[4], edges[5], edges[6], edges[7]),
            (edges[0], edges[9], -edges[4], -edges[8]),
            (edges[1], edges[10], -edges[5], -edges[9]),
            (edges[2], edges[11], -edges[6], -edges[10]),
            (edges[3], edges[8], -edges[7], -edges[11]),
        )
    ]
    return [geo.addPlaneSurface([loop]) for loop in loops]


def _prism_layer_indices(mesh: dict, wall_patch_name: str, layers: int) -> np.ndarray:
    """Recover exact prism-layer numbers from triangular through-thickness faces."""
    cell_types = np.asarray(mesh["cell_type_code"], dtype=np.int32)
    prism = cell_types == 6
    result = np.full(int(mesh["n_cells"]), -1, dtype=np.int16)
    wall = next(patch for patch in mesh["boundary"] if patch["name"] == wall_patch_name)
    start = int(wall["start_face"])
    stop = start + int(wall["n_faces"])
    wall_owners = np.unique(np.asarray(mesh["owners"], dtype=np.int32)[start:stop])
    if not np.all(prism[wall_owners]):
        raise RuntimeError("A general-body wall face is not owned by a prismatic layer cell")
    result[wall_owners] = 0

    n_internal = int(mesh["n_interior_faces"])
    owners = np.asarray(mesh["owners"], dtype=np.int32)[:n_internal]
    neighbours = np.asarray(mesh["neighbours"], dtype=np.int32)
    triangular = np.fromiter(
        (len(face) == 3 for face in mesh["faces"][:n_internal]),
        dtype=bool,
        count=n_internal,
    )
    through_layer = triangular & prism[owners] & prism[neighbours]
    owners = owners[through_layer]
    neighbours = neighbours[through_layer]
    for layer in range(layers - 1):
        owner_to_neighbour = (result[owners] == layer) & (result[neighbours] < 0)
        neighbour_to_owner = (result[neighbours] == layer) & (result[owners] < 0)
        result[neighbours[owner_to_neighbour]] = layer + 1
        result[owners[neighbour_to_owner]] = layer + 1

    missing = np.flatnonzero(prism & (result < 0))
    unexpected = np.flatnonzero(result >= layers)
    if missing.size or unexpected.size:
        raise RuntimeError(
            "Could not recover a complete general-body prism stack: "
            f"missing={missing[:10].tolist()}, unexpected={unexpected[:10].tolist()}"
        )
    counts = np.bincount(result[prism], minlength=layers)
    if len(counts) != layers or not np.all(counts == int(wall["n_faces"])):
        raise RuntimeError(
            f"Boundary-layer prism counts are incomplete: {counts.tolist()} vs "
            f"{wall['n_faces']} wall faces"
        )
    return result


class GeneralBodyMesher:
    """Mesh a box domain around any closed, watertight STL body.

    The body must lie strictly inside ``domain`` with enough clearance for the
    requested boundary-layer thickness.  Surface triangles are remeshed while
    remaining on the STL geometry, sharp edges are retained according to
    ``feature_angle_degrees``, and the returned object is a native FVM
    ``mesh_data`` dictionary.

    Parameters
    ----------
    domain:
        Fluid-domain bounds ``(xmin, xmax, ymin, ymax, zmin, zmax)``.
    max_cell_size:
        Far-field core size.
    surface_file:
        Closed, watertight STL file for the solid body.
    wall_patch_name:
        Name assigned to the body surface patch.
    surface_cell_size:
        Target tangential cell size at the body.
    boundary_layer:
        First height, layer count, geometric growth, and generic transition
        count for the compatibility Gmsh path.
    refinements:
        Optional cfMesh-style box refinements in the volume.
    """

    def __init__(
        self,
        domain: Bounds,
        max_cell_size: float,
        *,
        surface_file: str | Path,
        wall_patch_name: str,
        surface_cell_size: float,
        boundary_layer: BoundaryLayerSpec,
        refinements: tuple[BoxRefinement, ...] = (),
        feature_angle_degrees: float = 35.0,
        core_algorithm: int = 10,
    ) -> None:
        _validate_bounds(domain, "domain")
        if not math.isfinite(max_cell_size) or max_cell_size <= 0.0:
            raise ValueError("max_cell_size must be finite and positive")
        if not math.isfinite(surface_cell_size) or surface_cell_size <= 0.0:
            raise ValueError("surface_cell_size must be finite and positive")
        if surface_cell_size > max_cell_size:
            raise ValueError("surface_cell_size must not exceed max_cell_size")
        if not wall_patch_name.strip():
            raise ValueError("wall_patch_name must not be empty")
        if not math.isfinite(feature_angle_degrees) or not 0.0 < feature_angle_degrees < 180.0:
            raise ValueError("feature_angle_degrees must lie strictly between 0 and 180")
        if core_algorithm not in {1, 4, 10}:
            raise ValueError("core_algorithm must be Gmsh Delaunay (1), Frontal (4), or HXT (10)")

        boundary_layer.validate()
        surface = TriangulatedSurface.from_stl(surface_file)
        body_min = np.asarray(surface.bounds[::2], dtype=np.float64)
        body_max = np.asarray(surface.bounds[1::2], dtype=np.float64)
        domain_min = np.asarray(domain[::2], dtype=np.float64)
        domain_max = np.asarray(domain[1::2], dtype=np.float64)
        clearance = np.minimum(body_min - domain_min, domain_max - body_max)
        if np.any(clearance <= boundary_layer.thickness):
            raise ValueError(
                "The body needs clearance greater than the complete boundary-layer "
                f"thickness {boundary_layer.thickness:.6g}; per-axis clearance is "
                f"{clearance.tolist()}"
            )

        for refinement in refinements:
            _validate_bounds(refinement.bounds, f"{refinement.name}.bounds")
            if not math.isfinite(refinement.cell_size) or refinement.cell_size <= 0.0:
                raise ValueError(f"{refinement.name}.cell_size must be finite and positive")
            if not all(
                domain[2 * axis]
                <= refinement.bounds[2 * axis]
                < refinement.bounds[2 * axis + 1]
                <= domain[2 * axis + 1]
                for axis in range(3)
            ):
                raise ValueError(f"{refinement.name} must lie inside domain")

        self.domain = tuple(float(value) for value in domain)
        self.max_cell_size = float(max_cell_size)
        self.surface = surface
        self.wall_patch_name = wall_patch_name
        self.surface_cell_size = float(surface_cell_size)
        self.boundary_layer = boundary_layer
        self.refinements = tuple(refinements)
        self.feature_angle_degrees = float(feature_angle_degrees)
        self.core_algorithm = int(core_algorithm)

    def _configure_size_fields(self, body_surface_tags: list[int]) -> None:
        field = gmsh.model.mesh.field
        distance = field.add("Distance")
        field.setNumbers(distance, "FacesList", body_surface_tags)
        field.setNumber(distance, "Sampling", 100)

        transition = field.add("Threshold")
        field.setNumber(transition, "InField", distance)
        field.setNumber(transition, "SizeMin", self.surface_cell_size)
        field.setNumber(transition, "SizeMax", self.max_cell_size)
        field.setNumber(transition, "DistMin", self.boundary_layer.thickness)
        field.setNumber(
            transition,
            "DistMax",
            self.boundary_layer.thickness
            + self.boundary_layer.transition_layers * self.surface_cell_size,
        )
        fields = [transition]

        for refinement in self.refinements:
            box = field.add("Box")
            field.setNumber(box, "VIn", refinement.cell_size)
            field.setNumber(box, "VOut", self.max_cell_size)
            for name, value in zip(
                ("XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax"),
                refinement.bounds,
                strict=True,
            ):
                field.setNumber(box, name, value)
            fields.append(box)

        if len(fields) == 1:
            background = fields[0]
        else:
            background = field.add("Min")
            field.setNumbers(background, "FieldsList", fields)
        field.setAsBackgroundMesh(background)

    def _generate_current_model(self) -> dict:
        model = gmsh.model
        gmsh.merge(str(self.surface.path))
        raw_body_surfaces = model.getEntities(2)
        signed_volume = _signed_surface_volume(self.surface.triangles)
        if signed_volume < 0.0:
            model.mesh.reverse(raw_body_surfaces)

        angle = math.radians(self.feature_angle_degrees)
        model.mesh.classifySurfaces(angle, True, True, angle, True)
        model.mesh.createGeometry()
        body_surfaces = model.getEntities(2)
        body_surface_tags = [tag for _dimension, tag in body_surfaces]

        gmsh.option.setNumber("Geometry.ExtrudeReturnLateralEntities", 0)
        extrusion = model.geo.extrudeBoundaryLayer(
            body_surfaces,
            [1] * self.boundary_layer.layers,
            list(self.boundary_layer.cumulative_heights),
            True,
        )
        top_surfaces = [tag for dimension, tag in extrusion if dimension == 2]
        layer_volumes = [tag for dimension, tag in extrusion if dimension == 3]
        if not top_surfaces or not layer_volumes:
            raise RuntimeError("Boundary-layer extrusion did not create a closed prismatic shell")

        outer_surfaces = _add_outer_box(model, self.domain, self.max_cell_size)
        outer_loop = model.geo.addSurfaceLoop(outer_surfaces)
        layer_front_loop = model.geo.addSurfaceLoop(top_surfaces)
        core_volume = model.geo.addVolume([outer_loop, layer_front_loop])
        model.geo.synchronize()

        for name, surface_tag in zip(_OUTER_SURFACES, outer_surfaces, strict=True):
            model.addPhysicalGroup(2, [surface_tag], name=name)
        model.addPhysicalGroup(2, body_surface_tags, name=self.wall_patch_name)
        model.addPhysicalGroup(3, layer_volumes, name="boundary_layer")
        model.addPhysicalGroup(3, [core_volume], name="fluid_core")

        self._configure_size_fields(body_surface_tags)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(self.surface_cell_size, self.max_cell_size))
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.max_cell_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", self.core_algorithm)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        model.mesh.generate(3)

        importer = GmshImporter()
        importer.source_path = str(self.surface.path)
        mesh = importer.get_mesh_data()
        for patch in mesh["boundary"]:
            if patch["name"] == self.wall_patch_name:
                patch["type"] = "wall"
        unexpected = {patch["name"] for patch in mesh["boundary"]} - {
            "inlet",
            "outlet",
            "ymin",
            "ymax",
            "zmin",
            "zmax",
            self.wall_patch_name,
        }
        if unexpected:
            raise RuntimeError(f"General-body mesh contains unclassified patches: {unexpected}")

        cell_types = np.asarray(mesh["cell_type_code"], dtype=np.int32)
        prism_mask = cell_types == 6
        mesh["boundary_layer_index"] = _prism_layer_indices(
            mesh,
            self.wall_patch_name,
            self.boundary_layer.layers,
        )
        mesh["mesh_generation"] = {
            "method": "cfmesh_style_general_body",
            "surface_file": str(self.surface.path),
            "surface_sha256": self.surface.sha256,
            "surface_bounds": self.surface.bounds,
            "surface_triangle_count": len(self.surface.triangles),
            "wall_patch_name": self.wall_patch_name,
            "max_cell_size": self.max_cell_size,
            "surface_cell_size": self.surface_cell_size,
            "feature_angle_degrees": self.feature_angle_degrees,
            "core_algorithm": self.core_algorithm,
            "boundary_layer": {
                "method": "surface_normal_advancing_layer",
                "wall_layers": self.boundary_layer.layers,
                "transition_layers": self.boundary_layer.transition_layers,
                "first_cell_height": self.boundary_layer.first_cell_height,
                "growth_ratio": self.boundary_layer.growth_ratio,
                "layer_heights": list(self.boundary_layer.layer_heights),
                "layer_thickness": self.boundary_layer.thickness,
                "prismatic_cells": int(np.count_nonzero(prism_mask)),
            },
            "refinements": [
                {
                    "name": refinement.name,
                    "bounds": tuple(refinement.bounds),
                    "cell_size": refinement.cell_size,
                }
                for refinement in self.refinements
            ],
            "attribution": "cfMesh-style workflow; advancing layer/core fill via Gmsh API",
        }
        validate_topology(mesh)
        validate_curved_wall_conformance(
            mesh,
            self.surface.triangles,
            self.wall_patch_name,
        )
        return mesh

    def build(self) -> dict:
        """Generate the mesh in-process and return native FVM connectivity."""
        if gmsh is None:
            raise ImportError("GeneralBodyMesher requires the OpenONDA gmsh dependency")

        owned_session = not gmsh.isInitialized()
        if owned_session:
            gmsh.initialize()
        option_names = (
            "General.Terminal",
            "Geometry.ExtrudeReturnLateralEntities",
            "Mesh.MeshSizeMin",
            "Mesh.MeshSizeMax",
            "Mesh.MeshSizeFromPoints",
            "Mesh.MeshSizeExtendFromBoundary",
            "Mesh.Algorithm",
            "Mesh.Algorithm3D",
            "Mesh.ElementOrder",
        )
        prior_options = {name: gmsh.option.getNumber(name) for name in option_names}
        gmsh.option.setNumber("General.Terminal", 0)
        prior_model = gmsh.model.getCurrent() if gmsh.model.list() else None
        model_name = f"openonda-general-body-{uuid.uuid4().hex}"
        gmsh.model.add(model_name)
        try:
            return self._generate_current_model()
        finally:
            if model_name in gmsh.model.list():
                gmsh.model.setCurrent(model_name)
                gmsh.model.remove()
            if prior_model is not None and prior_model in gmsh.model.list():
                gmsh.model.setCurrent(prior_model)
            if not owned_session:
                for name, value in prior_options.items():
                    gmsh.option.setNumber(name, value)
            if owned_session and gmsh.isInitialized():
                gmsh.finalize()

    def __call__(self) -> dict:
        """Allow the mesher itself to be passed to ``create_fvm_solver``."""
        return self.build()


__all__ = ["GeneralBodyMesher"]
