"""Regression tests for general closed-STL boundary-layer meshing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from source.solvers.fvm.mesh.boundary_layer import BoundaryLayerSpec
from source.solvers.fvm.mesh.general_body import GeneralBodyMesher
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.validation import validate_geometry, validate_topology

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CUBE_STL = (
    REPOSITORY_ROOT
    / "tutorials"
    / "coupled_fvm_vpm"
    / "cube_flow"
    / "reference_flow"
    / "assets"
    / "cube.stl"
)


def _write_octahedron_stl(path: Path) -> None:
    """Write a closed non-axis-aligned faceted body for generality checks."""
    vertices = np.asarray(
        (
            (0.5, 0.0, 0.0),
            (-0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, -0.5, 0.0),
            (0.0, 0.0, 0.5),
            (0.0, 0.0, -0.5),
        )
    )
    faces = (
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    )
    lines = ["solid octahedron"]
    for face in faces:
        triangle = vertices[list(face)]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        lines.append(f"  facet normal {normal[0]} {normal[1]} {normal[2]}")
        lines.append("    outer loop")
        lines.extend(f"      vertex {point[0]} {point[1]} {point[2]}" for point in triangle)
        lines.extend(("    endloop", "  endfacet"))
    lines.append("endsolid octahedron")
    path.write_text("\n".join(lines) + "\n")


def test_general_body_mesher_builds_complete_prismatic_wall_layers():
    layers = 4
    first_height = 0.02
    mesh = GeneralBodyMesher(
        domain=(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0),
        max_cell_size=0.5,
        surface_file=CUBE_STL,
        wall_patch_name="body",
        surface_cell_size=0.15,
        boundary_layer=BoundaryLayerSpec(
            first_cell_height=first_height,
            layers=layers,
            growth_ratio=1.3,
            transition_layers=4,
        ),
    ).build()

    validate_topology(mesh)
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    quality = validate_geometry(mesh, geometry)

    patches = {patch["name"] for patch in mesh["boundary"]}
    assert patches == {"inlet", "outlet", "ymin", "ymax", "zmin", "zmax", "body"}
    wall = next(patch for patch in mesh["boundary"] if patch["name"] == "body")
    assert wall["type"] == "wall"

    cell_types = np.asarray(mesh["cell_type_code"])
    prism_count = int(np.count_nonzero(cell_types == 6))
    assert prism_count == int(wall["n_faces"]) * layers
    assert np.any(cell_types == 4)
    assert mesh["mesh_generation"]["boundary_layer"]["prismatic_cells"] == prism_count
    layer_index = np.asarray(mesh["boundary_layer_index"])
    assert all(np.count_nonzero(layer_index == layer) == wall["n_faces"] for layer in range(layers))

    start = int(wall["start_face"])
    stop = start + int(wall["n_faces"])
    wall_faces = mesh["faces"][start:stop]
    wall_points = np.asarray(mesh["vertex_position"])[
        np.unique(np.concatenate([np.asarray(face) for face in wall_faces]))
    ]
    assert np.allclose(np.max(np.abs(wall_points), axis=1), 0.5, atol=1.0e-10)

    owners = np.asarray(mesh["owners"])[start:stop]
    wall_distance = np.linalg.norm(
        np.asarray(geometry["cell_centre"])[owners]
        - np.asarray(geometry["face_centre"])[start:stop],
        axis=1,
    )
    assert np.max(wall_distance) <= 0.51 * first_height
    assert quality["out_of_bounds_interpolation_weights"] == 0
    assert quality["max_non_orthogonality_deg"] < 75.0


def test_general_body_mesher_handles_non_axis_aligned_facets(tmp_path):
    surface_file = tmp_path / "octahedron.stl"
    _write_octahedron_stl(surface_file)
    layers = 3
    mesh = GeneralBodyMesher(
        domain=(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5),
        max_cell_size=0.4,
        surface_file=surface_file,
        wall_patch_name="octahedron",
        surface_cell_size=0.18,
        boundary_layer=BoundaryLayerSpec(
            first_cell_height=0.02,
            layers=layers,
            growth_ratio=1.2,
            transition_layers=3,
        ),
    ).build()

    validate_topology(mesh)
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    quality = validate_geometry(mesh, geometry)
    wall = next(patch for patch in mesh["boundary"] if patch["name"] == "octahedron")
    assert np.count_nonzero(np.asarray(mesh["cell_type_code"]) == 6) == wall["n_faces"] * layers
    assert quality["min_volume"] > 0.0
    assert quality["out_of_bounds_interpolation_weights"] == 0
