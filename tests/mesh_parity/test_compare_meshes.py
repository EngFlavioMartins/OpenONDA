"""Numbering-independent topology comparison contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from tools.mesh_parity.compare_meshes import compare_meshes
from tools.mesh_parity.openfoam_poly_mesh import PolyMesh

from ._fixtures import two_cell_mesh


def test_comparator_accepts_equivalent_meshes_with_points_and_cells_renumbered():
    result = compare_meshes(two_cell_mesh(), two_cell_mesh(renumber=True))

    assert result.passed
    assert result.first_failure is None
    assert result.cell_mapping["complete"]
    assert result.topology["adjacency_mismatches"] == 0
    assert result.topology["patch_incidence_mismatches"] == 0
    assert result.topology["face_topology_mismatches"] == 0


def test_comparator_fails_before_graph_matching_when_patch_invariant_changes():
    mesh = two_cell_mesh()
    changed = PolyMesh(
        points=mesh.points,
        faces=mesh.faces,
        owner=mesh.owner,
        neighbour=mesh.neighbour,
        boundary=(
            replace(mesh.boundary[0], name="different_inlet"),
            *mesh.boundary[1:],
        ),
        n_cells=mesh.n_cells,
    )

    result = compare_meshes(mesh, changed)

    assert not result.passed
    assert result.first_failure == "level_a_invariants"
    assert "patches" in result.invariant_differences
    assert result.topology["adjacency_mismatches"] is None


def test_comparator_is_invariant_to_nonplanar_face_starting_vertices():
    mesh = two_cell_mesh()
    points = mesh.points.copy()
    points[6, 2] += 0.125
    reference = replace(mesh, points=points)
    rotated = replace(
        reference,
        faces=tuple(np.roll(face, index % len(face)) for index, face in enumerate(mesh.faces)),
    )

    result = compare_meshes(reference, rotated)

    assert result.passed
    assert result.first_failure is None
