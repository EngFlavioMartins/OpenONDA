"""Focused contracts for the cfMesh finite-volume optimization port."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from source.solvers.fvm.mesh.cartesian.cfmesh_mesh_optimisation import (
    _cfmesh_bad_faces,
    _cfmesh_cell_centres,
    _cfmesh_low_quality_faces,
    _face_geometry,
    _mesh_addressing,
    _optimise_part_boundary_volume,
    _optimise_part_knupp,
    _optimise_part_untangler,
    _optimise_part_volume,
    _PartTetMesh,
)
from source.solvers.fvm.mesh.cartesian.cfmesh_surface_optimisation import (
    _gradients,
    _optimise_point,
    _optimise_point_kernel,
    _smooth_partition_points,
)


def _unit_cube() -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ]
    )
    faces = [
        np.asarray(face, dtype=np.int32)
        for face in (
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 4, 7, 3),
            (1, 2, 6, 5),
            (0, 1, 5, 4),
            (3, 7, 6, 2),
        )
    ]
    return points, faces, np.zeros(6, dtype=np.int32), np.empty(0, dtype=np.int32)


def test_quality_scans_accept_an_orthogonal_unit_cube():
    points, faces, owners, neighbours = _unit_cube()

    assert not _cfmesh_bad_faces(points, faces, owners, neighbours, 1)
    assert not _cfmesh_low_quality_faces(points, faces, owners, neighbours, 1)


def test_bad_face_scan_detects_an_inward_boundary_face():
    points, faces, owners, neighbours = _unit_cube()
    faces[1] = faces[1][::-1].copy()

    assert 1 in _cfmesh_bad_faces(points, faces, owners, neighbours, 1)


@pytest.mark.parametrize("angle, expected", [(64.0, False), (67.0, True), (71.0, True)])
def test_low_quality_nonorthogonality_uses_native_65_degree_gate(angle, expected):
    shear = np.tan(np.deg2rad(angle))
    points = np.asarray(
        [(x, y + x * shear, z) for x in range(3) for y in range(2) for z in range(2)]
    )
    faces = [[4, 6, 7, 5], [0, 1, 3, 2], [8, 10, 11, 9]]
    owners = [0, 0, 1]
    for cell in range(2):
        base = 4 * cell
        faces.extend(
            [
                [base + j for j in row]
                for row in ((0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
            ]
        )
        owners.extend([cell] * 4)
    bad = _cfmesh_low_quality_faces(
        points,
        [np.asarray(face, dtype=np.int32) for face in faces],
        np.asarray(owners, dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        2,
    )
    assert (0 in bad) is expected


def test_mesh_addressing_preserves_and_validates_native_cell_face_order():
    points, faces, owners, neighbours = _unit_cube()
    native_order = [[5, 3, 1, 4, 2, 0]]

    cell_faces, point_cells = _mesh_addressing(
        faces,
        owners,
        neighbours,
        1,
        len(points),
        cell_face_order=native_order,
    )

    assert cell_faces == native_order
    assert all(cells == {0} for cells in point_cells)
    with pytest.raises(ValueError, match="inconsistent with mesh topology"):
        _mesh_addressing(
            faces,
            owners,
            neighbours,
            1,
            len(points),
            cell_face_order=[[0, 1, 2, 3, 4]],
        )


@pytest.mark.parametrize("optimizer", [_optimise_point, _optimise_point_kernel])
def test_surface_optimizer_uses_cfmesh_branch_for_symmetric_simplex(optimizer):
    points = np.asarray(
        [
            (0.0, 0.0, 0.0),
            (0.175925104244658, 8.18286074045122e-18, 0.0),
            (0.129393069484928, 0.0928852466733864, 0.0),
            (0.224662907280417, 0.269029853901011, 0.0),
            (0.175925104244681, 0.18430756870769, 0.0),
            (0.224662907280378, -0.269029853901026, 0.0),
            (0.129393069484909, -0.0928852466733874, 0.0),
            (0.175925104244635, -0.18430756870769, 0.0),
            (-0.152240294632501, 8.55172432869605e-15, 0.0),
            (-0.906319307392956, 3.91363527458225e-14, 0.0),
        ]
    )
    triangles = np.asarray(
        [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (0, 5, 6),
            (0, 5, 1),
            (0, 6, 1),
            (0, 3, 8),
            (0, 3, 5),
            (0, 8, 5),
        ],
        dtype=np.int32,
    )

    result = optimizer(points, triangles)

    # Regenerated from these exact (rounded) input coordinates through
    # tools/mesh_parity/native_surface_optimizer, linked to cfMesh 3ff85555.
    # A result captured before rounding is not a valid symmetric-branch oracle.
    assert result == pytest.approx(
        (0.22429483774729889, 0.02784879347021511, 0.0), abs=1.0e-14, rel=0.0
    )


@pytest.mark.parametrize("optimizer", [_optimise_point, _optimise_point_kernel])
def test_surface_optimizer_matches_native_irregular_fans(optimizer):
    """Native cfMesh results, with the meshSurfaceOptimizer tolerance (0.001)."""
    expected = (
        (0.3727348298805224, 0.10032591573101839, 0.0),
        (-0.5085080760734136, 0.7677796639746444, 0.0),
        (0.3178672175108489, -0.12855944761548485, 0.0),
        (0.09280813389262191, 0.041256139939869256, 0.0),
        (-0.29660451373602414, -0.0405374580847703, 0.0),
        (0.16372376942486866, -0.13360137044471412, 0.0),
        (0.24305118038827375, -0.03866898389590672, 0.0),
        (0.1600159603252536, -0.1688390631342039, 0.0),
    )
    rng = np.random.default_rng(614)
    for n_neighbours, native_result in zip(range(4, 12), expected, strict=True):
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, n_neighbours))
        radii = rng.uniform(0.5, 1.5, n_neighbours)
        points = np.vstack(
            (
                np.zeros((1, 3)),
                np.column_stack(
                    (radii * np.cos(angles), radii * np.sin(angles), np.zeros(n_neighbours))
                ),
            )
        )
        triangles = np.asarray(
            [(0, j + 1, (j + 1) % n_neighbours + 1) for j in range(n_neighbours)],
            dtype=np.int32,
        )

        assert optimizer(points, triangles) == pytest.approx(native_result, abs=1.0e-14, rel=0.0)


def test_surface_optimizer_skips_collapsed_opposite_edge_in_gradients():
    points = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    gradient, hessian = _gradients(points, np.asarray(((0, 1, 1),)), 1.0e-15)

    np.testing.assert_array_equal(gradient, np.zeros(2))
    np.testing.assert_array_equal(hessian, 1.0e-300 * np.eye(2))


@pytest.fixture(params=("cfmesh_volume_first_pass.npz", "cfmesh_volume_second_pass.npz"))
def native_volume_pass(request):
    path = Path(__file__).parent / "fixtures" / request.param
    with np.load(path, allow_pickle=False) as data:
        yield data


@pytest.mark.slow
def test_surface_partition_passes_match_native_on_identical_inputs():
    path = Path(__file__).parent / "fixtures" / "cfmesh_surface_partition_passes.npz"
    with np.load(path, allow_pickle=False) as data:
        mesh = {
            "vertex_position": data["initial_points"].copy(),
            "faces": list(np.split(data["faces"], data["face_offsets"][1:-1])),
            "n_interior_faces": 0,
        }
        _smooth_partition_points(mesh, data["selected_points"], iterations=5)
        np.testing.assert_allclose(
            mesh["vertex_position"], data["after_surface"], rtol=0.0, atol=1.0e-12
        )


def test_volume_geometry_matches_native_addressing(native_volume_pass):
    data = native_volume_pass
    faces = list(np.split(data["faces"], data["face_offsets"][1:-1]))
    order = [row.tolist() for row in np.split(data["cell_faces"], data["cell_offsets"][1:-1])]
    centres, _areas = _face_geometry(data["mesh_points"], faces)
    face_map = data["face_centre_nodes"]
    np.testing.assert_allclose(
        centres[face_map[:, 0]], data["initial_points"][face_map[:, 1]], rtol=0.0, atol=1.0e-14
    )
    cell_centres = _cfmesh_cell_centres(
        data["mesh_points"],
        faces,
        data["owners"],
        data["neighbours"],
        len(order),
        cell_face_order=order,
    )
    cell_map = data["cell_centre_nodes"]
    np.testing.assert_allclose(
        cell_centres[cell_map[:, 0]],
        data["initial_points"][cell_map[:, 1]],
        rtol=0.0,
        atol=1.0e-14,
    )


def test_volume_optimisation_sequence_matches_native(native_volume_pass):
    """Keep the actual cfMesh inputs, not rounded/simplex-reconstructed inputs."""
    data = native_volume_pass
    point_tets = [[] for _ in data["initial_points"]]
    for tet_id, tet in enumerate(data["tets"]):
        for node_id in tet:
            point_tets[int(node_id)].append(tet_id)
    part = _PartTetMesh(
        points=data["initial_points"].copy(),
        tets=data["tets"],
        smooth_nodes=data["smooth_nodes"],
        boundary_nodes=data["boundary_nodes"],
        node_to_original=data["node_to_original"],
        face_centre_nodes=dict(data["face_centre_nodes"].tolist()),
        cell_centre_nodes=dict(data["cell_centre_nodes"].tolist()),
        point_tets=point_tets,
    )
    for optimise, expected in (
        (_optimise_part_knupp, "after_knupp"),
        (_optimise_part_untangler, "after_untangler"),
        (_optimise_part_volume, "after_volume"),
    ):
        optimise(part)
        np.testing.assert_allclose(part.points, data[expected], rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize("iterations", [0, 1])
def test_boundary_volume_uses_native_iteration_count(iterations):
    path = Path(__file__).parent / "fixtures" / "cfmesh_boundary_volume_pass.npz"
    with np.load(path, allow_pickle=False) as data:
        point_tets = [[] for _ in data["initial_points"]]
        for tet_id, tet in enumerate(data["tets"]):
            for node_id in tet:
                point_tets[int(node_id)].append(tet_id)
        part = _PartTetMesh(
            points=data["initial_points"].copy(),
            tets=data["tets"],
            smooth_nodes=data["smooth_nodes"],
            boundary_nodes=data["boundary_nodes"],
            node_to_original=data["node_to_original"],
            face_centre_nodes=dict(data["face_centre_nodes"].tolist()),
            cell_centre_nodes=dict(data["cell_centre_nodes"].tolist()),
            point_tets=point_tets,
        )
        _optimise_part_boundary_volume(part, iterations=iterations)
        if iterations == 0:
            np.testing.assert_array_equal(part.points, data["initial_points"])
        else:
            # The native boundary routine explicitly requests the machine's
            # CPU count, overriding OMP_NUM_THREADS. Auxiliary cell/face-centre
            # refreshes race, but cannot feed back into its single pass. Compare
            # every point transferred to the original mesh, not these discarded
            # intermediate centres.
            originals = part.node_to_original >= 0
            np.testing.assert_allclose(
                part.points[originals], data["after_volume"][originals], rtol=0.0, atol=1.0e-12
            )
