import numpy as np
import pytest


@pytest.fixture(scope="session")
def hand_built_3d_mesh():
    points = np.array(
        [
            [-1.0, -1.0, -1.0],
            [0.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 1.0, -1.0],
            [0.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 1.0],
            [0.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    faces = [
        # Interior: x=0 plane (4 faces)
        np.array([4, 13, 10, 1], dtype=np.int32),
        np.array([7, 16, 13, 4], dtype=np.int32),
        np.array([13, 22, 19, 10], dtype=np.int32),
        np.array([16, 25, 22, 13], dtype=np.int32),
        # Interior: y=0 plane (4 faces)
        np.array([3, 12, 13, 4], dtype=np.int32),
        np.array([12, 21, 22, 13], dtype=np.int32),
        np.array([4, 13, 14, 5], dtype=np.int32),
        np.array([13, 22, 23, 14], dtype=np.int32),
        # Interior: z=0 plane (4 faces)
        np.array([13, 12, 9, 10], dtype=np.int32),
        np.array([14, 13, 10, 11], dtype=np.int32),
        np.array([16, 15, 12, 13], dtype=np.int32),
        np.array([17, 16, 13, 14], dtype=np.int32),
        # Boundary: xmin (x=-1) 4 faces
        np.array([0, 9, 12, 3], dtype=np.int32),
        np.array([3, 12, 15, 6], dtype=np.int32),
        np.array([9, 18, 21, 12], dtype=np.int32),
        np.array([12, 21, 24, 15], dtype=np.int32),
        # Boundary: xmax (x=1) 4 faces  (Sf = +x)
        np.array([5, 14, 11, 2], dtype=np.int32),
        np.array([8, 17, 14, 5], dtype=np.int32),
        np.array([14, 23, 20, 11], dtype=np.int32),
        np.array([17, 26, 23, 14], dtype=np.int32),
        # Boundary: ymin (y=-1) 4 faces
        np.array([1, 10, 9, 0], dtype=np.int32),
        np.array([2, 11, 10, 1], dtype=np.int32),
        np.array([10, 19, 18, 9], dtype=np.int32),
        np.array([11, 20, 19, 10], dtype=np.int32),
        # Boundary: ymax (y=1) 4 faces
        np.array([6, 15, 16, 7], dtype=np.int32),
        np.array([7, 16, 17, 8], dtype=np.int32),
        np.array([15, 24, 25, 16], dtype=np.int32),
        np.array([16, 25, 26, 17], dtype=np.int32),
        # Boundary: zmin (z=-1) 4 faces
        np.array([1, 0, 3, 4], dtype=np.int32),
        np.array([2, 1, 4, 5], dtype=np.int32),
        np.array([4, 3, 6, 7], dtype=np.int32),
        np.array([5, 4, 7, 8], dtype=np.int32),
        # Boundary: zmax (z=1) 4 faces
        np.array([22, 21, 18, 19], dtype=np.int32),
        np.array([23, 22, 19, 20], dtype=np.int32),
        np.array([25, 24, 21, 22], dtype=np.int32),
        np.array([26, 25, 22, 23], dtype=np.int32),
    ]

    owners = np.array(
        [
            0, 2, 4, 6,
            0, 4, 1, 5,
            0, 1, 2, 3,
            0, 2, 4, 6,
            1, 3, 5, 7,
            0, 1, 4, 5,
            2, 3, 6, 7,
            0, 1, 2, 3,
            4, 5, 6, 7,
        ],
        dtype=np.int32,
    )
    neighbours = np.array([1, 3, 5, 7, 2, 6, 3, 7, 4, 5, 6, 7], dtype=np.int32)

    boundary = [
        {"name": "xmin", "startFace": 12, "nFaces": 4, "type": "patch"},
        {"name": "xmax", "startFace": 16, "nFaces": 4, "type": "patch"},
        {"name": "ymin", "startFace": 20, "nFaces": 4, "type": "patch"},
        {"name": "ymax", "startFace": 24, "nFaces": 4, "type": "patch"},
        {"name": "zmin", "startFace": 28, "nFaces": 4, "type": "patch"},
        {"name": "zmax", "startFace": 32, "nFaces": 4, "type": "patch"},
    ]

    mesh_data = {
        "points": points,
        "faces": faces,
        "owners": owners,
        "neighbours": neighbours,
        "boundary": boundary,
        "n_elements": 8,
        "n_faces": 36,
        "n_interior_faces": 12,
        "n_points": 27,
    }
    return mesh_data


@pytest.fixture(scope="session", params=[0.5, 0.25])
def gmsh_unit_cube(request):
    import gmsh

    lcar = request.param
    gmsh.initialize()
    try:
        model = gmsh.model
        model.add("unit_cube")
        model.occ.addBox(0, 0, 0, 1, 1, 1)
        model.occ.synchronize()
        model.mesh.setSize(model.getEntities(0), lcar)
        model.mesh.generate(3)
        from source.solvers.FVM.mesh.gmsh_importer import GmshImporter

        importer = GmshImporter()
        mesh_data = importer.get_mesh_data()
    finally:
        gmsh.finalize()
    return mesh_data


@pytest.fixture(scope="session")
def golden_reference():
    from pathlib import Path

    import numpy as np

    ref_path = Path(__file__).resolve().parent / "fixtures" / "golden_reference.npz"
    return dict(np.load(ref_path))


def cavity_mesh_3d(n):
    import gmsh

    gmsh.initialize()
    try:
        model = gmsh.model
        model.add("cavity_3d")
        model.occ.addBox(0, 0, 0, 1, 1, 1)
        model.occ.synchronize()
        lcar = 1.0 / n
        model.mesh.setSize(model.getEntities(0), lcar)
        model.mesh.generate(3)
        from source.solvers.FVM.mesh.gmsh_importer import GmshImporter

        importer = GmshImporter()
        mesh_data = importer.get_mesh_data()
    finally:
        gmsh.finalize()
    return mesh_data
