import numpy as np

from source.solvers.FVM.mesh.geometry import compute_mesh_geometry


class TestHandBuilt3DMesh:
    """Analytic geometry checks for the 2×2×2 cube of unit hexes."""

    def test_element_count(self, hand_built_3d_mesh):
        assert hand_built_3d_mesh["n_cells"] == 8

    def test_face_counts(self, hand_built_3d_mesh):
        n_int = hand_built_3d_mesh["n_interior_faces"]
        n_bnd = hand_built_3d_mesh["n_faces"] - n_int
        assert n_int == 12  # 3 interior planes × 4 faces each
        assert n_bnd == 24  # 6 boundary patches × 4 faces each

    def test_all_faces_quad(self, hand_built_3d_mesh):
        for f in hand_built_3d_mesh["faces"]:
            assert len(f) == 4

    def test_cell_volumes_are_one(self, hand_built_3d_mesh):
        geo = compute_mesh_geometry(hand_built_3d_mesh)
        vols = geo["cell_volumes"]
        assert vols.shape == (8,)
        assert np.allclose(vols, 1.0)

    def test_cell_centroids(self, hand_built_3d_mesh):
        geo = compute_mesh_geometry(hand_built_3d_mesh)
        cents = geo["cell_centroids"]
        assert cents.shape == (8, 3)
        expected = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ]
        )
        assert np.allclose(cents, expected)

    def test_face_areas(self, hand_built_3d_mesh):
        geo = compute_mesh_geometry(hand_built_3d_mesh)
        areas = geo["face_areas"]
        assert areas.shape[0] == hand_built_3d_mesh["n_faces"]
        assert np.allclose(areas, 1.0)

    def test_face_normals_axis_aligned(self, hand_built_3d_mesh):
        geo = compute_mesh_geometry(hand_built_3d_mesh)
        sf = geo["face_sf"]
        for i in range(sf.shape[0]):
            nz = np.count_nonzero(np.abs(sf[i]) > 1e-12)
            assert nz == 1, f"Face {i} normal {sf[i]} not axis-aligned"
            # magnitude should be 1 (unit area)
            assert np.isclose(np.linalg.norm(sf[i]), 1.0)

    def test_boundary_patches(self, hand_built_3d_mesh):
        names = {b["name"] for b in hand_built_3d_mesh["boundary"]}
        assert names == {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
        for b in hand_built_3d_mesh["boundary"]:
            assert b["n_faces"] == 4
