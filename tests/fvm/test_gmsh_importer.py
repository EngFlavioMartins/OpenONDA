import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from source.solvers.FVM.assemble import diffusion, matrix_assembly
from source.solvers.FVM.fields import gradients
from source.solvers.FVM.mesh import geometry, mesh_io
from source.solvers.FVM.mesh.gmsh_importer import GmshImporter
from source.solvers.FVM.mesh.openfoam_writer import write_poly_mesh

gmsh = pytest.importorskip("gmsh", reason="Gmsh FVM test dependency is not installed")

FIRST_ORDER_CELLS = {
    4: [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
    5: [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ],
    6: [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)],
    7: [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0.5, 1)],
}


def _add_discrete_cell(element_type, points):
    gmsh.model.addDiscreteEntity(3, 1)
    node_tags = list(range(1, len(points) + 1))
    gmsh.model.mesh.addNodes(3, 1, node_tags, [value for point in points for value in point])
    gmsh.model.mesh.addElementsByType(1, element_type, [1], node_tags)


def _diffusion_solution(mesh):
    for patch in mesh["boundary"]:
        patch["bc_type"] = "fixedValue"
    geo = geometry.compute_mesh_geometry(mesh, gradient_scheme="lsq")
    n_cells = mesh["n_elements"]
    n_total = n_cells + mesh["n_faces"] - mesh["n_interior_faces"]
    field = np.empty(n_total)
    field[:n_cells] = np.sum(geo["element_centroids"], axis=1)
    field[n_cells:] = np.sum(geo["face_centroids"][mesh["n_interior_faces"] :], axis=1)
    grad = gradients.compute_gradient_lsq_vectorized(field, mesh, geo)
    flux = diffusion.assemble_diffusion_term(
        field, grad, np.ones(n_cells), mesh, geo, mesh["boundary"]
    )
    matrix = matrix_assembly.assemble_matrix_from_fluxes_vectorized(flux, mesh)
    rhs = matrix_assembly.assemble_rhs_from_fluxes_vectorized(flux, mesh)
    return geo, matrix, rhs, spsolve(matrix, rhs)


class TestGmshImporter:
    @pytest.mark.parametrize("element_type", [4, 5, 6, 7])
    def test_every_claimed_first_order_cell_family(self, element_type):
        gmsh.initialize()
        try:
            gmsh.model.add(f"first_order_{element_type}")
            _add_discrete_cell(element_type, FIRST_ORDER_CELLS[element_type])
            mesh = GmshImporter().get_mesh_data()
        finally:
            gmsh.finalize()

        assert mesh["n_elements"] == 1
        assert mesh["cell_type_codes"].tolist() == [element_type]
        assert mesh["cell_orders"].tolist() == [1]
        assert mesh["provenance"]["contract"] == "gmsh-api-first-order-3d-v1"

    def test_high_order_volume_cell_is_rejected(self):
        tetra10 = FIRST_ORDER_CELLS[4] + [
            (0.5, 0, 0),
            (0.5, 0.5, 0),
            (0, 0.5, 0),
            (0, 0, 0.5),
            (0.5, 0, 0.5),
            (0, 0.5, 0.5),
        ]
        gmsh.initialize()
        try:
            gmsh.model.add("high_order_tetra")
            _add_discrete_cell(11, tetra10)
            with pytest.raises(ValueError, match="high-order Gmsh element.*order 2"):
                GmshImporter().get_mesh_data()
        finally:
            gmsh.finalize()

    def test_imported_mesh_topology(self):
        gmsh.initialize()
        try:
            model = gmsh.model
            model.add("topo_test")
            model.occ.addBox(0, 0, 0, 1, 1, 1)
            model.occ.synchronize()
            model.mesh.setSize(model.getEntities(0), 0.5)
            model.mesh.generate(3)
            imp = GmshImporter()
            mesh = imp.get_mesh_data()
        finally:
            gmsh.finalize()

        owners = mesh["owners"]
        neighbours = mesh["neighbours"]
        n_int = mesh["n_interior_faces"]
        n_faces = mesh["n_faces"]

        assert len(owners) == n_faces
        assert len(neighbours) == n_int
        assert np.all(neighbours >= 0)
        assert np.all(owners[:n_int] != neighbours[:n_int])
        assert np.all(owners[n_int:] >= 0)
        assert np.all(owners[n_int:] < mesh["n_elements"])
        for f in mesh["faces"]:
            assert np.all(f >= 0)
        assert mesh["source_point_ids"].shape == (mesh["n_points"],)
        assert mesh["source_cell_ids"].shape == (mesh["n_elements"],)
        assert mesh["cell_type_codes"].shape == (mesh["n_elements"],)
        assert np.all(mesh["cell_orders"] == 1)
        assert mesh["provenance"]["format"] == "gmsh"

    def test_orphan_boundary_creates_default_patch(self):
        gmsh.initialize()
        try:
            model = gmsh.model
            model.add("simple_cube")
            model.occ.addBox(0, 0, 0, 1, 1, 1)
            model.occ.synchronize()
            model.mesh.setSize(model.getEntities(0), 0.5)
            model.mesh.generate(3)
            imp = GmshImporter()
            mesh = imp.get_mesh_data()
        finally:
            gmsh.finalize()

        names = {b["name"] for b in mesh["boundary"]}
        assert len(names) >= 1
        total_bnd_faces = sum(b["nFaces"] for b in mesh["boundary"])
        assert total_bnd_faces == mesh["n_faces"] - mesh["n_interior_faces"]

    def test_openfoam_roundtrip_has_equivalent_operators_and_solution(self, tmp_path):
        gmsh.initialize()
        try:
            model = gmsh.model
            model.add("format_equivalence")
            model.occ.addBox(0, 0, 0, 1, 1, 1)
            model.occ.synchronize()
            model.mesh.setSize(model.getEntities(0), 0.6)
            model.mesh.generate(3)
            gmsh_mesh = GmshImporter().get_mesh_data()
        finally:
            gmsh.finalize()

        write_poly_mesh(tmp_path, gmsh_mesh)
        foam_mesh = mesh_io.load_poly_mesh(tmp_path)

        assert np.array_equal(foam_mesh["points"], gmsh_mesh["points"])
        assert all(
            np.array_equal(foam_face, gmsh_face)
            for foam_face, gmsh_face in zip(foam_mesh["faces"], gmsh_mesh["faces"], strict=True)
        )
        assert np.array_equal(foam_mesh["owners"], gmsh_mesh["owners"])
        assert np.array_equal(foam_mesh["neighbours"], gmsh_mesh["neighbours"])
        assert [patch["name"] for patch in foam_mesh["boundary"]] == [
            patch["name"] for patch in gmsh_mesh["boundary"]
        ]
        for mesh in (gmsh_mesh, foam_mesh):
            assert mesh["source_point_ids"].shape == (mesh["n_points"],)
            assert mesh["source_cell_ids"].shape == (mesh["n_elements"],)
            assert mesh["cell_families"].shape == (mesh["n_elements"],)
            assert mesh["global_cell_ids"].shape == (mesh["n_elements"],)
            assert mesh["global_face_ids"].shape == (mesh["n_faces"],)

        gmsh_geo, gmsh_matrix, gmsh_rhs, gmsh_solution = _diffusion_solution(gmsh_mesh)
        foam_geo, foam_matrix, foam_rhs, foam_solution = _diffusion_solution(foam_mesh)
        for key in (
            "face_centroids",
            "face_sf",
            "element_centroids",
            "element_volumes",
            "face_weights",
            "face_cf_vector",
        ):
            assert np.allclose(foam_geo[key], gmsh_geo[key], rtol=0.0, atol=1e-14)
        assert np.allclose(foam_matrix.toarray(), gmsh_matrix.toarray(), rtol=0.0, atol=1e-13)
        assert np.allclose(foam_rhs, gmsh_rhs, rtol=0.0, atol=1e-13)
        assert np.allclose(foam_solution, gmsh_solution, rtol=0.0, atol=1e-12)
