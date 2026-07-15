import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from source.solvers.FVM.assemble import diffusion, matrix_assembly
from source.solvers.FVM.fields import gradients
from source.solvers.FVM.mesh import geometry, mesh_io
from source.solvers.FVM.mesh.gmsh_importer import GmshImporter

gmsh = pytest.importorskip("gmsh", reason="Gmsh FVM test dependency is not installed")


def _foam_header(object_name, class_name):
    return (
        "FoamFile\n"
        "{\n"
        "    version 2.0;\n"
        "    format ascii;\n"
        f"    class {class_name};\n"
        f"    object {object_name};\n"
        "}\n"
    )


def _write_poly_mesh(case_dir, mesh):
    poly_mesh = case_dir / "constant" / "polyMesh"
    poly_mesh.mkdir(parents=True)

    points = "\n".join(f"({x:.17g} {y:.17g} {z:.17g})" for x, y, z in mesh["points"])
    (poly_mesh / "points").write_text(
        _foam_header("points", "vectorField") + f"{mesh['n_points']}\n(\n{points}\n)\n"
    )
    faces = "\n".join(
        f"{len(face)}({' '.join(str(int(node)) for node in face)})" for face in mesh["faces"]
    )
    (poly_mesh / "faces").write_text(
        _foam_header("faces", "faceList") + f"{mesh['n_faces']}\n(\n{faces}\n)\n"
    )
    owners = "\n".join(str(int(value)) for value in mesh["owners"])
    (poly_mesh / "owner").write_text(
        _foam_header("owner", "labelList") + f"{mesh['n_faces']}\n(\n{owners}\n)\n"
    )
    neighbours = "\n".join(str(int(value)) for value in mesh["neighbours"])
    (poly_mesh / "neighbour").write_text(
        _foam_header("neighbour", "labelList") + f"{mesh['n_interior_faces']}\n(\n{neighbours}\n)\n"
    )
    patches = []
    for patch in mesh["boundary"]:
        patches.append(
            f"{patch['name']}\n"
            "{\n"
            f"    type {patch.get('type', 'patch')};\n"
            f"    nFaces {patch['nFaces']};\n"
            f"    startFace {patch['startFace']};\n"
            "}"
        )
    patch_text = "\n".join(patches)
    (poly_mesh / "boundary").write_text(
        _foam_header("boundary", "polyBoundaryMesh") + f"{len(patches)}\n(\n{patch_text}\n)\n"
    )


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

        _write_poly_mesh(tmp_path, gmsh_mesh)
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
