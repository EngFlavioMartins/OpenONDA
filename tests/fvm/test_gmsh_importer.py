import numpy as np
import pytest

from source.solvers.FVM.io.vtk_exporter import VTKExporter
from source.solvers.FVM.mesh.gmsh_importer import GmshImporter

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

    def test_hexahedral_mesh_exports_native_vtk_hexahedra(self, tmp_path):
        """Recombined hex meshes must not degrade to VTK_POLYHEDRON.

        ParaView interpolates and renders polyhedra far worse than native
        hexahedra, so a uniformly hexahedral import has to carry the
        explicit cell corners the exporter's fast path needs.
        """
        pyvista = pytest.importorskip("pyvista")
        gmsh.initialize()
        try:
            gmsh.model.add("hexahedral_cell_vertices")
            _add_discrete_cell(5, FIRST_ORDER_CELLS[5])
            mesh = GmshImporter().get_mesh_data()
        finally:
            gmsh.finalize()

        assert mesh["cell_vertices"].shape == (1, 8)

        target = tmp_path / "hexahedron.vtu"
        VTKExporter(mesh).export(str(target), {"p": np.zeros(1)})
        grid = pyvista.read(target)

        assert grid.celltypes.tolist() == [pyvista.CellType.HEXAHEDRON]
        assert list(grid.cell_data) == ["p"]
        # A permuted corner ordering would collapse or invert the unit cube.
        assert grid.volume == pytest.approx(1.0)

    @pytest.mark.parametrize("element_type", [4, 6, 7])
    def test_non_hexahedral_mesh_keeps_polyhedral_export(self, element_type):
        """Mixed and non-hex families stay on the general polyhedron path."""
        gmsh.initialize()
        try:
            gmsh.model.add(f"no_cell_vertices_{element_type}")
            _add_discrete_cell(element_type, FIRST_ORDER_CELLS[element_type])
            mesh = GmshImporter().get_mesh_data()
        finally:
            gmsh.finalize()

        assert "cell_vertices" not in mesh

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
