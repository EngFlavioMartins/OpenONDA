import numpy as np
import pytest

from source.solvers.FVM.mesh.gmsh_importer import GmshImporter


class TestGmshImporter:

    def test_imported_mesh_topology(self):
        import gmsh
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

    def test_orphan_boundary_creates_default_patch(self):
        import gmsh
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
