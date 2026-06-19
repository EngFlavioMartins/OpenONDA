import numpy as np
import pytest

from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.io.vtk_exporter import VTKExporter


@pytest.mark.skipif("not pytest.importorskip('pyvista', reason='pyvista not installed')")
class TestVTKExporter:

    def test_export_scalar_and_vector(self, gmsh_unit_cube, tmp_path):
        mesh = gmsh_unit_cube
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_elements"]
        n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]
        cents = geo["element_centroids"]

        fields = {
            "p": np.ones(n_elem),
            "U": np.tile([1.0, 0.0, 0.0], (n_elem, 1)),
            "Co": np.zeros(n_elem),
        }

        path = tmp_path / "test.vtu"
        exporter = VTKExporter(mesh)
        exporter.export(str(path), fields, interpolate_to_points=False)

        import pyvista as pv
        data = pv.read(str(path))
        assert "p" in data.cell_data
        assert "U" in data.cell_data
        assert "Co" in data.cell_data
        assert data.cell_data["p"].shape == (n_elem,)
        assert data.cell_data["U"].shape == (n_elem, 3)

    def test_export_point_interpolation(self, gmsh_unit_cube, tmp_path):
        mesh = gmsh_unit_cube
        fields = {"p": np.ones(mesh["n_elements"])}

        path = tmp_path / "test_interp.vtu"
        exporter = VTKExporter(mesh)
        exporter.export(str(path), fields, interpolate_to_points=True)

        import pyvista as pv
        data = pv.read(str(path))
        assert "p" in data.point_data

    def test_export_writes_vtu(self, gmsh_unit_cube, tmp_path):
        mesh = gmsh_unit_cube
        fields = {"p": np.ones(mesh["n_elements"])}
        exporter = VTKExporter(mesh)
        path = tmp_path / "test_write.vtu"
        exporter.export(str(path), fields, interpolate_to_points=False)
        assert path.exists()
        assert path.stat().st_size > 100
