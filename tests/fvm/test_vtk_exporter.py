import numpy as np
import pytest

pytest.importorskip("pyvista", reason="PyVista FVM test dependency is not installed")
pytest.importorskip("vtk", reason="VTK FVM test dependency is not installed")

from source.solvers.FVM.io.vtk_exporter import VTKExporter


class TestVTKExporter:
    def test_export_scalar_and_vector(self, gmsh_unit_cube, tmp_path):
        mesh = gmsh_unit_cube
        n_elem = mesh["n_elements"]

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
