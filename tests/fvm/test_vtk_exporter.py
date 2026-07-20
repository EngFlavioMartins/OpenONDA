import numpy as np
import pytest

pytest.importorskip("pyvista", reason="PyVista FVM test dependency is not installed")
pytest.importorskip("vtk", reason="VTK FVM test dependency is not installed")

from source.solvers.FVM.config.types import OutputSetup
from source.solvers.FVM.io.vtk_exporter import PVDManager, VTKExporter


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
        assert "p" not in data.point_data
        assert data.cell_data["p"].shape == (n_elem,)
        assert data.cell_data["U"].shape == (n_elem, 3)
        text = path.read_text(encoding="utf-8")
        assert 'compressor="vtkLZ4DataCompressor"' in text
        assert 'format="appended"' in text
        assert '<AppendedData encoding="base64">' in text
        assert not any(".tmp" in item.name for item in tmp_path.iterdir())

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

    def test_export_can_disable_compression(self, gmsh_unit_cube, tmp_path):
        path = tmp_path / "uncompressed.vtu"
        exporter = VTKExporter(gmsh_unit_cube, OutputSetup(compression="none"))
        exporter.export(str(path), {"p": np.ones(gmsh_unit_cube["n_elements"])})

        text = path.read_text(encoding="utf-8")
        assert 'format="appended"' in text
        assert "compressor=" not in text

    def test_pvd_publication_is_atomic_and_portable(self, tmp_path):
        pvd = PVDManager(str(tmp_path / "case.pvd"))
        pvd.add_step(0.25, str(tmp_path / "state&one.vtu"))

        text = (tmp_path / "case.pvd").read_text(encoding="utf-8")
        assert 'timestep="0.25"' in text
        assert 'file="state&amp;one.vtu"' in text
        assert not any(".tmp" in item.name for item in tmp_path.iterdir())

        resumed = PVDManager(str(tmp_path / "case.pvd"))
        assert resumed.entries == [(0.25, "state&one.vtu")]
