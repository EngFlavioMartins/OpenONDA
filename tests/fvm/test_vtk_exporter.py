import numpy as np
import pytest

pytest.importorskip("pyvista", reason="PyVista FVM test dependency is not installed")
pytest.importorskip("vtk", reason="VTK FVM test dependency is not installed")

from source.solvers.FVM.config.types import OutputConfig
from source.solvers.FVM.io.vtk_exporter import PVDManager, VTKExporter


class TestVTKExporter:
    def test_export_scalar_and_vector(self, gmsh_unit_cube, tmp_path):
        mesh = gmsh_unit_cube
        n_elem = mesh["n_cells"]

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
        fields = {"p": np.ones(mesh["n_cells"])}

        path = tmp_path / "test_interp.vtu"
        exporter = VTKExporter(mesh)
        exporter.export(str(path), fields, interpolate_to_points=True)

        import pyvista as pv

        data = pv.read(str(path))
        assert "p" in data.point_data

    def test_export_float32_fields(self, gmsh_unit_cube, tmp_path):
        path = tmp_path / "float32.vtu"
        fields = {
            "p": np.ones(gmsh_unit_cube["n_cells"]),
            "U": np.ones((gmsh_unit_cube["n_cells"], 3)),
        }

        VTKExporter(gmsh_unit_cube, OutputConfig(precision="float32")).export(
            str(path),
            fields,
        )

        import pyvista as pv

        data = pv.read(str(path))
        assert data.cell_data["p"].dtype == np.float32
        assert data.cell_data["U"].dtype == np.float32

    @staticmethod
    def _graded_box_with_linear_field():
        """A graded box plus a field that is exactly linear in space."""
        from source.solvers.FVM.mesh.rectilinear import box_mesh_3d

        axis = np.array([0.0, 0.05, 0.14, 0.30, 0.58, 1.0])
        mesh = box_mesh_3d(axis, np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
        points = np.asarray(mesh["points"])
        centroids = points[np.asarray(mesh["cell_vertices"])].mean(axis=1)
        faces = mesh["faces"]
        face_centroids = np.array(
            [
                points[np.asarray(faces[f])].mean(axis=0)
                for f in range(mesh["n_interior_faces"], mesh["n_faces"])
            ]
        )

        def linear(p):
            return 2.0 * p[:, 0] - 3.0 * p[:, 1] + 0.5 * p[:, 2] + 1.0

        field = np.concatenate([linear(centroids), linear(face_centroids)])
        return mesh, points, centroids, field, linear

    def test_boundary_weighted_point_data_is_linearly_exact(self, tmp_path):
        """Wall values must survive interpolation, on a graded mesh.

        A plain cell-to-point average sees only interior cells, so it
        cannot reproduce a boundary condition at the wall, and its
        unweighted stencil is biased wherever the mesh is graded.
        """
        import pyvista as pv

        mesh, points, centroids, field, linear = self._graded_box_with_linear_field()

        path = tmp_path / "graded.vtu"
        setup = OutputConfig(asynchronous=False, point_interpolation="boundary_weighted")
        VTKExporter(mesh, setup).export(str(path), {"phi": field})
        data = pv.read(str(path))

        exact = linear(points)
        assert data.point_data["phi"] == pytest.approx(exact, abs=1e-10)
        # Cell data stays the authoritative finite-volume result.
        assert data.cell_data["phi"] == pytest.approx(linear(centroids))

        # The stock filter is materially wrong on the same mesh.
        plain = pv.read(str(path))
        plain.point_data.clear()
        stock = plain.cell_data_to_point_data().point_data["phi"]
        assert np.abs(np.asarray(stock) - exact).max() > 1e-2

    def test_point_interpolation_is_opt_in(self, tmp_path):
        """The default output stays purely cell-centred."""
        import pyvista as pv

        mesh, _points, _centroids, field, _linear = self._graded_box_with_linear_field()

        path = tmp_path / "plain.vtu"
        VTKExporter(mesh, OutputConfig(asynchronous=False)).export(str(path), {"phi": field})

        data = pv.read(str(path))
        assert "phi" in data.cell_data
        assert "phi" not in data.point_data

    def test_rejects_unknown_point_interpolation(self):
        with pytest.raises(ValueError, match="point_interpolation"):
            OutputConfig(point_interpolation="idw")

    def test_export_writes_vtu(self, gmsh_unit_cube, tmp_path):
        mesh = gmsh_unit_cube
        fields = {"p": np.ones(mesh["n_cells"])}
        exporter = VTKExporter(mesh)
        path = tmp_path / "test_write.vtu"
        exporter.export(str(path), fields, interpolate_to_points=False)
        assert path.exists()
        assert path.stat().st_size > 100

    def test_export_can_disable_compression(self, gmsh_unit_cube, tmp_path):
        path = tmp_path / "uncompressed.vtu"
        exporter = VTKExporter(gmsh_unit_cube, OutputConfig(compression="none"))
        exporter.export(str(path), {"p": np.ones(gmsh_unit_cube["n_cells"])})

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


def test_preserved_body_geometry_survives_vtk_export(tmp_path):
    """A body-preserving adaptive mesh must stay conformal in the written VTU."""
    import pyvista as pv

    from source.solvers.FVM.mesh.adaptive_cartesian import AdaptiveCartesianMesher

    from .test_preserve_body_geometry import write_box_stl

    stl = tmp_path / "body.stl"
    write_box_stl(str(stl), (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    mesher = AdaptiveCartesianMesher(
        domain=(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5),
        max_cell_size=0.5,
        surface_file=str(stl),
        wall_patch_name="cube",
        surface_cell_size=0.25,
        merge_outer_patch="numericalBoundary",
    )
    mesh = mesher.build()
    path = tmp_path / "preserved.vtu"
    exporter = VTKExporter(mesh)
    exporter.export(str(path), {"p": np.zeros(mesh["n_cells"])})

    data = pv.read(str(path))
    points = np.asarray(data.points, dtype=np.float64)
    cells = np.asarray(data.cells_dict[data.celltypes[0]]).reshape(-1, 8)
    body = np.asarray(mesher.surface_bounds, dtype=np.float64)
    lo = points[cells].min(axis=1)
    hi = points[cells].max(axis=1)
    overlap = np.maximum(
        0.0,
        np.minimum(hi, body[1::2]) - np.maximum(lo, body[::2]),
    )
    volumes = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]
    assert np.count_nonzero(volumes > 1e-12) == 0
    assert data.n_cells == mesh["n_cells"]
