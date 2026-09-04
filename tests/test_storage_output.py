"""Regression tests for compact, ParaView-readable VTK and sampler output."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from source.solvers.vpm.io.sampling import LineSampler, SurfaceSampler
from source.solvers.vpm.io.sampling.field_samplers import SAMPLER_BASE_CSV_COLUMNS
from source.write_precision import cast_for_write


class _SamplerSolver:
    write_precision = "f16"
    particles = SimpleNamespace(n_particles_total=1)

    @staticmethod
    def compute_velocity_and_gradient_at_points(points, particle_spacing):
        del particle_spacing
        count = len(points)
        velocity = np.column_stack((points[:, 0], points[:, 1], np.ones(count)))
        gradient = np.repeat(np.eye(3)[None, :, :], count, axis=0)
        return velocity, gradient


def _assert_appended_raw(path):
    payload = path.read_bytes()
    assert b'<AppendedData encoding="raw">' in payload
    assert b'format="appended"' in payload
    assert b'compressor="vtkZLibDataCompressor"' in payload


def _load_lamb_oseen_diagnostics():
    assets = Path(__file__).resolve().parents[1] / "tutorials/vpm/lamb_oseen_vortex/assets"
    spec = importlib.util.spec_from_file_location(
        "lamb_oseen_diagnostics",
        assets / "postprocess.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_rotor_wake_plotter():
    assets = Path(__file__).resolve().parents[1] / "tutorials/vpm/rotor_flow/assets"
    spec = importlib.util.spec_from_file_location(
        "rotor_wake_plotter",
        assets / "plot_rotor_wake_planes.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(assets))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_write_precision_preserves_integers_and_makes_paraview_safe_float16():
    values = np.array([-1.0, 1.0e10, np.inf], dtype=np.float64)
    written = cast_for_write(values, "f16")

    assert written.dtype == np.float32
    assert np.isfinite(written).all()
    np.testing.assert_array_equal(cast_for_write(np.array([1, 2]), "f16"), [1, 2])


def test_surface_sampler_writes_compact_paraview_readable_vts(tmp_path):
    import pyvista as pv

    sampler = SurfaceSampler(
        point=[0.0, 0.0, 0.0],
        normal=[0.0, 0.0, 1.0],
        bounds=[-1.0, 1.0, -1.0, 1.0],
        spacing=1.0,
        include_derivatives=False,
    )

    output = sampler.save_vtp(_SamplerSolver(), tmp_path / "plane.vts")
    _assert_appended_raw(output)
    grid = pv.read(output)

    assert grid.n_points == 9
    assert set(grid.point_data) == {"velocity", "vorticity"}
    assert grid.point_data["velocity"].dtype == np.float32

    field = _load_lamb_oseen_diagnostics().read_surface_field(output)
    velocity_magnitude = np.hypot(field["velocity_x"], field["velocity_y"])
    assert velocity_magnitude.shape == (3, 3)
    assert np.isfinite(velocity_magnitude).all()

    assert field["velocity_x"].shape == (3, 3)
    assert field["vorticity_z"].shape == (3, 3)

    profile, disc_mean = _load_rotor_wake_plotter()._plane_statistics(
        grid,
        freestream_speed=1.0,
        rotor_radius=1.0,
        radial_edges=np.array([0.0, 1.0, 2.0]),
    )
    assert profile.shape == (2,)
    assert np.isfinite(disc_mean)


def test_fvm_export_writes_compact_paraview_readable_vtu(tmp_path):
    import pyvista as pv

    from source.solvers.fvm.config.types import OutputConfig
    from source.solvers.fvm.io.vtk_exporter import VTKExporter
    from source.solvers.fvm.mesh.cartesian import structured_box

    mesh = structured_box(2, 2, 2)
    exporter = VTKExporter(mesh, OutputConfig(precision="f16", compression="zlib"))
    velocity = np.arange(mesh["n_cells"] * 3, dtype=np.float64).reshape(-1, 3)
    output = tmp_path / "fields.vtu"

    exporter.export(str(output), {"velocity": velocity})
    _assert_appended_raw(output)
    grid = pv.read(output)

    assert grid.n_cells == mesh["n_cells"]
    assert grid.cell_data["velocity"].dtype == np.float32


def test_owned_only_partition_vtu_excludes_incomplete_halo_cells(tmp_path):
    import pyvista as pv

    from source.solvers.fvm.config.types import OutputConfig
    from source.solvers.fvm.io.vtk_exporter import VTKExporter
    from source.solvers.fvm.mesh.cartesian import structured_box
    from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
    from source.solvers.fvm.mesh.partition import localize_mesh_and_geometry

    mesh = structured_box(4, 2, 1)
    # Exercise the general-polyhedron path used by recovered Cartesian cells.
    mesh.pop("cell_vertex_indices", None)
    mesh.pop("cell_type_code", None)
    geometry = compute_mesh_geometry(mesh, compute_lsq=False)
    localized, _local_geometry, partition = localize_mesh_and_geometry(
        mesh,
        geometry,
        rank=0,
        size=2,
        include_visualization_ghosts=False,
    )

    visualization = localized["_visualization_mesh"]
    assert visualization["n_cells"] == len(partition.owned_global_ids)
    assert visualization["n_cells"] < localized["n_cells"]

    output = tmp_path / "owned.vtu"
    exporter = VTKExporter(visualization, OutputConfig(compression="zlib"))
    exporter.export(
        str(output),
        {"global_cell_id": partition.owned_global_ids},
    )
    grid = pv.read(output)
    assert grid.n_cells == len(partition.owned_global_ids)


def test_line_sampler_omits_derivatives_from_compact_csv(tmp_path):
    sampler = LineSampler(
        start=[0.0, 0.0, 0.0],
        end=[1.0, 0.0, 0.0],
        spacing=1.0,
        include_derivatives=False,
    )
    data = {
        name: np.arange(sampler.n_points, dtype=np.float64) for name in SAMPLER_BASE_CSV_COLUMNS
    }
    sampler.sample = lambda _solver: data

    output = sampler.save_csv(None, tmp_path / "line.csv", time=0.25)
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(row for row in stream if not row.startswith("#")))

    assert rows[0] == SAMPLER_BASE_CSV_COLUMNS
    assert len(rows[1]) == len(SAMPLER_BASE_CSV_COLUMNS)
