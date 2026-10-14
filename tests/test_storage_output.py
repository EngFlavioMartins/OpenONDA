"""Regression tests for compact, ParaView-readable VTK and sampler output."""

from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np

from openonda.core.sampling import FIELD_SAMPLE_COLUMNS
from openonda.core.write_precision import cast_for_write
from openonda.vpm.sampling import LineSampler, SurfaceSampler
from tests._tutorial_helpers import load_tutorial_module


class _SamplerSolver:
    write_precision = "f16"
    vlm_solver = None
    case = SimpleNamespace(output=SimpleNamespace(compression="zlib"))

    @staticmethod
    def compute_velocity_and_gradient_at_points(points, particle_spacing):
        del particle_spacing
        count = len(points)
        velocity = np.column_stack((points[:, 0], points[:, 1], np.ones(count)))
        gradient = np.repeat(np.eye(3)[None, :, :], count, axis=0)
        return velocity, gradient


def _assert_appended_raw(path):
    file_bytes = path.read_bytes()
    assert b'<AppendedData encoding="raw">' in file_bytes
    assert b'format="appended"' in file_bytes
    assert b'compressor="vtkZLibDataCompressor"' in file_bytes


def _load_lamb_oseen_diagnostics():
    return load_tutorial_module("vpm/lamb_oseen_vortex", "assets.postprocess")


def _load_rotor_wake_plotter():
    return load_tutorial_module("vpm/rotor_flow", "assets.plot_rotor_wake_planes")


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

    output = sampler.write_vtk(_SamplerSolver(), tmp_path / "plane.vts", time=0.0)
    _assert_appended_raw(output)
    grid = pv.read(output)

    assert grid.n_points == 9
    assert set(grid.point_data) == {"velocity", "vorticity"}
    assert grid.field_data["time"][0] == 0.0
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

    from openonda.fvm.config import Output
    from openonda.fvm.io.vtk_exporter import VTKExporter
    from tests.support.fvm_mesh import structured_box

    mesh = structured_box(2, 2, 2)
    exporter = VTKExporter(mesh, Output(precision="f16", compression="zlib"))
    velocity = np.arange(mesh["n_cells"] * 3, dtype=np.float64).reshape(-1, 3)
    output = tmp_path / "fields.vtu"

    exporter.export(str(output), {"velocity": velocity})
    _assert_appended_raw(output)
    grid = pv.read(output)

    assert grid.n_cells == mesh["n_cells"]
    assert grid.cell_data["velocity"].dtype == np.float32


def test_owned_only_partition_vtu_excludes_incomplete_halo_cells(tmp_path):
    import pyvista as pv

    from openonda.fvm.config import Output
    from openonda.fvm.io.vtk_exporter import VTKExporter
    from openonda.fvm.mesh.geometry import compute_mesh_geometry
    from openonda.fvm.mesh.partition import localize_mesh_and_geometry
    from tests.support.fvm_mesh import structured_box

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
    exporter = VTKExporter(visualization, Output(compression="zlib"))
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

    output = sampler.write_csv(_SamplerSolver(), tmp_path / "line.csv", time=0.25)
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))

    assert tuple(rows[0]) == FIELD_SAMPLE_COLUMNS
    assert len(rows) == 1 + sampler.n_points
    assert all(float(row[0]) == 0.25 for row in rows[1:])


def test_scheduled_line_retains_native_times_across_sampler_manager_restart(tmp_path):
    import pandas as pd
    import pytest

    from openonda.core.output_events import OutputEvent
    from openonda.vpm import EverySteps, Samplers
    from openonda.vpm.sampling.manager import SamplerManager

    line = LineSampler(
        [0, 0, 0],
        [1, 0, 0],
        spacing=0.5,
        include_derivatives=False,
        file_name="line",
        schedule=EverySteps(1),
    )
    solver = _SamplerSolver()
    solver.samples_dir = tmp_path / "samples"
    solver.time_step_size = 0.1
    solver.case = SimpleNamespace(
        output=SimpleNamespace(compression="zlib"), samplers=Samplers(samples=(line,))
    )
    solver._restart_output_time = None
    for step in (1, 2, 3):
        solver.step, solver.time = step, step * 0.1
        # A rebuilt manager appends only after an explicit numerical restart.
        solver._restart_loaded = step > 1
        SamplerManager(solver).dispatch(OutputEvent.accepted_step)
    path = solver.samples_dir / "line.csv"
    data = pd.read_csv(path)
    np.testing.assert_allclose(data.time.unique(), [0.1, 0.2, 0.3], rtol=1e-15)
    assert data.groupby("time").size().tolist() == [3, 3, 3]
    np.testing.assert_allclose(data.velocity_x, data.position_x)
    np.testing.assert_allclose(data.velocity_z, 1.0)
    # An incompatible header cannot silently become malformed history.
    invalid_schema = "time,position_x\n0.3,0\n"
    path.write_text(invalid_schema)
    solver.step, solver.time = 4, 0.4
    with pytest.raises(RuntimeError, match="CSV header"):
        SamplerManager(solver).dispatch(OutputEvent.accepted_step)
    assert path.read_text() == invalid_schema
