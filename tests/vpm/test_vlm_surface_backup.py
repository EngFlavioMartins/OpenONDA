"""Surface backup time series reproduce saved geometry and loads without a solve."""

import hashlib
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pytest
import pyvista as pv

from source.solvers.vpm.boundary_elements.vlm.solver.vtk_export import write_lattice_vtk
from source.solvers.vpm.core.solver import VPMSolver
from source.solvers.vpm.io.sampler import OutputManager
from source.solvers.vpm.io.solver_io import SolverIO
from source.solvers.vpm.io.vlm_backup import export_vlm_backup, migrate_vlm_surface_series


def _checkpoint(directory, step, *, unsteady=True, time=None):
    if time is None:
        time = step * 0.125
    # Two translated/rotated panels, with unequal and signed solution fields.
    corners = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
            [[0, 1, 0], [0, 2, 0], [1, 2, 0], [1, 1, 0]],
        ],
        dtype=np.float64,
    )
    angle = time * 0.2
    rotation = np.array(
        [[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]]
    )
    corners = corners @ rotation.T + [time, -time * 2, 0.1234567890123]
    fields = {
        "panel_corner_position": corners,
        "circulation": np.array([0.7, -0.3]) * time,
        "normal": np.tile([0.0, 0.0, 1.0], (2, 1)) @ rotation.T,
        "bound_vortex_velocity": np.array([[2.0, 0.0, -0.1], [2.0, 0.0, -0.2]]),
        "panel_force": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * time,
    }
    if unsteady:
        fields.update(
            unsteady_panel_force=fields["panel_force"] * 0.2,
            unsteady_pressure_jump_coefficient=np.array([0.02, -0.01]),
        )
    path = directory / f"vpm_{step:06d}.h5"
    with h5py.File(path, "w") as file:
        solver = file.create_group("solver")
        solver.attrs.update(step=step, time=time)
        state = solver.create_group("vlm")
        state.attrs.update(reference_speed=2.0, version=6 if unsteady else 4)
        for name, value in fields.items():
            state.create_dataset(name, data=value)
    return path, fields, time


def test_backfill_preserves_checkpoint_exact_fields_and_sorted_unique_times(tmp_path):
    # Backfill final first, then older files, then repeat the final checkpoint.
    for step in (9, 4, 9):
        path, fields, time = _checkpoint(tmp_path, step)
        before = hashlib.sha256(path.read_bytes()).digest()
        result = export_vlm_backup(path)
        assert hashlib.sha256(path.read_bytes()).digest() == before
        surface = pv.read(result)
        np.testing.assert_array_equal(
            surface.points, fields["panel_corner_position"].reshape(-1, 3)
        )
        for name in ("circulation", "panel_force", "bound_vortex_velocity", "unsteady_panel_force"):
            np.testing.assert_array_equal(surface.cell_data[name], fields[name])
        np.testing.assert_allclose(surface["area"], 1.0)
        np.testing.assert_allclose(
            surface["pressure_jump_coefficient"],
            fields["circulation"] + fields["unsteady_pressure_jump_coefficient"],
        )
        assert surface.field_data["time"][0] == time
        assert surface.field_data["TimeValue"][0] == time
        from vtk import vtkStreamingDemandDrivenPipeline, vtkXMLPolyDataReader

        native_reader = vtkXMLPolyDataReader()
        native_reader.SetFileName(str(result))
        native_reader.UpdateInformation()
        assert native_reader.GetOutputInformation(0).Get(
            vtkStreamingDemandDrivenPipeline.TIME_STEPS()
        ) == (time,)
        assert surface.active_scalars_name == "circulation"
        assert surface.n_cells == 2
    entries = ET.parse(tmp_path / "vlm.pvd").findall(".//DataSet")
    assert [(float(e.attrib["timestep"]), e.attrib["file"]) for e in entries] == [
        (0.5, "vlm_000004.vtp"),
        (1.125, "vlm_000009.vtp"),
    ]
    reader = pv.get_reader(tmp_path / "vlm.pvd")
    assert reader.time_values == [0.5, 1.125]
    reader.set_active_time_value(1.125)
    assert reader.read()[0].field_data["TimeValue"][0] == 1.125


def test_older_checkpoint_omits_unrecorded_properties(tmp_path):
    path, fields, _ = _checkpoint(tmp_path, 4, unsteady=False)
    surface = pv.read(export_vlm_backup(path))
    assert "unsteady_panel_force" not in surface.cell_data
    assert "unsteady_pressure_jump_coefficient" not in surface.cell_data
    assert "is_trailing_edge" not in surface.cell_data
    np.testing.assert_allclose(surface["pressure_jump_coefficient"], fields["circulation"])


def test_vlm_export_labels_velocity_frames_and_dimensional_loads(tmp_path):
    from source.solvers.vpm.boundary_elements.vlm.solver.vtk_export import CELL_FIELDS

    corners = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])
    fields = {
        "panel_corner_position": corners,
        "normal": np.array([[0.0, 0.0, 1.0]]),
        "area": np.array([1.0]),
        "circulation": np.array([0.4]),
        "velocity": np.array([[3.0, 4.0, 0.0]]),
        "kinematic_velocity": np.array([[1.0, 0.0, 0.0]]),
        "relative_velocity": np.array([[2.0, 4.0, 0.0]]),
        "bound_vortex_velocity": np.array([[3.0, 4.0, 0.0]]),
        "bound_kinematic_velocity": np.array([[1.0, 0.0, 0.0]]),
        "bound_relative_velocity": np.array([[2.0, 4.0, 0.0]]),
        "bound_external_velocity": np.array([[3.0, 4.0, 0.0]]),
        "panel_force": np.array([[0.0, 0.0, 2.0]]),
    }
    path = write_lattice_vtk(
        fields,
        tmp_path / "vlm_000001.vtp",
        reference_speed=2.0,
        time=0.1,
        force_density=2.0,
    )
    surface = pv.read(path)
    np.testing.assert_allclose(surface["relative_velocity"], fields["relative_velocity"])
    np.testing.assert_allclose(
        surface["bound_relative_velocity"], fields["bound_relative_velocity"]
    )
    # dot(F, n)/(0.5*rho*U_ref^2*A) = 2/(0.5*2*4*1) = 0.5.
    np.testing.assert_allclose(surface["panel_normal_load_coefficient"], [0.5])
    np.testing.assert_allclose(surface["circulation_pressure_jump_proxy"], [0.4])
    assert "pressure_jump_coefficient" not in CELL_FIELDS
    assert surface.field_data["force_density"][0] == 2.0
    assert "inertial fluid velocity" in surface.field_data["velocity_frame"][0]
    assert (
        "legacy circulation proxy" in surface.field_data["pressure_jump_coefficient_definition"][0]
    )
    assert "compatibility-only" in surface.field_data["pressure_jump_coefficient_status"][0]

    angle = np.pi / 2.0
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    rotated = {
        **fields,
        "panel_corner_position": corners @ rotation.T + [4.0, -2.0, 1.0],
    }
    for name in (
        "normal",
        "velocity",
        "kinematic_velocity",
        "relative_velocity",
        "bound_vortex_velocity",
        "bound_kinematic_velocity",
        "bound_relative_velocity",
        "bound_external_velocity",
        "panel_force",
    ):
        rotated[name] = fields[name] @ rotation.T
    rotated_surface = pv.read(
        write_lattice_vtk(
            rotated,
            tmp_path / "vlm_000002.vtp",
            reference_speed=2.0,
            time=0.2,
            force_density=2.0,
        )
    )
    np.testing.assert_allclose(
        rotated_surface["panel_normal_load_coefficient"],
        surface["panel_normal_load_coefficient"],
    )
    np.testing.assert_allclose(
        rotated_surface["relative_velocity"], fields["relative_velocity"] @ rotation.T
    )


def test_particle_only_checkpoint_does_not_create_surface(tmp_path):
    path = tmp_path / "vpm_000001.h5"
    with h5py.File(path, "w") as file:
        file.create_group("solver")
    assert export_vlm_backup(path) is None
    assert not (tmp_path / "vlm.pvd").exists()


def test_failed_surface_write_preserves_previous_file_and_index(tmp_path, monkeypatch):
    path, _, _ = _checkpoint(tmp_path, 4)
    result = export_vlm_backup(path)
    before = result.read_bytes(), (tmp_path / "vlm.pvd").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("source.solvers.vpm.io.vlm_backup.write_lattice_vtk", fail)
    with pytest.raises(OSError, match="disk full"):
        export_vlm_backup(path)
    assert (result.read_bytes(), (tmp_path / "vlm.pvd").read_bytes()) == before


def test_scheduled_and_manual_backups_share_surface_writer(tmp_path, monkeypatch):
    _, fields, _ = _checkpoint(tmp_path, 4)
    calls = []

    def save_results(filename, *, time):
        return write_lattice_vtk(fields, f"{filename}.vtp", reference_speed=2.0, time=time)

    solver = SimpleNamespace(
        _backup_path=tmp_path,
        step=4,
        time=0.5,
        vlm_solver=SimpleNamespace(save_results=save_results),
        _sync_restart_state=lambda: None,
        _refresh_backup_particle_fields=lambda: None,
        _write_run_manifest=lambda *args: None,
        _run_started=True,
    )
    solver.io = SolverIO(solver)
    monkeypatch.setattr(
        "source.solvers.vpm.io.solver_io._BackupIO.save", lambda *a, **k: calls.append(solver.step)
    )
    solver.io.write_backup()
    solver.step, solver.time = 9, 1.125
    VPMSolver.save_backup(solver)
    assert calls == [4, 9]
    assert pv.get_reader(tmp_path / "vlm.pvd").time_values == [0.5, 1.125]
    assert not (tmp_path / "samples").exists()


def test_migrate_legacy_surface_series_merges_and_deduplicates_only_identical_frames(tmp_path):
    samples = tmp_path / "samples" / "case"
    solution = tmp_path / "solution"
    samples.mkdir(parents=True)
    solution.mkdir()
    for step, time in ((1, 0.1), (2, 0.2), (3, 0.3)):
        _checkpoint(solution, step, time=time)
    (samples / "vlm_000001.vtp").write_bytes(b"frame-1")
    (samples / "vlm_000002.vtp").write_bytes(b"frame-2")
    (solution / "vlm_000002.vtp").write_bytes(b"frame-2")
    (solution / "vlm_000003.vtp").write_bytes(b"frame-3")
    OutputManager._write_pvd(
        samples,
        "vlm",
        [(0.1, "vlm_000001.vtp"), (0.2, "vlm_000002.vtp")],
    )
    OutputManager._write_pvd(
        solution,
        "vlm",
        [(0.2, "vlm_000002.vtp"), (0.3, "vlm_000003.vtp")],
    )

    report = migrate_vlm_surface_series(samples, solution)

    assert report.moved_frames == 1
    assert report.deduplicated_frames == 1
    assert report.solution_frames == 3
    assert sorted(path.name for path in solution.glob("vlm_*.vtp")) == [
        "vlm_000001.vtp",
        "vlm_000002.vtp",
        "vlm_000003.vtp",
    ]
    assert not (samples / "vlm.pvd").exists()
    assert not list(samples.glob("vlm_*.vtp"))
    entries = ET.parse(solution / "vlm.pvd").findall(".//DataSet")
    assert [(float(entry.attrib["timestep"]), entry.attrib["file"]) for entry in entries] == [
        (0.1, "vlm_000001.vtp"),
        (0.2, "vlm_000002.vtp"),
        (0.3, "vlm_000003.vtp"),
    ]


def test_migrate_legacy_surface_series_preserves_conflicting_source(tmp_path):
    samples = tmp_path / "samples"
    solution = tmp_path / "solution"
    samples.mkdir()
    solution.mkdir()
    _checkpoint(solution, 1, time=0.1)
    (samples / "vlm_000001.vtp").write_bytes(b"source")
    (solution / "vlm_000001.vtp").write_bytes(b"different")
    OutputManager._write_pvd(samples, "vlm", [(0.1, "vlm_000001.vtp")])
    OutputManager._write_pvd(solution, "vlm", [(0.1, "vlm_000001.vtp")])

    with pytest.raises(ValueError, match="not byte-identical"):
        migrate_vlm_surface_series(samples, solution)
    assert (samples / "vlm_000001.vtp").is_file()
    assert (samples / "vlm.pvd").is_file()


@pytest.mark.parametrize("location", ["samples", "solution"])
def test_migration_rejects_unpaired_surface_before_changing_files(tmp_path, location):
    samples = tmp_path / "samples"
    solution = tmp_path / "solution"
    samples.mkdir()
    solution.mkdir()
    _checkpoint(solution, 1, time=0.1)
    (solution / "vlm_000001.vtp").write_bytes(b"paired-frame")
    OutputManager._write_pvd(solution, "vlm", [(0.1, "vlm_000001.vtp")])
    unpaired = tmp_path / location
    (unpaired / "vlm_000002.vtp").write_bytes(b"surface-without-particles")
    entries = [(0.2, "vlm_000002.vtp")]
    if unpaired == solution:
        entries.insert(0, (0.1, "vlm_000001.vtp"))
    OutputManager._write_pvd(unpaired, "vlm", entries)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    with pytest.raises(ValueError, match="no matching VPM backup"):
        migrate_vlm_surface_series(samples, solution)

    assert {
        p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    } == before


@pytest.mark.parametrize("native_step,native_time", [(2, 0.1), (1, 0.2), (1, np.nan)])
def test_migration_rejects_conflicting_native_clock(tmp_path, native_step, native_time):
    samples = tmp_path / "samples"
    solution = tmp_path / "solution"
    samples.mkdir()
    solution.mkdir()
    checkpoint, _, _ = _checkpoint(solution, 1, time=0.1)
    with h5py.File(checkpoint, "r+") as file:
        file["solver"].attrs.update(step=native_step, time=native_time)
    (samples / "vlm_000001.vtp").write_bytes(b"frame")
    OutputManager._write_pvd(samples, "vlm", [(0.1, "vlm_000001.vtp")])
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    with pytest.raises(ValueError, match="step/time conflicts with VPM backup"):
        migrate_vlm_surface_series(samples, solution)

    assert {
        p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    } == before
