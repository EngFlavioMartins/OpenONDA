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
from source.solvers.vpm.io.solver_io import SolverIO
from source.solvers.vpm.io.vlm_backup import export_vlm_backup


def _checkpoint(directory, step, *, unsteady=True):
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
