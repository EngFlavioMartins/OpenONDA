"""The FVM backup must restore the complete transient state."""

from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import json

import numpy as np
import pytest

from openonda import fvm
from openonda.fvm.io.backup import decode_state, encode_state
from openonda.fvm.io.mesh_storage import load_native_mesh, save_native_mesh
from openonda.fvm.mesh.construction import save_mesh
from openonda.fvm.mesh.validation import validate_topology
from tests.support.fvm_mesh import structured_box


def _case(directory, linear: fvm.LinearSolverConfig | None = None, mesh=None, **fields):
    return fvm.FVMCase(
        name="restart_test",
        directory=directory,
        numerics=fvm.Numerics(
            time_step_size=0.01,
            schemes=fvm.DiscretizationConfig(convection_scheme="upwind", time_scheme="backward"),
            linear=linear or fvm.LinearSolverConfig(linear_solver="spsolve"),
            pimple=fvm.PimpleConfig(n_correctors=2),
            transport=fvm.TransportConfig(density=1.0, kinematic_viscosity=0.02),
        ),
        initial_conditions=fvm.InitialFields(velocity=(0.2, 0.0, 0.0)),
        run=fvm.RunPlan(end_time=0.1),
        logging=fvm.LoggingConfig(console=False),
        mesh=structured_box(2, 2, 2) if mesh is None else mesh,
        boundaries=(
            fvm.BoundaryConfig.inlet("xmin", (0.5, 0.0, 0.0)),
            fvm.BoundaryConfig.outlet("xmax"),
            fvm.BoundaryConfig.wall("ymin"),
            fvm.BoundaryConfig.wall("ymax"),
            fvm.BoundaryConfig.wall("zmin"),
            fvm.BoundaryConfig.wall("zmax"),
        ),
        **fields,
    )


def _solver(case_dir, linear=None, mesh=None):
    with contextlib.redirect_stdout(io.StringIO()):
        solver = fvm.FVMSolver(_case(case_dir, linear, mesh))
    solver.auto_write = False
    return solver


def test_restart_continues_the_uninterrupted_run_bit_for_bit(tmp_path):
    """Reused AMG/ILU preconditioners are rebuilt at each backup, as on restart."""
    linear = fvm.LinearSolverConfig(momentum_preconditioner="ilu")
    mesh = structured_box(6, 4, 4)
    reference = _solver(tmp_path / "reference", linear, mesh)
    interrupted = _solver(tmp_path / "interrupted", linear, mesh)
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(3):
            reference.advance()
            interrupted.advance()
        reference.save_backup(tmp_path / "reference.npz")
        for _ in range(3):
            reference.advance()

    backup = interrupted.save_backup()
    assert backup == tmp_path / "interrupted/solution/fvm/fvm_000003.npz"
    resumed = _solver(tmp_path / "resumed", linear, mesh)
    resumed.load_backup(backup)
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(3):
            resumed.advance()

    for field_name in (
        "velocity",
        "kinematic_pressure",
        "volumetric_face_flux",
        "volumetric_face_flux_old",
        "volumetric_face_flux_older",
        "velocity_old",
        "velocity_older",
    ):
        np.testing.assert_array_equal(getattr(resumed, field_name), getattr(reference, field_name))
    assert resumed.time == pytest.approx(reference.time)
    assert resumed.step == reference.step


def test_backup_records_configuration_and_mesh_digests(tmp_path):
    solver = _solver(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.advance()
    backup = solver.save_backup()
    solver.close()
    with np.load(backup) as archive:
        metadata = json.loads(str(archive["metadata"]))
    assert set(metadata) == {
        "format_version",
        "configuration_sha256",
        "mesh_sha256",
        "kinematic_viscosity",
    }


def test_memory_snapshot_replays_the_same_step_as_disk_restart(tmp_path):
    from openonda.fvm.io.backup import capture_restart_state, restore_restart_state

    solver = _solver(tmp_path / "memory")
    with contextlib.redirect_stdout(io.StringIO()):
        solver.advance()
        solver.advance()
        snapshot = capture_restart_state(solver)
        accepted = solver.save_backup()
        solver.advance()
        restore_restart_state(solver, snapshot)
        solver.advance()
        from_memory = capture_restart_state(solver)
        solver.load_backup(accepted)
        solver.advance()
        from_disk = capture_restart_state(solver)
    for name in from_disk.fields:
        np.testing.assert_array_equal(from_memory.fields[name], from_disk.fields[name])
    assert from_memory.time == from_disk.time
    assert from_memory.step == from_disk.step
    solver.close()


def test_solver_metadata_serializes_sampler_configuration(tmp_path):
    case = _case(
        tmp_path,
        samplers=fvm.Samplers(
            (fvm.LineSampler((0, 0, 0), (1, 0, 0), n_points=3, file_name="centreline"),)
        ),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = fvm.FVMSolver(case)
    metadata = tmp_path / "solution" / "fvm_metadata.json"
    assert json.loads(metadata.read_text(encoding="utf-8"))["run_status"] == {"status": "created"}
    solver.close()

    saved_data = json.loads(metadata.read_text(encoding="utf-8"))
    assert saved_data["solver"] == "fvm"
    assert saved_data["case_name"] == "restart_test"
    sampler = saved_data["configuration"]["samplers"][0]
    assert sampler["type"] == "LineSampler"
    assert sampler["file_name"] == "centreline"
    assert sampler["n_points"] == 3


def test_case_variant_selects_solution_and_sample_directories(tmp_path):
    case = _case(
        tmp_path,
        variant="coarse",
        samplers=fvm.Samplers(
            (fvm.LineSampler((0, 0, 0), (1, 0, 0), n_points=3, file_name="centreline"),)
        ),
    )
    solution = tmp_path / "solution" / "coarse"
    samples = tmp_path / "samples" / "coarse"
    with contextlib.redirect_stdout(io.StringIO()):
        solver = fvm.FVMSolver(case)
        assert solution.is_dir()
        assert samples.is_dir()
        solver.advance()
    solver.close()

    assert solver.solution_dir == solution
    assert solver.samples_dir == samples
    assert (solution / "fvm.log").is_file()
    assert (solution / "fvm_metadata.json").is_file()
    assert (samples / "centreline.csv").is_file()


def test_native_mesh_archive_round_trips_without_pickle(tmp_path):
    mesh = structured_box(2, 3, 2)
    path = save_native_mesh(mesh, tmp_path / "mesh.npz")

    restored = load_native_mesh(path)

    validate_topology(restored)
    np.testing.assert_array_equal(restored["vertex_position"], mesh["vertex_position"])
    np.testing.assert_array_equal(restored["faces"], mesh["faces"])
    assert restored["boundary"] == mesh["boundary"]


@pytest.mark.parametrize("source_kind", ["callable", "mesher", "file"])
def test_solver_keeps_the_case_mesh_with_its_assets(tmp_path, source_kind):
    class Mesher:
        def build(self):
            return structured_box(2, 2, 2)

    if source_kind == "file":
        source = save_native_mesh(structured_box(2, 2, 2), tmp_path / "input.npz")
    elif source_kind == "mesher":
        source = Mesher()
    else:
        source = lambda: structured_box(2, 2, 2)  # noqa: E731
    with contextlib.redirect_stdout(io.StringIO()):
        solver = fvm.FVMSolver(_case(tmp_path, mesh=source))
    solver.close()

    native = solver.case.mesh_path
    assert native == tmp_path / "assets" / "restart_test_mesh.npz"
    assert native.with_suffix(".vtu").is_file()
    validate_topology(load_native_mesh(native))

    import pyvista as pv

    grid = pv.read(native.with_suffix(".vtu"))
    assert grid.n_cells == 8
    assert "cell_volume" in grid.cell_data
    mesher_log = (tmp_path / "solution" / "mesher.log").read_text(encoding="utf-8")
    assert "mesh construction" in mesher_log and "mesh export" in mesher_log
    assert "complete" in mesher_log


def test_mesher_failure_is_recorded_in_the_mesher_log(tmp_path):
    def fail_mesh():
        raise RuntimeError("deliberate mesher failure")

    with pytest.raises(RuntimeError, match="deliberate mesher failure"):
        fvm.FVMSolver(_case(tmp_path, mesh=fail_mesh))

    mesher_log = (tmp_path / "solution" / "mesher.log").read_text(encoding="utf-8")
    assert "mesh construction" in mesher_log and "failed" in mesher_log
    assert "meshing session" in mesher_log and "failed" in mesher_log
    assert "RuntimeError" in mesher_log and "deliberate mesher failure" in mesher_log


def test_rejected_mesh_is_stored_before_geometric_validation(tmp_path, monkeypatch):
    native = _case(tmp_path).mesh_path

    def reject(*args, **kwargs):
        import pyvista as pv

        assert pv.read(native.with_suffix(".vtu")).n_cells == 8
        validate_topology(load_native_mesh(native))
        raise ValueError("deliberate production validation failure")

    monkeypatch.setattr("openonda.fvm.mesh.validation.validate_topology", reject)
    with pytest.raises(ValueError, match="deliberate production validation"):
        fvm.FVMSolver(_case(tmp_path))


def test_repeated_startup_replaces_the_case_mesh(tmp_path):
    path = tmp_path / "assets" / "case_mesh.npz"
    save_mesh(structured_box(2, 2, 2), path, fvm.Output())
    save_mesh(structured_box(3, 2, 2), path, fvm.Output())
    assert sorted(item.name for item in path.parent.iterdir()) == ["case_mesh.npz", "case_mesh.vtu"]
    assert load_native_mesh(path)["n_cells"] == 12


def test_surface_frames_merge_into_the_existing_collection_after_restart(tmp_path):
    from openonda.core.vtk_output import read_collection

    surface = fvm.SurfaceSampler((0.5, 0.5, 0.5), (0, 0, 1), (0, 1, 0, 1), 0.5, file_name="slice")
    case = replace(
        _case(tmp_path, samplers=fvm.Samplers((surface,))),
        backup=fvm.Backup(schedule=fvm.EverySteps(1)),
        run=fvm.RunPlan(end_time=0.02),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        with fvm.FVMSolver(case) as first:
            first.run(start_from="latest")
        with fvm.FVMSolver(replace(case, run=fvm.RunPlan(end_time=0.03))) as second:
            second.run(start_from="latest")
    frames = read_collection(tmp_path / "samples" / "slice.pvd")
    assert [name for _, name in frames] == [f"slice_{step:06d}.vts" for step in range(4)]


def test_backup_storage_codec_is_bit_exact_for_history_and_scalars():
    velocity = np.linspace(-2.0, 3.0, 4096, dtype=np.float64).reshape(-1, 1)
    flux = np.linspace(-1.0, 1.0, 4096, dtype=np.float64)
    state = {
        "metadata": np.asarray('{"format_version": 8}'),
        "velocity": velocity,
        "velocity_old": velocity * (1.0 + np.finfo(np.float64).eps),
        "velocity_older": velocity * (1.0 - np.finfo(np.float64).eps),
        "volumetric_face_flux": flux,
        "volumetric_face_flux_old": flux + np.finfo(np.float64).eps,
        "volumetric_face_flux_older": flux - np.finfo(np.float64).eps,
        "step": np.asarray(42, dtype=np.int64),
    }

    stored = encode_state(state)
    restored = decode_state(stored)

    assert set(stored) == {*state, "storage_layout"}
    assert restored["metadata"].shape == ()
    for name, expected in state.items():
        np.testing.assert_array_equal(restored[name], expected)

    direct, compact = io.BytesIO(), io.BytesIO()
    np.savez_compressed(direct, **state)
    np.savez_compressed(compact, **stored)
    assert len(compact.getvalue()) < len(direct.getvalue()) / 2
