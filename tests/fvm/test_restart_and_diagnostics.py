"""The FVM backup must restore the complete transient state."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from source.solvers.fvm import (
    BoundaryConfig,
    DiscretizationConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    LineSampler,
    PimpleControl,
    RunSchedule,
    TimeConfig,
    TransportConfig,
)
from source.solvers.fvm.factory import create_fvm_solver
from source.solvers.fvm.io.backup import decode_state, encode_state
from source.solvers.fvm.io.mesh_storage import load_native_mesh, save_native_mesh
from source.solvers.fvm.mesh.cartesian import structured_box
from source.solvers.fvm.mesh.validation import validate_topology
from source.solvers.fvm.sampling.base import write_pvd


def _setup() -> FVMSetup:
    return FVMSetup(
        case_name="restart_test",
        time=TimeConfig(
            time_step_size=0.01,
            end_time=0.1,
            output_schedule=RunSchedule(every_n_steps=100),
        ),
        schemes=DiscretizationConfig(convection_scheme="upwind", time_scheme="backward"),
        linear=LinearSolverConfig(linear_solver="spsolve"),
        pimple=PimpleControl(n_correctors=2),
        transport=TransportConfig(density=1.0, kinematic_viscosity=0.02),
        boundaries=[
            BoundaryConfig.inlet("xmin", [0.5, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax"),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        initial_velocity=[0.2, 0.0, 0.0],
    )


def _solver(case_dir):
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(_setup(), str(case_dir), mesh_data=structured_box(2, 2, 2))
    solver.auto_write = False
    return solver


def test_restart_restores_backward_time_history(tmp_path):
    reference = _solver(tmp_path / "reference")
    interrupted = _solver(tmp_path / "interrupted")
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(3):
            reference.advance()
        for _ in range(2):
            interrupted.advance()

    backup = tmp_path / "restart.npz"
    interrupted.save_state(backup)
    resumed = _solver(tmp_path / "resumed")
    resumed.load_state(backup)
    with contextlib.redirect_stdout(io.StringIO()):
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
        np.testing.assert_allclose(
            getattr(resumed, field_name), getattr(reference, field_name), atol=1e-13
        )
    assert resumed.time == pytest.approx(reference.time)
    assert resumed.step == reference.step


def test_run_manifest_serializes_sampler_configuration(tmp_path):
    setup = _setup()
    setup.samplers = (
        LineSampler(start=[0, 0, 0], end=[1, 0, 0], n_points=3, file_name="centreline"),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(setup, str(tmp_path), mesh_data=structured_box(2, 2, 2))
    destination = tmp_path / "run_manifest.json"
    solver.write_run_manifest(destination)
    solver.close()

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["active_components"]["immersed_boundary"] is False
    sampler = payload["configuration"]["samplers"][0]
    assert sampler["type"] == "LineSampler"
    assert sampler["file_name"] == "centreline"
    assert sampler["n_points"] == 3


def test_solver_owns_named_solution_and_sample_directories(tmp_path):
    setup = _setup()
    setup.samplers = (
        LineSampler(start=[0, 0, 0], end=[1, 0, 0], n_points=3, file_name="centreline"),
    )
    solution = tmp_path / "solution" / "coarse"
    samples = tmp_path / "samples" / "coarse"
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(
            setup,
            str(tmp_path),
            solution_dir=str(solution),
            samples_dir=str(samples),
            mesh_data=structured_box(2, 2, 2),
        )
        assert solution.is_dir()
        assert samples.is_dir()
        solver.advance()
    solver.write_run_manifest()
    solver.close()

    assert Path(solver.solution_dir) == solution
    assert Path(solver.samples_dir) == samples
    assert (solution / "fvm.log").is_file()
    assert (solution / "run_manifest.json").is_file()
    assert (samples / "centreline.csv").is_file()


def test_native_mesh_archive_round_trips_without_pickle(tmp_path):
    mesh = structured_box(2, 3, 2)
    path = save_native_mesh(mesh, tmp_path / "mesh.npz")

    restored = load_native_mesh(path)

    validate_topology(restored)
    np.testing.assert_array_equal(restored["vertex_position"], mesh["vertex_position"])
    np.testing.assert_array_equal(restored["faces"], mesh["faces"])
    assert restored["boundary"] == mesh["boundary"]


def test_solver_factory_persists_generated_mesh_in_solution_directory(tmp_path):
    solution = tmp_path / "solution" / "generated"
    with contextlib.redirect_stdout(io.StringIO()):
        solver = create_fvm_solver(
            _setup(),
            case_dir=tmp_path,
            solution_dir=solution,
            mesh=lambda: structured_box(2, 2, 2),
        )
    solver.close()

    native = solution / "mesh.npz"
    paraview = solution / "mesh.vtu"
    assert native.is_file()
    assert paraview.is_file()
    validate_topology(load_native_mesh(native))

    import pyvista as pv

    grid = pv.read(paraview)
    assert grid.n_cells == 8
    assert "cell_volume" in grid.cell_data


def test_solver_factory_builds_mesher_objects_and_persists_the_result(tmp_path):
    class Mesher:
        def build(self):
            return structured_box(2, 2, 2)

    solution = tmp_path / "solution" / "mesher-object"
    with contextlib.redirect_stdout(io.StringIO()):
        solver = create_fvm_solver(
            _setup(),
            case_dir=tmp_path,
            solution_dir=solution,
            mesh=Mesher(),
        )
    solver.close()

    assert (solution / "mesh.npz").is_file()
    assert (solution / "mesh.vtu").is_file()


def test_solver_factory_prepares_output_directories_and_log_before_meshing(tmp_path):
    solution = tmp_path / "solution" / "startup"
    samples = tmp_path / "samples" / "startup"
    observed = {}

    class Mesher:
        def build(self):
            observed["solution_exists"] = solution.is_dir()
            observed["samples_exists"] = samples.is_dir()
            observed["log"] = (solution / "fvm.log").read_text(encoding="utf-8")
            return structured_box(2, 2, 2)

    with contextlib.redirect_stdout(io.StringIO()):
        solver = create_fvm_solver(
            _setup(),
            case_dir=tmp_path,
            solution_dir=solution,
            samples_dir=samples,
            mesh=Mesher(),
        )
    solver.close()

    assert observed["solution_exists"]
    assert observed["samples_exists"]
    assert "FVM STARTUP" in observed["log"]
    assert "materializing mesh" in observed["log"]


@pytest.mark.parametrize("source_kind", ["dictionary", "file"])
def test_factory_backs_up_loaded_mesh_before_solver_admission(tmp_path, monkeypatch, source_kind):
    from source.solvers.fvm.core import solver as solver_module

    mesh = structured_box(2, 2, 2)
    source = save_native_mesh(mesh, tmp_path / "input.npz") if source_kind == "file" else mesh
    solution = tmp_path / "solution" / source_kind

    def reject(*args, **kwargs):
        import pyvista as pv

        assert pv.read(solution / "mesh.vtu").n_cells == 8
        validate_topology(load_native_mesh(solution / "mesh.npz"))
        raise ValueError("deliberate production admission failure")

    monkeypatch.setattr(solver_module, "FVMSolver", reject)
    with pytest.raises(ValueError, match="deliberate production admission"):
        create_fvm_solver(_setup(), case_dir=tmp_path, solution_dir=solution, mesh=source)


def test_mesh_backup_preserves_previous_pair_on_repeated_startup(tmp_path):
    from source.solvers.fvm.factory import _save_generated_mesh

    _save_generated_mesh(structured_box(2, 2, 2), tmp_path, _setup().output)
    previous = {name: (tmp_path / name).read_bytes() for name in ("mesh.npz", "mesh.vtu")}
    _save_generated_mesh(structured_box(3, 2, 2), tmp_path, _setup().output)
    archives = list(tmp_path.glob("mesh-backup-*"))
    assert len(archives) == 1
    for name, content in previous.items():
        assert (archives[0] / name).read_bytes() == content
    assert load_native_mesh(tmp_path / "mesh.npz")["n_cells"] == 12


def test_pvd_index_merges_existing_frames_across_restart(tmp_path):
    write_pvd(tmp_path, "slice", [(0.5, "slice_000050.vts"), (1.0, "slice_000100.vts")])
    write_pvd(tmp_path, "slice", [(1.5, "slice_000150.vts")])
    write_pvd(tmp_path, "slice", [(1.0, "slice_000100.vts")])

    datasets = ET.parse(Path(tmp_path) / "slice.pvd").findall(".//DataSet")
    assert [(float(item.attrib["timestep"]), item.attrib["file"]) for item in datasets] == [
        (0.5, "slice_000050.vts"),
        (1.0, "slice_000100.vts"),
        (1.5, "slice_000150.vts"),
    ]


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
