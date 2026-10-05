"""Current pressure settings are restored with strict numerical configuration."""

import contextlib
from dataclasses import replace
import io
import json

import numpy as np
import pytest

from source.solvers.fvm import (
    BoundaryConfig,
    DiscretizationConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    PimpleControl,
    RunSchedule,
    TimeConfig,
    TransportConfig,
)
from source.solvers.fvm.io.backup import config_hash
from tests.support.fvm_mesh import structured_box


def _setup() -> FVMSetup:
    return FVMSetup(
        case_name="pressure_default_restart",
        time=TimeConfig(
            time_step_size=0.01,
            end_time=0.05,
            output_schedule=RunSchedule(every_n_steps=100),
        ),
        schemes=DiscretizationConfig(convection_scheme="upwind", time_scheme="backward"),
        linear=LinearSolverConfig(pressure_solver="amg"),
        pimple=PimpleControl(n_correctors=2),
        transport=TransportConfig(density=1.0, kinematic_viscosity=0.02),
        boundaries=[
            BoundaryConfig.inlet("xmin", [0.5, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", kinematic_pressure=0.0),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        initial_velocity=[0.2, 0.0, 0.0],
    )


def _solver(setup: FVMSetup, directory) -> FVMSolver:
    with contextlib.redirect_stdout(io.StringIO()):
        solver = FVMSolver(setup, str(directory), mesh_data=structured_box(2, 2, 2))
    solver.auto_write = False
    return solver


def test_pressure_amg_is_default_and_explicit_bicgstab_remains_available():
    assert LinearSolverConfig().pressure_solver == "amg"
    assert LinearSolverConfig(pressure_solver="bicgstab").pressure_solver == "bicgstab"
    setup = _setup()
    assert config_hash(
        replace(setup, linear=LinearSolverConfig(pressure_solver="bicgstab"))
    ) != config_hash(setup)


def test_current_pressure_backup_continues_with_identical_amg_settings(tmp_path):
    writer = _solver(_setup(), tmp_path / "writer")
    uninterrupted = _solver(_setup(), tmp_path / "uninterrupted")
    with contextlib.redirect_stdout(io.StringIO()):
        writer.advance()
        uninterrupted.advance()
        uninterrupted.advance()
    backup = writer.save_state(tmp_path / "accepted.npz")
    resumed = _solver(_setup(), tmp_path / "resumed")
    resumed.load_state(backup)
    assert resumed.step == writer.step == 1
    assert resumed.time == writer.time
    for name in ("velocity", "kinematic_pressure", "volumetric_face_flux"):
        np.testing.assert_array_equal(getattr(resumed, name), getattr(writer, name))
    with contextlib.redirect_stdout(io.StringIO()):
        resumed.advance()
    assert resumed.step == uninterrupted.step == 2
    assert resumed.time == pytest.approx(uninterrupted.time)
    assert all(result.converged for result in resumed.algorithm.last_linear_results)
    assert all(result.converged for result in uninterrupted.algorithm.last_linear_results)
    for name in ("velocity", "kinematic_pressure", "volumetric_face_flux"):
        np.testing.assert_allclose(
            getattr(resumed, name), getattr(uninterrupted, name), rtol=2e-5, atol=2e-5
        )
    for solver in (writer, uninterrupted, resumed):
        solver.close()


def test_restart_rejects_tampered_hash_and_changed_numerical_controls(tmp_path):
    writer = _solver(_setup(), tmp_path / "writer")
    with contextlib.redirect_stdout(io.StringIO()):
        writer.advance()
    backup = writer.save_state(tmp_path / "accepted.npz")
    with np.load(backup, allow_pickle=False) as archive:
        content = {name: np.array(archive[name], copy=True) for name in archive.files}
    metadata = json.loads(str(content["metadata"].item()))
    metadata["config_hash"] = "0" * 64
    content["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **content)
    for suffix, setup, candidate in (
        ("hash", _setup(), tampered),
        (
            "tolerance",
            replace(_setup(), linear=replace(_setup().linear, pressure_tolerance=1e-7)),
            backup,
        ),
        (
            "time_scheme",
            replace(_setup(), schemes=replace(_setup().schemes, time_scheme="euler_implicit")),
            backup,
        ),
    ):
        solver = _solver(setup, tmp_path / suffix)
        with pytest.raises(RuntimeError, match="configuration hash"):
            solver.load_state(candidate)
        assert solver.step == 0
        solver.close()
    writer.close()
