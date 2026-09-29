"""The pressure AMG default preserves strict legacy BiCGStab restarts."""

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
from source.solvers.fvm.mesh.cartesian import structured_box


def _setup(*, legacy: bool = False) -> FVMSetup:
    return FVMSetup(
        case_name="pressure_default_restart",
        time=TimeConfig(
            time_step_size=0.01,
            end_time=0.05,
            output_schedule=RunSchedule(every_n_steps=100),
        ),
        schemes=DiscretizationConfig(convection_scheme="upwind", time_scheme="backward"),
        linear=LinearSolverConfig(pressure_solver=None if legacy else "amg"),
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
    assert config_hash(_setup(legacy=True)) != config_hash(_setup())


def test_authenticated_legacy_pressure_backup_continues_with_amg(tmp_path):
    old = _solver(_setup(legacy=True), tmp_path / "old")
    uninterrupted = _solver(_setup(), tmp_path / "uninterrupted")
    with contextlib.redirect_stdout(io.StringIO()):
        old.advance()
        uninterrupted.advance()
        uninterrupted.advance()
    backup = old.save_state(tmp_path / "old_accepted.npz")
    resumed = _solver(_setup(), tmp_path / "resumed")
    resumed.load_state(backup)
    assert resumed.step == old.step == 1
    assert resumed.time == old.time
    for name in ("velocity", "kinematic_pressure", "volumetric_face_flux"):
        np.testing.assert_array_equal(getattr(resumed, name), getattr(old, name))
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
    for solver in (old, uninterrupted, resumed):
        solver.close()


def test_legacy_migration_rejects_tampered_hash_and_changed_controls(tmp_path):
    old = _solver(_setup(legacy=True), tmp_path / "old")
    with contextlib.redirect_stdout(io.StringIO()):
        old.advance()
    backup = old.save_state(tmp_path / "old_accepted.npz")
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
            replace(_setup(), schemes=replace(_setup().schemes, time_scheme="euler")),
            backup,
        ),
    ):
        solver = _solver(setup, tmp_path / suffix)
        with pytest.raises(RuntimeError, match="configuration hash"):
            solver.load_state(candidate)
        assert solver.step == 0
        solver.close()
    old.close()
