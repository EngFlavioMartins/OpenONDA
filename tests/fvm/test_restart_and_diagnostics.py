"""The FVM checkpoint must restore the complete transient state."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from source.solvers.fvm import (
    BoundaryConfig,
    DiscretizationConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    PimpleControl,
    TimeConfig,
    TransportConfig,
)
from source.solvers.fvm.mesh.cartesian import structured_box


def _setup() -> FVMSetup:
    return FVMSetup(
        case_name="restart_contract",
        time=TimeConfig.transient(time_step_size=0.01, duration=0.1, output_interval_steps=100),
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

    checkpoint = tmp_path / "restart.npz"
    interrupted.save_state(checkpoint)
    resumed = _solver(tmp_path / "resumed")
    resumed.load_state(checkpoint)
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
