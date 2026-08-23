"""FVM console/file logging contract."""

from __future__ import annotations

import contextlib
import io

import numpy as np
from scipy import sparse

from source.solvers.fvm import (
    BoundaryConfig,
    DiscretizationConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    LoggingConfig,
    PimpleControl,
    TimeConfig,
    TransportConfig,
)
from source.solvers.fvm.io.logging import Logging
from source.solvers.fvm.solve import linear_interface

from ._structured_mesh import structured_box


def _logging_config(log: LoggingConfig | None = None, steps: int = 1) -> FVMSetup:
    return FVMSetup(
        case_name="logging-contract",
        time=TimeConfig.transient(
            time_step_size=0.01, duration=0.01 * steps, output_interval_steps=100
        ),
        schemes=DiscretizationConfig(convection_scheme="upwind"),
        linear=LinearSolverConfig(
            momentum_solver="bicgstab",
            pressure_solver="cg",
            momentum_tolerance=1e-10,
            pressure_tolerance=1e-10,
        ),
        pimple=PimpleControl(n_correctors=2),
        transport=TransportConfig(density=1.0, kinematic_viscosity=0.02),
        logging=log or LoggingConfig(),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", 0.0),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        initial_velocity=[1.0, 0.0, 0.0],
        initial_kinematic_pressure=0.0,
    )


def _run(tmp_path, config, steps: int = 1) -> str:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        solver = FVMSolver(config, case_dir=str(tmp_path), mesh_data=structured_box(2, 2, 2))
        solver.auto_write = False
        for _ in range(steps):
            solver.advance(0.01)
        solver.close()
    return stdout.getvalue()


def test_simple_mode_writes_one_row_per_step(tmp_path, monkeypatch):
    monkeypatch.delenv("FVM_LOG", raising=False)
    monkeypatch.setenv("FVM_PROFILE", "0")

    console = _run(tmp_path, _logging_config(steps=3), steps=3)

    log_path = tmp_path / "solution" / "fvm.log"
    assert log_path.is_file()
    assert log_path.read_text(encoding="utf-8") == console

    for marker in (
        "FVM Solver: Finite Volume Method",
        "FVM SOLVER INFO",
        "BOUNDARY CONDITIONS",
        "MONITORING & I/O",
        "res(U)",
        "s/step",
        "[FVM][RunTiming]",
    ):
        assert marker in console

    for verbose_marker in (
        "TIME STEP  (step 1,",
        "Solver Convergence",
        "Turbulence Diagnostics",
        "PERFORMANCE PROFILE",
        "[FVM][Timing]",
    ):
        assert verbose_marker not in console

    # One header, then one row per step.
    assert console.count("res(U)") == 1
    rows = [line for line in console.splitlines() if line.strip().startswith(("1  ", "2  ", "3  "))]
    assert len(rows) == 3


def test_debug_mode_writes_the_block_and_the_profile(tmp_path, monkeypatch):
    monkeypatch.delenv("FVM_LOG", raising=False)
    monkeypatch.delenv("FVM_PROFILE", raising=False)

    console = _run(tmp_path, _logging_config(LoggingConfig(mode="debug")))

    for marker in (
        "TIME STEP  (step 1,",
        "Solver Convergence",
        "Conservation",
        "Time Control",
        "[FVM][Timing]",
        "PERFORMANCE PROFILE",
        "Momentum Predictor",
        "Linear solvers",
        "Resident, all ranks",
        "Memory by subsystem",
    ):
        assert marker in console

    assert "res(U)" not in console
    for slop in (
        "critical path",
        "Unattributed / profiler gap",
        "Stable allocation inventory",
        "native_python_petsc_rss_remainder",
        "Aggregate RSS now",
    ):
        assert slop not in console


def test_env_variable_overrides_the_configured_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("FVM_LOG", "debug")
    monkeypatch.setenv("FVM_PROFILE", "0")

    console = _run(tmp_path, _logging_config(LoggingConfig(mode="simple")))

    assert "TIME STEP  (step 1," in console
    assert "res(U)" not in console


def test_interval_reports_the_first_step_and_every_nth(tmp_path, monkeypatch):
    monkeypatch.delenv("FVM_LOG", raising=False)
    monkeypatch.setenv("FVM_PROFILE", "0")

    console = _run(tmp_path, _logging_config(LoggingConfig(interval_steps=3), steps=4), steps=4)

    rows = [
        line.split()[0]
        for line in console.splitlines()
        if line.startswith("  ") and line.strip()[:1].isdigit()
    ]
    assert rows == ["1", "3"]


def test_debug_interval_suppresses_the_profile_of_unreported_steps(tmp_path, monkeypatch):
    monkeypatch.delenv("FVM_LOG", raising=False)
    monkeypatch.delenv("FVM_PROFILE", raising=False)

    console = _run(
        tmp_path, _logging_config(LoggingConfig(mode="debug", interval_steps=3), steps=4), steps=4
    )

    assert console.count("TIME STEP  (step ") == 2
    assert console.count("PERFORMANCE PROFILE") == 2
    assert "TIME STEP  (step 2," not in console

    written = (tmp_path / "solution" / "performance.jsonl").read_text().splitlines()
    assert len(written) == 4


def test_acceptance_warning_forces_a_report_and_is_logged(tmp_path, monkeypatch):
    monkeypatch.delenv("FVM_LOG", raising=False)
    monkeypatch.setenv("FVM_PROFILE", "0")

    config = _logging_config(LoggingConfig(interval_steps=100), steps=1)
    config.acceptance.max_courant_number_warning = 1e-12

    console = _run(tmp_path, config)

    assert "! max_courant_number=" in console
    assert "exceeds warning threshold" in console


def test_linear_fallback_warning_uses_fvm_sink(tmp_path, monkeypatch):
    monkeypatch.setattr(
        linear_interface,
        "cg",
        lambda matrix, rhs, **kwargs: (np.zeros_like(rhs), 1),
    )
    monkeypatch.setattr(
        linear_interface,
        "spsolve",
        lambda matrix, rhs: rhs.copy(),
    )
    logger = Logging(tmp_path, console=False)

    linear_interface.solve_linear_system(
        sparse.eye(3, format="csr"),
        np.ones(3),
        method="cg",
        failure_policy="direct_fallback",
        log_sink=logger,
    )
    logger.close()

    log_text = (tmp_path / "solution" / "fvm.log").read_text(encoding="utf-8")
    assert (
        "[FVM][Warning] component=linear_solver method=cg status=not_converged info=1" in log_text
    )
