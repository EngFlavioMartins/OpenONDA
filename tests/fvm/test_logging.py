"""FVM console/file logging contract."""

from __future__ import annotations

import contextlib
import io

import numpy as np
from scipy import sparse

from source.solvers.FVM import (
    BoundaryConfig,
    FVMSetup,
    LinearSolverConfig,
    LogConfig,
    PimpleControl,
    SchemesConfig,
    Solver,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.io.logging import Logging
from source.solvers.FVM.solve import linear_interface

from ._structured_mesh import structured_box


def _logging_config(log: LogConfig | None = None, steps: int = 1) -> FVMSetup:
    return FVMSetup(
        case_name="logging-contract",
        time=TimeConfig.transient(dt=0.01, duration=0.01 * steps, write_interval=100),
        schemes=SchemesConfig(convection_scheme="upwind"),
        linear=LinearSolverConfig(
            momentum_solver="bicgstab",
            pressure_solver="cg",
            momentum_tol=1e-10,
            pressure_tol=1e-10,
        ),
        pimple=PimpleControl(n_correctors=2),
        transport=TransportConfig(density=1.0, nu=0.02),
        logging=log or LogConfig(),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", 0.0),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        initial_U=[1.0, 0.0, 0.0],
        initial_p=0.0,
    )


def _run(tmp_path, config, steps: int = 1) -> str:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        solver = Solver(config, case_dir=str(tmp_path), mesh_data=structured_box(2, 2, 2))
        solver.auto_write = False
        for _ in range(steps):
            solver.evolve(0.01)
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
        "Simulation time:",
    ):
        assert marker in console

    for verbose_marker in (
        "TIME STEP  (step 1,",
        "Solver Convergence",
        "Turbulence Diagnostics",
        "PERFORMANCE PROFILE",
        "Time for this step:",
    ):
        assert verbose_marker not in console

    # One header, then one row per step.
    assert console.count("res(U)") == 1
    rows = [line for line in console.splitlines() if line.strip().startswith(("1  ", "2  ", "3  "))]
    assert len(rows) == 3


def test_debug_mode_writes_the_block_and_the_profile(tmp_path, monkeypatch):
    monkeypatch.delenv("FVM_LOG", raising=False)
    monkeypatch.delenv("FVM_PROFILE", raising=False)

    console = _run(tmp_path, _logging_config(LogConfig(mode="debug")))

    for marker in (
        "TIME STEP  (step 1,",
        "Solver Convergence",
        "Conservation",
        "Time Control",
        "Time for this step:",
        "Total simulation time:",
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

    console = _run(tmp_path, _logging_config(LogConfig(mode="simple")))

    assert "TIME STEP  (step 1," in console
    assert "res(U)" not in console


def test_interval_reports_the_first_step_and_every_nth(tmp_path, monkeypatch):
    monkeypatch.delenv("FVM_LOG", raising=False)
    monkeypatch.setenv("FVM_PROFILE", "0")

    console = _run(tmp_path, _logging_config(LogConfig(interval=3), steps=4), steps=4)

    rows = [
        line.split()[0]
        for line in console.splitlines()
        if line.startswith("  ") and line.strip()[:1].isdigit()
    ]
    assert rows == ["1", "3"]


def test_debug_interval_suppresses_the_profile_of_unreported_steps(tmp_path, monkeypatch):
    monkeypatch.delenv("FVM_LOG", raising=False)
    monkeypatch.delenv("FVM_PROFILE", raising=False)

    console = _run(tmp_path, _logging_config(LogConfig(mode="debug", interval=3), steps=4), steps=4)

    assert console.count("TIME STEP  (step ") == 2
    assert console.count("PERFORMANCE PROFILE") == 2
    assert "TIME STEP  (step 2," not in console

    written = (tmp_path / "solution" / "performance.jsonl").read_text().splitlines()
    assert len(written) == 4


def test_acceptance_warning_forces_a_report_and_is_logged(tmp_path, monkeypatch):
    monkeypatch.delenv("FVM_LOG", raising=False)
    monkeypatch.setenv("FVM_PROFILE", "0")

    config = _logging_config(LogConfig(interval=100), steps=1)
    config.acceptance.cfl_warning = 1e-12

    console = _run(tmp_path, config)

    assert "! cfl=" in console
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
    assert "(Warning) cg did not converge, info=1" in log_text
