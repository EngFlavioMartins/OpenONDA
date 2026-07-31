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
    PimpleControl,
    SchemesConfig,
    Solver,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.io.logging import Logging
from source.solvers.FVM.solve import linear_interface

from ._structured_mesh import structured_box


def _logging_config() -> FVMSetup:
    return FVMSetup(
        case_name="logging-contract",
        time=TimeConfig.transient(dt=0.01, duration=0.01, write_interval=100),
        schemes=SchemesConfig(convection_scheme="upwind"),
        linear=LinearSolverConfig(
            momentum_solver="bicgstab",
            pressure_solver="cg",
            momentum_tol=1e-10,
            pressure_tol=1e-10,
        ),
        pimple=PimpleControl(n_correctors=2),
        transport=TransportConfig(density=1.0, nu=0.02),
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


def test_solver_tees_vpm_style_output_to_solution_log(tmp_path):
    stdout = io.StringIO()

    with contextlib.redirect_stdout(stdout):
        solver = Solver(
            _logging_config(),
            case_dir=str(tmp_path),
            mesh_data=structured_box(2, 2, 2),
        )
        solver.auto_write = False
        solver.evolve(0.01)
        solver.close()

    console = stdout.getvalue()
    log_path = tmp_path / "solution" / "fvm.log"
    assert log_path.is_file()
    assert log_path.read_text(encoding="utf-8") == console

    for marker in (
        "FVM Solver: Finite Volume Method",
        "FVM SOLVER INFO",
        "CONFIGURATION",
        "BOUNDARY CONDITIONS",
        "MONITORING & I/O",
        "TIME STEP  (step 1,",
        "Solver Convergence",
        "Conservation",
        "Time for this step:",
        "Total simulation time:",
    ):
        assert marker in console

    for legacy_or_verbose_marker in (
        ">>> [Time-step:",
        "Step completed in",
        "Momentum Predictor",
    ):
        assert legacy_or_verbose_marker not in console


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
