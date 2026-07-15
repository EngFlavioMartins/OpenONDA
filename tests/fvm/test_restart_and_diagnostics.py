from __future__ import annotations

import contextlib
import io
import json

import numpy as np
import pytest
from scipy import sparse

from source.solvers.FVM import (
    BoundaryConfig,
    FVMConfig,
    RunAcceptancePolicy,
    Solver,
    SolverParams,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.solve.linear_interface import solve_linear_system

from ._structured_mesh import structured_box


def _config(time_scheme="euler_implicit", **solver_overrides):
    solver = SolverParams.pimple(
        n_correctors=2,
        linear_solver="spsolve",
        convection_scheme="upwind",
        time_scheme=time_scheme,
    )
    for name, value in solver_overrides.items():
        setattr(solver, name, value)
    return FVMConfig(
        case_name="restart_contract",
        time=TimeConfig.transient(dt=0.01, duration=0.1, write_interval=100),
        solver=solver,
        transport=TransportConfig(density=1.0, nu=0.02),
        boundaries=[
            BoundaryConfig.inlet("xmin", [0.5, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax"),
            BoundaryConfig.wall("ymin"),
            BoundaryConfig.wall("ymax"),
            BoundaryConfig.wall("zmin"),
            BoundaryConfig.wall("zmax"),
        ],
        initial_U=[0.2, 0.0, 0.0],
    )


def _solver(config, path):
    with contextlib.redirect_stdout(io.StringIO()):
        result = Solver(config, str(path), mesh_data=structured_box(2, 2, 2))
    result.auto_write = False
    return result


@pytest.mark.parametrize("time_scheme", ["euler_implicit", "backward"])
def test_restart_matches_uninterrupted_bdf_integration(tmp_path, time_scheme):
    reference = _solver(_config(time_scheme), tmp_path / "reference")
    split = _solver(_config(time_scheme), tmp_path / "split")
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(3):
            reference.evolve()
        for _ in range(2):
            split.evolve()
    checkpoint = tmp_path / f"{time_scheme}.restart.npz"
    split.save_state(checkpoint)

    resumed = _solver(_config(time_scheme), tmp_path / "resumed")
    resumed.load_state(checkpoint)
    with contextlib.redirect_stdout(io.StringIO()):
        resumed.evolve()

    for name in ("U", "p", "phi", "U_old", "U_old_old"):
        np.testing.assert_allclose(getattr(resumed, name), getattr(reference, name), atol=1e-13)
    assert resumed.flow_time == pytest.approx(reference.flow_time)
    assert resumed.time_step == reference.time_step
    assert resumed._n_committed == reference._n_committed


def test_restart_rejects_incompatible_config_and_mesh(tmp_path):
    original = _solver(_config(), tmp_path / "original")
    checkpoint = tmp_path / "state.npz"
    original.save_state(checkpoint)

    changed_config = _config()
    changed_config.transport.nu *= 2.0
    with pytest.raises(ValueError, match="configuration hash"):
        _solver(changed_config, tmp_path / "changed").load_state(checkpoint)

    with contextlib.redirect_stdout(io.StringIO()):
        changed_mesh = Solver(_config(), str(tmp_path / "mesh"), mesh_data=structured_box(3, 2, 2))
    with pytest.raises(ValueError, match="mesh hash"):
        changed_mesh.load_state(checkpoint)


def test_scipy_linear_result_discloses_solver_health():
    matrix = sparse.diags([-np.ones(3), 4.0 * np.ones(4), -np.ones(3)], [-1, 0, 1])
    solution, result = solve_linear_system(
        matrix.tocsr(), np.ones(4), method="bicgstab", return_info=True, tol=1e-11
    )
    assert np.all(np.isfinite(solution))
    assert result.backend == "scipy"
    assert result.method == "bicgstab"
    assert result.preconditioner == "ilu"
    assert result.converged
    assert result.iterations >= 0
    assert result.final_residual < 1e-9
    assert not result.used_fallback


def test_pimple_step_exposes_structured_diagnostics_and_outer_stop(tmp_path):
    config = _config(
        n_outer_correctors=4,
        outer_residual_tolerance=1e6,
        outer_continuity_tolerance=1e6,
    )
    solver = _solver(config, tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve_pimple()

    record = solver.last_diagnostics
    assert record.algorithm == "PIMPLE"
    assert len(record.outer_correctors) == 1
    assert len(record.linear_solves) == 5
    assert all(result.converged for result in record.linear_solves)
    assert record.nonfinite_count == 0
    assert np.isfinite(record.boundary_mass_balance)
    assert np.isfinite(record.kinetic_energy) and record.kinetic_energy >= 0.0
    assert np.isfinite(record.enstrophy) and record.enstrophy >= 0.0


def test_acceptance_policy_uses_sustained_window(tmp_path):
    config = _config()
    config.acceptance = RunAcceptancePolicy(residual_abort=0.5, sustained_steps=2)
    solver = _solver(config, tmp_path)

    def unhealthy_step(U, p, phi, *args, **kwargs):
        return U, p, phi, {"U": 1.0, "p": 1.0}

    solver.algorithm.step = unhealthy_step
    solver.algorithm.last_linear_results = ()
    solver.algorithm.last_outer_diagnostics = ()
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve_pimple()
        with pytest.raises(RuntimeError, match="2 consecutive"):
            solver.solve_pimple()


def test_piso_factory_and_validation_are_distinct():
    piso = SolverParams.piso(n_correctors=3)
    assert piso.algorithm == "PISO"
    assert piso.n_correctors == 3
    assert piso.n_outer_correctors == 1


def test_run_manifest_records_reproducibility_identity(tmp_path):
    solver = _solver(_config(), tmp_path)
    destination = tmp_path / "manifest.json"
    solver.write_run_manifest(destination)
    data = json.loads(destination.read_text())

    assert len(data["config_hash"]) == 64
    assert len(data["mesh_hash"]) == 64
    assert data["execution"]["operator_backend"] == "numpy"
    assert data["mesh"]["cells"] == solver.mesh_data["n_elements"]
    assert data["packages"]["numpy"]
