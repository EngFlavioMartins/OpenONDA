from __future__ import annotations

import contextlib
import errno
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from source.solvers.FVM import (
    BoundaryConfig,
    DiscretizationConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    PimpleControl,
    RunAcceptancePolicy,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.io.storage import InsufficientStorageError
from source.solvers.FVM.sampling.base import SamplingSchedule
from source.solvers.FVM.sampling.forces import ForceSampler
from source.solvers.FVM.solve.linear_interface import solve_linear_system
from source.solvers.FVM.solve.simple_solver import SIMPLESolver

from ._structured_mesh import structured_box


def _config(time_scheme="euler_implicit", samplers=None, **solver_overrides):
    solver_schemes = DiscretizationConfig(convection_scheme="upwind", time_scheme=time_scheme)
    solver_linear = LinearSolverConfig(linear_solver="spsolve")
    solver_pimple = PimpleControl(n_correctors=2)
    for name, value in solver_overrides.items():
        for group in (solver_schemes, solver_linear, solver_pimple):
            if hasattr(group, name):
                setattr(group, name, value)
                break
        else:
            raise AttributeError(f"no solver group owns field {name!r}")
    return FVMSetup(
        case_name="restart_contract",
        time=TimeConfig.transient(time_step_size=0.01, duration=0.1, output_interval_steps=100),
        schemes=solver_schemes,
        linear=solver_linear,
        pimple=solver_pimple,
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
        samplers=samplers,
    )


def _solver(config, path):
    with contextlib.redirect_stdout(io.StringIO()):
        result = FVMSolver(config, str(path), mesh_data=structured_box(2, 2, 2))
    result.auto_write = False
    return result


@pytest.mark.parametrize("time_scheme", ["euler_implicit", "backward"])
def test_restart_matches_uninterrupted_bdf_integration(tmp_path, time_scheme):
    reference = _solver(_config(time_scheme), tmp_path / "reference")
    split = _solver(_config(time_scheme), tmp_path / "split")
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(3):
            reference.advance()
        for _ in range(2):
            split.advance()
    checkpoint = tmp_path / f"{time_scheme}.restart.npz"
    split.save_state(checkpoint)

    resumed = _solver(_config(time_scheme), tmp_path / "resumed")
    resumed.load_state(checkpoint)
    with contextlib.redirect_stdout(io.StringIO()):
        resumed.advance()

    for name in (
        "velocity",
        "kinematic_pressure",
        "face_flux",
        "face_flux_old",
        "face_flux_older",
        "velocity_old",
        "velocity_older",
    ):
        np.testing.assert_allclose(getattr(resumed, name), getattr(reference, name), atol=1e-13)
    assert resumed.time == pytest.approx(reference.time)
    assert resumed.step == reference.step
    assert resumed._n_committed == reference._n_committed


def test_restart_rejects_incompatible_config_and_mesh(tmp_path):
    original = _solver(_config(), tmp_path / "original")
    checkpoint = tmp_path / "state.npz"
    original.save_state(checkpoint)

    changed_config = _config()
    changed_config.transport.kinematic_viscosity *= 2.0
    with pytest.raises(ValueError, match="configuration hash"):
        _solver(changed_config, tmp_path / "changed").load_state(checkpoint)

    with contextlib.redirect_stdout(io.StringIO()):
        changed_mesh = FVMSolver(
            _config(), str(tmp_path / "mesh"), mesh_data=structured_box(3, 2, 2)
        )
    with pytest.raises(ValueError, match="mesh hash"):
        changed_mesh.load_state(checkpoint)


def test_restart_allows_an_explicit_end_time_extension(tmp_path):
    original = _solver(_config(), tmp_path / "original")
    checkpoint = tmp_path / "state.npz"
    original.save_state(checkpoint)

    extended = _config()
    extended.time.end_time = 0.2
    restored = _solver(extended, tmp_path / "extended")
    restored.load_state(checkpoint, allow_config_change=True)

    assert restored.time == original.time
    np.testing.assert_array_equal(restored.velocity, original.velocity)


def _force_sampler():
    return ForceSampler(
        patch_names=["___none__"],
        ref_velocity=1.0,
        ref_area=1.0,
        ref_length=1.0,
        schedule=SamplingSchedule(every_n_steps=1),
    )


def test_checkpoint_round_trips_sampler_config(tmp_path):
    samplers = [_force_sampler()]
    original = _solver(_config(samplers=samplers), tmp_path / "original")
    checkpoint = tmp_path / "state.npz"
    original.save_state(checkpoint)

    restored = _solver(_config(samplers=samplers), tmp_path / "restored")
    restored.load_state(checkpoint)

    assert restored.setup.samplers == original.setup.samplers


def test_checkpoint_rejects_missing_sampler_config(tmp_path):
    original = _solver(_config(samplers=[_force_sampler()]), tmp_path / "original")
    checkpoint = tmp_path / "state.npz"
    original.save_state(checkpoint)

    with pytest.raises(ValueError, match="configuration hash"):
        _solver(_config(), tmp_path / "restored").load_state(checkpoint)


def test_restart_rewinds_append_only_histories(tmp_path):
    solver = _solver(_config(), tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(2):
            solver.advance()
    checkpoint = tmp_path / "state.npz"
    solver.save_state(checkpoint)

    samples = tmp_path / "samples"
    samples.mkdir(exist_ok=True)
    solution = tmp_path / "solution"
    solution.mkdir(exist_ok=True)
    (samples / "forces_history.csv").write_text(
        "time,patch,Cd\n0.01,cube,1.0\n0.02,cube,1.1\n0.03,cube,9.9\n"
    )
    (solution / "diagnostics.jsonl").write_text('{"time": 0.01}\n{"time": 0.02}\n{"time": 0.03}\n')

    solver.load_state(checkpoint)

    assert "0.03" not in (samples / "forces_history.csv").read_text()
    assert "0.03" not in (solution / "diagnostics.jsonl").read_text()


def test_checkpoint_capacity_preflight_preserves_previous_restart(tmp_path, monkeypatch):
    solver = _solver(_config(), tmp_path)
    checkpoint = tmp_path / "state.npz"
    checkpoint.write_bytes(b"known-good-prior-generation")
    usage = SimpleNamespace(total=100, used=99, free=1)
    monkeypatch.setattr("source.solvers.FVM.io.storage.shutil.disk_usage", lambda _path: usage)

    with pytest.raises(InsufficientStorageError):
        solver.save_state(checkpoint)

    assert checkpoint.read_bytes() == b"known-good-prior-generation"


def test_diagnostics_disk_full_preserves_step_and_disables_future_writes(tmp_path, monkeypatch):
    solver = _solver(_config(), tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve_pimple()

    calls = 0

    def disk_full(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise OSError(errno.ENOSPC, "test disk full")

    monkeypatch.setattr("source.solvers.FVM.io.solver_io.append_line_recoverably", disk_full)
    solver.io.write_step_diagnostics()
    solver.io.write_step_diagnostics()

    assert calls == 1
    assert solver.io._diagnostics_write_disabled


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


def test_steady_simple_does_not_confuse_linear_and_nonlinear_convergence(monkeypatch):
    solver = SIMPLESolver.__new__(SIMPLESolver)
    solver.params = {
        "max_iter": 2,
        "tolerance": 1e-3,
        "velocity_relaxation": 0.7,
        "pressure_relaxation": 0.3,
    }
    solver.mesh_data = {}
    solver.geo_data = {}
    solver.residuals = []
    increments = iter((1.0, 0.1))

    def step(velocity, p, face_flux, **kwargs):
        solver.last_res_p = 1e-14
        solver.last_res_u = 1e-14
        solver.last_outer_diagnostics = (SimpleNamespace(continuity_max=1e-14),)
        return velocity, p, face_flux, {"U_increment": next(increments)}

    solver.step = step
    monkeypatch.setattr(
        "source.solvers.FVM.assemble.convection.compute_volumetric_face_flux",
        lambda *args: np.zeros(1),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        *_, converged = solver.solve(np.zeros((1, 3)), np.zeros(1))

    assert not converged
    assert len(solver.residuals) == 2
    assert solver.residuals[-1]["R_u"] == pytest.approx(0.1)


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
    # The criteria are met on the first corrector, so the loop stops well short
    # of the configured four.  It still runs one more: ``pimpleControl::loop``
    # flags the corrector *after* the satisfied one as final and runs it
    # unrelaxed, so the committed step is never a relaxed one.
    assert len(record.outer_correctors) == 2
    assert len(record.linear_solves) == 10
    assert all(result.converged for result in record.linear_solves)
    assert sum(result.solve_seconds > 0.0 for result in record.linear_solves[:3]) == 1
    assert record.nonfinite_count == 0
    assert np.isfinite(record.boundary_mass_balance)
    assert np.isfinite(record.kinetic_energy) and record.kinetic_energy >= 0.0
    assert np.isfinite(record.enstrophy) and record.enstrophy >= 0.0


def test_pimple_releases_previous_step_derived_fields_before_allocating(monkeypatch, tmp_path):
    solver = _solver(_config(), tmp_path)
    solver._derived_fields.update(
        {
            "velocity_gradient": np.empty((8, 3, 3)),
            "vorticity": np.empty((8, 3)),
            ("courant", 0.01): np.empty(8),
        }
    )
    original = solver.compute_effective_viscosity
    observed = False

    def compute_effective_viscosity():
        nonlocal observed
        observed = True
        assert not solver._derived_fields
        return original()

    monkeypatch.setattr(solver, "compute_effective_viscosity", compute_effective_viscosity)
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve_pimple()

    assert observed


def test_acceptance_policy_uses_sustained_window(tmp_path):
    config = _config()
    config.acceptance = RunAcceptancePolicy(residual_abort=0.5, sustained_steps=2)
    solver = _solver(config, tmp_path)

    def unhealthy_step(velocity, p, face_flux, *args, **kwargs):
        return velocity, p, face_flux, {"U": 1.0, "p": 1.0}

    solver.algorithm.step = unhealthy_step
    solver.algorithm.last_linear_results = ()
    solver.algorithm.last_outer_diagnostics = ()
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve_pimple()
        with pytest.raises(RuntimeError, match="2 consecutive"):
            solver.solve_pimple()


def test_piso_factory_and_validation_are_distinct():
    piso_pimple = PimpleControl(algorithm="PISO", n_correctors=3)
    assert piso_pimple.algorithm == "PISO"
    assert piso_pimple.n_correctors == 3
    assert piso_pimple.n_outer_correctors == 1


def test_run_manifest_records_reproducibility_identity(tmp_path):
    solver = _solver(_config(), tmp_path)
    destination = tmp_path / "manifest.json"
    solver.write_run_manifest(destination)
    data = json.loads(destination.read_text())

    assert len(data["config_hash"]) == 64
    assert len(data["mesh_hash"]) == 64
    assert data["execution"]["operator_backend"] == "numpy"
    assert data["mesh"]["cells"] == solver.mesh_data["n_cells"]
    assert data["packages"]["numpy"]
