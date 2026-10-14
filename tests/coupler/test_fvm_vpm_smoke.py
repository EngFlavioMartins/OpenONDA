"""End-to-end coupled FVM/VPM run in a uniform stream, with a restart."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from openonda import coupler, vpm
from openonda.core.vtk_output import read_collection
from tests.support.coupler import FREESTREAM_VELOCITY, uniform_stream_cases

FVM_TIME_STEP_SIZE = 0.05
EXCHANGE_TIME_STEP_SIZE = 0.15


def _solver(directory):
    flow, particles = uniform_stream_cases(
        directory,
        fvm_time_step_size=FVM_TIME_STEP_SIZE,
        exchange_time_step_size=EXCHANGE_TIME_STEP_SIZE,
        end_time=3 * EXCHANGE_TIME_STEP_SIZE,
        particle_spacing=0.125,
        vpm_output=vpm.Output(schedule=vpm.EverySteps(3)),
        max_n_particles=50_000,
    )
    config = coupler.CouplerConfig(
        freestream_velocity=FREESTREAM_VELOCITY,
        backup=coupler.Backup(schedule=coupler.EverySteps(2)),
    )
    return coupler.CoupledSolver(flow, particles, config)


def test_coupled_run_writes_matching_frames_and_restarts(tmp_path):
    with _solver(tmp_path) as solver:
        solver.run()
        expected_velocity = solver.fvm_solver.velocity.copy()
        expected_pressure = solver.fvm_solver.kinematic_pressure.copy()

        # Three FVM substeps per exchange; uniform inflow keeps a uniform field.
        assert solver.n_fvm_substeps == 3
        assert solver.fvm_solver.step == 9
        assert solver.step == 3
        assert np.allclose(expected_velocity.mean(axis=0), FREESTREAM_VELOCITY, atol=1e-6)
        assert solver.vpm_solver.particles.n_particles_total == 0

    solution = tmp_path / "solution"
    metadata = json.loads((solution / "coupler_metadata.json").read_text())
    assert metadata["solver"] == "coupler"
    assert metadata["run_status"] == {"status": "completed"}
    assert metadata["coupling"]["n_fvm_substeps"] == 3
    assert metadata["state"] == {"step": 3, "time": 3 * EXCHANGE_TIME_STEP_SIZE}

    # Scheduled backup at exchange 2 and the final backup at exchange 3.
    backups = sorted(path.name for path in (solution / "coupled").iterdir())
    assert backups == ["coupled_000002", "coupled_000003"]
    info = json.loads((solution / "coupled/coupled_000002/checkpoint_info.json").read_text())
    assert info["kind"] == "openonda.coupled_backup"
    assert (info["step"], info["fvm_step"], info["vpm_step"]) == (2, 6, 2)
    assert info["files"] == {
        "fvm": "fvm_000002.npz",
        "vpm": "vpm_000002.h5",
        "boundary_condition": "boundary_condition_000002.npz",
    }

    # Every particle frame has an FVM frame at exactly the same time.
    fvm_times = {repr(time) for time, _ in read_collection(solution / "fvm.pvd")}
    vpm_times = [repr(time) for time, _ in read_collection(solution / "vpm.pvd")]
    assert len(vpm_times) == 3  # initial, backup at exchange 2, schedule and backup at 3
    assert set(vpm_times) <= fvm_times

    coupler_log = (solution / "coupler.log").read_text()
    assert "COUPLED BACKUP" in coupler_log
    history = (solution / "coupler_diagnostics.jsonl").read_text().splitlines()
    assert [json.loads(row)["step"] for row in history] == [1, 2, 3]

    # Restart from the exchange-2 backup of a copied case.
    restart = tmp_path / "restart"
    restart.mkdir()
    shutil.copytree(solution, restart / "solution")
    shutil.rmtree(restart / "solution/coupled/coupled_000003")
    with _solver(restart) as restored:
        restored.run(start_from="latest")
        assert restored.vorticity_transfer.step == 4
        np.testing.assert_allclose(
            restored.fvm_solver.velocity, expected_velocity, rtol=0.0, atol=1e-13
        )
        np.testing.assert_allclose(
            restored.fvm_solver.kinematic_pressure, expected_pressure, rtol=0.0, atol=1e-13
        )
    history = (restart / "solution/coupler_diagnostics.jsonl").read_text().splitlines()
    assert [json.loads(row)["step"] for row in history] == [1, 2, 3]


def test_cases_must_share_their_directory(tmp_path):
    flow, _ = uniform_stream_cases(
        tmp_path / "a", fvm_time_step_size=0.05, exchange_time_step_size=0.15, end_time=0.3
    )
    _, particles = uniform_stream_cases(
        tmp_path / "b", fvm_time_step_size=0.05, exchange_time_step_size=0.15, end_time=0.3
    )
    with pytest.raises(ValueError, match="directory and variant"):
        coupler.CoupledSolver(flow, particles, coupler.CouplerConfig())


def test_exchange_must_hold_whole_fvm_steps(tmp_path):
    flow, particles = uniform_stream_cases(
        tmp_path, fvm_time_step_size=0.04, exchange_time_step_size=0.15, end_time=0.3
    )
    with pytest.raises(ValueError, match="whole multiple"):
        coupler.CoupledSolver(flow, particles, coupler.CouplerConfig())


def test_backup_times_must_fall_on_exchanges(tmp_path):
    flow, particles = uniform_stream_cases(
        tmp_path, fvm_time_step_size=0.05, exchange_time_step_size=0.15, end_time=0.3
    )
    config = coupler.CouplerConfig(backup=coupler.Backup(schedule=coupler.EveryTime(0.2)))
    with pytest.raises(ValueError, match="whole number of exchange time steps"):
        coupler.CoupledSolver(flow, particles, config)
