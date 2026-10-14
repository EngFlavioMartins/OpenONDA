"""Coupled-solver construction owns both solvers and releases them on failure."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from openonda import coupler
from openonda.core.simulation.parallel import collective_phase
import openonda.fvm as fvm_api
import openonda.vpm as vpm_api
from tests.support.coupler import uniform_stream_cases


def test_vpm_construction_failure_closes_the_fvm_solver(tmp_path, monkeypatch):
    closed = []

    def fail(case):
        raise ValueError("bad particle configuration")

    original_close = fvm_api.FVMSolver.close

    def close(solver, *, failure=None):
        closed.append(failure)
        original_close(solver, failure=failure)

    monkeypatch.setattr(fvm_api.FVMSolver, "close", close)
    monkeypatch.setattr(vpm_api, "VPMSolver", fail)
    flow, particles = uniform_stream_cases(
        tmp_path, fvm_time_step_size=0.05, exchange_time_step_size=0.15, end_time=0.3
    )
    with pytest.raises(ValueError, match="bad particle configuration"):
        coupler.CoupledSolver(flow, particles, coupler.CouplerConfig())
    assert len(closed) == 1 and isinstance(closed[0], ValueError)


def test_solver_backups_belong_to_the_coupled_case(tmp_path):
    flow, particles = uniform_stream_cases(
        tmp_path, fvm_time_step_size=0.05, exchange_time_step_size=0.15, end_time=0.3
    )
    particles = replace(particles, backup=vpm_api.Backup(schedule=vpm_api.EverySteps(1)))
    with pytest.raises(ValueError, match="CouplerConfig.backup"):
        coupler.CoupledSolver(flow, particles, coupler.CouplerConfig())


def test_worker_receives_local_phase_failure():
    comm = SimpleNamespace(
        Get_size=lambda: 2,
        Ibarrier=lambda: SimpleNamespace(Test=lambda: True),
        allgather=lambda value: ["ParticleStateError: strain limit exceeded", value],
    )
    with (
        pytest.raises(RuntimeError, match="rank 0.*strain limit exceeded"),
        collective_phase(comm, "VPM output"),
    ):
        pass
