"""Time-based VPM output scheduling tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.VPM.config import VPMSetup
from source.solvers.VPM.io.sampler import SamplerExecutor
from source.solvers.VPM.io.sampling import SamplingSchedule
from source.solvers.VPM.io.solver_io import SolverIO


def test_nearest_time_schedule_has_no_cumulative_drift() -> None:
    dt = 0.015
    schedule = SamplingSchedule(every_time=1.0)
    due = [step * dt for step in range(1, 1334) if schedule.is_due(step, step * dt, dt)]

    assert len(due) == 20
    assert np.allclose(due[:3], [1.005, 1.995, 3.0])
    assert (
        max(abs(actual - target) for actual, target in zip(due, range(1, 21), strict=True))
        <= dt / 2
    )


def test_vpm_checkpoint_uses_time_schedule() -> None:
    solver = SimpleNamespace(
        step=0,
        time=0.0,
        time_step_size=0.015,
        checkpoint_interval_steps=0,
        checkpoint_interval_time=1.0,
    )
    io = SolverIO.__new__(SolverIO)
    io.solver = solver

    due_steps = [step for step in range(1, 201) if io.should_checkpoint(step)]
    assert due_steps == [67, 133, 200]


def test_vpm_checkpoint_cadences_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="only one"):
        VPMSetup(checkpoint_interval_steps=10, checkpoint_interval_time=1.0)


def test_non_due_scheduled_sampler_does_not_touch_particle_state() -> None:
    class Particles:
        @property
        def n_particles(self):
            raise AssertionError("particle state was accessed for a non-due sampler")

    solver = SimpleNamespace(
        setup=SimpleNamespace(
            samplers=(SimpleNamespace(schedule=SamplingSchedule(every_time=0.05)),)
        ),
        particles=Particles(),
        step=1,
        time=0.015,
        time_step_size=0.015,
    )

    SamplerExecutor.execute(solver, scheduled_only=True)


def test_due_scheduled_sampler_is_selected(monkeypatch, tmp_path) -> None:
    sampler = SimpleNamespace(
        schedule=SamplingSchedule(every_time=0.05),
        file_name="probe",
    )
    solver = SimpleNamespace(
        setup=SimpleNamespace(samplers=(sampler,), sample_subdirectory=None),
        particles=SimpleNamespace(n_particles=2),
        particle_vortex_strength=np.ones((2, 3)),
        case_dir=tmp_path,
        step=3,
        time=0.045,
        time_step_size=0.015,
    )
    calls = []
    monkeypatch.setattr(SamplerExecutor, "_save_output", lambda *args: calls.append(args))

    SamplerExecutor.execute(solver, scheduled_only=True)

    assert len(calls) == 1
    assert sampler._call_count == 1
