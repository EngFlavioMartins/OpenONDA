"""Output contracts: numerical and lifecycle contracts."""

from __future__ import annotations

import contextlib
import inspect
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import openonda.vpm as vpm
from source.solvers.vpm.config.artifacts import Backup, Samplers
from source.solvers.vpm.io.sampler import OutputEvent, OutputManager
from source.solvers.vpm.io.sampling import EverySteps, EveryTime, FinalOnly
from source.solvers.vpm.io.solver_io import SolverIO


class _Sample:
    def sample(self, _solver):
        return None


def test_backup_defaults_and_custom_directories():
    assert Backup() == Backup(0, "solution", "solution")
    configured = Backup(25, "results/state", "results/logs")
    assert configured.interval_steps == 25
    assert configured.directory == "results/state"
    assert configured.log_directory == "results/logs"


@pytest.mark.parametrize("interval", [-1, True, 1.5])
def test_backup_rejects_invalid_intervals(interval):
    error = ValueError if interval == -1 else TypeError
    with pytest.raises(error):
        Backup(interval_steps=interval)


def test_samplers_accept_a_tuple_and_own_the_directory():
    first = _Sample()
    second = _Sample()
    configured = Samplers((first, second), "ring/run_a")

    assert configured.samples == (first, second)
    assert configured.directory == "ring/run_a"


@pytest.mark.parametrize("directory", ["", "/absolute", ".", "../escape"])
def test_sampler_directory_must_stay_below_samples(directory):
    with pytest.raises(ValueError):
        Samplers(samples=(_Sample(),), directory=directory)


def test_public_case_owns_backup_and_sampler_construction_objects():
    case = vpm.VPMCase(
        numerics=vpm.Numerics(),
        backup=Backup(interval_steps=10),
        samplers=Samplers(samples=(_Sample(),)),
    )

    assert case.backup.interval_steps == 10
    assert len(case.samplers.samples) == 1
    assert {"backup", "samplers"} <= set(inspect.signature(vpm.VPMCase).parameters)


def test_vpm_solver_creates_solution_and_samples_directories_on_startup(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        solver = vpm.VPMSolver(
            vpm.VPMCase(
                directory=tmp_path,
                backup=vpm.Backup(interval_steps=0),
                numerics=vpm.Numerics(
                    time_step_size=0.01,
                    compute_device="CPU",
                    precision="f32",
                    max_n_particles=16,
                    max_evaluation_points=16,
                    induction=vpm.DirectInduction(),
                    viscous=vpm.ViscousConfig.inviscid(),
                    verbose=False,
                ),
            )
        )
    try:
        assert (tmp_path / "solution").is_dir()
        assert (tmp_path / "samples").is_dir()
        assert (tmp_path / "solution" / "vpm.log").is_file()
    finally:
        solver.close()


class _TableSampler:
    file_name = "table"
    schedule = EverySteps(1)
    include_derivatives = False

    def sample(self, _solver):
        return {
            name: np.asarray([1.0])
            for name in (
                "position_x",
                "position_y",
                "position_z",
                "velocity_x",
                "velocity_y",
                "velocity_z",
                "vorticity_x",
                "vorticity_y",
                "vorticity_z",
            )
        }


class _VtkSampler:
    file_name = "surface"
    schedule = EverySteps(1)

    def save_vtp(self, _solver, path: Path, time: float | None = None) -> None:
        del time
        path.write_text("vts", encoding="utf-8")


def _solver(tmp_path, samplers: Samplers):
    return SimpleNamespace(
        case_dir=tmp_path,
        case=SimpleNamespace(backup=Backup(), samplers=samplers),
        step=1,
        time=0.1,
        time_step_size=0.1,
        backups=0,
    )


def test_every_time_dispatches_after_crossing_a_physical_cadence():
    schedule = EveryTime(0.25)

    assert not schedule.is_due(2, 0.2, 0.1)
    assert schedule.is_due(3, 0.3, 0.1)


def test_final_only_is_not_an_accepted_step_schedule():
    schedule = FinalOnly()

    assert schedule.is_final_only
    assert not schedule.is_due(10, 1.0, 0.1)


def test_csv_series_is_atomic_and_preserved_on_resume(tmp_path):
    solver = _solver(tmp_path, Samplers(samples=(_TableSampler(),)))
    manager = OutputManager(solver)

    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    solver.step, solver.time = 2, 0.2
    manager.dispatch(OutputEvent.ACCEPTED_STEP)

    assert len((tmp_path / "samples" / "table.csv").read_text().splitlines()) == 3


def test_pvd_series_is_atomic_and_preserved_on_resume(tmp_path):
    solver = _solver(tmp_path, Samplers(samples=(_VtkSampler(),)))
    manager = OutputManager(solver)

    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    solver.step, solver.time = 2, 0.2
    manager.dispatch(OutputEvent.ACCEPTED_STEP)

    assert (tmp_path / "samples" / "surface.pvd").read_text().count("<DataSet") == 2


def test_resume_rejects_nonmonotonic_csv_event(tmp_path):
    solver = _solver(tmp_path, Samplers(samples=(_TableSampler(),)))
    manager = OutputManager(solver)
    manager.dispatch(OutputEvent.ACCEPTED_STEP)

    with pytest.raises(RuntimeError, match="duplicate or nonmonotonic"):
        manager.dispatch(OutputEvent.ACCEPTED_STEP)


def test_output_manager_is_the_only_backup_cadence_owner(tmp_path):
    solver = _solver(tmp_path, Samplers())
    solver.case.backup = Backup(interval_steps=2)
    solver._write_backup = lambda: setattr(solver, "backups", solver.backups + 1)
    manager = OutputManager(solver)

    solver.step = 1
    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    solver.step = 2
    manager.dispatch(OutputEvent.ACCEPTED_STEP)

    assert solver.backups == 1


def test_budget_terminal_sampling_keeps_due_rows_and_adds_missing_current_rows(tmp_path):
    sampler = _TableSampler()
    solver = _solver(tmp_path, Samplers(samples=(sampler,)))
    manager = OutputManager(solver)
    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    manager.write_all(OutputEvent.FINAL, skip_current=True)
    path = tmp_path / "samples/table.csv"
    assert len(path.read_text().splitlines()) == 2
    solver.step, solver.time = 2, 0.2
    manager.write_all(OutputEvent.FINAL, skip_current=True)
    assert len(path.read_text().splitlines()) == 3


class _ExecutionSample:
    file_name = "probe"

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def save_csv(self, _solver_17, path, *, time: float) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error
        path.write_text(f"time={time}\n", encoding="utf-8")


def _solver_17(tmp_path, samples: tuple[object, ...]):
    samplers = Samplers(samples=samples)
    return SimpleNamespace(
        case_dir=tmp_path,
        case=SimpleNamespace(
            samplers=samplers,
            backup=SimpleNamespace(interval_steps=0),
        ),
        setup=SimpleNamespace(samplers=samplers),
        step=4,
        time=0.2,
        time_step_size=0.05,
        particles=SimpleNamespace(n_particles_total=0),
        particle_vortex_strength=np.empty((0, 3)),
    )


def test_empty_particle_state_still_executes_configured_sampler(tmp_path):
    sample = _ExecutionSample()

    OutputManager(_solver_17(tmp_path, (sample,))).write_all()

    assert sample.calls == 1


def test_sampler_failure_is_fatal_by_default(tmp_path):
    sample = _ExecutionSample(error=OSError("disk full"))

    with pytest.raises(RuntimeError, match="Sampler 'probe' failed at step 4") as error:
        OutputManager(_solver_17(tmp_path, (sample,))).write_all()

    assert isinstance(error.value.__cause__, OSError)


def test_sampler_failure_is_reported(tmp_path):
    sample = _ExecutionSample(error=OSError("disk full"))
    with pytest.raises(RuntimeError, match="disk full"):
        OutputManager(_solver_17(tmp_path, (sample,))).write_all()
    assert sample.calls == 1


def test_individual_sampler_can_declare_an_unmet_prerequisite(tmp_path):
    class _PrerequisiteSample(_ExecutionSample):
        def is_applicable(self, _solver_17) -> bool:
            return False

    sample = _PrerequisiteSample()

    OutputManager(_solver_17(tmp_path, (sample,))).write_all()

    assert sample.calls == 0


def test_scheduled_backup_does_not_dispatch_scientific_output(tmp_path, monkeypatch):
    calls: list[str] = []
    solver = SimpleNamespace(
        _backup_path=tmp_path / "solution",
        setup=SimpleNamespace(backup=SimpleNamespace(interval_steps=2)),
        step=2,
        time=0.1,
    )
    io = SolverIO(solver)

    monkeypatch.setattr(
        "source.solvers.vpm.io.solver_io._BackupIO.save",
        lambda *_args, **_kwargs: calls.append("backup"),
    )
    monkeypatch.setattr(io, "export_state", lambda *_args, **_kwargs: calls.append("state"))
    monkeypatch.setattr(io, "_export_panel_loads", lambda *_args: calls.append("panel"))
    monkeypatch.setattr(io, "_export_vlm_results", lambda *_args: calls.append("vlm"))

    io.write_backup()

    assert calls == ["backup"]


TIME_STEP_SIZE = 0.006


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval": 0},
        {"interval": 20, "first_step": 0},
        {"interval": 20, "start_time": -1.0},
    ],
)
def test_invalid_schedules_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        EverySteps(**kwargs)


def test_step_schedule_obeys_cadence_and_both_offsets():
    # Independent predicate over a range of cadences and aligned/unaligned
    # offsets covers the combinations formerly split across five examples.
    dt = 0.006
    for interval in (1, 7, 20):
        for first in (1, 9, 20):
            for start in (0.0, 0.5, 1.5):
                schedule = EverySteps(interval=interval, first_step=first, start_time=start)
                for step in range(501):
                    expected = step >= first and step % interval == 0 and step * dt >= start
                    assert schedule.is_due(step, step * dt, dt) == expected
