"""Focused tests for the framework-owned VPM output runtime."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.config.artifacts import Backup, Samplers
from source.solvers.vpm.io.sampler import OutputEvent, OutputManager
from source.solvers.vpm.io.sampling import EverySteps, EveryTime, FinalOnly


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
        setup=SimpleNamespace(backup=Backup(), samplers=samplers),
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
    solver.setup.backup = Backup(interval_steps=2)
    solver._write_backup = lambda: setattr(solver, "backups", solver.backups + 1)
    manager = OutputManager(solver)

    solver.step = 1
    manager.dispatch(OutputEvent.ACCEPTED_STEP)
    solver.step = 2
    manager.dispatch(OutputEvent.ACCEPTED_STEP)

    assert solver.backups == 1
