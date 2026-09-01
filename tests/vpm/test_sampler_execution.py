"""Reliability tests for VPM sampler dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from source.solvers.vpm.config.artifacts import Samplers
from source.solvers.vpm.io.sampler import OutputManager
from source.solvers.vpm.io.solver_io import SolverIO


class _Sample:
    file_name = "probe"

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def save_csv(self, _solver, path, *, time: float) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error
        path.write_text(f"time={time}\n", encoding="utf-8")


def _solver(tmp_path, samples: tuple[object, ...]):
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
    sample = _Sample()

    OutputManager(_solver(tmp_path, (sample,))).write_all()

    assert sample.calls == 1


def test_sampler_failure_is_fatal_by_default(tmp_path):
    sample = _Sample(error=OSError("disk full"))

    with pytest.raises(RuntimeError, match="Sampler 'probe' failed at step 4") as error:
        OutputManager(_solver(tmp_path, (sample,))).write_all()

    assert isinstance(error.value.__cause__, OSError)


def test_sampler_failure_is_reported(tmp_path):
    sample = _Sample(error=OSError("disk full"))
    with pytest.raises(RuntimeError, match="disk full"):
        OutputManager(_solver(tmp_path, (sample,))).write_all()
    assert sample.calls == 1


def test_individual_sampler_can_declare_an_unmet_prerequisite(tmp_path):
    class _PrerequisiteSample(_Sample):
        def is_applicable(self, _solver) -> bool:
            return False

    sample = _PrerequisiteSample()

    OutputManager(_solver(tmp_path, (sample,))).write_all()

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
