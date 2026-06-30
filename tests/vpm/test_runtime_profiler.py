"""
Runtime profiler tests — the ``RuntimeProfiler`` abstraction and its
integration with the VPM ``Solver`` time loop.

These verify the *bookkeeping* of the timing system (sections accumulate, calls
count, steps tally, reports format, reset clears) without asserting on absolute
wall-clock values, which are machine-dependent.  Taichi sync is exercised
implicitly via the CPU solver run; the unit-level checks use ``sync=None``.
"""

import time

import numpy as np
import pytest

from source.solvers.VPM import Solver, SolverConfig
from source.solvers.VPM.config.types import (
    AdvectionConfig,
    StretchingConfig,
    ViscousConfig,
)
from source.solvers.VPM.io import RuntimeProfiler

_SIGMA = 0.05


# ── Unit: the RuntimeProfiler abstraction (no GPU, sync=None) ──────────────────


def test_section_accumulates_calls_and_time():
    prof = RuntimeProfiler(sync=None)
    with prof.step():
        with prof.section("A"):
            time.sleep(0.005)
        with prof.section("B"):
            pass
        with prof.section("A"):  # second call to A
            pass

    assert prof.n_steps == 1
    assert prof._calls["A"] == 2
    assert prof._calls["B"] == 1
    assert prof._cumulative["A"] >= prof._cumulative["B"]  # A slept once
    assert prof.wall_time >= prof._cumulative["A"]  # full step bounds its sections


def test_disabled_section_is_noop_but_step_still_times():
    prof = RuntimeProfiler(enabled=False, sync=None)
    with prof.step():
        with prof.section("ignored"):
            pass
    # Disabled: no section recorded, but the step is still counted/timed.
    assert prof._cumulative == {}
    assert prof.n_steps == 1
    assert prof.wall_time >= 0.0


def test_reset_clears_all_statistics():
    prof = RuntimeProfiler(sync=None)
    with prof.step():
        with prof.section("A"):
            pass
    prof.reset()
    assert prof.n_steps == 0
    assert prof.wall_time == 0.0
    assert prof._cumulative == {} and prof._calls == {}


def test_format_report_contains_sections_and_footer():
    prof = RuntimeProfiler(sync=None)
    for _ in range(3):
        with prof.step():
            with prof.section("Velocity"):
                pass
    report = "\n".join(prof.format_report())
    assert "VPM RUNTIME PROFILE" in report
    assert "Velocity" in report
    assert "Step total" in report
    assert "3 steps" in report


def test_format_report_handles_empty_profiler():
    prof = RuntimeProfiler(sync=None)
    report = "\n".join(prof.format_report())
    assert "no sections recorded" in report


# ── Integration: profiler wired into the solver time loop ──────────────────────


def _tiny_solver(tmp_path, timing_frequency=0):
    config = SolverConfig(
        time_step_size=0.05,
        processing_unit="CPU",
        advection=AdvectionConfig(scheme="RK2"),
        stretching=StretchingConfig.disabled(),
        viscous=ViscousConfig.inviscid(),
        backup_frequency=0,
        logging_frequency=0,
        timing_frequency=timing_frequency,
        backup_directory=str(tmp_path),
    )
    solver = Solver(config=config)
    volume = (4.0 / 3.0) * np.pi * _SIGMA**3
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        circulation=np.array([[0.0, 0.0, 1.0]]),
        radius=np.array([_SIGMA]),
        volume=np.array([volume]),
        viscosity=np.array([1e-5]),
    )
    return solver


def test_solver_exposes_profiler_and_records_steps(tmp_path):
    solver = _tiny_solver(tmp_path)
    assert isinstance(solver.profiler, RuntimeProfiler)

    n_steps = 4
    for _ in range(n_steps):
        solver.update_state()

    assert solver.profiler.n_steps == n_steps
    # Core stages must have been timed.
    assert "Advection" in solver.profiler._cumulative
    assert "Stretching + diffusion" in solver.profiler._cumulative
    # The public mirror tracks the profiler's cumulative wall time.
    assert solver.simulation_time == pytest.approx(solver.profiler.wall_time)


def test_print_timing_runs_without_error(tmp_path):
    # The solver redirects stdout to its own log file, so assert on the formatted
    # report content (and that print_timing() does not raise) rather than capsys.
    solver = _tiny_solver(tmp_path)
    for _ in range(2):
        solver.update_state()
    report = "\n".join(solver.profiler.format_report())
    assert "VPM RUNTIME PROFILE" in report
    assert "2 steps" in report
    solver.print_timing()  # must not raise


def test_timing_frequency_triggers_periodic_report(tmp_path, monkeypatch):
    solver = _tiny_solver(tmp_path, timing_frequency=2)
    calls = {"n": 0}
    monkeypatch.setattr(solver.profiler, "report", lambda: calls.__setitem__("n", calls["n"] + 1))
    for _ in range(4):
        solver.update_state()
    # freq=2 over 4 steps → fires on steps 2 and 4.
    assert calls["n"] == 2


def test_timing_frequency_zero_never_reports(tmp_path, monkeypatch):
    solver = _tiny_solver(tmp_path, timing_frequency=0)
    calls = {"n": 0}
    monkeypatch.setattr(solver.profiler, "report", lambda: calls.__setitem__("n", calls["n"] + 1))
    for _ in range(3):
        solver.update_state()
    assert calls["n"] == 0
