"""
Regression tests for two VPM solver correctness fixes.

* ``Solver._advance_time_step`` must accumulate ``flow_time`` (adding each
  applied step size) rather than recompute ``time_step * time_step_size``, so
  the clock stays monotonic and physically correct when the step size changes
  mid-run.  Failing that, ``set_time_step_size`` could make time run backwards.

* Core Spreading (CS) diffusion is analytic (Gaussian core growth, σ² += 4νt),
  so it has no parabolic-CFL/stability upper bound.  A configured dt larger
  than h²/(4ν) must not raise a "stability limit" warning.
"""

import contextlib
import io

import numpy as np
import pytest

from source.solvers.VPM import Solver, VPMSetup
from source.solvers.VPM.config.types import AdvectionConfig, StretchingConfig, ViscousConfig

_SIGMA = 0.05
_VOLUME = (4.0 / 3.0) * np.pi * _SIGMA**3


def _cpu_solver(tmp_path, dt, viscous):
    config = VPMSetup(
        time_step_size=dt,
        processing_unit="CPU",
        stretching=StretchingConfig.disabled(),
        viscous=viscous,
        advection=AdvectionConfig(scheme="NONE"),
        backup_frequency=0,
        logging_frequency=0,
        backup_directory=str(tmp_path),
    )
    solver = Solver(setup=config)
    solver.add_vortex_particles(
        position=np.array([[0.0, 0.0, 0.0]]),
        velocity=np.zeros((1, 3)),
        circulation=np.array([[0.0, 0.0, 1.0]]),
        radius=np.array([_SIGMA]),
        volume=np.array([_VOLUME]),
        viscosity=np.array([1e-3]),
    )
    return solver


def test_flow_time_accumulates_when_dt_changes(tmp_path):
    """A step-size change mid-run must not make the clock jump or run backwards."""
    solver = _cpu_solver(tmp_path, dt=0.02, viscous=ViscousConfig(scheme="CS"))
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(3):
            solver.update_state()
    assert solver.flow_time == pytest.approx(0.06, abs=1e-12)

    solver.set_time_step_size(0.01)
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(2):
            solver.update_state()

    # Physically 3×0.02 + 2×0.01 = 0.08 s.  The old step*dt formulation would
    # report 5×0.01 = 0.05 s — i.e. the clock running backwards.
    assert solver.flow_time == pytest.approx(0.08, abs=1e-12)
    assert solver.flow_time > 0.06


def test_flow_time_is_monotonic_across_a_dt_change(tmp_path):
    solver = _cpu_solver(tmp_path, dt=0.1, viscous=ViscousConfig(scheme="CS"))
    times = []
    with contextlib.redirect_stdout(io.StringIO()):
        for step in range(6):
            solver.update_state()
            times.append(solver.flow_time)
            if step == 2:
                solver.set_time_step_size(0.001)

    assert np.all(np.diff(times) > 0)
    assert times[-1] == pytest.approx(0.1 + 0.1 + 0.1 + 0.001 * 3, abs=1e-12)


def test_cs_does_not_warn_about_stability_limit(tmp_path, capsys):
    """CS is analytic; a dt > h²/(4ν) is not an instability."""
    # h²/(4nu) = 0.0025 / (4e-3) = 0.625 s.  dt = 10 s is far beyond it.
    viscous = ViscousConfig.cs(viscosity=1e-3, characteristic_distance=_SIGMA)
    solver = _cpu_solver(tmp_path, dt=10.0, viscous=viscous)

    with contextlib.redirect_stdout(io.StringIO()):
        solver.update_state()

    assert not hasattr(solver, "_cs_dt_info")
    captured = capsys.readouterr()
    assert "[CS] WARNING" not in captured.out
    assert "stability" not in captured.out.lower()