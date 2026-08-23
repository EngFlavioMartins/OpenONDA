"""Regression tests for PanelSolver state history across advance()/advance_time()."""

from types import SimpleNamespace

import numpy as np
import pytest

taichi = pytest.importorskip("taichi", reason="panel solver requires taichi")

from source.solvers.vpm.boundary_elements.panels.solver.panel_solver import PanelSolver


def _bare_solver():
    solver = PanelSolver.__new__(PanelSolver)
    solver._mesh_generated = True
    solver._current_time = 0.0
    solver.coupling_scope = "vpm_boundary_condition"
    solver.results = {"time_history": []}
    solver.step = 0
    solver.logging_interval_steps = 1
    solver.lattice = SimpleNamespace(save_old_doublet_strength=lambda: None, bodies=[], n_panels=4)
    return solver


def test_advance_time_reuses_wake_velocity_stored_by_advance():
    solver = _bare_solver()
    wake = np.full((4, 3), 0.37)
    freestream = np.array([2.5, -1.0, 0.125])

    recorded = {}
    solver.solve = lambda fs, wk, t: recorded.update(fs=fs, wk=wk)
    solver.advance(
        freestream_velocity=freestream,
        wake_velocity=wake,
        time_step_size=0.01,
        time=0.01,
    )
    assert recorded["wk"] is wake

    solver.advance_time(time_step_size=0.01, current_time=0.02)

    assert recorded["wk"] is wake
    assert np.array_equal(recorded["fs"], freestream)


def test_advance_time_without_history_uses_no_wake_field():
    solver = _bare_solver()
    recorded = {}
    solver.solve = lambda fs, wk, t: recorded.update(wk=wk)
    solver.advance_time(time_step_size=0.01, current_time=0.01)

    assert recorded["wk"] is None
