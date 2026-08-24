"""Contracts for fixed-time panel refresh after external particle replacement."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from source.solvers.vpm.boundary_elements.panels.solver.panel_solver import PanelSolver
from source.solvers.vpm.core.solver import VPMSolver


def test_panel_refresh_resolves_current_state_without_advancing_history():
    calls: list[tuple] = []
    wake_velocity = object()
    panel = SimpleNamespace(
        step=7,
        _current_time=0.3,
        results={"time_history": [0.1, 0.2, 0.3]},
        ensure_mesh_generated=lambda: calls.append(("mesh",)),
        _set_coupled_wake_velocity=lambda particles, physics: (
            calls.append(("wake", particles, physics)) or wake_velocity
        ),
        solve=lambda freestream, wake, time: calls.append(
            ("solve", np.asarray(freestream).copy(), wake, time)
        ),
    )
    particles = object()
    physics = object()

    PanelSolver.refresh_coupled_solution(
        panel,
        particles=particles,
        physics=physics,
        freestream_velocity=np.array([1.0, 0.0, 0.0]),
        time=0.3,
    )

    assert calls[0] == ("mesh",)
    assert calls[1] == ("wake", particles, physics)
    assert calls[2][0] == "solve"
    np.testing.assert_array_equal(calls[2][1], [1.0, 0.0, 0.0])
    assert calls[2][2:] == (wake_velocity, 0.3)
    assert panel.step == 7
    assert panel._current_time == 0.3
    assert panel.results["time_history"] == [0.1, 0.2, 0.3]


def test_vpm_refresh_is_limited_to_boundary_only_panel_scope():
    calls: list[dict] = []
    panel = SimpleNamespace(
        coupling_scope="vpm_boundary_condition",
        refresh_coupled_solution=lambda **kwargs: calls.append(kwargs),
    )
    solver = SimpleNamespace(
        panel_solver=panel,
        particles=object(),
        physics=object(),
        freestream_velocity=np.array([1.0, 0.0, 0.0]),
        time=0.4,
    )

    VPMSolver.refresh_boundary_element_solution(solver)

    assert calls == [
        {
            "particles": solver.particles,
            "physics": solver.physics,
            "freestream_velocity": solver.freestream_velocity,
            "time": 0.4,
        }
    ]

    panel.coupling_scope = "full"
    VPMSolver.refresh_boundary_element_solution(solver)
    assert len(calls) == 1
