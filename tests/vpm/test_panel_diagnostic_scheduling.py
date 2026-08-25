"""The panel-induced-velocity diagnostic must be opt-in and bounded in cost."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from source.solvers.vpm.boundary_elements.panels.solver.panel_solver import PanelSolver


def _panel_stub(*, diagnostic_interval_steps: int, refresh_count: int, sample_size: int = 4096):
    """A PanelSolver-shaped stub that records induced-velocity query sizes."""
    queried: list[int] = []
    panel = SimpleNamespace(
        diagnostic_interval_steps=diagnostic_interval_steps,
        diagnostic_sample_size=sample_size,
        refresh_count=refresh_count,
        results={"diagnostic_history": [{}]},
        compute_induced_velocity=lambda points: (
            queried.append(points.shape[0]) or np.zeros_like(points)
        ),
    )
    # Exercise the real scheduling predicate against the stub's state.
    panel.induced_velocity_diagnostic_is_due = lambda: (
        PanelSolver.induced_velocity_diagnostic_is_due(panel)
    )
    return panel, queried


def _particles(n_particles_total: int):
    position = np.tile(np.arange(n_particles_total, dtype=np.float32)[:, None], (1, 3))
    return SimpleNamespace(
        n_particles_total=n_particles_total,
        _np_float_dtype=np.float32,
        position=SimpleNamespace(to_numpy=lambda: position),
    )


def test_diagnostic_is_off_by_default():
    panel, queried = _panel_stub(diagnostic_interval_steps=0, refresh_count=1)

    PanelSolver._record_particle_velocity_diagnostic(panel, _particles(500_000))

    # Off by default means a coupled production run never pays the
    # n_panels * n_particles evaluation cost for diagnostics.
    assert queried == []
    assert panel.results["diagnostic_history"][-1] == {}


def test_diagnostic_runs_only_on_its_schedule():
    panel, queried = _panel_stub(diagnostic_interval_steps=10, refresh_count=7)
    PanelSolver._record_particle_velocity_diagnostic(panel, _particles(1000))
    assert queried == []

    panel.refresh_count = 10
    PanelSolver._record_particle_velocity_diagnostic(panel, _particles(1000))
    assert len(queried) == 1


def test_diagnostic_cost_is_bounded_by_the_sample_size():
    panel, queried = _panel_stub(diagnostic_interval_steps=1, refresh_count=1, sample_size=1000)

    PanelSolver._record_particle_velocity_diagnostic(panel, _particles(543_276))

    # The evaluated point count must track the sample size, not the half-
    # million particles a coupled cube run carries.
    assert queried[0] <= 1000
    entry = panel.results["diagnostic_history"][-1]
    assert entry["induced_velocity_sample_size"] == queried[0]
    assert entry["induced_velocity_sample_stride"] > 1


def test_diagnostic_sample_is_deterministic():
    sizes = []
    for _ in range(3):
        panel, queried = _panel_stub(diagnostic_interval_steps=1, refresh_count=1, sample_size=256)
        PanelSolver._record_particle_velocity_diagnostic(panel, _particles(10_000))
        sizes.append(queried[0])

    # A fixed stride, not a random draw, so restarts reproduce the same sample.
    assert len(set(sizes)) == 1
