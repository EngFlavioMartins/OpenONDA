"""Accepted-clock contract for VPM evolution orchestration."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from source.solvers.vpm.core.evolution import EvolutionStepper


class _Particles:
    n_particles_total = 0
    step = 0

    def touch_state(self) -> None:
        raise AssertionError("a failed step must not publish particle state")


class _Profiler:
    wall_time = 0.0

    def step(self):
        return nullcontext()

    def section(self, _name: str):
        return nullcontext()

    def report_step(self) -> None:
        raise AssertionError("a failed step must not report completion")


class _Stabilization:
    def begin_step(self, **_kwargs) -> None:
        return None

    def stage_clock(self, **_kwargs) -> None:
        return None

    def refresh_metrics(self, **_kwargs) -> None:
        return None

    def run_phase(self, _phase: str, **_kwargs) -> None:
        return None

    def update_residual_viscosity(self) -> None:
        return None


def test_failed_physical_phase_does_not_commit_solver_clock():
    solver = SimpleNamespace(
        step=4,
        time=0.4,
        time_step_size=0.1,
        particles=_Particles(),
        profiler=_Profiler(),
        stabilization=_Stabilization(),
        vlm_solver=None,
        panel_solver=None,
        setup=SimpleNamespace(
            advection=SimpleNamespace(scheme="RK3"),
            diagnostics=SimpleNamespace(validate_stages=False),
        ),
        physics=SimpleNamespace(velocity_override=None),
        stretching_enabled=False,
        flow_model="POTENTIAL",
        time_integration="FRACTIONAL",
        n_sources=0,
        stabilization_config=SimpleNamespace(
            stretching_viscosity_coefficient=0.0,
            pedrizzetti_relaxation_enabled=False,
        ),
        kinetic_energy_rate=0.0,
        viscous_kinetic_energy_rate=0.0,
        viscous_scheme="NONE",
        _is_particle_regeneration_pending=False,
    )
    stepper = EvolutionStepper(solver)
    stepper._update_velocities = lambda: None
    stepper._update_les_state = lambda: None
    stepper._update_positions = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        stepper.advance()

    assert (solver.step, solver.time) == (4, 0.4)
