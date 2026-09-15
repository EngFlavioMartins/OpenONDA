"""Regression tests for the shared VPM block logger."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from source.solvers.vpm.io.logging import Logging


@pytest.fixture(autouse=True)
def reset_progress_state():
    Logging._last_progress_wall = None
    Logging._active_step = None
    Logging._reported_step = None
    Logging._pending_sections.clear()
    Logging.set_routine_messages_enabled(True)
    yield
    Logging._last_progress_wall = None
    Logging._active_step = None
    Logging._reported_step = None
    Logging._pending_sections.clear()
    Logging.set_routine_messages_enabled(True)


def test_routine_suppression_keeps_warnings_visible(capsys) -> None:
    try:
        Logging._pending_sections.clear()
        Logging.set_routine_messages_enabled(False)
        Logging.message("routine detail")
        Logging.warning("important warning")
    finally:
        Logging.set_routine_messages_enabled(True)

    output = capsys.readouterr().out
    assert "routine detail" not in output
    assert "WARNINGS" in output and "Important warning" in output


def test_progress_has_accepted_flow_and_wall_times(capsys) -> None:
    Logging.set_routine_messages_enabled(True)
    Logging.time_step(61, 0.4757141, 929.9, total_steps=100, n_particles=14080)

    output = capsys.readouterr().out
    assert "VPM TIME STEP 61 / 100" in output
    assert "FLOW TIME 4.757141e-01 s" in output
    assert "ELAPSED 00:15:29.9" in output
    assert "Active particles" in output and "14,080" in output
    assert max(map(len, output.splitlines())) <= 88
    assert "BEGIN" not in output
    assert "COMPLETED" not in output
    assert "time at start" not in output.lower()


def test_progress_is_throttled_but_final_state_is_visible(capsys):
    Logging.set_routine_messages_enabled(True)
    for step, wall in [(1, 1.0), (2, 2.0), (3, 30.0), (4, 31.0), (5, 32.0)]:
        Logging.time_step(step, step * 0.1, wall, total_steps=5)
    output = capsys.readouterr().out
    assert "VPM TIME STEP 1 / 5" in output
    assert "VPM TIME STEP 2 / 5" not in output
    assert "VPM TIME STEP 3 / 5" not in output
    assert "VPM TIME STEP 4 / 5" in output
    assert "VPM TIME STEP 5 / 5" in output


def test_begin_step_does_not_claim_that_work_has_completed(capsys):
    Logging.begin_step(7)
    assert capsys.readouterr().out == ""
    Logging.warning("failed before acceptance")
    output = capsys.readouterr().out
    assert "WARNINGS" in output and "VPM step" in output
    assert "7" in output and "Failed before acceptance" in output


@pytest.mark.parametrize(
    "source, expected_label",
    [
        ("fourier_transition_viscous_rate", "Viscous estimate / density"),
        ("direct_transition_viscous_rate", "Viscous estimate / density"),
        ("free_space_fft_energy_backward_difference", "Energy rate / density"),
    ],
)
def test_energy_rate_label_distinguishes_transition_estimates(source, expected_label, capsys):
    from source.solvers.vpm.core.solver import VPMSolver

    solver = VPMSolver.__new__(VPMSolver)
    solver._flow_integrals = {"kinetic_energy_rate_source": source}
    system = SimpleNamespace(
        step=16,
        time=0.4,
        particles=SimpleNamespace(n_particles_total=200),
        total_kinetic_energy=0.8,
        kinetic_energy_rate=-0.1,
        viscous_kinetic_energy_rate=-0.1,
        kinetic_energy_rate_source=solver.kinetic_energy_rate_source,
        vortex_strength_magnitude_sum=1.0,
        net_vortex_strength=[0.0, 0.0, 0.0],
        total_linear_impulse=[0.0, 0.0, 1.0],
        total_angular_impulse=[0.0, 0.0, 0.0],
        total_enstrophy=10.0,
        total_helicity=0.0,
    )
    system._flow_integrals = {
        "kinetic_energy_rate_source": source,
        "total_kinetic_energy": 0.8,
        "kinetic_energy_rate": -0.1,
        "viscous_kinetic_energy_rate": -0.1,
        "vortex_strength_magnitude_sum": 1.0,
        "net_vortex_strength": (0.0, 0.0, 0.0),
        "linear_impulse": (0.0, 0.0, 1.0),
        "angular_impulse": (0.0, 0.0, 0.0),
        "total_enstrophy": 10.0,
        "total_helicity": 0.0,
    }
    system._diagnostics_history = {"time": [], "vortex_centroid": []}
    Logging.flow_diagnostics(system)
    energy_line = next(
        line for line in capsys.readouterr().out.splitlines() if expected_label in line
    )
    assert expected_label in energy_line
    assert ("Viscous estimate / density" in energy_line) != ("Energy rate / density" in energy_line)
