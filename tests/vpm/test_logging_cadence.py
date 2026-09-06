"""Regression tests for the compact VPM block logger."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from source import log_style
from source.solvers.vpm.io.logging import Logging


@pytest.fixture(autouse=True)
def reset_progress_state():
    Logging._last_progress_wall = None
    Logging._active_step = None
    yield
    Logging._last_progress_wall = None
    Logging._active_step = None


def test_routine_suppression_keeps_warnings_visible(capsys) -> None:
    try:
        Logging._last_block_section = None
        Logging.set_routine_messages_enabled(False)
        Logging.message("routine detail")
        Logging.warning("important warning")
    finally:
        Logging.set_routine_messages_enabled(True)

    output = capsys.readouterr().out
    assert "routine detail" not in output
    assert "Warning     | important warning" in output


def test_progress_is_one_line_with_accepted_flow_and_wall_times(capsys) -> None:
    Logging.set_routine_messages_enabled(True)
    Logging.time_step(61, 0.4757141, 929.9, total_steps=100, n_particles=14080)

    output = capsys.readouterr().out
    assert "Progress    | step=     61/100" in output
    assert "t=    0.4757 s" in output
    assert "elapsed=00:15:29.9" in output
    assert "N=   14,080" in output
    assert len(output.splitlines()) == 1
    assert "BEGIN" not in output
    assert "COMPLETED" not in output
    assert "time at start" not in output.lower()


def test_progress_is_throttled_but_final_state_is_visible(capsys):
    Logging.set_routine_messages_enabled(True)
    for step, wall in [(1, 1.0), (2, 2.0), (3, 30.0), (4, 31.0), (5, 32.0)]:
        Logging.time_step(step, step * 0.1, wall, total_steps=5)
    output = capsys.readouterr().out
    assert "step=        1/5" in output
    assert "step=        2/5" not in output
    assert "step=        3/5" not in output
    assert "step=        4/5" in output
    assert "step=        5/5" in output


def test_begin_step_does_not_claim_that_work_has_completed(capsys):
    Logging.begin_step(7)
    assert capsys.readouterr().out == ""
    Logging.warning("failed before acceptance")
    assert "Warning     | step 7 | failed before acceptance" in capsys.readouterr().out


def test_event_values_share_one_compact_value_column(capsys):
    Logging.record(
        "discrete vortex heat method regeneration",
        ("threshold", "1.021e-07"),
        ("nodes retained", "75,796"),
        ("utilization", "18.9", "%"),
    )
    assert capsys.readouterr().out.splitlines() == [
        "Event       | discrete vortex heat method regeneration",
        "  threshold                  |    1.021e-07",
        "  nodes retained             |       75,796",
        "  utilization                |         18.9 %",
    ]


def test_block_spacing_is_only_between_uppercase_sections() -> None:
    report = log_style.block_report(
        "VPM solver configuration",
        [
            (
                "flow model",
                [
                    ("governing model", "large-eddy simulation"),
                    ("particle kernel", "Gaussian"),
                ],
            ),
            ("output and monitoring", [("backup interval", "20", "time steps")]),
        ],
    )

    assert " FLOW MODEL\n  Governing model" in report
    assert "Governing model" in report and "\n\n  Governing model" not in report
    assert "Gaussian\n\n OUTPUT AND MONITORING" in report


def test_flow_diagnostics_are_compact_and_do_not_repeat_static_models(capsys) -> None:
    Logging.set_routine_messages_enabled(True)
    Logging.time_step(61, 0.4757141, 929.9, n_particles=14_080)
    system = SimpleNamespace(
        step=61,
        time=0.4757141,
        wall_time=929.9,
        particles=SimpleNamespace(n_particles_total=14_080),
        vortex_strength_magnitude_sum=38.85,
        net_vortex_strength=np.array([2.016e-7, 8.458e-8, 1.024e-5]),
        total_linear_impulse=np.array([18.80161, 9.41327e-5, -8.048337e-7]),
        total_angular_impulse=np.array([3.000175e-3, -2.930725e-5, -1.180542e-5]),
        total_kinetic_energy=23.46,
        viscous_kinetic_energy_rate=-2.273,
        kinetic_energy_rate=-3.463,
        total_enstrophy=1493.0,
        total_helicity=0.1024,
        vortex_centroid=np.array([0.5814, 1.343e-6, 1.325e-6]),
        vortex_centroids_by_group={0: np.array([0.0821, -2.464e-6, 6.505e-7])},
        vlm_solver=None,
    )

    Logging.flow_diagnostics(system)
    output = capsys.readouterr().out

    assert "Diagnostics | step=         61 | t=    0.4757 s" in output
    assert "N=   14,080" in output
    assert "net=" in output
    assert "[ 2.016e-07,  8.458e-08,  1.024e-05]" in output
    assert "Centroid" in output
    assert "Group 0" not in output
    assert len(output.splitlines()) == 8  # One progress line and seven diagnostics.
    assert Logging._last_progress_wall == 929.9
    assert "FLOW QUANTITIES AT" not in output
    assert "treecode" not in output
    assert "transposed" not in output
    assert "observed" not in output.lower()
