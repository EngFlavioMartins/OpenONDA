"""Regression tests for the compact VPM block logger."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from source import log_style
from source.solvers.vpm.io.logging import Logging


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
    assert " WARNINGS\n  Important warning" in output


def test_time_step_header_contains_only_current_flow_and_wall_times(capsys) -> None:
    Logging.set_routine_messages_enabled(True)
    Logging.time_step(61, 0.4757141, 929.9)

    output = capsys.readouterr().out
    assert "VPM TIME STEP 61" in output
    assert "FLOW TIME 4.757141e-01 s" in output
    assert "WALL TIME 00:15:29.9" in output
    assert "BEGIN" not in output
    assert "COMPLETED" not in output
    assert "time at start" not in output.lower()


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
    Logging.time_step(61, 0.4757141, 929.9)
    system = SimpleNamespace(
        step=61,
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

    assert " FLOW QUANTITIES\n  Particle system" in output
    assert "Net vector" in output
    assert "[2.016e-07, 8.458e-08, 1.024e-05]" in output
    assert " VORTEX POSITION\n  All particles" in output
    assert "FLOW QUANTITIES AT" not in output
    assert "treecode" not in output
    assert "transposed" not in output
    assert "observed" not in output.lower()
