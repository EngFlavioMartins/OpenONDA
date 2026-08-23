"""Tests for time-history gates used by the relaxed vortex-ring benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments.stage_5b_ring_relaxation_history import (  # noqa: E402
    evaluate,
    local_speeds,
)


def test_local_speeds_recovers_quadratic_path() -> None:
    time = np.linspace(0.0, 2.0, 5)
    centre = 0.2 * time + 0.03 * time**2
    np.testing.assert_allclose(local_speeds(time, centre), 0.2 + 0.06 * time, atol=1.0e-14)


def test_gate_requires_small_residuals_not_only_a_plateau() -> None:
    rows = []
    for time in (0.0, 0.5, 1.0):
        rows.append(
            {
                "collapse_residual": 0.20,
                "advective_residual": 0.20,
                "fitted_translation_speed": 0.25,
                "fitted_speed_relative_difference": 0.01,
                "tube_circulation": 1.0,
                "linear_impulse_x": 3.0,
                "max_mode_amplitude": 1.0e-8,
                "energy_balance_relative_residual": 0.0,
                "invariant_projection_correction_ratio": 0.0,
                "time_star": time,
            }
        )

    gate = evaluate(rows)

    assert gate["status"] == "PLATEAUED_ABOVE_TARGET"
    assert gate["checks"]["time_plateau"]
    assert not gate["checks"]["single_valued_relation"]
