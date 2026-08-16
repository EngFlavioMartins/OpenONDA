"""Tests for the vortex-ring q(psi) quasi-steady diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments.stage_5b_ring_quasi_steady import (
    cumulative_streamfunction,
    manufactured_controls,
)


def test_streamfunction_recovers_uniform_axial_flow() -> None:
    x = np.linspace(-1.0, 1.0, 17)
    r = np.linspace(0.0, 2.0, 65)
    ux = np.full((len(x), len(r)), 1.7)
    psi = cumulative_streamfunction(ux, r, translation_speed=0.2)

    expected = np.broadcast_to(0.5 * 1.5 * r[None, :] ** 2, psi.shape)
    np.testing.assert_allclose(psi, expected, atol=2.0e-15)


def test_exact_single_value_control_passes_and_broken_control_separates() -> None:
    controls = manufactured_controls(129)
    exact = controls["exact_q_equals_one_plus_0p8_psi"]
    broken = controls["broken_q_adds_0p45_x"]

    assert exact["collapse_residual"] < 5.0e-3
    assert exact["advective_residual"] < 2.0e-2
    assert broken["collapse_residual"] > max(0.05, 10.0 * exact["collapse_residual"])
    assert broken["advective_residual"] > max(0.05, 10.0 * exact["advective_residual"])
