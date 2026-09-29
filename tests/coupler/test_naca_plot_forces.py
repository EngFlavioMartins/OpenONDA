"""NACA force plots use the saved run's freestream direction."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tutorials/coupled_fvm_vpm/03_naca4412_flow/assets/plot_forces.py"
)
SPEC = importlib.util.spec_from_file_location("naca_plot_forces", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
plot_forces = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_forces)


@pytest.mark.parametrize("degrees", [10.0, -25.0, 90.0])
def test_wind_axis_uses_saved_freestream(degrees, tmp_path):
    angle = np.deg2rad(degrees)
    direction = np.array([np.cos(angle), np.sin(angle), 0.0])
    solution = tmp_path / "solution"
    solution.mkdir()
    (solution / "run_metadata.json").write_text(
        json.dumps({"physics": {"freestream_velocity": (3.0 * direction).tolist()}})
    )
    body_drag = np.array([1.0, 2.0])
    body_lift = np.array([2.0, -1.0])
    drag, lift = plot_forces._wind_axis_coefficients(
        body_drag, body_lift, plot_forces._wind_direction(tmp_path)
    )
    np.testing.assert_allclose(drag, body_drag * direction[0] + body_lift * direction[1])
    np.testing.assert_allclose(lift, -body_drag * direction[1] + body_lift * direction[0])


def test_csv_only_results_use_setup_freestream(tmp_path):
    (tmp_path / "setup.py").write_text("FREESTREAM_VELOCITY = (0.0, 2.0, 0.0)\n")
    np.testing.assert_allclose(plot_forces._wind_direction(tmp_path), [0.0, 1.0])


def test_invalid_saved_freestream_does_not_fall_back(tmp_path):
    solution = tmp_path / "solution"
    solution.mkdir()
    (solution / "run_metadata.json").write_text(
        json.dumps({"physics": {"freestream_velocity": [0.0, 0.0, 0.0]}})
    )
    (tmp_path / "setup.py").write_text("FREESTREAM_VELOCITY = (1.0, 0.0, 0.0)\n")
    with pytest.raises(ValueError, match="nonzero airfoil plane"):
        plot_forces._wind_direction(tmp_path)
