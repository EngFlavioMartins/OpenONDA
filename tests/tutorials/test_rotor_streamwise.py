"""Streamwise induction averages preserve signs, coordinates and physical time."""

import numpy as np
import pandas as pd
import pytest

from tests._tutorial_helpers import load_tutorial_module

plot = load_tutorial_module("vpm/rotor_flow", "assets.plot_rotor_streamwise")


def _samples():
    return pd.DataFrame(
        [
            {
                "time": t,
                "position_x": x,
                "position_y": 2.0,
                "position_z": 0.0,
                "velocity_x": 7.0 - x + 2 * t,
                "velocity_y": -0.5 * t,
                "velocity_z": x - t,
            }
            for t in (0.0, 0.3, 1.0, 2.0)
            for x in (-12.0, 0.0, 12.0, 36.0)
        ]
    )


def test_window_mean_uses_time_weights_and_preserves_crossflow_sign():
    profile = plot.mean_profile(_samples().sample(frac=1, random_state=2), 0.2, 1.6)
    mean_time = 0.9
    np.testing.assert_allclose(profile.velocity_x, 7 - profile.position_x + 2 * mean_time)
    np.testing.assert_allclose(profile.velocity_y, -0.5 * mean_time)
    np.testing.assert_allclose(profile.velocity_z, profile.position_x - mean_time)


@pytest.mark.parametrize("defect", ["short", "missing", "duplicate", "nan"])
def test_incomplete_native_line_cannot_be_plotted_as_valid(defect):
    data = _samples()
    if defect == "short":
        data = data[data.time > 0]
    elif defect == "missing":
        data = data.drop(3)
    elif defect == "duplicate":
        data = pd.concat([data, data.iloc[[0]]])
    else:
        data.loc[0, "velocity_y"] = np.nan
    with pytest.raises(ValueError):
        plot.mean_profile(data, 0.2, 1.6)
