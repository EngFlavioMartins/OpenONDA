"""Profile comparison must respect physical time and the observed line."""

import numpy as np
import pytest

from openonda.cylinder_campaign import compare_profiles, profile_statistics


def test_profile_statistics_weight_irregular_times_and_interpolate_endpoints(tmp_path):
    path = tmp_path / "profile.csv"
    rows = [
        [time, y, 1 + y + time, 0, time] for time in (0, 0.1, 0.8, 2.0, 3.0) for y in (-1, 0, 1)
    ]
    np.savetxt(
        path,
        rows,
        delimiter=",",
        header="time,position_y,velocity_x,velocity_y,velocity_z",
        comments="",
    )
    statistics = profile_statistics(path, 0.2, 1.8)
    np.testing.assert_allclose(statistics["mean_velocity"], [[1, 0, 1], [2, 0, 1], [3, 0, 1]])
    np.testing.assert_allclose(
        statistics["rms_velocity"], [[1.6 / np.sqrt(12), 0, 1.6 / np.sqrt(12)]] * 3
    )
    assert compare_profiles(statistics, statistics) == {
        "mean_velocity_l2": 0.0,
        "rms_velocity_l2": 0.0,
    }
    with pytest.raises(ValueError, match="does not cover"):
        profile_statistics(path, 0.2, 4)


def test_profile_error_requires_matching_observation_lines():
    values = {"y": [-1, 0, 1], "mean_velocity": [[1, 0, 0]] * 3, "rms_velocity": [[0, 0, 0]] * 3}
    different = {**values, "y": [-0.5, 0, 0.5]}
    with pytest.raises(ValueError, match="same physical observation"):
        compare_profiles(values, different)
