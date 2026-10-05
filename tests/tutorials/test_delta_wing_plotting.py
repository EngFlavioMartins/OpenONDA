"""Physical delta-wing averaging uses interpolated integration endpoints."""

import numpy as np
import pyvista as pv

from tests._tutorial_helpers import load_tutorial_module

postprocess = load_tutorial_module("vpm/delta_wing", "assets.postprocess")


def test_mean_wake_integrates_complete_period_with_irregular_endpoints(monkeypatch):
    points = np.zeros((3, 3))

    class Grid:
        def __init__(self, time):
            self.points = points
            self.time = time

        def __getitem__(self, key):
            assert key == "velocity"
            return np.full((3, 3), 2 * self.time + 3)

    monkeypatch.setattr(pv, "read", Grid)
    frames = [(time, time, "source") for time in (0.0, 0.3, 0.8, 1.2)]
    result = postprocess._wake_average([frames], end=1.1, period=0.9)
    # Integral mean of 2t+3 over [0.2, 1.1] is 4.3.
    np.testing.assert_allclose(result[0][1], 4.3, rtol=0, atol=1e-12)
