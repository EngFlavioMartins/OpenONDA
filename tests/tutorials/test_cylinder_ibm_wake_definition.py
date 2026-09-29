"""Wake length uses the cylinder rear and a dimensionally correct marker."""

import numpy as np
import pytest

from tests._tutorial_helpers import load_tutorial_module


def test_wake_length_and_marker_are_translation_and_scale_invariant():
    wake = load_tutorial_module("fvm/cylinder_ibm", "assets.plot_wake")
    # Rear at center + D/2; downstream zero crossing at center + 2.1 D.
    for center, diameter in ((0.0, 1.0), (4.0, 2.0), (-3.0, 0.5)):
        x = center + diameter * np.array([0.45, 0.75, 1.5, 2.0, 2.2])
        u = np.array([0.0, -0.3, -0.2, -0.1, 0.1])
        length = wake.recirculation_length(x, u, D=diameter, center_x=center)
        assert length / diameter == pytest.approx(1.6)
        assert wake._wake_endpoint_over_d(length, diameter, center_x=center) == pytest.approx(
            center / diameter + 2.1
        )


def test_wake_length_does_not_relabel_center_distance_as_rear_distance():
    wake = load_tutorial_module("fvm/cylinder_ibm", "assets.plot_wake")
    x = np.array([0.75, 1.5, 2.0, 2.1, 2.2])
    u = np.array([-0.2, -0.2, -0.1, 0.0, 0.1])
    assert wake.recirculation_length(x, u, D=1.0) == pytest.approx(1.6)
