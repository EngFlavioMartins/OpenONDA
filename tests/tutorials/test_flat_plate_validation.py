"""The tutorial's qualification detects frame, reflection and vector-budget defects."""

import numpy as np
import pandas as pd
import pytest

from tutorials.vpm.flat_plate.assets.validate_results import check_polar, vector_strength_closure


def symmetric_polar():
    polar = {}
    for frame in ("moving", "static"):
        polar[frame, "aoa00"] = np.zeros(3)
        for angle in (2, 5, 10):
            for sign in (-1, 1):
                tag = f"aoan{angle:02}" if sign < 0 else f"aoa{angle:02}"
                polar[frame, tag] = np.array(
                    [sign * 0.08 * angle, 0.00024 * angle**2, sign * 0.0004 * angle]
                )
    return polar


def test_symmetric_frame_equivalent_polar_passes():
    assert check_polar(symmetric_polar()) == []


@pytest.mark.parametrize("coefficient", range(3))
def test_detects_wrong_reflection_parity_even_when_frames_agree(coefficient):
    polar = symmetric_polar()
    for frame in ("moving", "static"):
        polar[frame, "aoan05"][coefficient] *= -1
    failures = check_polar(polar)
    assert len(failures) == 2
    assert all("reflection" in failure for failure in failures)


def test_detects_frame_bias_that_preserves_reflection_symmetry():
    polar = symmetric_polar()
    for (frame, _), values in polar.items():
        if frame == "moving":
            values *= 1.01
    failures = check_polar(polar)
    assert failures
    assert all("moving/static" in failure for failure in failures)


def test_zero_incidence_is_checked_independently_of_frame_agreement():
    polar = symmetric_polar()
    for frame in ("moving", "static"):
        polar[frame, "aoa00"][1] = 1e-5
    failures = check_polar(polar)
    assert len(failures) == 2
    assert all("zero incidence" in failure for failure in failures)


def test_full_vector_budget_detects_drift_outside_spanwise_component():
    flow = pd.DataFrame(
        {
            "bound_vortex_strength_x": [0, 0],
            "bound_vortex_strength_y": [2, 4],
            "bound_vortex_strength_z": [0, 0],
            "coupled_vortex_strength_x": [0, 0.003],
            "coupled_vortex_strength_y": [0, 0],
            "coupled_vortex_strength_z": [0, 0.004],
        }
    )
    assert vector_strength_closure(flow) == pytest.approx(0.00125)
    flow.loc[1, "coupled_vortex_strength_z"] = np.nan
    assert vector_strength_closure(flow) == float("inf")
    assert vector_strength_closure(flow.iloc[:0]) == float("inf")
    assert vector_strength_closure(flow.drop(columns="coupled_vortex_strength_x")) == float("inf")
    assert vector_strength_closure(flow.fillna(0) * 0) == 0
