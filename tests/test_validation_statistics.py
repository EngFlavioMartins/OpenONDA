"""Analytical averages and weak-reference errors used in scientific reports."""

import numpy as np
import pytest

from openonda.validation import profile_error, time_mean


def test_irregular_samples_recover_exact_linear_vector_mean():
    times = np.array([0.0, 0.1, 0.7, 1.8, 2.0])
    values = np.column_stack([2 * times + 1, -3 * times])
    np.testing.assert_allclose(time_mean(times, values, 0.15, 1.3), [2.45, -2.175])


@pytest.mark.parametrize(
    "times,start,end", [([0, 1], -0.1, 0.5), ([0, 1], 0, 2), ([0, 1, 1], 0, 1), ([1, 0], 0, 1)]
)
def test_mean_rejects_extrapolation_and_bad_clocks(times, start, end):
    with pytest.raises(ValueError):
        time_mean(times, np.ones(len(times)), start, end)


def test_reference_floor_retains_error_in_unloaded_regions():
    # Only the hub/outer region is wrong; masking small references would return 0.
    assert profile_error([0.2, 0.1, -0.1], [0.2, 0, 0], 0.05) == pytest.approx(np.sqrt(8 / 3))
    assert profile_error([np.nan, 0.1], [0, 0], 0.05) == pytest.approx(2)
