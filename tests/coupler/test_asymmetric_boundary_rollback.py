"""The unequal trial must change endpoint data while preserving its trial clock."""

import numpy as np
import pytest

from tests.support.cylinder.run_asymmetric_boundary_rollback import (
    interpolated_trial_trace,
    trace_difference,
)


def test_unequal_trial_uses_same_start_and_declared_fraction_but_a_different_physical_endpoint():
    start = {"velocity": np.array([[1., .1, 0], [1., -.1, 0]]),
             "normal_velocity": np.array([1., -1.]),
             "tangential_gradient": np.array([[0, .2, 0], [0, -.2, 0]])}
    a = {name: values + .03 for name, values in start.items()}
    b = {name: values - .05 for name, values in start.items()}
    old_start = {name: values.copy() for name, values in start.items()}
    for endpoint in (a, b):
        zero = interpolated_trial_trace(start, endpoint, 0)
        final = interpolated_trial_trace(start, endpoint, 5)
        for name in start:
            np.testing.assert_array_equal(zero[name], start[name])
            np.testing.assert_array_equal(final[name], endpoint[name])
            np.testing.assert_array_equal(start[name], old_start[name])
    first = interpolated_trial_trace(start, a, 1)
    unequal = interpolated_trial_trace(start, b, 1)
    for name in first:
        np.testing.assert_allclose(first[name] - unequal[name], .016)
    assert trace_difference(a, b)["normal_velocity"]["maximum_absolute_component"] == pytest.approx(.08)
    assert interpolated_trial_trace(start, a, 5)["normal_velocity"].shape == (2,)
    with pytest.raises(ValueError, match="five native substeps"):
        interpolated_trial_trace(start, a, 6)
    broken = {name: value.copy() for name, value in b.items()}
    broken["normal_velocity"][0] = np.nan
    with pytest.raises(ValueError, match="Finite matching"):
        interpolated_trial_trace(start, broken, 1)
