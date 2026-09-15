"""Invalid output times fail before simulation and valid cadences stay unchanged."""

import numpy as np
import pytest

from openonda.vpm import EverySteps, EveryTime


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.1])
def test_schedules_reject_nonfinite_or_negative_times(value):
    with pytest.raises(ValueError):
        EveryTime(value)
    with pytest.raises(ValueError):
        EveryTime(0.2, start_time=value)
    with pytest.raises(ValueError):
        EverySteps(1, start_time=value)


@pytest.mark.parametrize("value", [True, False, "0.2", [0.2]])
def test_schedule_times_require_real_numbers(value):
    with pytest.raises(TypeError):
        EveryTime(value)
    with pytest.raises(TypeError):
        EveryTime(0.2, start_time=value)
    with pytest.raises(TypeError):
        EverySteps(1, start_time=value)


def test_schedule_valid_zero_origin_and_numpy_times_cross_once():
    with pytest.raises(ValueError):
        EveryTime(0.0)
    schedule = EveryTime(np.float64(0.25), start_time=0.0)
    assert not schedule.is_due(2, 0.2, 0.1)
    assert schedule.is_due(3, 0.3, 0.1)
    assert not schedule.is_due(4, 0.4, 0.1)
    assert EverySteps(1, start_time=0.0).is_due(1, 0.1, 0.1)
