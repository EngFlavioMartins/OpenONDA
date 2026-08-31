"""A sampler schedule must hold off until the flow is worth recording."""

from __future__ import annotations

import pytest

from source.solvers.vpm.io.sampling import EverySteps

TIME_STEP_SIZE = 0.006


def _time(step: int) -> float:
    return step * TIME_STEP_SIZE


def test_bare_cadence_fires_on_every_multiple_after_the_first_step() -> None:
    schedule = EverySteps(interval=20)

    assert not schedule.is_due(0, _time(0), TIME_STEP_SIZE)
    assert not schedule.is_due(19, _time(19), TIME_STEP_SIZE)
    assert schedule.is_due(20, _time(20), TIME_STEP_SIZE)
    assert schedule.is_due(2400, _time(2400), TIME_STEP_SIZE)


def test_first_step_suppresses_the_cadence_until_the_offset() -> None:
    schedule = EverySteps(interval=20, first_step=1640)

    assert not schedule.is_due(20, _time(20), TIME_STEP_SIZE)
    assert not schedule.is_due(1620, _time(1620), TIME_STEP_SIZE)
    assert schedule.is_due(1640, _time(1640), TIME_STEP_SIZE)
    assert schedule.is_due(1660, _time(1660), TIME_STEP_SIZE)


def test_start_time_suppresses_the_cadence_until_the_flow_time() -> None:
    schedule = EverySteps(interval=20, start_time=9.8)

    assert not schedule.is_due(1620, _time(1620), TIME_STEP_SIZE)
    assert schedule.is_due(1640, _time(1640), TIME_STEP_SIZE)


def test_an_offset_never_relaxes_the_cadence() -> None:
    schedule = EverySteps(interval=20, start_time=9.8)

    assert not schedule.is_due(1635, _time(1635), TIME_STEP_SIZE)


def test_both_offsets_wait_for_the_later_one() -> None:
    schedule = EverySteps(interval=20, first_step=2000, start_time=9.8)

    assert not schedule.is_due(1640, _time(1640), TIME_STEP_SIZE)
    assert schedule.is_due(2000, _time(2000), TIME_STEP_SIZE)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interval": 0},
        {"interval": 20, "first_step": 0},
        {"interval": 20, "start_time": -1.0},
    ],
)
def test_invalid_schedules_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        EverySteps(**kwargs)
