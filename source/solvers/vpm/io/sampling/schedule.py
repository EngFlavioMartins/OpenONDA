"""Typed schedules for output events.

Schedules are value objects. They retain no event history: exactly-once
delivery and restart reconciliation are responsibilities of ``OutputManager``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class OutputSchedule(Protocol):
    """A pure predicate selecting accepted solver states for an output event."""

    @property
    def is_final_only(self) -> bool: ...

    def is_due(self, step: int, time: float, time_step_size: float) -> bool: ...


@dataclass(frozen=True)
class EverySteps:
    """Run at a positive accepted-step cadence."""

    interval: int
    first_step: int | None = None
    start_time: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.interval, bool)
            or not isinstance(self.interval, int)
            or self.interval < 1
        ):
            raise ValueError("EverySteps.interval must be a positive integer")
        if self.first_step is not None and (
            isinstance(self.first_step, bool)
            or not isinstance(self.first_step, int)
            or self.first_step < 1
        ):
            raise ValueError("EverySteps.first_step must be a positive integer")
        if self.start_time is not None and self.start_time < 0.0:
            raise ValueError("EverySteps.start_time must be non-negative")

    @property
    def is_final_only(self) -> bool:
        return False

    @property
    def at_end(self) -> bool:
        """Compatibility spelling for legacy samplers."""
        return False

    def is_due(self, step: int, time: float, time_step_size: float) -> bool:
        del time_step_size
        return (
            step > 0
            and step % self.interval == 0
            and (self.first_step is None or step >= self.first_step)
            and (self.start_time is None or time >= self.start_time)
        )


@dataclass(frozen=True)
class EveryTime:
    """Run when an accepted state lands on a physical-time cadence.

    A due event is detected when the accepted state crosses a cadence boundary.
    It does not interpolate fields; when a boundary falls between accepted
    states, the first state after it is sampled exactly once.
    """

    interval: float
    start_time: float = 0.0

    def __post_init__(self) -> None:
        if self.interval <= 0.0:
            raise ValueError("EveryTime.interval must be positive")
        if self.start_time < 0.0:
            raise ValueError("EveryTime.start_time must be non-negative")

    @property
    def is_final_only(self) -> bool:
        return False

    @property
    def at_end(self) -> bool:
        return False

    def is_due(self, step: int, time: float, time_step_size: float) -> bool:
        if step <= 0 or time < self.start_time:
            return False
        previous_time = time - time_step_size
        epsilon = max(abs(self.interval) * 1.0e-12, abs(time) * 1.0e-14)
        current_bucket = int((time - self.start_time + epsilon) // self.interval)
        previous_bucket = int((previous_time - self.start_time + epsilon) // self.interval)
        return current_bucket >= 0 and current_bucket > previous_bucket


@dataclass(frozen=True)
class FinalOnly:
    """Run exactly once when the framework dispatches its final event."""

    @property
    def is_final_only(self) -> bool:
        return True

    @property
    def at_end(self) -> bool:
        return True

    def is_due(self, step: int, time: float, time_step_size: float) -> bool:
        del step, time, time_step_size
        return False
