"""Typed schedules for output events.

Schedules are value objects. They retain no event history: exactly-once
delivery and restart reconciliation are responsibilities of ``OutputManager``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class OutputSchedule(Protocol):
    """Pure predicate selecting accepted solver states for output.

    Implementations retain no event history. They receive the current accepted
    step/time and the step duration in seconds; exactly-once delivery and
    restart reconciliation belong to :class:`OutputManager`.
    """

    @property
    def is_final_only(self) -> bool:
        """Whether the framework should dispatch this schedule only at final."""
        ...

    def is_due(self, step: int, time: float, time_step_size: float) -> bool:
        """Return whether the accepted state is selected for output."""
        ...


@dataclass(frozen=True)
class EverySteps:
    """Select every positive integer number of accepted steps.

    Parameters
    ----------
    interval : int
        Accepted-step spacing, at least one.
    first_step : int or None
        Optional first eligible accepted step.
    start_time : float or None
        Optional physical-time floor in seconds.
    """

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
        """Return ``False`` because this schedule is driven by accepted steps."""
        return False

    @property
    def at_end(self) -> bool:
        """Compatibility spelling for legacy samplers."""
        return False

    def is_due(self, step: int, time: float, time_step_size: float) -> bool:
        """Return true when ``step`` reaches this schedule's cadence."""
        epsilon = max(abs(self.interval * time_step_size) * 1.0e-12, abs(time) * 1.0e-14)
        return (
            step > 0
            and step % self.interval == 0
            and (self.first_step is None or step >= self.first_step)
            and (self.start_time is None or time + epsilon >= self.start_time)
        )


@dataclass(frozen=True)
class EveryTime:
    """Run when an accepted state lands on a physical-time cadence.

    A due event is detected when the accepted state crosses a cadence boundary.
    It does not interpolate fields; when a boundary falls between accepted
    states, the first state after it is sampled exactly once.

    Parameters
    ----------
    interval : float
        Positive physical-time cadence in seconds.
    start_time : float, default=0.0
        First cadence origin in seconds.
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
        """Return ``False`` because this schedule is driven by physical time."""
        return False

    @property
    def at_end(self) -> bool:
        """Compatibility alias indicating that no forced final sample is requested."""
        return False

    def is_due(self, step: int, time: float, time_step_size: float) -> bool:
        """Return true when the accepted step crosses a time boundary."""
        epsilon = max(abs(self.interval) * 1.0e-12, abs(time) * 1.0e-14)
        if step <= 0 or time + epsilon < self.start_time:
            return False
        previous_time = time - time_step_size
        current_bucket = int((time - self.start_time + epsilon) // self.interval)
        previous_bucket = int((previous_time - self.start_time + epsilon) // self.interval)
        return current_bucket >= 0 and current_bucket > previous_bucket


@dataclass(frozen=True)
class FinalOnly:
    """Select exactly one explicit framework final-output event.

    `is_due` is always false because final dispatch is an event-level decision,
    not an accepted-step cadence.

    Notes
    -----
    This parameterless value object retains no state. ``OutputManager`` owns
    exactly-once final dispatch and restart reconciliation.
    """

    @property
    def is_final_only(self) -> bool:
        """Return ``True`` so the output manager dispatches only at finalization."""
        return True

    @property
    def at_end(self) -> bool:
        """Compatibility alias returning ``True`` for legacy sampler code."""
        return True

    def is_due(self, step: int, time: float, time_step_size: float) -> bool:
        """Return false; final events are dispatched explicitly by the manager."""
        del step, time, time_step_size
        return False
