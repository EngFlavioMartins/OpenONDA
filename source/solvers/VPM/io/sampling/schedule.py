"""Sampling schedules for fixed-step VPM states."""

from __future__ import annotations

from dataclasses import dataclass

from source.utilities import nearest_time_event_due


@dataclass(frozen=True)
class SamplingSchedule:
    """Run a sampler at a step cadence or nearest to a flow-time cadence."""

    every_n_steps: int | None = None
    every_time: float | None = None

    def __post_init__(self) -> None:
        if (self.every_n_steps is None) == (self.every_time is None):
            raise ValueError("Provide exactly one of every_n_steps or every_time")
        if self.every_n_steps is not None and self.every_n_steps < 1:
            raise ValueError("every_n_steps must be a positive integer")
        if self.every_time is not None and self.every_time <= 0.0:
            raise ValueError("every_time must be positive")

    def is_due(self, step: int, time: float, time_step_size: float) -> bool:
        """Return whether the sampler should run on this accepted state."""
        if self.every_n_steps is not None:
            return step > 0 and step % self.every_n_steps == 0
        assert self.every_time is not None
        return nearest_time_event_due(time, time_step_size, self.every_time)
