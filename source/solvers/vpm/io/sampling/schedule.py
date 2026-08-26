"""Sampling schedules for fixed-step VPM states."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingSchedule:
    """Run a sampler at a fixed positive step cadence.

    ``first_step`` and ``start_time`` are optional inclusive lower bounds that
    delay the first sample; when both are given the later one applies.

    Examples
    --------
    >>> SamplingSchedule(every_n_steps=10).is_due(10, 0.1, 0.01)
    True
    >>> SamplingSchedule(every_n_steps=10, first_step=100).is_due(10, 0.1, 0.01)
    False
    >>> SamplingSchedule(every_n_steps=10, start_time=5.0).is_due(600, 6.0, 0.01)
    True
    """

    every_n_steps: int
    first_step: int | None = None
    start_time: float | None = None

    def __post_init__(self) -> None:
        if self.every_n_steps < 1:
            raise ValueError("every_n_steps must be a positive integer")
        if self.first_step is not None and self.first_step < 1:
            raise ValueError("first_step must be a positive integer")
        if self.start_time is not None and self.start_time < 0.0:
            raise ValueError("start_time must be non-negative")

    def is_due(self, step: int, time: float, time_step_size: float) -> bool:
        """Return whether the sampler should run on this accepted state."""
        if step <= 0 or step % self.every_n_steps != 0:
            return False
        if self.first_step is not None and step < self.first_step:
            return False
        return not (self.start_time is not None and time < self.start_time)
