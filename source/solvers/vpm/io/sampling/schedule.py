"""Sampling schedules for fixed-step VPM states."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingSchedule:
    """Run a sampler at a fixed positive step cadence."""

    every_n_steps: int

    def __post_init__(self) -> None:
        if self.every_n_steps < 1:
            raise ValueError("every_n_steps must be a positive integer")

    def is_due(self, step: int, time: float, time_step_size: float) -> bool:
        """Return whether the sampler should run on this accepted state."""
        return step > 0 and step % self.every_n_steps == 0
