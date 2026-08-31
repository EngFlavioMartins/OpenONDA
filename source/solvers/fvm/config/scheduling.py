"""Immutable accepted-step and physical-time schedules for FVM run events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from typing import Self


@dataclass(frozen=True, slots=True)
class RunSchedule:
    """Cadence for a solver-owned event.

    Provide exactly one of ``every_n_steps`` or ``every_time``.  Step-based
    schedules count accepted FVM steps.  Physical-time schedules are crossing
    based and, when maximum-Courant time-step control is active, participate in
    step selection so the solver lands exactly on each event time.  This is the
    ``timeStep`` / ``adjustableRunTime`` pattern used by OpenFOAM.

    The object is immutable because output orchestration is fixed when the
    solver is constructed.

    Examples
    --------
    >>> RunSchedule(every_n_steps=20)
    RunSchedule(every_n_steps=20, every_time=None)
    >>> RunSchedule(every_time=0.25)
    RunSchedule(every_n_steps=None, every_time=0.25)
    """

    every_n_steps: int | None = None
    every_time: float | None = None

    def __post_init__(self) -> None:
        if (self.every_n_steps is None) == (self.every_time is None):
            raise ValueError("Provide exactly one of every_n_steps or every_time")
        if self.every_n_steps is not None:
            if isinstance(self.every_n_steps, bool) or not isinstance(self.every_n_steps, int):
                raise TypeError("every_n_steps must be an integer")
            if self.every_n_steps < 1:
                raise ValueError("every_n_steps must be at least one")
        if self.every_time is not None:
            if isinstance(self.every_time, bool) or not isinstance(self.every_time, Real):
                raise TypeError("every_time must be a real number")
            if not math.isfinite(self.every_time) or self.every_time <= 0.0:
                raise ValueError("every_time must be finite and positive")
            object.__setattr__(self, "every_time", float(self.every_time))

    def to_dict(self) -> dict[str, int | float | None]:
        """Return a JSON-safe representation."""
        return {"every_n_steps": self.every_n_steps, "every_time": self.every_time}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Reconstruct a schedule from :meth:`to_dict` output."""
        unknown = sorted(set(data) - {"every_n_steps", "every_time"})
        if unknown:
            raise ValueError("Unknown RunSchedule field(s): " + ", ".join(unknown))
        every_n_steps = data.get("every_n_steps")
        every_time = data.get("every_time")
        if every_n_steps is not None and (
            isinstance(every_n_steps, bool) or not isinstance(every_n_steps, int)
        ):
            raise TypeError("every_n_steps must be an integer or null")
        if every_time is not None and (
            isinstance(every_time, bool) or not isinstance(every_time, Real)
        ):
            raise TypeError("every_time must be a real number or null")
        return cls(
            every_n_steps=every_n_steps,
            every_time=None if every_time is None else float(every_time),
        )

    def is_due(self, step: int, time: float, time_step_size: float | None = None) -> bool:
        """Return whether the accepted state triggers this schedule."""
        if self.every_n_steps is not None:
            return int(step) % self.every_n_steps == 0
        if time_step_size is None or time_step_size <= 0.0:
            return False
        assert self.every_time is not None
        interval = float(self.every_time)
        epsilon = 1.0e-9
        current_bucket = math.floor(float(time) / interval + epsilon)
        previous_bucket = math.floor((float(time) - float(time_step_size)) / interval + epsilon)
        return current_bucket != previous_bucket

    def next_time_after(self, time: float) -> float | None:
        """Return the next physical-time event strictly after ``time``."""
        if self.every_time is None:
            return None
        interval = self.every_time
        bucket = math.floor(float(time) / interval + 1.0e-9)
        return (bucket + 1) * interval

    def describe(self) -> str:
        """Return a compact human-readable cadence."""
        if self.every_n_steps is not None:
            return f"every {self.every_n_steps} accepted step(s)"
        return f"every {self.every_time:g} s"
