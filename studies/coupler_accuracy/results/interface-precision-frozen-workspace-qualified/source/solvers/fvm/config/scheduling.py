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
    RunSchedule(every_n_steps=20, every_time=None, final_only=False)
    >>> RunSchedule(every_time=0.25)
    RunSchedule(every_n_steps=None, every_time=0.25, final_only=False)
    >>> RunSchedule(final_only=True)
    RunSchedule(every_n_steps=None, every_time=None, final_only=True)
    """

    every_n_steps: int | None = None
    every_time: float | None = None
    final_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.final_only, bool):
            raise TypeError("final_only must be a boolean")
        if self.final_only:
            if self.every_n_steps is not None or self.every_time is not None:
                raise ValueError("A final-only schedule cannot also define a cadence")
            return
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

    def to_dict(self) -> dict[str, int | float | bool | None]:
        """Return a JSON-safe representation."""
        return {
            "every_n_steps": self.every_n_steps,
            "every_time": self.every_time,
            "final_only": self.final_only,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Reconstruct a schedule from :meth:`to_dict` output."""
        unknown = sorted(set(data) - {"every_n_steps", "every_time", "final_only"})
        if unknown:
            raise ValueError("Unknown RunSchedule field(s): " + ", ".join(unknown))
        every_n_steps = data.get("every_n_steps")
        every_time = data.get("every_time")
        final_only = data.get("final_only", False)
        if not isinstance(final_only, bool):
            raise TypeError("final_only must be a boolean")
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
            final_only=final_only,
        )

    @property
    def is_final_only(self) -> bool:
        """Whether this schedule is selected only by a terminal event."""
        return self.final_only

    @property
    def at_end(self) -> bool:
        """Compatibility spelling used by the VPM output protocol."""
        return self.final_only

    def is_due(self, step: int, time: float, time_step_size: float | None = None) -> bool:
        """Return whether the accepted state triggers this schedule."""
        if self.final_only:
            return False
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
        if self.final_only or self.every_time is None:
            return None
        interval = self.every_time
        bucket = math.floor(float(time) / interval + 1.0e-9)
        return (bucket + 1) * interval

    def describe(self) -> str:
        """Return a compact human-readable cadence."""
        if self.final_only:
            return "final state only"
        if self.every_n_steps is not None:
            return f"every {self.every_n_steps} accepted step(s)"
        return f"every {self.every_time:g} s"
