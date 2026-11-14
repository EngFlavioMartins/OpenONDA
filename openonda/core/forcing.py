"""Serializable velocity histories evaluated on native accepted clocks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class VelocityRamp:
    """Hold an initial velocity, then make a C2 transition to a final velocity.

    Velocities use m/s and transition endpoints use seconds. The quintic
    transition has zero first and second derivatives at both endpoints. It
    serves boundary velocity, where the acceleration should also leave and
    reach its end values smoothly; body motion uses the sin² ramp of
    :class:`~openonda.core.kinematics.SmoothRampMotion`.
    """

    initial: tuple[float, float, float]
    final: tuple[float, float, float]
    start_time: float
    end_time: float

    def __post_init__(self) -> None:
        for name in ("initial", "final"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"VelocityRamp.{name} must contain three finite velocities")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        if not all(math.isfinite(value) for value in (self.start_time, self.end_time)):
            raise ValueError("VelocityRamp transition times must be finite")
        if not 0 <= self.start_time < self.end_time:
            raise ValueError("VelocityRamp requires 0 <= start_time < end_time")

    def at(self, time: float) -> np.ndarray:
        """Return the velocity at ``time`` in s, in m/s.

        The initial velocity holds before ``start_time`` and the final velocity
        after ``end_time``.
        """
        fraction = np.clip((time - self.start_time) / (self.end_time - self.start_time), 0.0, 1.0)
        blend = fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))
        first = np.asarray(self.initial)
        return first + blend * (np.asarray(self.final) - first)
