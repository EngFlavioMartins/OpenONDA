"""Time-based event scheduling on fixed-step solver states."""

from __future__ import annotations

import math


def nearest_time_event_due(time: float, time_step_size: float, interval: float) -> bool:
    """Return whether this state is nearest to a positive interval boundary.

    The centred interval avoids cumulative drift when ``interval`` is not an
    integer multiple of ``time_step_size``. Exact half-step ties select the
    earlier state.
    """
    if not all(math.isfinite(value) for value in (time, time_step_size, interval)):
        raise ValueError("time, time_step_size, and interval must be finite")
    if time_step_size <= 0.0 or interval <= 0.0:
        raise ValueError("time_step_size and interval must be positive")
    if interval + 1.0e-14 < time_step_size:
        raise ValueError("interval must not be smaller than time_step_size")
    if time <= 0.0:
        return False

    half_step = 0.5 * time_step_size
    epsilon = 32.0 * math.ulp(max(1.0, abs(time), interval))

    def bucket(value: float) -> int:
        return math.floor(value / interval + epsilon)

    previous = bucket(time - half_step)
    current = bucket(time + half_step)
    return current > 0 and current != previous
