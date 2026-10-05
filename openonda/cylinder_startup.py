"""Shared accepted-clock background for the cylinder startup trigger."""

from __future__ import annotations


def startup_velocity(time, duration, transition_duration, startup, steady) -> tuple[float, ...]:
    """Hold the startup flow, then remove it with a C2 quintic taper.

    The transition occupies the final ``transition_duration`` seconds of
    startup. Its first and second time derivatives vanish at both ends, so
    removing the crossflow does not introduce an impulsive acceleration.
    A zero transition retains the original instantaneous switch policy.
    Caller-side schedule validation keeps this hot-path function small.
    """
    if time >= duration:
        return tuple(float(value) for value in steady)
    if transition_duration <= 0 or time <= duration - transition_duration:
        return tuple(float(value) for value in startup)
    fraction = (float(time) - (duration - transition_duration)) / transition_duration
    blend = fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))
    return tuple(
        float(first + blend * (last - first)) for first, last in zip(startup, steady, strict=True)
    )


__all__ = ["startup_velocity"]
