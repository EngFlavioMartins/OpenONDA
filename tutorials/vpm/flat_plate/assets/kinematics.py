"""Analytical plate displacement for moving- and wind-frame comparisons."""

import math

import numpy as np


def distance_travelled(
    times: np.ndarray, kinematics: str, ramp_time: float, freestream_speed: float, chord: float
) -> np.ndarray:
    """Return plate travel in chord lengths at sampled physical times."""
    if kinematics != "ramp":
        return times * freestream_speed / chord

    ramp_distance = (
        0.5 * freestream_speed * (times - ramp_time / math.pi * np.sin(math.pi * times / ramp_time))
    )
    cruise_distance = 0.5 * freestream_speed * ramp_time + freestream_speed * (times - ramp_time)
    return np.where(times < ramp_time, ramp_distance, cruise_distance) / chord
