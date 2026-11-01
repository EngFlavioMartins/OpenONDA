"""Pitch motion and pitch phase, to compare the two runs at equal phase.

The wake follows the prescribed pitch, so the phase of the motion is the phase
of shedding in both runs.
"""

import numpy as np

PITCH_AMPLITUDE = 10.0  # degrees
FREQUENCY = 0.2  # Hz


def pitch_angle(time):
    """Angle of attack alpha(t) = alpha0 sin(2 pi f t) in degrees."""
    return PITCH_AMPLITUDE * np.sin(2.0 * np.pi * FREQUENCY * np.asarray(time))


def harmonic(time, values):
    """Complex amplitude of the first pitch harmonic of ``values``; zero phase is sin(2 pi f t)."""
    omega = 2.0 * np.pi * FREQUENCY
    fluctuation = values - values.mean()
    return 2.0j * np.sum(fluctuation * np.exp(-1j * omega * time)) / len(time)
