"""Heave motion and heave phase, to compare the two runs at equal phase.

The wake locks onto the prescribed heave, so the phase of the motion is the
phase of shedding in both runs. Grouping each run's states by heave phase
compares like with like.
"""

import numpy as np

AMPLITUDE = 0.2  # m, heave amplitude
FREQUENCY = 0.18  # Hz, heave frequency
N_PHASES = 12  # phase bins of 30 degrees


def heave_displacement(time):
    """Transverse cylinder and mesh displacement y(t) = A sin(2 pi f t) in m."""
    return AMPLITUDE * np.sin(2.0 * np.pi * FREQUENCY * np.asarray(time))


def heave_velocity(time):
    """Transverse cylinder and mesh velocity in m/s."""
    return 2.0 * np.pi * FREQUENCY * AMPLITUDE * np.cos(2.0 * np.pi * FREQUENCY * np.asarray(time))


def heave_phase(time):
    """Heave phase in [0, 2 pi); zero at the mean position moving up."""
    return np.mod(2.0 * np.pi * FREQUENCY * np.asarray(time), 2.0 * np.pi)


def phase_bin(time):
    """Index of the heave-phase bin of each time; bin k is centred on the phase 2 pi k / N_PHASES.

    Centred bins keep their edges off the saved times: every 25 s, a time
    saved every 0.2 s falls on a whole multiple of 30 degrees.
    """
    return np.floor(heave_phase(time) / (2.0 * np.pi) * N_PHASES + 0.5).astype(int) % N_PHASES


def phase_averaged_velocity(time, velocity_x, velocity_y, start_time):
    """Probe velocity averaged in each phase bin after ``start_time``.

    ``time`` has one entry per sample time and the velocity arrays the shape
    ``(n_times, n_positions)``. Returns the streamwise and transverse velocity,
    each of shape ``(N_PHASES, n_positions)``.
    """
    developed = time.round(6) >= start_time
    bins = phase_bin(time[developed])
    return [
        np.array([values[developed][bins == k].mean(axis=0) for k in range(N_PHASES)])
        for values in (velocity_x, velocity_y)
    ]
