"""Shedding phase from the lift coefficient, to compare the two runs at equal phase.

Each run sheds at its own Strouhal number, so states saved at the same time
drift apart in phase. Grouping each run's states by its own shedding phase
compares like with like.
"""

import numpy as np
from scipy.signal import hilbert

N_PHASES = 12  # phase bins of 30 degrees


def shedding_phase(forces_csv):
    """Return the shedding phase in [0, 2 pi) of the run, as a function of time.

    The phase is the angle of the analytic signal of the lift coefficient;
    it is zero at maximum lift.
    """
    forces = np.genfromtxt(forces_csv, delimiter=",", names=True)
    analytic = hilbert(forces["lift_coefficient"] - forces["lift_coefficient"].mean())

    def phase(time):
        real = np.interp(time, forces["time"], analytic.real)
        imaginary = np.interp(time, forces["time"], analytic.imag)
        return np.mod(np.angle(real + 1j * imaginary), 2.0 * np.pi)

    return phase


def phase_bin(phase):
    """Index of the phase bin, from 0 at maximum lift."""
    return (np.asarray(phase) / (2.0 * np.pi) * N_PHASES).astype(int) % N_PHASES


def phase_averaged_velocity(samples, phase, start_time):
    """Probe velocity averaged in each phase bin after ``start_time``.

    Returns the sorted probe positions and the streamwise and transverse
    velocity, each of shape ``(N_PHASES, n_positions)``.
    """
    developed = samples["time"] >= start_time
    positions, point = np.unique(samples["position_y"][developed].round(6), return_inverse=True)
    bins = phase_bin(phase(samples["time"][developed]))
    averages = []
    for name in ("velocity_x", "velocity_y"):
        values = samples[name][developed]
        averages.append(
            np.array(
                [
                    np.bincount(point[bins == k], values[bins == k], len(positions))
                    / np.bincount(point[bins == k], minlength=len(positions))
                    for k in range(N_PHASES)
                ]
            )
        )
    return positions, averages[0], averages[1]


def phase_averaged_difference(samples, phase, reference_samples, reference_phase, start_time):
    """Velocity difference from the reference at equal phase, per bin and position.

    Both probes are averaged over their own phases after ``start_time``; the
    reference is interpolated to the probe positions. Returns the probe
    positions and ``|Delta u|`` with shape ``(N_PHASES, n_positions)``.
    """
    positions, u, v = phase_averaged_velocity(samples, phase, start_time)
    reference = phase_averaged_velocity(reference_samples, reference_phase, start_time)
    reference_u = np.array([np.interp(positions, reference[0], row) for row in reference[1]])
    reference_v = np.array([np.interp(positions, reference[0], row) for row in reference[2]])
    return positions, np.hypot(u - reference_u, v - reference_v)
