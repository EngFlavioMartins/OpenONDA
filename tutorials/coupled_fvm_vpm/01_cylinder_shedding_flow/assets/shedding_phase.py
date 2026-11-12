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
