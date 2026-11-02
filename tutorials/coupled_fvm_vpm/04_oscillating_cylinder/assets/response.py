"""Vibration response of one run: amplitudes, frequency and the statistics window.

The paper defines the reduced amplitude from the top 20 % of the response
peaks in the steady state; the window here is the last ten natural periods,
or whatever shorter history a run has produced.
"""

from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parents[1]
DIAMETER = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s
REYNOLDS_PER_REDUCED_VELOCITY = 250.0
REDUCED_VELOCITIES = (4.0, 6.0, 8.0)
STATISTICS_PERIODS = 10  # natural periods before the end of the run
PEAK_FRACTION = 0.2  # the paper's top 20 % of the peaks

PAPER = CASE_DIR / "assets/references/martins_avila_2019_low_re_zeta_0p5.csv"


def variant(reduced_velocity):
    """Directory name of one reduced velocity below samples/ and solution/."""
    return f"vstar_{reduced_velocity:g}"


def completed_runs(samples_root):
    """Reduced velocities with a motion history below ``samples_root``, in increasing order."""
    return sorted(
        float(path.parent.name.removeprefix("vstar_"))
        for path in samples_root.glob("vstar_*/motion_history.csv")
    )


def natural_period(reduced_velocity):
    """Natural period in water in s: V* D / U."""
    return reduced_velocity * DIAMETER / FREESTREAM_SPEED


def read_table(path):
    """Named columns of a sampler CSV."""
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def read_run(samples_dir, reduced_velocity):
    """Motion and force histories of one run restricted to its statistics window."""
    motion = read_table(samples_dir / "motion_history.csv")
    forces = read_table(samples_dir / "forces_history.csv")
    end = motion["time"].max()
    start = max(motion["time"].min(), end - STATISTICS_PERIODS * natural_period(reduced_velocity))
    window = motion["time"] >= start - 1e-9
    return motion[window], forces[forces["time"] >= start - 1e-9], (start, end)


def paper_points():
    """Digitized and tabulated cross-flow amplitudes of the paper, with their reduced velocity."""
    data = read_table(PAPER)
    return (
        data["reynolds_number"] / REYNOLDS_PER_REDUCED_VELOCITY,
        data["cross_flow_amplitude"],
        data["source"],
    )


def peak_amplitude(time, signal):
    """Mean of the highest 20 % of the oscillation peaks of ``signal`` about its mean."""
    centred = signal - signal.mean()
    interior = centred[1:-1]
    maxima = interior[(interior > centred[:-2]) & (interior >= centred[2:])]
    minima = -interior[(interior < centred[:-2]) & (interior <= centred[2:])]
    peaks = np.sort(np.concatenate((maxima, minima)))[::-1]
    if len(peaks) == 0:
        return 0.5 * np.ptp(centred)
    return peaks[: max(1, round(PEAK_FRACTION * len(peaks)))].mean()


def oscillation_frequency(time, signal):
    """Frequency in Hz from the upward zero crossings of ``signal`` about its mean."""
    centred = signal - signal.mean()
    up = np.flatnonzero((centred[:-1] < 0.0) & (centred[1:] >= 0.0))
    if len(up) < 2:
        return np.nan
    crossing = time[up] - centred[up] * np.diff(time)[up] / np.diff(centred)[up]
    return 1.0 / np.diff(crossing).mean()


def statistics(samples_dir, reduced_velocity):
    """Reduced amplitudes, frequency ratio, mean drag and lift amplitude of one run."""
    motion, forces, (start, end) = read_run(samples_dir, reduced_velocity)
    frequency = oscillation_frequency(motion["time"], motion["displacement_y"])
    return {
        "reduced_velocity": reduced_velocity,
        "reynolds_number": REYNOLDS_PER_REDUCED_VELOCITY * reduced_velocity,
        "window_start": start,
        "window_end": end,
        "window_periods": (end - start) / natural_period(reduced_velocity),
        "cross_flow_amplitude": peak_amplitude(motion["time"], motion["displacement_y"]) / DIAMETER,
        "streamwise_amplitude": peak_amplitude(motion["time"], motion["displacement_x"]) / DIAMETER,
        "mean_streamwise_offset": motion["displacement_x"].mean() / DIAMETER,
        "frequency_ratio": frequency * natural_period(reduced_velocity),
        "mean_drag": forces["drag_coefficient"].mean(),
        "lift_amplitude": peak_amplitude(forces["time"], forces["lift_coefficient"]),
    }
