#!/usr/bin/env python3
"""Wake velocity profiles at x = 2 D and 4 D in the inertial frame: time-mean, at every
common sample time and averaged over equal heave phase."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, REFERENCE_STYLE, figure_size, save_fig
from openonda.plotting import centered_subplots_adjust, set_thesis_style
from openonda.saved_times import match_saved_times

from assets.heave_phase import N_PHASES, heave_displacement, heave_velocity
from assets.heave_phase import phase_averaged_velocity

CASE_DIR = Path(__file__).resolve().parents[1]
LOCKED_IN = 25.0  # s, start of the averaging window
DIAMETER = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s

# Probe data from the reference, the coupled FVM box, and the particle wake.
SOURCES = {
    2: {
        "Reference FVM": "reference_flow/samples/transverse_x2.csv",
        "Coupled FVM": "samples/fvm_transverse_x2.csv",
        "Coupled VPM": "samples/vpm_transverse_x2.csv",
    },
    4: {
        "Reference FVM": "reference_flow/samples/transverse_x4.csv",
        "Coupled VPM": "samples/vpm_transverse_x4.csv",
    },
}
STYLES = {
    "Reference FVM": REFERENCE_STYLE,
    "Coupled FVM": {"color": COLORS["fvm"], "linewidth": LINE_WIDTH},
    "Coupled VPM": {"color": COLORS["vpm"], "linewidth": LINE_WIDTH},
}


def read_profiles(path, label):
    """Sample times, transverse positions and the velocity of a probe line in the inertial frame.

    FVM probes stay at their inertial positions. The coupled particles are
    sampled in the frame of the heaving box, so their positions and transverse
    velocity are shifted back by the heave and interpolated to the inertial
    positions. Velocities have the shape ``(n_times, n_positions)``.
    """
    samples = np.genfromtxt(path, delimiter=",", names=True)
    times = np.unique(samples["time"])
    positions = np.unique(samples["position_y"].round(6))
    positions = positions[np.abs(positions) <= 2.0]
    velocity_x, velocity_y = [], []
    for time in times:
        rows = samples["time"] == time
        order = np.argsort(samples["position_y"][rows])
        y = samples["position_y"][rows][order]
        u = samples["velocity_x"][rows][order]
        v = samples["velocity_y"][rows][order]
        if label == "Coupled VPM":
            y = y + heave_displacement(time)
            v = v + heave_velocity(time)
        velocity_x.append(np.interp(positions, y, u))
        velocity_y.append(np.interp(positions, y, v))
    return times, positions, np.array(velocity_x), np.array(velocity_y)


def profile_figure(curves, labels):
    """Two rows of transverse profiles at both stations, with a shared legend."""
    figure, axes = plt.subplots(2, 2, figsize=figure_size("stacked"), sharex=True, sharey="row")
    centered_subplots_adjust(figure, outer=0.13, bottom=0.2, top=0.97, hspace=0.12, wspace=0.12)
    for column, station in enumerate(SOURCES):
        for label, y, upper, lower in curves[station]:
            axes[0, column].plot(y, upper, **STYLES[label], label=label)
            axes[1, column].plot(y, lower, **STYLES[label], label=label)
        axes[1, column].set_xlabel(f"$y/D$ at $x/D={station}$")
    axes[0, 0].set_ylabel(labels[0])
    axes[1, 0].set_ylabel(labels[1])
    handles, names = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=3, frameon=False, handlelength=1.5)
    return figure, axes


def save_velocity_figure(curves, limits, name):
    """Streamwise and transverse velocity on the y-range shared by every figure of a series."""
    figure, axes = profile_figure(curves, (r"$u_x/U_\infty$", r"$u_y/U_\infty$"))
    for row, (low, high) in enumerate(limits):
        margin = 0.05 * (high - low)
        axes[row, 0].set_ylim((low - margin) / FREESTREAM_SPEED, (high + margin) / FREESTREAM_SPEED)
    save_fig(figure, CASE_DIR / "figures" / name)


def mean_profiles(profiles):
    """Time-mean streamwise velocity and rms transverse velocity over the locked-in wake."""
    curves = {station: [] for station in SOURCES}
    for (station, label), (times, y, u, v) in profiles.items():
        window = times.round(6) >= LOCKED_IN
        curves[station].append(
            (
                label,
                y / DIAMETER,
                u[window].mean(axis=0) / FREESTREAM_SPEED,
                v[window].std(axis=0) / FREESTREAM_SPEED,
            )
        )
    figure, _ = profile_figure(curves, (r"$\overline{u}/U_\infty$", r"$v_\mathrm{rms}/U_\infty$"))
    save_fig(figure, CASE_DIR / "figures/velocity_profiles")


def instantaneous_profiles(profiles, limits):
    """Streamwise and transverse velocity at every sample time shared by all probes."""
    common = match_saved_times(*(times for times, *_ in profiles.values()))
    for frame, time in enumerate(common.times):
        curves = {station: [] for station in SOURCES}
        for ((station, label), (_, y, u, v)), indices in zip(profiles.items(), common.indices):
            index = indices[frame]
            curves[station].append(
                (label, y / DIAMETER, u[index] / FREESTREAM_SPEED, v[index] / FREESTREAM_SPEED)
            )
        save_velocity_figure(curves, limits, f"velocity_profiles/t{time:06.2f}")


def phase_averaged_profiles(profiles, limits):
    """Velocity averaged over equal heave phase, and its difference from the reference."""
    averages = {
        key: (y, *phase_averaged_velocity(times, u, v, LOCKED_IN))
        for key, (times, y, u, v) in profiles.items()
    }

    rows = []
    for k in range(N_PHASES):
        curves = {station: [] for station in SOURCES}
        for station, files in SOURCES.items():
            for label in files:
                positions, u, v = averages[station, label]
                curves[station].append(
                    (label, positions / DIAMETER, u[k] / FREESTREAM_SPEED, v[k] / FREESTREAM_SPEED)
                )
                if label == "Reference FVM":
                    continue
                reference = averages[station, "Reference FVM"]
                difference = (
                    100.0
                    * np.hypot(
                        u[k] - np.interp(positions, reference[0], reference[1][k]),
                        v[k] - np.interp(positions, reference[0], reference[2][k]),
                    )
                    / FREESTREAM_SPEED
                )
                rows.append((k, station, label, np.sqrt(np.mean(difference**2)), difference.max()))
        save_velocity_figure(curves, limits, f"phase_averaged_profiles/phase{k:02d}")

    lines = [
        "phase_bin,phase_degrees,station_x_over_D,source,velocity_rms_percent,velocity_max_percent"
    ]
    lines += [
        f"{k},{360.0 * k / N_PHASES:g},{station},{label},{rms:.4g},{largest:.4g}"
        for k, station, label, rms, largest in rows
    ]
    (CASE_DIR / "figures/auxiliary").mkdir(parents=True, exist_ok=True)
    (CASE_DIR / "figures/auxiliary/profile_differences.csv").write_text("\n".join(lines) + "\n")
    for station, files in SOURCES.items():
        for label in files:
            values = [
                rms
                for _, row_station, row_label, rms, _ in rows
                if (row_station, row_label) == (station, label)
            ]
            if values:
                print(
                    f"x/D = {station}, {label}: phase-averaged velocity difference "
                    f"rms {np.sqrt(np.mean(np.square(values))):.2g}% U_inf"
                )


def main():
    set_thesis_style()
    profiles = {
        (station, label): read_profiles(CASE_DIR / path, label)
        for station, files in SOURCES.items()
        for label, path in files.items()
    }
    mean_profiles(profiles)

    # All profile frames share one range for each velocity component.
    limits = [
        (
            min(profile[component].min() for profile in profiles.values()),
            max(profile[component].max() for profile in profiles.values()),
        )
        for component in (2, 3)
    ]
    phase_averaged_profiles(profiles, limits)
    instantaneous_profiles(profiles, limits)


if __name__ == "__main__":
    main()
