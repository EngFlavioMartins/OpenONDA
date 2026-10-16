#!/usr/bin/env python3
"""Wake velocity profiles at x = 2 D and 4 D: time-mean, at every common sample time and
averaged over equal shedding phase."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, REFERENCE_STYLE, figure_size, save_fig
from openonda.plotting import centered_subplots_adjust, set_thesis_style
from openonda.saved_times import match_saved_times

from assets.shedding_phase import N_PHASES, phase_bin, shedding_phase

CASE_DIR = Path(__file__).resolve().parents[1]
DEVELOPED_SHEDDING = 40.0  # s, start of the averaging window
DIAMETER = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s
REFERENCE_FORCES = CASE_DIR / "reference_flow/samples/forces_history.csv"
COUPLED_FORCES = CASE_DIR / "samples/forces_history.csv"

# Reference FVM, coupled FVM (inside its box at x = 2 m) and the coupled particles
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


def wake_profile(path):
    """Time-mean streamwise velocity and rms transverse velocity at each sampled y."""
    samples = np.genfromtxt(path, delimiter=",", names=True)
    window = samples["time"] >= DEVELOPED_SHEDDING
    y, point = np.unique(samples["position_y"][window].round(6), return_inverse=True)
    count = np.bincount(point)

    u = samples["velocity_x"][window]
    v = samples["velocity_y"][window]
    mean_u = np.bincount(point, u) / count
    mean_v = np.bincount(point, v) / count
    rms_v = np.sqrt(np.bincount(point, (v - mean_v[point]) ** 2) / count)
    return y / DIAMETER, mean_u / FREESTREAM_SPEED, rms_v / FREESTREAM_SPEED


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


def instantaneous_profiles(samples, limits):
    """Streamwise and transverse velocity at every sample time shared by all probes."""
    clocks = {key: np.unique(table["time"]) for key, table in samples.items()}
    common = match_saved_times(*clocks.values())
    for frame, time in enumerate(common.times):
        curves = {station: [] for station in SOURCES}
        for (station, label), indices in zip(samples, common.indices):
            table = samples[station, label]
            rows = table["time"] == clocks[station, label][indices[frame]]
            order = np.argsort(table["position_y"][rows])
            curves[station].append(
                (
                    label,
                    table["position_y"][rows][order] / DIAMETER,
                    table["velocity_x"][rows][order] / FREESTREAM_SPEED,
                    table["velocity_y"][rows][order] / FREESTREAM_SPEED,
                )
            )
        save_velocity_figure(curves, limits, f"velocity_profiles/t{time:06.2f}")


def save_velocity_figure(curves, limits, name):
    """Streamwise and transverse velocity on the y-range shared by every figure of a series."""
    figure, axes = profile_figure(curves, (r"$u_x/U_\infty$", r"$u_y/U_\infty$"))
    for row, (low, high) in enumerate(limits):
        margin = 0.05 * (high - low)
        axes[row, 0].set_ylim((low - margin) / FREESTREAM_SPEED, (high + margin) / FREESTREAM_SPEED)
    save_fig(figure, CASE_DIR / "figures" / name)


def phase_averaged_profiles(samples, limits):
    """Velocity averaged over equal shedding phase, and its difference from the reference."""
    reference_phase, coupled_phase = (
        shedding_phase(REFERENCE_FORCES),
        shedding_phase(COUPLED_FORCES),
    )
    averages = {}
    for (station, label), table in samples.items():
        phase = reference_phase if label == "Reference FVM" else coupled_phase
        developed = table["time"] >= DEVELOPED_SHEDDING
        positions, point = np.unique(table["position_y"][developed].round(6), return_inverse=True)
        bins = phase_bin(phase(table["time"][developed]))
        for k in range(N_PHASES):
            chosen = bins == k
            count = np.bincount(point[chosen], minlength=len(positions))
            averages[station, label, k] = (
                positions,
                *(
                    np.bincount(point[chosen], table[name][developed][chosen], len(positions))
                    / count
                    for name in ("velocity_x", "velocity_y")
                ),
            )

    rows = []
    for k in range(N_PHASES):
        curves = {station: [] for station in SOURCES}
        for station, files in SOURCES.items():
            reference = averages[station, "Reference FVM", k]
            for label in files:
                positions, u, v = averages[station, label, k]
                curves[station].append(
                    (label, positions / DIAMETER, u / FREESTREAM_SPEED, v / FREESTREAM_SPEED)
                )
                if label == "Reference FVM":
                    continue
                difference = (
                    100.0
                    * np.hypot(
                        u - np.interp(positions, reference[0], reference[1]),
                        v - np.interp(positions, reference[0], reference[2]),
                    )
                    / FREESTREAM_SPEED
                )
                rows.append((k, station, label, np.sqrt(np.mean(difference**2)), difference.max()))
        save_velocity_figure(curves, limits, f"phase_averaged_profiles/phase{k:02d}")

    lines = [
        "phase_bin,phase_start_degrees,station_x_over_D,source,velocity_rms_percent,velocity_max_percent"
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
    curves = {
        station: [(label, *wake_profile(CASE_DIR / path)) for label, path in files.items()]
        for station, files in SOURCES.items()
    }
    figure, _ = profile_figure(curves, (r"$\overline{u}/U_\infty$", r"$v_\mathrm{rms}/U_\infty$"))
    save_fig(figure, CASE_DIR / "figures/velocity_profiles")

    samples = {
        (station, label): np.genfromtxt(CASE_DIR / path, delimiter=",", names=True)
        for station, files in SOURCES.items()
        for label, path in files.items()
    }
    # One y-range per row for all figures, so they can be compared frame by frame
    limits = [
        (
            min(table[name].min() for table in samples.values()),
            max(table[name].max() for table in samples.values()),
        )
        for name in ("velocity_x", "velocity_y")
    ]
    phase_averaged_profiles(samples, limits)
    instantaneous_profiles(samples, limits)


if __name__ == "__main__":
    main()
