#!/usr/bin/env python3
"""Measure wake scales from the developed FVM reference solutions.

The study reports recirculation and vortex-formation length, shear-layer
thickness, wall strain rate, Strouhal number and wake width.
"""

from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay

from openonda.plotting import fit_thesis_y_label_margins
from openonda.plotting import COLORS, LINE_WIDTH, MARKER_SIZE
from openonda.plotting import save_validation_figure, validation_subplots

from assets.plot_fields import saved_frames
from assets.plot_forces import read_forces, shedding_statistics

CASE_DIR = Path(__file__).resolve().parents[1]
DIAMETER = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s
WINDOW = (40.0, 55.0)  # s

# Reference data: Reynolds number, body cell size [m] and time step [s].
REFERENCES = {
    "reference_flow": (150.0, 0.04, 0.008),
    "sensitivity/reference_re100": (100.0, 0.04, 0.008),
    "sensitivity/reference_re250": (250.0, 0.04, 0.008),
    "sensitivity/reference_h0.08": (150.0, 0.08, 0.016),
    "sensitivity/reference_h0.056569": (150.0, 0.056569, 0.0113138),
    "sensitivity/reference_h0.028284": (150.0, 0.028284, 0.0056568),
}
FIGURE_COLORS = {
    "sensitivity/reference_re100": COLORS["blue"],
    "reference_flow": COLORS["teal"],
    "sensitivity/reference_re250": COLORS["amber"],
}

# Centreline, base-plane and transverse sampling lines.
SPACING = 0.01  # m
CENTRELINE = np.arange(0.52, 8.0, SPACING)  # x [m] at y = 0
BASE_PLANE = np.arange(0.0, 1.5, SPACING)  # y [m] at x = D/2
STATIONS = (2.0, 4.0)  # x [m]
TRANSVERSE = np.arange(-3.0, 3.0 + SPACING / 2, SPACING)  # y [m] at each station

PEAK_FIT = 0.15  # m, half-width of the parabola fitted through the rms peak


def sample_points():
    """In-plane points of the centreline, the base plane and the stations, in that order."""
    lines = [np.column_stack((CENTRELINE, np.zeros_like(CENTRELINE)))]
    lines.append(np.column_stack((np.full_like(BASE_PLANE, 0.5 * DIAMETER), BASE_PLANE)))
    lines += [np.column_stack((np.full_like(TRANSVERSE, x), TRANSVERSE)) for x in STATIONS]
    return np.vstack(lines)


def line_interpolation(cell_centres, points):
    """Cells and barycentric weights that interpolate cell values linearly to the points."""
    wake = (cell_centres[:, 0] > -3.0) & (cell_centres[:, 0] < 12.0)
    wake &= np.abs(cell_centres[:, 1]) < 6.0
    triangles = Delaunay(cell_centres[wake])
    simplex = triangles.find_simplex(points)
    affine = triangles.transform[simplex]
    weights = np.einsum("nij,nj->ni", affine[:, :2], points - affine[:, 2])
    weights = np.column_stack((weights, 1.0 - weights.sum(axis=1)))
    return np.flatnonzero(wake)[triangles.simplices[simplex]], weights


def run_statistics(directory, cell_size, end_time):
    """Mean and rms velocity on the sample lines and the largest wall strain rate per frame."""
    times, paths = saved_frames(CASE_DIR / directory / "solution/fvm.pvd")
    frames = [path for time, path in zip(times, paths) if WINDOW[0] <= round(time, 6) < end_time]

    # The static mesh uses one interpolation stencil for every frame.
    cell_centres = pv.read(frames[0]).cell_centers().points[:, :2]
    corners, weights = line_interpolation(cell_centres, sample_points())
    wall_cells = np.hypot(cell_centres[:, 0], cell_centres[:, 1]) < 0.5 * DIAMETER + cell_size

    velocity_sum, velocity_squares, strain_rates = 0.0, 0.0, []
    for path in frames:
        frame = pv.read(path)
        velocity = frame.cell_data["velocity"][:, :2]

        # Interpolate velocity to the sample lines.
        sampled = np.einsum("ni,nic->nc", weights, velocity[corners])
        velocity_sum = velocity_sum + sampled
        velocity_squares = velocity_squares + sampled**2

        # Planar wall shear strain rate, |omega_z|/2.
        strain_rates.append(0.5 * np.abs(frame.cell_data["vorticity"][wall_cells, 2]).max())

    mean = velocity_sum / len(frames)
    rms = np.sqrt(np.maximum(velocity_squares / len(frames) - mean**2, 0.0))
    return mean, rms, np.array(strain_rates)


def split_lines(values):
    """Values on the centreline, the base plane and each station."""
    edges = np.cumsum([len(CENTRELINE), len(BASE_PLANE), *[len(TRANSVERSE)] * len(STATIONS)])
    return np.split(values, edges[:-1])


def peak_position(x, values):
    """Position of the maximum, refined by a parabola through the values around it."""
    near = np.abs(x - x[np.argmax(values)]) <= PEAK_FIT
    curvature, slope, _ = np.polyfit(x[near], values[near], 2)
    return -slope / (2.0 * curvature)


def recirculation_end(x, mean_u):
    """Last upward zero crossing of the mean streamwise velocity on the centreline."""
    i = np.flatnonzero((mean_u[:-1] < 0.0) & (mean_u[1:] >= 0.0))[-1]
    return x[i] - mean_u[i] * (x[i + 1] - x[i]) / (mean_u[i + 1] - mean_u[i])


def vorticity_thickness(y, mean_u):
    """Velocity jump across the separated shear layer over its largest mean velocity gradient."""
    return np.ptp(mean_u) / np.abs(np.gradient(mean_u, y)).max()


def wake_half_width(y, mean_u):
    """Half the distance between the points where the velocity defect is half its peak."""
    defect = mean_u.max() - mean_u
    centre = np.argmax(defect)
    half = 0.5 * defect[centre]
    upper = centre + np.flatnonzero(defect[centre:] < half)[0]
    lower = centre - np.flatnonzero(defect[centre::-1] < half)[0]
    y_upper = np.interp(half, defect[[upper, upper - 1]], y[[upper, upper - 1]])
    y_lower = np.interp(half, defect[[lower, lower + 1]], y[[lower, lower + 1]])
    return 0.5 * (y_upper - y_lower)


def flow_scales(directory, reynolds_number, cell_size):
    """Flow scales of one reference, lengths in D and strain rates in 1/s, and the centreline
    mean and rms streamwise velocity."""
    forces = read_forces(CASE_DIR / directory / "samples/forces_history.csv")
    strouhal_number = shedding_statistics(forces[forces["time"] <= WINDOW[1]])["Strouhal number"]

    # Average over whole shedding periods, so that no phase is weighted more than another
    period = DIAMETER / (strouhal_number * FREESTREAM_SPEED)
    end_time = WINDOW[0] + np.floor((WINDOW[1] - WINDOW[0]) / period) * period
    mean, rms, strain_rates = run_statistics(directory, cell_size, end_time)
    centreline, base_plane, *stations = zip(split_lines(mean), split_lines(rms))
    (centreline_mean, centreline_rms), (base_mean, _) = centreline, base_plane

    thickness = vorticity_thickness(BASE_PLANE, base_mean[:, 0])
    scales = {
        "recirculation_length": (recirculation_end(CENTRELINE, centreline_mean[:, 0]) - 0.5)
        / DIAMETER,
        "formation_length": peak_position(CENTRELINE, centreline_rms[:, 0]) / DIAMETER,
        "transverse_rms_peak": peak_position(CENTRELINE, centreline_rms[:, 1]) / DIAMETER,
        "vorticity_thickness": thickness / DIAMETER,
        "vorticity_thickness_sqrt_re": thickness * np.sqrt(reynolds_number) / DIAMETER,
        "strain_rate_mean": strain_rates.mean(),
        "strain_rate_peak": strain_rates.max(),
        "strouhal_number": strouhal_number,
        **{
            f"half_width_x{x:g}": wake_half_width(TRANSVERSE, station_mean[:, 0]) / DIAMETER
            for x, (station_mean, _) in zip(STATIONS, stations)
        },
    }
    return scales, centreline_mean[:, 0] / FREESTREAM_SPEED, centreline_rms[:, 0] / FREESTREAM_SPEED


def centreline_figure(profiles, measured):
    """Mean and rms streamwise velocity on the wake centreline, with L_r and L_f marked."""
    figure, axes = validation_subplots(2, sharex=True, height_cm=9.2, top_padding_cm=1.6)
    x = CENTRELINE / DIAMETER
    for directory, color in FIGURE_COLORS.items():
        mean_u, rms_u = profiles[directory]
        scales = measured[directory]
        label = f"$Re={REFERENCES[directory][0]:g}$"
        axes[0].plot(x, mean_u, color=color, linewidth=LINE_WIDTH, label=label)
        axes[1].plot(x, rms_u, color=color, linewidth=LINE_WIDTH, label=label)

        # End of the recirculation region on the zero line, formation point at the rms peak
        end = 0.5 + scales["recirculation_length"]
        formation = scales["formation_length"]
        axes[0].plot(end, 0.0, "o", color=color, markersize=1.5 * MARKER_SIZE)
        axes[1].plot(
            formation,
            np.interp(formation, x, rms_u),
            "D",
            color=color,
            markersize=1.5 * MARKER_SIZE,
        )
    axes[0].axhline(0.0, color=COLORS["background_light"], linewidth=LINE_WIDTH, zorder=0)
    axes[0].plot([], [], "o", color=COLORS["gray"], label="$x=D/2+L_r$")
    axes[0].plot([], [], "D", color=COLORS["gray"], label="$x=L_f$")
    axes[0].set_ylabel(r"$\overline{u}/U_\infty$")
    axes[1].set_ylabel(r"$u_\mathrm{rms}/U_\infty$")
    axes[1].set_xlabel("$x/D$ on $y=0$")
    axes[1].set_xlim(0.5, 6.0)
    axes[1].margins(y=0.1)  # room for the formation-point markers

    # Reynolds numbers in the first legend row, the two markers below them
    handles, labels = axes[0].get_legend_handles_labels()
    order = (0, 3, 1, 4, 2)
    figure.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.8,
    )
    fit_thesis_y_label_margins(figure, figure.axes)
    save_validation_figure(figure, CASE_DIR / "figures/flow_scales")


def main():
    measured, profiles = {}, {}
    for directory, (reynolds_number, cell_size, _) in REFERENCES.items():
        measured[directory], *profiles[directory] = flow_scales(
            directory, reynolds_number, cell_size
        )

    # One row per reference
    names = list(next(iter(measured.values())))
    lines = [",".join(["name", "reynolds_number", "cell_size", "time_step", *names])]
    for directory, (reynolds_number, cell_size, time_step) in REFERENCES.items():
        row = [Path(directory).name, f"{reynolds_number:g}", f"{cell_size:g}", f"{time_step:g}"]
        row += [f"{measured[directory][name]:.5g}" for name in names]
        lines.append(",".join(row))
    (CASE_DIR / "figures/auxiliary").mkdir(parents=True, exist_ok=True)
    (CASE_DIR / "figures/auxiliary/flow_scales.csv").write_text("\n".join(lines) + "\n")

    print(f"Flow scales over whole shedding periods in {WINDOW[0]:g}-{WINDOW[1]:g} s, lengths in D")
    print(
        f"  {'reference':18s} {'Re':>5s} {'h':>8s} {'L_r':>6s} {'L_f':>6s} {'v_rms':>6s}"
        f" {'d_w':>6s} {'d_w√Re':>7s} {'|S|mean':>8s} {'|S|peak':>8s} {'St':>7s}"
        f" {'b(2D)':>6s} {'b(4D)':>6s}"
    )
    for directory, (reynolds_number, cell_size, _) in REFERENCES.items():
        s = measured[directory]
        print(
            f"  {Path(directory).name:18s} {reynolds_number:5g} {cell_size:8.6g}"
            f" {s['recirculation_length']:6.3f} {s['formation_length']:6.3f}"
            f" {s['transverse_rms_peak']:6.3f} {s['vorticity_thickness']:6.3f}"
            f" {s['vorticity_thickness_sqrt_re']:7.3f} {s['strain_rate_mean']:8.2f}"
            f" {s['strain_rate_peak']:8.2f} {s['strouhal_number']:7.4f}"
            f" {s['half_width_x2']:6.3f} {s['half_width_x4']:6.3f}"
        )

    centreline_figure(profiles, measured)


if __name__ == "__main__":
    main()
