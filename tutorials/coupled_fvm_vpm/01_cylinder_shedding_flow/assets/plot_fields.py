#!/usr/bin/env python3
"""Phase-averaged velocity and vorticity of the reference and the coupled flow, and their difference."""

from pathlib import Path
import xml.etree.ElementTree as ET

from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from openonda.plotting import CM, COLORMAPS, FONT_SIZE_PT, export_figure, set_thesis_style
from openonda.saved_times import match_saved_times

from assets.shedding_phase import N_PHASES, phase_bin, shedding_phase

CASE_DIR = Path(__file__).resolve().parents[1]
FIGURES = CASE_DIR / "figures"
DIAMETER = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s
FVM_BOX = (-2.4, 4.4, -2.4, 2.4)  # m
WINDOW = (-1.5, 9.5, -1.5, 1.5)  # m
SPACING = 0.02  # m, comparison lattice
DEVELOPED_SHEDDING = 40.0  # s, start of the phase averages

# Reference FVM, coupled FVM inside its box and the coupled particles outside it
REFERENCE = CASE_DIR / "reference_flow/solution/fvm.pvd"
COUPLED_FVM = CASE_DIR / "solution/fvm.pvd"
COUPLED_VPM = CASE_DIR / "solution/vpm.pvd"
REFERENCE_FORCES = CASE_DIR / "reference_flow/samples/forces_history.csv"
COUPLED_FORCES = CASE_DIR / "samples/forces_history.csv"

# Field label and colour map, and the label of the difference panel
QUANTITIES = {
    "velocity": ("$u_x/U_\\infty$", COLORMAPS["velocity"], r"$\|\Delta\mathbf{u}\|/U_\infty$ [\%]"),
    "vorticity": (
        "$\\omega_z D/U_\\infty$",
        COLORMAPS["vorticity"],
        r"$|\Delta\omega_z| D/U_\infty$",
    ),
}

# Differences in amber, pale at zero, apart from the field colour maps
DIFFERENCE_COLOURS = colormaps["thesis_amber"].reversed()

# Figure layout in cm, measured from the top of the canvas; the plotting area is centred
LEFT = 1.4
WIDTH = 12.5 - 2 * LEFT
PANEL = WIDTH * (WINDOW[3] - WINDOW[2]) / (WINDOW[1] - WINDOW[0])
TITLE, PANEL_GAP, BAR_GAP, BAR_HEIGHT, BAR_LABEL = 0.7, 0.75, 1.25, 0.16, 1.25

x, y = np.meshgrid(
    np.arange(WINDOW[0], WINDOW[1] + SPACING / 2, SPACING),
    np.arange(WINDOW[2], WINDOW[3] + SPACING / 2, SPACING),
)
body = np.hypot(x, y) < DIAMETER / 2


def saved_frames(collection):
    """Times and frame paths of a VTK collection."""
    entries = sorted(
        (float(item.attrib["timestep"]), collection.parent / item.attrib["file"])
        for item in ET.parse(collection).iter("DataSet")
    )
    return [time for time, _ in entries], [path for _, path in entries]


def triangulation(points):
    """Delaunay triangulation of the sources near the comparison window."""
    near = (
        (points[:, 0] > WINDOW[0] - 0.5)
        & (points[:, 0] < WINDOW[1] + 0.5)
        & (np.abs(points[:, 1]) < WINDOW[3] + 0.5)
    )
    return Delaunay(points[near]), near


def on_lattice(mesh_triangulation, velocity, vorticity):
    """Velocity (u_x, u_y) and vorticity on the lattice; NaN outside the sources."""
    triangles, near = mesh_triangulation
    sampled = LinearNDInterpolator(
        triangles, np.column_stack((velocity[near, :2], vorticity[near, 2]))
    )(x, y)
    sampled[body] = np.nan
    return sampled[..., :2] / FREESTREAM_SPEED, sampled[..., 2] * DIAMETER / FREESTREAM_SPEED


def cell_centres(path):
    """In-plane FVM cell centres; the mesh does not move."""
    return pv.read(path).cell_centers().points[:, :2]


def comparison_figure(name, quantity, reference, coupled, difference):
    """Reference, coupled and difference panels with a colour bar for each scale."""
    label, cmap, difference_label = QUANTITIES[quantity]
    valid = np.isfinite(difference)
    if quantity == "velocity":
        low, high = (
            np.nanmin([reference[valid], coupled[valid]]),
            np.nanmax([reference[valid], coupled[valid]]),
        )
    else:
        # Saturate at the wake's peak, which lies well below the wall layer's
        wake = valid & (x > 1.0)
        high = max(np.percentile(np.abs([reference[wake], coupled[wake]]), 99.5), 0.01)
        low = -high
    largest = max(np.nanmax(difference), 1e-6)
    panels = (
        ("(a) Reference FVM", reference, np.linspace(low, high, 41), cmap),
        ("(b) Coupled FVM and VPM", coupled, np.linspace(low, high, 41), cmap),
        ("(c) Difference", difference, np.linspace(0.0, largest, 41), DIFFERENCE_COLOURS),
    )

    set_thesis_style()
    height = 3 * (TITLE + PANEL) + PANEL_GAP + 2 * (BAR_GAP + BAR_HEIGHT + BAR_LABEL)
    figure = plt.figure(figsize=(12.5 * CM, height * CM))

    def axes_at(top, panel_height):
        return figure.add_axes(
            (LEFT / 12.5, 1 - (top + panel_height) / height, WIDTH / 12.5, panel_height / height)
        )

    top = 0.0
    for row, (title, values, levels, colours) in enumerate(panels):
        top += TITLE
        axes = axes_at(top, PANEL)
        plot = axes.contourf(x, y, values, levels=levels, cmap=colours, extend="both")
        axes.add_patch(
            plt.Circle((0, 0), DIAMETER / 2, facecolor="white", edgecolor="black", lw=0.6)
        )
        if row > 0:
            axes.axvline(FVM_BOX[1], color="black", linestyle="--", linewidth=0.6)
        axes.set(
            xlim=WINDOW[:2], ylim=WINDOW[2:], aspect="equal", yticks=[-1, 0, 1], ylabel="$y/D$"
        )
        axes.set_title(title, pad=1.0, fontsize=FONT_SIZE_PT)
        if row == 0:
            axes.tick_params(labelbottom=False)
            top += PANEL + PANEL_GAP - TITLE
            continue
        axes.set_xlabel("$x/D$")
        top += PANEL + BAR_GAP
        bar_low, bar_high = levels[0], levels[-1]
        figure.colorbar(
            plot,
            cax=axes_at(top, BAR_HEIGHT),
            orientation="horizontal",
            ticks=[bar_low, 0.5 * (bar_low + bar_high), bar_high],
            format="%.2g",
            label=label if row == 1 else difference_label,
        )
        top += BAR_HEIGHT + BAR_LABEL
    export_figure(figure, FIGURES / f"{quantity}_fields" / name, figure_format="both")


def phase_sums(frames, fields_of, phase):
    """Sums of the lattice velocity and vorticity in each phase bin, and the frame counts."""
    velocity = np.zeros((N_PHASES, *x.shape, 2))
    vorticity = np.zeros((N_PHASES, *x.shape))
    count = np.zeros(N_PHASES, dtype=int)
    for time, paths in frames:
        if time < DEVELOPED_SHEDDING:
            continue
        k = phase_bin(phase(time))
        frame_velocity, frame_vorticity = fields_of(paths)
        velocity[k] += frame_velocity
        vorticity[k] += frame_vorticity
        count[k] += 1
    return velocity, vorticity, count


def main():
    reference_times, reference_paths = saved_frames(REFERENCE)
    fvm_times, fvm_paths = saved_frames(COUPLED_FVM)
    vpm_times, vpm_paths = saved_frames(COUPLED_VPM)
    coupled_times = match_saved_times(fvm_times, vpm_times)

    # Both FVM meshes are fixed, so their triangulations are built once
    reference_triangles = triangulation(cell_centres(reference_paths[0]))
    box_triangles = triangulation(cell_centres(fvm_paths[0]))

    def reference_fields(path):
        mesh = pv.read(path)
        return on_lattice(
            reference_triangles, mesh.cell_data["velocity"], mesh.cell_data["vorticity"]
        )

    def coupled_fields(paths):
        box_mesh, particles = pv.read(paths[0]), pv.read(paths[1])
        box = on_lattice(
            box_triangles, box_mesh.cell_data["velocity"], box_mesh.cell_data["vorticity"]
        )
        outside = on_lattice(
            triangulation(particles.points[:, :2]),
            particles.point_data["velocity"],
            particles.point_data["vorticity"],
        )
        # The FVM supplies the coupled flow inside its box, the particles outside it
        return [np.where(np.isfinite(inner), inner, outer) for inner, outer in zip(box, outside)]

    # Each run is averaged over the states of equal shedding phase, by its own lift
    reference_sums = phase_sums(
        zip(reference_times, reference_paths), reference_fields, shedding_phase(REFERENCE_FORCES)
    )
    coupled_sums = phase_sums(
        (
            (time, (fvm_paths[f], vpm_paths[v]))
            for time, (f, v) in zip(coupled_times.times, zip(*coupled_times.indices))
        ),
        coupled_fields,
        shedding_phase(COUPLED_FORCES),
    )

    rows = []
    for k in range(N_PHASES):
        n_reference, n_coupled = reference_sums[2][k], coupled_sums[2][k]
        if n_reference == 0 or n_coupled == 0:
            continue
        reference = [field[k] / n_reference for field in reference_sums[:2]]
        coupled = [field[k] / n_coupled for field in coupled_sums[:2]]
        velocity_difference = 100.0 * np.linalg.norm(coupled[0] - reference[0], axis=-1)
        vorticity_difference = np.abs(coupled[1] - reference[1])
        name = f"phase{k:02d}"
        comparison_figure(
            name, "velocity", reference[0][..., 0], coupled[0][..., 0], velocity_difference
        )
        comparison_figure(name, "vorticity", reference[1], coupled[1], vorticity_difference)

        row = [k, 360.0 * k / N_PHASES, n_reference, n_coupled]
        for difference in (velocity_difference, vorticity_difference):
            valid = difference[np.isfinite(difference)]
            row += [np.sqrt(np.mean(valid**2)), valid.max()]
        rows.append(row)
        print(
            f"Phase {row[1]:3.0f} deg ({n_reference} and {n_coupled} frames): velocity difference "
            f"rms {row[4]:.2g}%, max {row[5]:.2g}%; vorticity difference rms {row[6]:.2g}, "
            f"max {row[7]:.2g} U/D"
        )

    (FIGURES / "auxiliary").mkdir(parents=True, exist_ok=True)
    np.savetxt(
        FIGURES / "auxiliary/field_differences.csv",
        rows,
        delimiter=",",
        fmt="%.6g",
        header=(
            "phase_bin,phase_start_degrees,reference_frames,coupled_frames,velocity_rms_percent,"
            "velocity_max_percent,vorticity_rms,vorticity_max"
        ),
        comments="",
    )


if __name__ == "__main__":
    main()
