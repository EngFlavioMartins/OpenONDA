#!/usr/bin/env python3
"""Mid-plane velocity and vorticity of the FVM-only reference and the coupled flow, and their
difference: at every saved time the runs share, and averaged over the statistics window.

The reference and particle slices share one lattice; the coupled FVM slice is interpolated onto
it inside the FVM box. Writes figures/velocity_fields/t<time>, figures/vorticity_fields/t<time>,
figures/mean_velocity_field, figures/mean_vorticity_field and the difference norms in
figures/auxiliary/. Waits, with a message, until both runs have slices in the statistics window.
"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator

from openonda.plotting import CM, COLORMAPS, FONT_SIZE_PT, export_figure, set_thesis_style
from openonda.saved_times import match_saved_times

CASE_DIR = Path(__file__).absolute().parents[1]
FIGURES = CASE_DIR / "figures"
CUBE_SIDE = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s
WINDOW = (-1.8, 12.0, -2.0, 2.0)  # m, bounds of the reference and particle slices
STATISTICS_WINDOW = (30.0, 60.0)  # s, the time-mean fields
MINIMUM_MEAN_FRAMES = 5  # slices both runs must have in the window before the means are drawn

# Mid-plane slices: reference FVM, coupled FVM in the box, and particles in the outer wake.
REFERENCE_SLICES = CASE_DIR / "reference_flow/samples/grid_h0045_dt0.02/slice_z0.pvd"
COUPLED_FVM = CASE_DIR / "samples/fvm_slice_z0.pvd"
COUPLED_VPM = CASE_DIR / "samples/vpm_slice_z0.pvd"
COUPLER_METADATA = CASE_DIR / "solution/coupler_metadata.json"

# Field labels, colour maps, and difference labels.
QUANTITIES = {
    "velocity": ("u_x", "/U_\\infty", COLORMAPS["velocity"], r"$\|\Delta\mathbf{u}\|/U_\infty$ ($\%$)"),
    "vorticity": ("\\omega_z", " D/U_\\infty", COLORMAPS["vorticity"], r"$|\Delta\omega_z| D/U_\infty$"),
}

# Absolute differences use amber, with white at zero.
DIFFERENCE_COLOURS = colormaps["thesis_amber"].reversed()

# Fixed thesis layout, in cm.
LEFT = 1.4
WIDTH = 12.5 - 2 * LEFT
PANEL = WIDTH * (WINDOW[3] - WINDOW[2]) / (WINDOW[1] - WINDOW[0])
TITLE, PANEL_GAP, BAR_GAP, BAR_HEIGHT, BAR_LABEL = 0.7, 0.75, 1.25, 0.16, 1.25


def saved_frames(collection):
    """Times and frame paths of a VTK collection."""
    entries = sorted(
        (float(item.attrib["timestep"]), collection.parent / item.attrib["file"])
        for item in ET.parse(collection).iter("DataSet")
    )
    return [time for time, _ in entries], [path for _, path in entries]


def read_slice(path):
    """Lattice coordinates, velocity and vorticity z of a slice frame, in units of U and D."""
    grid = pv.read(path)
    ni, nj, _ = grid.dimensions
    points = grid.points.reshape(nj, ni, 3)
    velocity = grid.point_data["velocity"].reshape(nj, ni, 3) / FREESTREAM_SPEED
    vorticity = grid.point_data["vorticity"][:, 2].reshape(nj, ni) * CUBE_SIDE / FREESTREAM_SPEED
    return points[0, :, 0], points[:, 0, 1], velocity, vorticity


class Lattice:
    """Comparison lattice of the particle slice, with the cube and the FVM box marked."""

    def __init__(self, path, box):
        x_axis, y_axis = read_slice(path)[:2]
        self.x, self.y = np.meshgrid(x_axis, y_axis)
        self.box = box
        self.cube = (np.abs(self.x) < CUBE_SIDE / 2) & (np.abs(self.y) < CUBE_SIDE / 2)
        self.in_box = ((self.x >= box[0]) & (self.x <= box[1]) & (self.y >= box[2])
                       & (self.y <= box[3]))

    def onto(self, path):
        """Velocity and vorticity of a slice interpolated onto the lattice; NaN outside it."""
        x_axis, y_axis, velocity, vorticity = read_slice(path)
        values = []
        for field in (velocity, vorticity):
            field = RegularGridInterpolator((y_axis, x_axis), field, bounds_error=False,
                                            fill_value=np.nan)((self.y, self.x))
            field[self.cube] = np.nan
            values.append(field)
        return values

    def coupled(self, paths):
        """The FVM inside its box and the particles outside it."""
        box_values = self.onto(paths[0])
        values = self.onto(paths[1])
        for inner, outer in zip(box_values, values):
            outer[self.in_box] = inner[self.in_box]
        return values


def differences(reference, coupled):
    """Velocity difference in % of U and vorticity difference on the lattice."""
    return (
        100.0 * np.linalg.norm(coupled[0] - reference[0], axis=-1),
        np.abs(coupled[1] - reference[1]),
    )


def comparison_figure(lattice, path, quantity, reference, coupled, difference, mean=False):
    """Reference, coupled and difference panels with a colour bar for each scale."""
    x, y = lattice.x, lattice.y
    symbol, scale, cmap, difference_label = QUANTITIES[quantity]
    label = f"$\\overline{{{symbol}}}{scale}$" if mean else f"${symbol}{scale}$"
    valid = np.isfinite(difference)
    if quantity == "velocity":
        low = np.nanmin([reference[valid], coupled[valid]])
        high = np.nanmax([reference[valid], coupled[valid]])
    else:
        # Set the vorticity scale from the wake rather than the wall layer.
        wake = valid & (x > CUBE_SIDE)
        high = max(np.percentile(np.abs([reference[wake], coupled[wake]]), 99.5), 0.01)
        low = -high
    # Widen each scale to round ticks so the colour bar ends on a labelled value.
    field_ticks = MaxNLocator(nbins=5).tick_values(low, high)
    difference_ticks = MaxNLocator(nbins=5).tick_values(0.0, max(np.nanmax(difference), 1e-6))
    panels = (
        ("(a) FVM only", reference, field_ticks, cmap),
        ("(b) Coupled FVM and VPM", coupled, field_ticks, cmap),
        ("(c) Difference", difference, difference_ticks, DIFFERENCE_COLOURS),
    )

    set_thesis_style()
    height = 3 * (TITLE + PANEL) + PANEL_GAP + 2 * (BAR_GAP + BAR_HEIGHT + BAR_LABEL)
    figure = plt.figure(figsize=(12.5 * CM, height * CM))

    def axes_at(top, panel_height):
        return figure.add_axes(
            (LEFT / 12.5, 1 - (top + panel_height) / height, WIDTH / 12.5, panel_height / height)
        )

    box = lattice.box
    top = 0.0
    for row, (title, values, ticks, colours) in enumerate(panels):
        top += TITLE
        axes = axes_at(top, PANEL)
        levels = np.linspace(ticks[0], ticks[-1], 41)
        plot = axes.contourf(x, y, values, levels=levels, cmap=colours, extend="both")
        axes.add_patch(plt.Rectangle((-CUBE_SIDE / 2, -CUBE_SIDE / 2), CUBE_SIDE, CUBE_SIDE,
                                     facecolor="white", edgecolor="black", lw=0.6))
        if row > 0:
            # The FVM field occupies the box; particles occupy the outer wake.
            axes.add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2],
                                         fill=False, edgecolor="black", linestyle="--",
                                         linewidth=0.6))
        axes.set(xlim=WINDOW[:2], ylim=WINDOW[2:], aspect="equal", yticks=[-1, 0, 1], ylabel="$y/D$")
        axes.set_title(title, pad=4.0, fontsize=FONT_SIZE_PT)
        if row == 0:
            axes.tick_params(labelbottom=False)
            top += PANEL + PANEL_GAP - TITLE
            continue
        axes.set_xlabel("$x/D$")
        top += PANEL + BAR_GAP
        # The default formatter writes a typeset minus sign.
        figure.colorbar(plot, cax=axes_at(top, BAR_HEIGHT), orientation="horizontal", ticks=ticks,
                        label=label if row == 1 else difference_label)
        top += BAR_HEIGHT + BAR_LABEL
    export_figure(figure, FIGURES / path, figure_format="png")


def difference_norms(velocity_difference, vorticity_difference):
    """Rms and maximum of the velocity and vorticity differences over the lattice."""
    row = []
    for difference in (velocity_difference, vorticity_difference):
        valid = difference[np.isfinite(difference)]
        row += [np.sqrt(np.mean(valid**2)), valid.max()]
    return row


def write_table(name, rows, header):
    (FIGURES / "auxiliary").mkdir(parents=True, exist_ok=True)
    np.savetxt(FIGURES / "auxiliary" / name, rows, delimiter=",", fmt="%.6g", header=header,
               comments="")


def main():
    missing = [path for path in (REFERENCE_SLICES, COUPLED_FVM, COUPLED_VPM, COUPLER_METADATA)
               if not path.exists()]
    if missing:
        print("fields: wait for " + ", ".join(str(path.relative_to(CASE_DIR)) for path in missing))
        return
    reference_times, reference_paths = saved_frames(REFERENCE_SLICES)
    fvm_times, fvm_paths = saved_frames(COUPLED_FVM)
    vpm_times, vpm_paths = saved_frames(COUPLED_VPM)
    coupled_times = match_saved_times(fvm_times, vpm_times)
    coupled_paths = [(fvm_paths[f], vpm_paths[v]) for f, v in zip(*coupled_times.indices)]
    common = match_saved_times(reference_times, coupled_times.times)
    in_window = [STATISTICS_WINDOW[0] <= time <= STATISTICS_WINDOW[1] for time in common.times]
    if sum(in_window) < MINIMUM_MEAN_FRAMES:
        print(f"fields: wait for {MINIMUM_MEAN_FRAMES} shared slices from t = {STATISTICS_WINDOW[0]:g} s "
              f"(have {sum(in_window)}; coupled slices to t = {coupled_times.times[-1]:g} s)")
        return

    bounds = json.loads(COUPLER_METADATA.read_text())["coupling"]["fvm_domain_bounds"]
    lattice = Lattice(vpm_paths[0], (bounds["xmin"], bounds["xmax"], bounds["ymin"], bounds["ymax"]))

    # Compare instantaneous fields at the saved times shared by both runs.
    rows, sums = [], None
    for time, r, c, inside in zip(common.times, *common.indices, in_window):
        reference = lattice.onto(reference_paths[r])
        coupled = lattice.coupled(coupled_paths[c])
        velocity_difference, vorticity_difference = differences(reference, coupled)
        name = f"t{time:06.2f}"
        comparison_figure(lattice, f"velocity_fields/{name}", "velocity", reference[0][..., 0],
                          coupled[0][..., 0], velocity_difference)
        comparison_figure(lattice, f"vorticity_fields/{name}", "vorticity", reference[1], coupled[1],
                          vorticity_difference)
        rows.append([time, *difference_norms(velocity_difference, vorticity_difference)])
        if inside:
            pair = (reference, coupled)
            sums = pair if sums is None else [[s + f for s, f in zip(total, new)]
                                              for total, new in zip(sums, pair)]
    write_table("field_differences.csv", rows,
                "time,velocity_rms_percent,velocity_max_percent,vorticity_rms,vorticity_max")

    # Turbulent wakes decorrelate, so compare time means over the statistics window as well.
    frames = sum(in_window)
    reference, coupled = ([field / frames for field in total] for total in sums)
    velocity_difference, vorticity_difference = differences(reference, coupled)
    comparison_figure(lattice, "mean_velocity_field", "velocity", reference[0][..., 0],
                      coupled[0][..., 0], velocity_difference, mean=True)
    comparison_figure(lattice, "mean_vorticity_field", "vorticity", reference[1], coupled[1],
                      vorticity_difference, mean=True)
    mean_row = [frames, *difference_norms(velocity_difference, vorticity_difference)]
    write_table("mean_field_differences.csv", [mean_row],
                "frames,velocity_rms_percent,velocity_max_percent,vorticity_rms,vorticity_max")
    window_end = max(t for t, inside in zip(common.times, in_window) if inside)
    print(f"Mid-plane time-mean fields over {STATISTICS_WINDOW[0]:g}-{window_end:g} s ({frames} slices): "
          f"velocity difference rms {mean_row[1]:.2g} %, max {mean_row[2]:.2g} %; vorticity "
          f"difference rms {mean_row[3]:.2g}, max {mean_row[4]:.2g} U/D")


if __name__ == "__main__":
    main()
