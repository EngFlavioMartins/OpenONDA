#!/usr/bin/env python3
"""Velocity and vorticity of the reference and the coupled flow, and their difference, on the
mid-span plane and two cross-flow planes: at every common sample time and time-mean."""

from pathlib import Path
import xml.etree.ElementTree as ET

from matplotlib import colormaps
from matplotlib.path import Path as Outline
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator

from openonda.plotting import CM, COLORMAPS, FONT_SIZE_PT, export_figure, set_thesis_style
from openonda.saved_times import match_saved_times

from assets.naca_surface import section_outline

CASE_DIR = Path(__file__).resolve().parents[1]
FIGURES = CASE_DIR / "figures"
CHORD = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s
ANGLE_OF_ATTACK = 10.0  # degrees, nose-up pitch of the wing
FVM_DOMAIN_BOUNDS = (-1.6, 2.72, -1.28, 1.24, -3.6, 3.6)  # m
WING_SECTION = section_outline("4412", CHORD, 161, ANGLE_OF_ATTACK)

# Each plane: its slices, the axes drawn across and up, the plotting window, the vorticity
# component normal to it, and whether the coupled FVM samples it
PLANES = {
    "z0": {"axes": (0, 1), "window": (-1.0, 8.0, -1.5, 1.5), "normal": 2, "fvm": True},
    "x2": {"axes": (2, 1), "window": (-3.6, 3.6, -1.2, 1.2), "normal": 0, "fvm": True},
    "x5": {"axes": (2, 1), "window": (-3.6, 3.6, -1.2, 1.2), "normal": 0, "fvm": False},
}
AXIS_NAMES = "xyz"

# Differences in amber, pale at zero, apart from the field colour maps
DIFFERENCE_COLOURS = colormaps["thesis_amber"].reversed()

# Figure layout in cm, measured from the top of the canvas; the plotting area is centred
LEFT = 1.4
WIDTH = 12.5 - 2 * LEFT
TITLE, PANEL_GAP, BAR_GAP, BAR_HEIGHT, BAR_LABEL = 0.7, 0.75, 1.25, 0.16, 1.25


def saved_frames(collection):
    """Times and frame paths of a VTK collection."""
    entries = sorted(
        (float(item.attrib["timestep"]), collection.parent / item.attrib["file"])
        for item in ET.parse(collection).iter("DataSet")
    )
    return [time for time, _ in entries], [path for _, path in entries]


def on_plane(path, plane):
    """Plane coordinates (across, up) and the velocity and vorticity, as (up, across) arrays."""
    grid = pv.read(path)
    n_first, n_second, _ = grid.dimensions
    across, up = plane["axes"]

    # Frames list the points with the first in-plane axis fastest
    def lattice(values):
        values = values.reshape(n_second, n_first, *values.shape[1:])
        return values if across < up else values.swapaxes(0, 1)

    points = lattice(grid.points)
    return (
        points[..., across],
        points[..., up],
        lattice(grid.point_data["velocity"]) / FREESTREAM_SPEED,
        lattice(grid.point_data["vorticity"]) * CHORD / FREESTREAM_SPEED,
    )


def onto(fvm_plane, across, up):
    """FVM values interpolated to the comparison lattice; NaN outside the FVM slice."""
    fvm_across, fvm_up, velocity, vorticity = fvm_plane
    points = np.stack((up, across), axis=-1)
    return [
        RegularGridInterpolator(
            (fvm_up[:, 0], fvm_across[0]), values, bounds_error=False, fill_value=np.nan
        )(points)
        for values in (velocity, vorticity)
    ]


def comparison_figure(name, plane_name, quantity, across, up, panels):
    """Reference, coupled and difference panels with a colour bar for each scale."""
    plane = PLANES[plane_name]
    window = plane["window"]
    labels = {
        "velocity": ("$u_x/U_\\infty$", r"$\|\Delta\mathbf{u}\|/U_\infty$ [\%]"),
        "vorticity": (
            f"$\\omega_{AXIS_NAMES[plane['normal']]} c/U_\\infty$",
            f"$|\\Delta\\omega_{AXIS_NAMES[plane['normal']]}| c/U_\\infty$",
        ),
    }[quantity]
    reference, coupled, difference = panels
    visible = (across >= window[0]) & (across <= window[1]) & (up >= window[2]) & (up <= window[3])
    valid = visible & np.isfinite(difference)
    if quantity == "velocity":
        low, high = (
            np.nanmin([reference[valid], coupled[valid]]),
            np.nanmax([reference[valid], coupled[valid]]),
        )
        # A plane the wake has not reached yet holds the uniform freestream
        high = max(high, low + 0.01)
        cmap = COLORMAPS["velocity"]
    else:
        # Saturate at the wake's peak, which lies well below the wall layer's
        wake = valid & ((across > 0.6 * CHORD) if plane_name == "z0" else True)
        high = max(np.percentile(np.abs([reference[wake], coupled[wake]]), 99.5), 0.01)
        low = -high
        cmap = COLORMAPS["vorticity"]
    largest = max(np.nanmax(np.where(valid, difference, np.nan)), 1e-6)
    rows = (
        ("(a) Reference FVM", reference, np.linspace(low, high, 41), cmap),
        ("(b) Coupled FVM and VPM", coupled, np.linspace(low, high, 41), cmap),
        ("(c) Difference", difference, np.linspace(0.0, largest, 41), DIFFERENCE_COLOURS),
    )

    set_thesis_style()
    panel = WIDTH * (window[3] - window[2]) / (window[1] - window[0])
    height = 3 * (TITLE + panel) + PANEL_GAP + 2 * (BAR_GAP + BAR_HEIGHT + BAR_LABEL)
    figure = plt.figure(figsize=(12.5 * CM, height * CM))

    def axes_at(top, panel_height):
        return figure.add_axes(
            (LEFT / 12.5, 1 - (top + panel_height) / height, WIDTH / 12.5, panel_height / height)
        )

    across_name, up_name = (AXIS_NAMES[axis] for axis in plane["axes"])
    top = 0.0
    for row, (title, values, levels, colours) in enumerate(rows):
        top += TITLE
        axes = axes_at(top, panel)
        plot = axes.contourf(across, up, values, levels=levels, cmap=colours, extend="both")
        if plane_name == "z0":
            axes.add_patch(plt.Polygon(WING_SECTION, facecolor="white", edgecolor="black", lw=0.6))
        if row > 0 and plane["fvm"]:
            # The FVM box: the coupled flow is the FVM inside it, the particles outside
            box = (
                FVM_DOMAIN_BOUNDS[2 * plane["axes"][0] : 2 * plane["axes"][0] + 2],
                FVM_DOMAIN_BOUNDS[2 * plane["axes"][1] : 2 * plane["axes"][1] + 2],
            )
            axes.add_patch(
                plt.Rectangle(
                    (box[0][0], box[1][0]),
                    box[0][1] - box[0][0],
                    box[1][1] - box[1][0],
                    fill=False,
                    color="black",
                    linestyle="--",
                    linewidth=0.6,
                )
            )
        axes.set(xlim=window[:2], ylim=window[2:], aspect="equal", ylabel=f"${up_name}/c$")
        axes.set_title(title, pad=1.0, fontsize=FONT_SIZE_PT)
        if row == 0:
            axes.tick_params(labelbottom=False)
            top += panel + PANEL_GAP - TITLE
            continue
        axes.set_xlabel(f"${across_name}/c$")
        top += panel + BAR_GAP
        bar_low, bar_high = levels[0], levels[-1]
        figure.colorbar(
            plot,
            cax=axes_at(top, BAR_HEIGHT),
            orientation="horizontal",
            ticks=[bar_low, 0.5 * (bar_low + bar_high), bar_high],
            format="%.2g",
            label=labels[0] if row == 1 else labels[1],
        )
        top += BAR_HEIGHT + BAR_LABEL
    export_figure(figure, FIGURES / f"fields/{plane_name}_{quantity}" / name, figure_format="both")


def compare(name, plane_name, across, up, reference, coupled, inside_wing):
    """Draw the velocity and vorticity comparisons; return the rms and largest differences."""
    normal = PLANES[plane_name]["normal"]
    velocity_difference = 100.0 * np.linalg.norm(coupled[0] - reference[0], axis=-1)
    vorticity_difference = np.abs(coupled[1][..., normal] - reference[1][..., normal])
    for values in (velocity_difference, vorticity_difference):
        values[inside_wing] = np.nan
    fields = (
        ("velocity", reference[0][..., 0], coupled[0][..., 0], velocity_difference),
        ("vorticity", reference[1][..., normal], coupled[1][..., normal], vorticity_difference),
    )
    for quantity, reference_field, coupled_field, difference in fields:
        comparison_figure(
            name, plane_name, quantity, across, up, (reference_field, coupled_field, difference)
        )
    statistics = []
    for difference in (velocity_difference, vorticity_difference):
        valid = difference[np.isfinite(difference)]
        statistics += [np.sqrt(np.mean(valid**2)), valid.max()]
    return statistics


def main():
    rows = []
    for plane_name, plane in PLANES.items():
        samples = CASE_DIR / "samples"
        reference_times, reference_paths = saved_frames(
            CASE_DIR / f"reference_flow/samples/slice_{plane_name}.pvd"
        )
        vpm_times, vpm_paths = saved_frames(samples / f"vpm_slice_{plane_name}.pvd")
        clocks, paths = [reference_times, vpm_times], [reference_paths, vpm_paths]
        if plane["fvm"]:
            fvm_times, fvm_paths = saved_frames(samples / f"fvm_slice_{plane_name}.pvd")
            clocks.append(fvm_times)
            paths.append(fvm_paths)
        common = match_saved_times(*clocks)

        # The particles and the reference share one lattice; the FVM slice covers the box only
        across, up = on_plane(reference_paths[0], plane)[:2]
        inside_wing = np.zeros(across.shape, dtype=bool)
        if plane_name == "z0":
            inside_wing = Outline(WING_SECTION).contains_points(
                np.column_stack((across.ravel(), up.ravel()))
            )
            inside_wing = inside_wing.reshape(across.shape)

        sums = None
        for time, indices in zip(common.times, zip(*common.indices)):
            frames = [on_plane(source[index], plane) for source, index in zip(paths, indices)]
            reference = frames[0][2:]
            coupled = frames[1][2:]
            if plane["fvm"]:
                # The FVM supplies the coupled flow inside its box, the particles outside it
                coupled = [
                    np.where(np.isfinite(inner), inner, outer)
                    for inner, outer in zip(onto(frames[2], across, up), coupled)
                ]
            statistics = compare(
                f"t{time:06.2f}", plane_name, across, up, reference, coupled, inside_wing
            )
            rows.append([plane_name, f"{time:.6g}", *(f"{value:.6g}" for value in statistics)])
            sums = (
                [reference, coupled]
                if sums is None
                else [
                    [s + f for s, f in zip(total, new)]
                    for total, new in zip(sums, (reference, coupled))
                ]
            )

        # Time-mean over the common frames of the statistics window
        n_frames = len(common.times)
        mean_reference, mean_coupled = ([field / n_frames for field in total] for total in sums)
        statistics = compare(
            "mean", plane_name, across, up, mean_reference, mean_coupled, inside_wing
        )
        rows.append([plane_name, "mean", *(f"{value:.6g}" for value in statistics)])
        print(
            f"Plane {plane_name}, time-mean over {n_frames} frames: velocity difference rms "
            f"{statistics[0]:.2g}%, max {statistics[1]:.2g}%; vorticity difference rms "
            f"{statistics[2]:.2g}, max {statistics[3]:.2g} U/c"
        )

    (FIGURES / "auxiliary").mkdir(parents=True, exist_ok=True)
    header = "plane,time,velocity_rms_percent,velocity_max_percent,vorticity_rms,vorticity_max"
    lines = [header, *(",".join(row) for row in rows)]
    (FIGURES / "auxiliary/field_differences.csv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
