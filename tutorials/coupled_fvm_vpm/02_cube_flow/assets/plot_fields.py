#!/usr/bin/env python3
"""Mid-plane velocity and vorticity of the reference and the coupled flow, and their difference:
at every common sample time and averaged over developed flow."""

from pathlib import Path
import xml.etree.ElementTree as ET

from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator

from openonda.plotting import CM, COLORMAPS, FONT_SIZE_PT, export_figure, set_thesis_style
from openonda.saved_times import match_saved_times

CASE_DIR = Path(__file__).resolve().parents[1]
FIGURES = CASE_DIR / "figures"
CUBE_SIDE = 1.0  # m
FREESTREAM_SPEED = 1.0  # m/s
FVM_DOMAIN_BOUNDS = (-1.8, 4.3, -1.8, 1.8)  # m, x and y bounds in the mid-plane
WINDOW = (-1.8, 12.0, -2.0, 2.0)  # m, bounds of the reference and particle slices
DEVELOPED_FLOW = 30.0  # s, start of the time averages
STATISTICS_END = 60.0  # s, end of the coupled run; the reference is cut to the same window

# Mid-plane fields: reference FVM, coupled FVM in the box, and particles in the outer wake.
REFERENCE_FIELDS = CASE_DIR / "reference_flow/solution/grid_h0045/fvm.pvd"
COUPLED_FVM = CASE_DIR / "samples/fvm_slice_z0.pvd"
COUPLED_VPM = CASE_DIR / "samples/vpm_slice_z0.pvd"

# Field labels, colour maps, and difference scales.
QUANTITIES = {
    "velocity": (
        "u_x",
        "/U_\\infty",
        COLORMAPS["velocity"],
        r"$\|\Delta\mathbf{u}\|/U_\infty$ [\%]",
    ),
    "vorticity": (
        "\\omega_z",
        " D/U_\\infty",
        COLORMAPS["vorticity"],
        r"$|\Delta\omega_z| D/U_\infty$",
    ),
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


# The particle slice defines the comparison lattice.
x_axis, y_axis = read_slice(saved_frames(COUPLED_VPM)[1][0])[:2]
x, y = np.meshgrid(x_axis, y_axis)
sample_points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
cube = (np.abs(x) < CUBE_SIDE / 2) & (np.abs(y) < CUBE_SIDE / 2)
in_fvm_domain = (x <= FVM_DOMAIN_BOUNDS[1]) & (np.abs(y) <= FVM_DOMAIN_BOUNDS[3])


def reference_volume_fields(path):
    """Fine-grid reference fields sampled onto the coupled slice lattice."""
    mesh = pv.read(path)
    velocity = mesh.cell_data["velocity"].copy()
    vorticity = mesh.cell_data["vorticity"].copy()
    mesh.cell_data.clear()
    mesh.cell_data["velocity"] = velocity
    mesh.cell_data["vorticity"] = vorticity
    fields = pv.PolyData(sample_points).sample(mesh.cell_data_to_point_data(pass_cell_data=False))
    valid = np.asarray(fields.point_data["vtkValidPointMask"], dtype=bool).reshape(x.shape)
    velocity = fields.point_data["velocity"].reshape(*x.shape, 3) / FREESTREAM_SPEED
    vorticity = fields.point_data["vorticity"][:, 2].reshape(x.shape) * CUBE_SIDE / FREESTREAM_SPEED
    velocity[~valid], vorticity[~valid] = np.nan, np.nan
    velocity[cube], vorticity[cube] = np.nan, np.nan
    return velocity, vorticity


def coupled_fields(paths):
    """The FVM inside its box and the particles outside it, on the lattice."""
    box_x, box_y, box_velocity, box_vorticity = read_slice(paths[0])
    _, _, velocity, vorticity = read_slice(paths[1])
    for box_values, values in ((box_velocity, velocity), (box_vorticity, vorticity)):
        on_lattice = RegularGridInterpolator(
            (box_y, box_x), box_values, bounds_error=False, fill_value=np.nan
        )((y, x))
        values[in_fvm_domain] = on_lattice[in_fvm_domain]
        values[cube] = np.nan
    return velocity, vorticity


def differences(reference, coupled):
    """Velocity difference in % of U and vorticity difference on the lattice."""
    return (
        100.0 * np.linalg.norm(coupled[0] - reference[0], axis=-1),
        np.abs(coupled[1] - reference[1]),
    )


def comparison_figure(path, quantity, reference, coupled, difference, mean=False):
    """Reference, coupled and difference panels with a colour bar for each scale."""
    symbol, scale, cmap, difference_label = QUANTITIES[quantity]
    label = f"$\\overline{{{symbol}}}{scale}$" if mean else f"${symbol}{scale}$"
    valid = np.isfinite(difference)
    if quantity == "velocity":
        low, high = (
            np.nanmin([reference[valid], coupled[valid]]),
            np.nanmax([reference[valid], coupled[valid]]),
        )
    else:
        # Set the vorticity scale from the wake rather than the wall layer.
        wake = valid & (x > CUBE_SIDE)
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
            plt.Rectangle(
                (-CUBE_SIDE / 2, -CUBE_SIDE / 2),
                CUBE_SIDE,
                CUBE_SIDE,
                facecolor="white",
                edgecolor="black",
                lw=0.6,
            )
        )
        if row > 0:
            axes.add_patch(
                plt.Rectangle(
                    FVM_DOMAIN_BOUNDS[::2],
                    FVM_DOMAIN_BOUNDS[1] - FVM_DOMAIN_BOUNDS[0],
                    FVM_DOMAIN_BOUNDS[3] - FVM_DOMAIN_BOUNDS[2],
                    fill=False,
                    edgecolor="black",
                    linestyle="--",
                    linewidth=0.6,
                )
            )
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
    export_figure(figure, FIGURES / path, figure_format="both")


def difference_norms(velocity_difference, vorticity_difference):
    """Rms and maximum of the velocity and vorticity differences over the lattice."""
    row = []
    for difference in (velocity_difference, vorticity_difference):
        valid = difference[np.isfinite(difference)]
        row += [np.sqrt(np.mean(valid**2)), valid.max()]
    return row


def write_table(name, rows, header):
    (FIGURES / "auxiliary").mkdir(parents=True, exist_ok=True)
    np.savetxt(
        FIGURES / "auxiliary" / name,
        rows,
        delimiter=",",
        fmt="%.6g",
        header=header,
        comments="",
    )


def main():
    reference_times, reference_paths = saved_frames(REFERENCE_FIELDS)
    fvm_times, fvm_paths = saved_frames(COUPLED_FVM)
    vpm_times, vpm_paths = saved_frames(COUPLED_VPM)
    coupled_times = match_saved_times(fvm_times, vpm_times)
    coupled_paths = [(fvm_paths[f], vpm_paths[v]) for f, v in zip(*coupled_times.indices)]

    # Compare instantaneous fields at the saved times shared by both runs.
    common = match_saved_times(reference_times, coupled_times.times)
    rows = []
    for time, r, c in zip(common.times, *common.indices):
        reference = reference_volume_fields(reference_paths[r])
        coupled = coupled_fields(coupled_paths[c])
        velocity_difference, vorticity_difference = differences(reference, coupled)
        name = f"t{time:06.2f}"
        comparison_figure(
            f"velocity_fields/{name}",
            "velocity",
            reference[0][..., 0],
            coupled[0][..., 0],
            velocity_difference,
        )
        comparison_figure(
            f"vorticity_fields/{name}", "vorticity", reference[1], coupled[1], vorticity_difference
        )
        rows.append([time, *difference_norms(velocity_difference, vorticity_difference)])
    write_table(
        "field_differences.csv",
        rows,
        "time,velocity_rms_percent,velocity_max_percent,vorticity_rms,vorticity_max",
    )

    # Turbulent wakes decorrelate, so compare developed time means as well.
    def time_mean(times, paths, fields_of):
        states = [fields_of(path) for time, path in zip(times, paths) if DEVELOPED_FLOW <= time <= STATISTICS_END]
        return [np.mean([state[k] for state in states], axis=0) for k in range(2)], len(states)

    reference, n_reference = time_mean(reference_times, reference_paths, reference_volume_fields)
    coupled, n_coupled = time_mean(coupled_times.times, coupled_paths, coupled_fields)
    velocity_difference, vorticity_difference = differences(reference, coupled)
    comparison_figure(
        "mean_velocity_field",
        "velocity",
        reference[0][..., 0],
        coupled[0][..., 0],
        velocity_difference,
        mean=True,
    )
    comparison_figure(
        "mean_vorticity_field",
        "vorticity",
        reference[1],
        coupled[1],
        vorticity_difference,
        mean=True,
    )
    mean_row = [
        n_reference,
        n_coupled,
        *difference_norms(velocity_difference, vorticity_difference),
    ]
    write_table(
        "mean_field_differences.csv",
        [mean_row],
        "reference_frames,coupled_frames,velocity_rms_percent,velocity_max_percent,"
        "vorticity_rms,vorticity_max",
    )

    times = np.array(common.times)
    instantaneous = np.array(rows)[(times >= DEVELOPED_FLOW) & (times <= STATISTICS_END), 1:]
    print(f"Mid-plane differences from the reference over {DEVELOPED_FLOW:g} <= t <= {STATISTICS_END:g} s:")
    print(
        f"  time-mean fields ({n_reference} and {n_coupled} frames): velocity rms "
        f"{mean_row[2]:.2g}%, max {mean_row[3]:.2g}%; vorticity rms {mean_row[4]:.2g}, "
        f"max {mean_row[5]:.2g} U/D"
    )
    print(
        f"  instantaneous fields ({len(instantaneous)} times, mean of each norm): velocity rms "
        f"{instantaneous[:, 0].mean():.2g}%, max {instantaneous[:, 1].mean():.2g}%; vorticity "
        f"rms {instantaneous[:, 2].mean():.2g}, max {instantaneous[:, 3].mean():.2g} U/D"
    )


if __name__ == "__main__":
    main()
