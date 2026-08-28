#!/usr/bin/env python3
"""Publication-style FVM/VPM and reference streamwise-velocity fields."""

from pathlib import Path
import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402

FIGURE_FORMAT = "png"
FIGURE_DPI = util.FIGURE_DPI
FIGURE_HEIGHT_CM = 7.5
FIGURE_SIZE = util.figure_size(FIGURE_HEIGHT_CM)

# Manual layout controls (fractions of the fixed 12.5 cm canvas).
LAYOUT_LEFT = 0.11
LAYOUT_RIGHT = 0.99
LAYOUT_BOTTOM = 0.14
LAYOUT_TOP = 0.90
LAYOUT_WSPACE = 0.10
LAYOUT_HSPACE = 0.72
COLORBAR_HEIGHT_RATIO = 0.055

BODY_HALF = 0.5
BODY_MARGIN = 0.05
OUTLET_BAND = 0.12


def _body_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.abs(x) <= BODY_HALF + BODY_MARGIN) & (np.abs(y) <= BODY_HALF + BODY_MARGIN)


def _add_body(ax) -> None:
    ax.add_patch(
        plt.Rectangle(
            (-BODY_HALF, -BODY_HALF),
            2 * BODY_HALF,
            2 * BODY_HALF,
            facecolor=util.COLORS["background_light"],
            edgecolor=util.COLORS["DarkText"],
            lw=0.3,
        )
    )


def _on_grid(source: dict, target: dict, key: str) -> np.ndarray:
    points = np.column_stack((source["x"].ravel(), source["y"].ravel()))
    values = source[key].ravel()
    result = griddata(points, values, (target["x"], target["y"]), method="linear")
    missing = ~np.isfinite(result)
    if np.any(missing):
        result[missing] = griddata(
            points,
            values,
            (target["x"][missing], target["y"][missing]),
            method="nearest",
        )
    return result


def _style_axes(
    fig,
    axes,
    box,
    velocity_plot,
    error_plot,
    velocity_colorbar_ax,
    error_colorbar_ax,
    vmax: float,
    p95: float,
) -> None:
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(box["xmin"], box["xmax"])
        ax.set_ylim(box["ymin"], box["ymax"])
        _add_body(ax)
        ax.set_xlabel(r"$x/D$")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[0].set_ylabel(r"$y/D$")

    fig.colorbar(
        velocity_plot,
        cax=velocity_colorbar_ax,
        orientation="horizontal",
        format="%.1f",
        label=r"$u_x/U_\infty$",
    ).set_ticks(np.linspace(-vmax, vmax, 3))
    fig.colorbar(
        error_plot,
        cax=error_colorbar_ax,
        orientation="horizontal",
        format="%.1f",
        extend="max",
        label=r"$|\Delta u_x|/U_\infty$ [%]",
    ).set_ticks(np.linspace(0, p95, 3))


def _field_figure(
    time: float,
    x: np.ndarray,
    y: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    left_title: str,
    right_title: str,
    name: str,
    box: dict,
    figure_format: str = FIGURE_FORMAT,
    dpi: int = FIGURE_DPI,
) -> tuple[float, float, float]:
    error = np.abs(left - right) * 100.0
    error[_body_mask(x, y)] = np.nan

    vmax = max(float(np.nanmax(np.abs(left))), float(np.nanmax(np.abs(right))))
    levels = np.linspace(-vmax, vmax, 41)
    valid = np.isfinite(error)
    p95 = float(np.nanpercentile(error[valid], 95)) if np.any(valid) else 1.0
    p95 = max(p95, 1e-3)
    maximum = float(np.nanmax(error[valid])) if np.any(valid) else 0.0

    fig = plt.figure(figsize=FIGURE_SIZE, dpi=dpi)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, COLORBAR_HEIGHT_RATIO))
    axes = np.asarray(
        [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[0, 2]),
        ]
    )
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])
    axes[2].sharex(axes[0])
    axes[2].sharey(axes[0])
    velocity_colorbar_ax = fig.add_subplot(grid[1, :2])
    error_colorbar_ax = fig.add_subplot(grid[1, 2])
    fig.subplots_adjust(
        left=LAYOUT_LEFT,
        right=LAYOUT_RIGHT,
        bottom=LAYOUT_BOTTOM,
        top=LAYOUT_TOP,
        wspace=LAYOUT_WSPACE,
        hspace=LAYOUT_HSPACE,
    )
    axes[0].contourf(x, y, left, levels=levels, cmap=util.COLORMAPS["velocity"], extend="both")
    axes[0].set_title(left_title)
    velocity_plot = axes[1].contourf(
        x, y, right, levels=levels, cmap=util.COLORMAPS["velocity"], extend="both"
    )
    axes[1].set_title(right_title)
    error_plot = axes[2].pcolormesh(
        x, y, error, cmap=util.COLORMAPS["error"], vmin=0, vmax=p95, shading="auto"
    )
    axes[2].set_title("Error")
    axes[2].text(
        0.03,
        0.97,
        rf"$p_{{95}}={p95:.1f}\%$" "\n" rf"max $={maximum:.1f}\%$",
        transform=axes[2].transAxes,
        va="top",
        fontsize=util.FONT_SIZE_PT,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
    )
    _style_axes(
        fig,
        axes,
        box,
        velocity_plot,
        error_plot,
        velocity_colorbar_ax,
        error_colorbar_ax,
        vmax,
        p95,
    )

    util.save(fig, f"{name}_t{time:.2f}", figure_format, dpi)
    plt.close(fig)
    return float(np.nanmean(error[valid])), p95, maximum


def plot_frame(
    time: float,
    consts: dict,
    figure_format: str = FIGURE_FORMAT,
    dpi: int = FIGURE_DPI,
) -> None:
    fvm = util.load_slice("fvm", time)
    vpm = util.load_slice("vpm", time)
    reference = util.load_slice("reference", time)
    if fvm is None or vpm is None or reference is None:
        return

    freestream_speed = consts["freestream_speed"]
    reference_length = consts["reference_length"]
    x, y = fvm["x"] / reference_length, fvm["y"] / reference_length
    fvm_velocity_x = fvm["velocity_x"] / freestream_speed
    vpm_velocity_x = _on_grid(vpm, fvm, "velocity_x") / freestream_speed

    mean, p95, maximum = _field_figure(
        time,
        x,
        y,
        fvm_velocity_x,
        vpm_velocity_x,
        "FVM",
        "VPM",
        "velocity_fields",
        consts["box"],
        figure_format,
        dpi,
    )
    valid = np.isfinite(fvm_velocity_x) & np.isfinite(vpm_velocity_x)
    outlet = valid & (x >= consts["box"]["xmax"] - OUTLET_BAND)
    outlet_error = np.abs(fvm_velocity_x[outlet] - vpm_velocity_x[outlet]) * 100.0
    print(
        f"  velocity_fields t={time:.2f}: mean={mean:.1f}%, p95={p95:.1f}%, "
        f"max={maximum:.1f}%, outlet mean={np.nanmean(outlet_error):.1f}%"
    )

    reference_x = reference["x"] / reference_length
    reference_y = reference["y"] / reference_length
    reference_velocity_x = reference["velocity_x"] / freestream_speed
    vpm_reference_velocity_x = _on_grid(vpm, reference, "velocity_x") / freestream_speed
    mean, p95, maximum = _field_figure(
        time,
        reference_x,
        reference_y,
        reference_velocity_x,
        vpm_reference_velocity_x,
        "Reference",
        "VPM",
        "reference_vpm_fields",
        consts["box"],
        figure_format,
        dpi,
    )
    print(
        f"  reference_vpm_fields t={time:.2f}: mean={mean:.1f}%, p95={p95:.1f}%, max={maximum:.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=util.EXPORT_FORMATS, default=FIGURE_FORMAT)
    parser.add_argument("--dpi", type=int, default=FIGURE_DPI, help="PNG resolution in dpi.")
    args = parser.parse_args()

    times = util.common_times(
        util.slice_times("fvm"),
        util.slice_times("vpm"),
        util.slice_times("reference"),
    )
    if times.size == 0:
        raise SystemExit("No coincident field samples found in samples/.")
    consts = util.run_constants()
    for time in times:
        plot_frame(float(time), consts, args.format, args.dpi)


if __name__ == "__main__":
    main()
