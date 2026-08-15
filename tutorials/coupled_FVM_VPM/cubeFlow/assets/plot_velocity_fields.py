#!/usr/bin/env python3
"""Publication-style FVM/VPM and reference velocity fields."""

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
FIGURE_DPI = 300
FIGURE_HEIGHT = 7.0 / 2.54
FIGURE_WIDTH = 12.5 / 2.54
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


def _style_axes(fig, axes, box, velocity_plot, error_plot, vmax: float, p95: float) -> None:
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(box["xmin"], box["xmax"])
        ax.set_ylim(box["ymin"], box["ymax"])
        _add_body(ax)
        ax.set_xlabel(r"$x/D$")
    axes[0].set_ylabel(r"$y/D$")

    fig.colorbar(
        velocity_plot,
        ax=axes[:2].tolist(),
        orientation="horizontal",
        shrink=0.8,
        pad=0.15,
        aspect=40,
        format="%.1f",
        label=r"$u_x/U_\infty$",
    ).set_ticks(np.linspace(-vmax, vmax, 3))
    fig.colorbar(
        error_plot,
        ax=axes[2],
        orientation="horizontal",
        shrink=0.8,
        pad=0.15,
        aspect=20,
        format="%.1f",
        label=r"$\varepsilon$ [\%]",
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
) -> tuple[float, float]:
    error = np.abs(left - right) * 100.0
    error[np.abs(left) < 0.01] = np.nan
    error[_body_mask(x, y)] = np.nan

    vmax = max(float(np.nanmax(np.abs(left))), float(np.nanmax(np.abs(right))))
    levels = np.linspace(-vmax, vmax, 41)
    valid = np.isfinite(error)
    p95 = float(np.nanpercentile(error[valid], 95)) if np.any(valid) else 1.0
    p95 = max(p95, 1e-3)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        dpi=FIGURE_DPI,
        constrained_layout=True,
        sharex=True,
        sharey=True,
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
    axes[2].set_title(r"$\varepsilon$ [\%]")
    _style_axes(fig, axes, box, velocity_plot, error_plot, vmax, p95)

    util.save(fig, f"{name}_t{time:.2f}", figure_format, FIGURE_DPI)
    plt.close(fig)
    return float(np.nanmean(error[valid])), p95


def plot_frame(time: float, consts: dict, figure_format: str = FIGURE_FORMAT) -> None:
    fvm = util.load_slice("fvm", time)
    vpm = util.load_slice("vpm", time)
    reference = util.load_slice("reference", time)
    if fvm is None or vpm is None or reference is None:
        return

    U_inf, D = consts["U_inf"], consts["D"]
    x, y = fvm["x"] / D, fvm["y"] / D
    ux_fvm = fvm["Ux"] / U_inf
    ux_vpm = _on_grid(vpm, fvm, "Ux") / U_inf
    ux_ref = _on_grid(reference, fvm, "Ux") / U_inf

    mean, p95 = _field_figure(
        time,
        x,
        y,
        ux_fvm,
        ux_vpm,
        r"Hybrid FVM, $u_x^\mathrm{FVM}$",
        r"VPM, $u_x^\mathrm{VPM}$",
        "velocity_fields",
        consts["box"],
        figure_format,
    )
    valid = np.isfinite(ux_fvm) & np.isfinite(ux_vpm)
    outlet = valid & (x >= consts["box"]["xmax"] - OUTLET_BAND)
    outlet_error = np.abs(ux_fvm[outlet] - ux_vpm[outlet]) * 100.0
    print(
        f"  velocity_fields t={time:.2f}: mean={mean:.1f}%, p95={p95:.1f}%, "
        f"outlet mean={np.nanmean(outlet_error):.1f}%"
    )

    mean, p95 = _field_figure(
        time,
        x,
        y,
        ux_ref,
        ux_fvm,
        r"Reference FVM, $u_x^\mathrm{ref}$",
        r"VPM, $u_x^\mathrm{VPM}$",
        "Stitched_RefVsHybrid",
        consts["box"],
        figure_format,
    )
    print(f"  Stitched_RefVsHybrid t={time:.2f}: mean={mean:.1f}%, p95={p95:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
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
        plot_frame(float(time), consts, args.format)


if __name__ == "__main__":
    main()
