#!/usr/bin/env python3
"""Publication-style VPM/reference wake error fields."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402

FIGURE_FORMAT = "png"
FIGURE_DPI = 400
FIGURE_HEIGHT = 11.0 / 2.54
FIGURE_WIDTH = 12.5 / 2.54
SLICE_NAME = "wake_slice_z0"
BODY_HALF = 0.5
BODY_MARGIN = 0.05


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


def plot_frame(time: float, vpm: dict, reference: dict, consts: dict) -> None:
    U_inf, D = consts["U_inf"], consts["D"]
    x, y = vpm["x"] / D, vpm["y"] / D
    ux_vpm = vpm["Ux"] / U_inf
    omega_vpm = vpm["omega_z"] * D / U_inf
    ux_ref = _on_grid(reference, vpm, "Ux") / U_inf
    omega_ref = _on_grid(reference, vpm, "omega_z") * D / U_inf

    body = (np.abs(x) <= BODY_HALF + BODY_MARGIN) & (np.abs(y) <= BODY_HALF + BODY_MARGIN)
    for field in (ux_vpm, omega_vpm, ux_ref, omega_ref):
        field[body] = np.nan

    velocity_error = (ux_vpm - ux_ref) * 100.0
    vorticity_error = (omega_vpm - omega_ref) * 100.0
    velocity_limit = np.nanpercentile(np.abs(ux_ref), 99) or 1.0
    vorticity_limit = np.nanpercentile(np.abs(omega_ref), 99)
    if not np.isfinite(vorticity_limit) or vorticity_limit <= 0:
        vorticity_limit = np.nanpercentile(np.abs(omega_vpm), 95)
    velocity_error_limit = np.nanpercentile(np.abs(velocity_error), 99) or 1.0
    vorticity_error_limit = np.nanpercentile(np.abs(vorticity_error), 99) or 1.0

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        dpi=FIGURE_DPI,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panels = (
        (axes[0, 0], ux_vpm, -velocity_limit, velocity_limit, "velocity", r"VPM  $u_x/U_\infty$"),
        (
            axes[0, 1],
            omega_vpm,
            -vorticity_limit,
            vorticity_limit,
            "vorticity",
            r"VPM  $\omega_z D/U_\infty$",
        ),
        (
            axes[1, 0],
            ux_ref,
            -velocity_limit,
            velocity_limit,
            "velocity",
            r"Reference  $u_x/U_\infty$",
        ),
        (
            axes[1, 1],
            omega_ref,
            -vorticity_limit,
            vorticity_limit,
            "vorticity",
            r"Reference  $\omega_z D/U_\infty$",
        ),
        (
            axes[2, 0],
            velocity_error,
            -velocity_error_limit,
            velocity_error_limit,
            "error_diverging",
            r"Error  $\Delta u_x/U_\infty$",
        ),
        (
            axes[2, 1],
            vorticity_error,
            -vorticity_error_limit,
            vorticity_error_limit,
            "error_diverging",
            r"Error  $\Delta\omega_z D/U_\infty$",
        ),
    )

    plots = []
    for ax, field, vmin, vmax, cmap, title in panels:
        plot = ax.pcolormesh(
            x, y, field, cmap=util.COLORMAPS[cmap], vmin=vmin, vmax=vmax, shading="auto"
        )
        ax.axvline(consts["box"]["xmax"], color=util.COLORS["DarkText"], ls="--", lw=1.0)
        ax.set(title=title, xlim=(0, 5), ylim=(-1.5, 1.5))
        ax.set_aspect("equal")
        plots.append(plot)

    fig.colorbar(plots[0], ax=[axes[0, 0], axes[1, 0]], pad=0.02, aspect=40)
    fig.colorbar(plots[1], ax=[axes[0, 1], axes[1, 1]], pad=0.02, aspect=40)
    fig.colorbar(plots[4], ax=axes[2, 0], pad=0.02, aspect=20)
    fig.colorbar(plots[5], ax=axes[2, 1], pad=0.02, aspect=20)
    for ax in axes[2, :]:
        ax.set_xlabel(r"$x/D$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$y/D$")
    fig.suptitle(f"Wake fields VPM vs reference ($z=0$, $t={time:.2f}$ s)")

    util.save(fig, f"wake_errors_t{time:.2f}", FIGURE_FORMAT, FIGURE_DPI)
    plt.close(fig)

    x_interface = consts["box"]["xmax"]
    for label, mask in (
        (f"x<{x_interface:g}", x < x_interface),
        (f"{x_interface:g}-3", (x >= x_interface) & (x < 3)),
        ("x>3", x >= 3),
    ):
        print(
            f"  t={time:.2f} {label:>6}: "
            f"|Delta u_x|mean={np.nanmean(np.abs(velocity_error[mask])):.3f}, "
            f"|Delta omega|mean={np.nanmean(np.abs(vorticity_error[mask])):.3f}"
        )


def main() -> None:
    times = util.common_times(
        util.slice_times("vpm", SLICE_NAME),
        util.slice_times("reference", SLICE_NAME),
    )
    if times.size == 0:
        raise SystemExit("No coincident wake samples found in samples/.")
    consts = util.run_constants()
    for time in times:
        vpm = util.load_slice("vpm", float(time), SLICE_NAME)
        reference = util.load_slice("reference", float(time), SLICE_NAME)
        if vpm is not None and reference is not None:
            plot_frame(float(time), vpm, reference, consts)


if __name__ == "__main__":
    main()
