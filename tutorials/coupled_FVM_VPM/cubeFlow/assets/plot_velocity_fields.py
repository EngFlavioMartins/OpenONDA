#!/usr/bin/env python3
"""z = 0 velocity and vorticity fields for the three solutions, with errors.

Reference / coupled-FVM / coupled-VPM are read straight from the slice
samplers. The two coupled slices share one grid, so their error map is a
direct difference; the reference plane is interpolated onto that grid once.
"""

from __future__ import annotations

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
FIGURE_DPI = 300
BODY_HALF = 0.5


def _add_body(ax):
    ax.add_patch(
        plt.Rectangle(
            (-BODY_HALF, -BODY_HALF),
            2 * BODY_HALF,
            2 * BODY_HALF,
            facecolor=util.COLORS["MaskGray"],
            edgecolor=util.COLORS["AxisBlack"],
            lw=0.6,
            zorder=5,
        )
    )


def _mask_body(x, y, values):
    inside = (np.abs(x) <= BODY_HALF + 1e-9) & (np.abs(y) <= BODY_HALF + 1e-9)
    out = np.array(values, dtype=float)
    out[inside] = np.nan
    return out


def _to_grid(source_slice, target):
    """Interpolate a slice onto the target slice's grid."""
    points = np.column_stack([source_slice["x"].ravel(), source_slice["y"].ravel()])
    target_points = (target["x"], target["y"])
    out = {}
    for key in ("Ux", "omega_z"):
        values = source_slice[key]
        if values is None:
            out[key] = None
            continue
        out[key] = griddata(points, values.ravel(), target_points, method="linear")
    return out


def _panel(ax, x, y, values, cmap, vmin, vmax, title):
    mesh = ax.pcolormesh(x, y, values, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    _add_body(ax)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/D$")
    ax.set_ylabel(r"$y/D$")
    ax.set_title(title, fontsize=9)
    return mesh


def _row(fig, grid, row, panels, cmap, err_cmap, label, err_label):
    finite = [p[1][np.isfinite(p[1])] for p in panels[:-1] if p[1] is not None]
    stacked = np.concatenate(finite) if finite else np.array([0.0, 1.0])
    vmax = np.nanpercentile(np.abs(stacked), 99) or 1.0
    vmin = stacked.min() if stacked.min() < 0 else 0.0

    err = panels[-1][1]
    emax = np.nanpercentile(np.abs(err[np.isfinite(err)]), 95) if err is not None else 1.0
    emax = emax or 1.0

    for column, (title, values, x, y) in enumerate(panels):
        if values is None:
            continue
        ax = fig.add_subplot(grid[row, column])
        is_error = column == len(panels) - 1
        mesh = _panel(
            ax,
            x,
            y,
            values,
            err_cmap if is_error else cmap,
            -emax if is_error else vmin,
            emax if is_error else vmax,
            title,
        )
        bar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
        bar.set_label(err_label if is_error else label, fontsize=8)


def plot_frame(time: float, consts) -> None:
    U_inf, D = consts["U_inf"], consts["D"]
    fvm = util.load_slice("fvm", time)
    vpm = util.load_slice("vpm", time)
    reference = util.load_slice("reference", time)
    target = fvm or vpm or reference
    if target is None:
        return

    x, y = target["x"] / D, target["y"] / D
    ref_on_grid = _to_grid(reference, target) if reference is not None else None

    def series(key, scale):
        out = {}
        for name, source in (("fvm", fvm), ("vpm", vpm)):
            out[name] = (
                _mask_body(x, y, source[key] / scale)
                if source is not None and source[key] is not None
                else None
            )
        out["reference"] = (
            _mask_body(x, y, ref_on_grid[key] / scale)
            if ref_on_grid is not None and ref_on_grid[key] is not None
            else None
        )
        return out

    velocity = series("Ux", U_inf)
    vorticity = series("omega_z", U_inf / D)

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.42)

    for row, (data, cmap, label, err_label) in enumerate(
        (
            (velocity, util.COLORMAPS["velocity"], r"$u_x/U_\infty$", r"$\Delta u_x/U_\infty$"),
            (
                vorticity,
                util.COLORMAPS["vorticity"],
                r"$\omega_z D/U_\infty$",
                r"$\Delta\omega_z D/U_\infty$",
            ),
        )
    ):
        error = None
        if data["fvm"] is not None and data["vpm"] is not None:
            error = data["fvm"] - data["vpm"]
        _row(
            fig,
            grid,
            row,
            [
                ("Reference FVM", data["reference"], x, y),
                ("Coupled FVM", data["fvm"], x, y),
                ("Coupled VPM", data["vpm"], x, y),
                ("FVM $-$ VPM", error, x, y),
            ],
            cmap,
            util.COLORMAPS["error_diverging"],
            label,
            err_label,
        )

    fig.suptitle(rf"$z = 0$ plane,  $t = {time:.2f}\,$s", y=0.98, fontsize=11)
    util.save(fig, f"velocity_fields_t{time:.2f}", FIGURE_FORMAT, FIGURE_DPI)
    plt.close(fig)


def main() -> None:
    consts = util.run_constants()
    times = util.slice_times("fvm")
    if times.size == 0:
        times = util.slice_times("reference")
    times = times[times > 1e-9]
    if times.size == 0:
        raise SystemExit("No sampled slices found; run the case or the resampler first.")
    for time in times:
        plot_frame(float(time), consts)


if __name__ == "__main__":
    main()
