#!/usr/bin/env python3
"""Centreline / off-axis velocity profiles and drag history from sampler output."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402

SERIES = ("reference", "fvm", "vpm")
LINES = (("centerline", "Centreline, $y = 0$"), ("offaxis_y075", "Off-axis, $y = 0.75D$"))
FIGURE_FORMAT = "png"
FIGURE_DPI = 400


def _profile_panel(ax, name, title, time, consts):
    U_inf, D = consts["U_inf"], consts["D"]
    for source in SERIES:
        frame = util.load_line(source, name, time)
        if frame is None:
            continue
        ax.plot(
            frame["x"] / D,
            frame["Ux"] / U_inf,
            color=util.colour(source),
            label=util.label(source),
            lw=1.4,
        )
    box = consts["box"]
    ax.axvspan(
        box["xmin"] / D,
        box["xmax"] / D,
        color=util.COLORS["box"],
        alpha=0.25,
        lw=0,
        label="FVM box",
    )
    ax.axvspan(-0.5, 0.5, color=util.COLORS["MaskGray"], alpha=0.55, lw=0)
    ax.axhline(1.0, color=util.COLORS["decor_light"], lw=0.6, ls=":")
    ax.set_xlim(-3.0, 10.0)
    ax.set_xlabel(r"$x/D$")
    ax.set_ylabel(r"$u_x/U_\infty$")
    ax.set_title(title)


def _forces_panel(ax, consts):
    U_inf, D = consts["U_inf"], consts["D"]
    for source in ("reference", "fvm"):
        forces = util.load_forces(source)
        if forces is None:
            continue
        ax.plot(
            forces["time"] * U_inf / D,
            forces["Cd"],
            color=util.colour(source),
            label=util.label(source),
            lw=1.2,
        )
    vpm = util.load_vpm_forces()
    if vpm is not None:
        ax.plot(
            vpm["time"] * U_inf / D,
            vpm["CD"],
            color=util.colour("vpm"),
            label="Coupled VPM (panels)",
            lw=1.2,
        )
    ax.set_xlabel(r"$t\,U_\infty/D$")
    ax.set_ylabel(r"$C_D$")
    ax.set_title("Drag history")
    ax.legend(frameon=False, fontsize=8)


def plot_frame(time: float, consts) -> None:
    fig = plt.figure(figsize=(11, 7))
    grid = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.26)

    axes = []
    for column, (name, title) in enumerate(LINES):
        ax = fig.add_subplot(grid[0, column])
        _profile_panel(ax, name, title, time, consts)
        axes.append(ax)

    _forces_panel(fig.add_subplot(grid[1, :]), consts)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.suptitle(rf"$t = {time:.2f}\,$s", y=1.02, fontsize=10)
    util.save(fig, f"velocity_profiles_t{time:.2f}", FIGURE_FORMAT, FIGURE_DPI)
    plt.close(fig)


def main() -> None:
    consts = util.run_constants()
    times = util.comparison_times()
    if times.size == 0:
        raise SystemExit("No sampled line data found; run the case or the resampler first.")
    for time in times:
        plot_frame(float(time), consts)


if __name__ == "__main__":
    main()
