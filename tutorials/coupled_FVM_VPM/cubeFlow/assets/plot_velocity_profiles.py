#!/usr/bin/env python3
"""Publication-style velocity profiles and drag history."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402

FIGURE_FORMAT = "png"
FIGURE_DPI = 400
FIGURE_HEIGHT = 6.0 / 2.54
FIGURE_WIDTH = 12.5 / 2.54
PROFILE_MARK_EVERY = 5
FORCE_MARKERS = 20


def _force_series(source: str) -> tuple[np.ndarray, np.ndarray]:
    data = util.load_forces(source)
    if data is None:
        return np.empty(0), np.empty(0)
    return np.asarray(data["time"]), np.asarray(data["Cd"])


def _profile(ax, name: str, time: float, consts: dict, title: str, ylim: tuple[float, float]):
    U_inf, D = consts["U_inf"], consts["D"]
    box = consts["box"]

    ax.axvspan(box["xmin"] / D, box["xmax"] / D, color=util.COLORS["background_light"])
    if name == "centerline":
        ax.axvspan(-0.5, 0.5, color=util.COLORS["background_strong"], zorder=1)

    styles = {
        "reference": dict(color=util.COLORS["reference"], ls="-.", label="Reference FVM"),
        "fvm": dict(color=util.COLORS["hybrid"], ls="-", label="FVM"),
        "vpm": dict(
            color=util.COLORS["vpm"],
            ls="-",
            marker="o",
            ms=1.5,
            markevery=PROFILE_MARK_EVERY,
            label="VPM",
        ),
    }
    for source in ("reference", "fvm", "vpm"):
        frame = util.load_line(source, name, time)
        if frame is not None:
            ax.plot(frame["x"] / D, frame["Ux"] / U_inf, zorder=2, **styles[source])

    ax.set(xlabel=r"$x/D$", ylabel="", xlim=(-3, 10), ylim=ylim, title=title)


def plot_frame(time: float, consts: dict) -> None:
    U_inf, D = consts["U_inf"], consts["D"]
    fig = plt.figure(figsize=(FIGURE_WIDTH, 2 * FIGURE_HEIGHT), dpi=FIGURE_DPI)
    grid = GridSpec(2, 2, figure=fig)
    ax_center = fig.add_subplot(grid[0, 0])
    ax_offaxis = fig.add_subplot(grid[0, 1])
    ax_drag = fig.add_subplot(grid[1, :])

    t_fvm, cd_fvm = _force_series("fvm")
    t_ref, cd_ref = _force_series("reference")
    if t_fvm.size:
        ax_drag.plot(
            t_fvm * U_inf / D,
            cd_fvm,
            color=util.COLORS["hybrid"],
            ls="-",
            marker="o",
            ms=2,
            markevery=max(1, t_fvm.size // FORCE_MARKERS),
            label="FVM",
        )
    if t_ref.size:
        ax_drag.plot(
            t_ref * U_inf / D,
            cd_ref,
            color=util.COLORS["reference"],
            ls="-.",
            label="Reference FVM",
        )
    ax_drag.set(
        xlabel=r"$t U_\infty / D$",
        ylabel=r"$C_D$",
        xlim=(0, 20),
        ylim=(0.5, 2),
        title=r"Drag Coefficient, $C_D$",
    )
    ax_drag.legend(loc="upper right")

    _profile(
        ax_center,
        "centerline",
        time,
        consts,
        f"Centerline ($t={time:.2f}$)",
        (-1.2, 1.2),
    )
    _profile(
        ax_offaxis,
        "offaxis_y075",
        time,
        consts,
        f"Off-axis $y=0.75D$ ($t={time:.2f}$)",
        (-0.5, 1.5),
    )
    ax_center.set_ylabel(r"$u_x/U_\infty$")
    ax_offaxis.legend(loc="lower right")

    fig.tight_layout()
    util.save(fig, f"velocity_profiles_t{time:.2f}", FIGURE_FORMAT, FIGURE_DPI)
    plt.close(fig)


def main() -> None:
    times = util.common_times(
        util.line_times("fvm", "centerline"),
        util.line_times("vpm", "centerline"),
        util.line_times("reference", "centerline"),
        util.line_times("fvm", "offaxis_y075"),
        util.line_times("vpm", "offaxis_y075"),
        util.line_times("reference", "offaxis_y075"),
    )
    if times.size == 0:
        raise SystemExit("No coincident profile samples found in samples/.")
    consts = util.run_constants()
    for time in times:
        plot_frame(float(time), consts)


if __name__ == "__main__":
    main()
