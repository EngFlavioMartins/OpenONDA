#!/usr/bin/env python3
"""Common-reconstruction velocity profiles and unfiltered wall drag."""

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from . import postprocess as util


def _force_series(source: str, end_time: float):
    data = util.load_forces(source)
    if data is None:
        raise ValueError(f"Missing {source} forces")
    selected = data["time"] <= end_time + util.TIME_ATOL
    # Raw accepted samples: no smoothing, outlier removal, or time interpolation.
    return data["time"][selected], data["drag_coefficient"][selected]


def _profile(ax, name, time, consts, title):
    box, speed = consts["box"], consts["freestream_speed"]
    ax.axvspan(box["xmin"], box["xmax"], color=util.COLORS["background_light"])
    if name == "centreline":
        ax.axvspan(-0.5, 0.5, color=util.COLORS["background_strong"])
    for source, style in (("reference", "-."), ("fvm", "-"), ("vpm", "--")):
        frame = util.load_line(source, name, time)
        if frame is None:
            raise ValueError(f"No exact {source} {name} sample at t={time:g}")
        values = np.array(frame["velocity_x"], dtype=float, copy=True) / speed
        if name == "centreline":
            values[np.abs(frame["position_x"]) <= 0.5 + 1e-12] = np.nan
        ax.plot(
            frame["position_x"],
            values,
            color=util.colour(source),
            ls=style,
            label=util.label(source),
            lw=1 if source == "reference" else 1.1,
        )
    ax.set(
        title=title,
        ylabel=r"$u_x/U_\infty$",
        xlabel=r"$x/D$",
        xlim=(-3, 10),
        xticks=[-2, 0, 2, 4, 6, 8, 10],
    )
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.margins(y=0.08)


def plot_frame(time, consts, figure_format="png", dpi=util.FIGURE_DPI):
    util._THEME.set_thesis_style()
    fig, axes = plt.subplots(3, 1, figsize=util.figure_size(16), dpi=dpi)
    axes[1].sharex(axes[0])
    util._THEME.centered_subplots_adjust(fig, outer=0.16, bottom=0.095, top=0.95, hspace=0.5)
    _profile(axes[0], "centreline", time, consts, r"(a) Centrreline, $y/D=0$")
    _profile(axes[1], "offaxis_y075", time, consts, r"(b) Off-axis, $y/D=0.75$")
    axes[0].set_xlabel("")
    axes[0].tick_params(labelbottom=False)
    for source, style in (("fvm", "-"), ("reference", "-.")):
        t, cd = _force_series(source, time)
        axes[2].plot(
            t * consts["freestream_speed"],
            cd,
            color=util.colour(source),
            ls=style,
            label=util.label(source),
        )
    axes[2].set(
        title=r"(c) Raw wall drag",
        ylabel=r"$C_D$",
        xlabel=r"$tU_\infty/D$",
        xlim=(0, time * consts["freestream_speed"]),
    )
    axes[2].yaxis.set_major_locator(MaxNLocator(4))
    axes[2].xaxis.set_major_locator(MaxNLocator(6))
    axes[2].margins(y=0.08)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(
        handles,
        labels,
        loc="upper right",
        ncol=1,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.5,
        handletextpad=0.4,
    )
    util._THEME.fit_thesis_y_label_margins(fig, axes)
    util.save(fig, f"velocity_profiles_t{time:.2f}", figure_format, dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=util.EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=util.FIGURE_DPI)
    args = parser.parse_args()
    util.validate_plot_inputs()
    times = util.common_times(
        *(
            util.line_times(source, name)
            for source in ("fvm", "vpm", "reference")
            for name in ("centreline", "offaxis_y075")
        )
    )
    if not len(times):
        raise SystemExit("No exactly coincident profile states")
    consts = util.run_constants()
    for time in times:
        plot_frame(float(time), consts, args.format, args.dpi)
    util.remove_obsolete_frames("velocity_profiles", times, args.format)


if __name__ == "__main__":
    main()
