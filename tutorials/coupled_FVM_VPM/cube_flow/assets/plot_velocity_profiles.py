#!/usr/bin/env python3
"""Publication-style velocity profiles and drag history."""

from pathlib import Path
import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _plotutil as util  # noqa: E402

FIGURE_FORMAT = "png"
FIGURE_DPI = util.FIGURE_DPI
FIGURE_HEIGHT_CM = 10.0
FIGURE_SIZE = util.figure_size(FIGURE_HEIGHT_CM)

# Manual layout controls (fractions of the fixed 12.5 cm canvas).
# Adjust these six values to tune the margins and inter-panel spacing.
LAYOUT_LEFT = 0.12
LAYOUT_RIGHT = 0.98
LAYOUT_BOTTOM = 0.10
LAYOUT_TOP = 0.95
LAYOUT_WSPACE = 0.30
LAYOUT_HSPACE = 0.48

PROFILE_MARK_EVERY = 5
FORCE_MARKERS = 20
DRAG_ZOOM_START = 0.20


def _force_series(source: str, end_time: float) -> tuple[np.ndarray, np.ndarray]:
    data = util.load_forces(source)
    if data is None:
        return np.empty(0), np.empty(0)
    time = np.asarray(data["time"], dtype=float)
    cd = np.asarray(data["Cd"], dtype=float)
    selected = np.isfinite(time) & np.isfinite(cd) & (time <= end_time + util.TIME_ATOL)
    return time[selected], cd[selected]


def _drag_y_limits(series: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float] | None:
    settled = [cd[t >= DRAG_ZOOM_START] for t, cd in series if np.any(t >= DRAG_ZOOM_START)]
    if not settled:
        return None
    values = np.concatenate(settled)
    low, high = float(np.min(values)), float(np.max(values))
    padding = max(0.05, 0.08 * max(high - low, 0.1))
    return low - padding, high + padding


def _full_range_inset(ax, series: list[tuple[np.ndarray, np.ndarray]], end_time: float) -> None:
    limits = ax.get_ylim()
    available = [cd for _, cd in series if cd.size]
    if not available:
        return
    all_values = np.concatenate(available)
    if np.min(all_values) >= limits[0] and np.max(all_values) <= limits[1]:
        return
    inset = ax.inset_axes((0.73, 0.50, 0.25, 0.42))
    styles = ((util.COLORS["hybrid"], "-"), (util.COLORS["reference"], "-."))
    for (time, cd), (colour, linestyle) in zip(series, styles, strict=True):
        inset.plot(time, cd, color=colour, linestyle=linestyle, linewidth=0.7)
    inset.set_xlim(0.0, max(end_time, 0.1))
    inset.tick_params(labelsize=5, length=2)


def _profile(ax, name: str, time: float, consts: dict, title: str, ylim: tuple[float, float]):
    U_inf, D = consts["freestream_speed"], consts["D"]
    box = consts["box"]

    ax.axvspan(box["xmin"] / D, box["xmax"] / D, color=util.COLORS["background_light"])
    if name == "centerline":
        ax.axvspan(-0.5, 0.5, color=util.COLORS["background_strong"], zorder=1)

    styles = {
        "reference": dict(color=util.COLORS["reference"], ls="-.", label="Reference"),
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
        if frame is None:
            raise RuntimeError(f"Missing exact {source} {name} sample at t={time:.12g} s")
        ax.plot(frame["x"] / D, frame["Ux"] / U_inf, zorder=2, **styles[source])

    ax.set(xlabel=r"$x/D$", ylabel="", xlim=(-3, 10), ylim=ylim, title=title)


def plot_frame(
    time: float,
    consts: dict,
    figure_format: str = FIGURE_FORMAT,
    dpi: int = FIGURE_DPI,
) -> None:
    U_inf, D = consts["freestream_speed"], consts["D"]
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=dpi)
    grid = GridSpec(2, 2, figure=fig, height_ratios=(1.0, 0.85))
    ax_centre = fig.add_subplot(grid[0, 0])
    ax_offaxis = fig.add_subplot(grid[0, 1])
    ax_drag = fig.add_subplot(grid[1, :])
    fig.subplots_adjust(
        left=LAYOUT_LEFT,
        right=LAYOUT_RIGHT,
        bottom=LAYOUT_BOTTOM,
        top=LAYOUT_TOP,
        wspace=LAYOUT_WSPACE,
        hspace=LAYOUT_HSPACE,
    )

    t_fvm, cd_fvm = _force_series("fvm", time)
    t_ref, cd_ref = _force_series("reference", time)
    t_fvm_nd = t_fvm * U_inf / D
    t_ref_nd = t_ref * U_inf / D
    if t_fvm.size:
        ax_drag.plot(
            t_fvm_nd,
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
            t_ref_nd,
            cd_ref,
            color=util.COLORS["reference"],
            ls="-.",
            label="Reference",
        )
    ax_drag.set(
        xlabel=r"$t U_\infty / D$",
        ylabel=r"$C_D$",
        xlim=(0, max(time * U_inf / D, 0.1)),
        title=rf"$C_D$ history, $t\leq {time:.2f}$",
    )
    drag_series = [(t_fvm_nd, cd_fvm), (t_ref_nd, cd_ref)]
    drag_limits = _drag_y_limits(drag_series)
    if drag_limits is not None:
        ax_drag.set_ylim(*drag_limits)
        _full_range_inset(ax_drag, drag_series, time * U_inf / D)
    ax_drag.legend(loc="upper left", handlelength=1.8, borderpad=0.3, labelspacing=0.3)

    _profile(
        ax_centre,
        "centerline",
        time,
        consts,
        "Centerline",
        (-1.2, 1.2),
    )
    _profile(
        ax_offaxis,
        "offaxis_y075",
        time,
        consts,
        r"$y/D=0.75$",
        (-0.5, 1.5),
    )
    ax_centre.set_ylabel(r"$u_x/U_\infty$")
    ax_offaxis.legend(loc="lower right", handlelength=1.8, borderpad=0.3, labelspacing=0.3)

    util.save(fig, f"velocity_profiles_t{time:.2f}", figure_format, dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=util.EXPORT_FORMATS, default=FIGURE_FORMAT)
    parser.add_argument("--dpi", type=int, default=FIGURE_DPI, help="PNG resolution in dpi.")
    args = parser.parse_args()

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
        plot_frame(float(time), consts, args.format, args.dpi)


if __name__ == "__main__":
    main()
