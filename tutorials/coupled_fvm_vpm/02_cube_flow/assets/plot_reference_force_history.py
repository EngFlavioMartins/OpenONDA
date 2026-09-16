#!/usr/bin/env python3
"""Plot raw reference and coupled force histories with their accepted timesteps."""

from __future__ import annotations

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from . import postprocess as util


def plot_force_history(report, fmt, dpi):
    util._THEME.set_thesis_style()
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=util.figure_size(11.8), dpi=dpi)
    util._THEME.centered_subplots_adjust(fig, outer=0.18, bottom=0.14, top=0.84, hspace=0.58)
    for source, style in (("reference", "-."), ("fvm", "-")):
        data = util.load_forces(source)
        keep = data["time"] <= report["comparison_end_time"] + util.TIME_ATOL
        axes[0].plot(
            data["time"][keep],
            data["drag_coefficient"][keep],
            color=util.colour(source),
            ls=style,
            label=util.label(source),
        )
        axes[1].semilogy(
            data["time"][keep],
            data["accepted_time_step_size"][keep],
            color=util.colour(source),
            ls=style,
        )
    axes[0].set(ylabel=r"$C_D$", title="(a) Raw wall drag")
    axes[0].yaxis.set_major_locator(MaxNLocator(4))
    axes[1].set(
        ylabel=r"$\Delta t$ [s]",
        xlabel="Flow time [s]",
        title="(b) Accepted timestep at force samples",
        xlim=(0, report["comparison_end_time"]),
    )
    axes[1].xaxis.set_major_locator(MaxNLocator(6))
    axes[1].xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    for ax in axes:
        ax.axvline(report["largest_reference_Cd_after_t1"][0]["time"], color=".4", lw=0.5, ls=":")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False
    )
    util.save(fig, "reference_force_history", fmt, dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=util.EXPORT_FORMATS, default="png")
    parser.add_argument("--dpi", type=int, default=util.FIGURE_DPI)
    args = parser.parse_args()
    report = util.build_comparison_report()
    util.write_comparison_report(report)
    plot_force_history(report, args.format, args.dpi)


if __name__ == "__main__":
    main()
