#!/usr/bin/env python3
"""Compare cylinder forces over the actually available common interval."""

import argparse

import matplotlib.pyplot as plt

from openonda.plotting import (
    CM,
    COLORS,
    LINE_WIDTH,
    REFERENCE_LINE_WIDTH,
    centered_subplots_adjust,
    set_thesis_style,
)

from . import postprocess as data

FORCE_COLUMNS = ("drag_coefficient", "lift_coefficient")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    arguments = parser.parse_args()

    coupled = data.history(data.CASE_DIR / "samples" / "forces_history.csv", FORCE_COLUMNS)
    reference = data.history(data.reference_directory() / "forces_history.csv", FORCE_COLUMNS)
    time, coupled_values, reference_values, errors = data.common_history(
        coupled, reference, FORCE_COLUMNS
    )
    data.write_json(
        "reference_force_errors.json",
        {
            "reference": str(data.reference_directory().relative_to(data.CASE_DIR)),
            "time_interval": [float(time[0]), float(time[-1])],
            "errors": errors,
            **data.history_coverage(coupled, reference),
        },
    )
    print(
        f"Force comparison: common saved coverage {time[0]:g}–{time[-1]:g} s; "
        "interpolated only within that interval, without time shifts."
    )

    set_thesis_style()
    figure, axes = plt.subplots(2, 1, figsize=(12.5 * CM, 8.3 * CM), sharex=True)
    labels = (r"$C_D$", r"$C_L$")
    for index, axis in enumerate(axes):
        axis.plot(
            time,
            reference_values[:, index],
            color=COLORS["reference"],
            linewidth=REFERENCE_LINE_WIDTH,
            label="Reference FVM",
            linestyle="--",
        )
        axis.plot(
            time,
            coupled_values[:, index],
            color=COLORS["hybrid"],
            linewidth=LINE_WIDTH,
            label="Coupled FVM",
        )
        axis.set_ylabel(labels[index])
        axis.grid(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.025),
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.8",
    )
    axes[1].set_xlabel(r"$tU_\infty/D$")
    centered_subplots_adjust(figure, outer=0.135, bottom=0.28, top=0.93, hspace=0.12)
    data.save_figure(figure, axes, "reference_forces", arguments.format)


if __name__ == "__main__":
    main()
