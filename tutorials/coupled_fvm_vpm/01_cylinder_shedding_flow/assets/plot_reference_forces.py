#!/usr/bin/env python3
"""Compare coupled and fine-reference cylinder force histories."""

import argparse

import matplotlib.pyplot as plt

from openonda.plotting import (
    CM,
    COLORS,
    LINE_WIDTH,
    REFERENCE_LINE_WIDTH,
    centered_subplots_adjust,
    fit_thesis_y_label_margins,
    set_thesis_style,
)

import postprocess as data

FORCE_COLUMNS = ("drag_coefficient", "lift_coefficient")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    arguments = parser.parse_args()

    coupled = data.history(data.CASE_DIR / "samples" / "forces_history.csv", FORCE_COLUMNS)
    reference = data.history(data.REFERENCE / "forces_history.csv", FORCE_COLUMNS)
    time, coupled_values, reference_values, errors = data.common_history(
        coupled, reference, FORCE_COLUMNS
    )
    data.write_json(
        "reference_force_errors.json",
        {
            "reference": "fine",
            "time_interval": [float(time[0]), float(time[-1])],
            "errors": errors,
        },
    )

    set_thesis_style()
    figure, axes = plt.subplots(2, 1, figsize=(12.5 * CM, 8.3 * CM), sharex=True)
    labels = (r"$C_D$", r"$C_L$")
    for index, axis in enumerate(axes):
        axis.plot(
            time,
            reference_values[:, index],
            color=COLORS["RefGray"],
            linewidth=REFERENCE_LINE_WIDTH,
            label="Reference FVM",
        )
        axis.plot(
            time,
            coupled_values[:, index],
            color=COLORS["FVMorange"],
            linewidth=LINE_WIDTH,
            label="Coupled FVM",
        )
        axis.set_ylabel(labels[index])
        axis.grid(alpha=0.22)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.025),
        frameon=False,
    )
    axes[1].set_xlabel(r"$tU_\infty/D$")
    centered_subplots_adjust(figure, outer=0.17, bottom=0.20, top=0.95, hspace=0.12)
    fit_thesis_y_label_margins(figure, axes)
    data.save_figure(figure, axes, "reference_forces", arguments.format)


if __name__ == "__main__":
    main()
