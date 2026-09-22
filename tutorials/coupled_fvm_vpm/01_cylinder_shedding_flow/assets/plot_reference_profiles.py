#!/usr/bin/env python3
"""Compare fine-reference, coupled-FVM, and VPM cylinder profiles."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    arguments = parser.parse_args()

    reference = data.reference_directory()
    reference_paths = tuple(reference / f"transverse_x{x}.csv" for x in (1, 2, 4))
    vpm_paths = tuple(data.CASE_DIR / "samples" / f"vpm_transverse_x{x}.csv" for x in (1, 2, 4))
    fvm_path = data.CASE_DIR / "samples" / "fvm_transverse_x1.csv"
    time = data.latest_common_profile_time((*reference_paths, *vpm_paths, fvm_path))

    set_thesis_style()
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12.5 * CM, 12.0 * CM),
        sharex="col",
        sharey="row",
    )
    errors: dict[str, dict[str, float]] = {}
    for column, x_position in enumerate((1, 2, 4)):
        reference = data.profile(reference_paths[column], time)
        vpm = data.profile(vpm_paths[column], time)
        candidates = [("VPM", vpm, COLORS["VPMpurple"])]
        if x_position == 1:
            candidates.insert(0, ("Coupled FVM", data.profile(fvm_path, time), COLORS["FVMorange"]))
        for row, velocity in enumerate(data.VELOCITY_COLUMNS[:2]):
            axis = axes[row, column]
            axis.plot(
                reference.position_y,
                reference[velocity],
                color=COLORS["RefGray"],
                linewidth=REFERENCE_LINE_WIDTH,
                label="Reference FVM",
            )
            for label, candidate, color in candidates:
                axis.plot(
                    candidate.position_y,
                    candidate[velocity],
                    color=color,
                    linewidth=LINE_WIDTH,
                    label=label,
                )
                lower = max(float(reference.position_y.min()), float(candidate.position_y.min()))
                upper = min(float(reference.position_y.max()), float(candidate.position_y.max()))
                y = candidate.position_y.to_numpy(dtype=float)
                keep = (y >= lower) & (y <= upper)
                y = y[keep]
                actual = candidate[velocity].to_numpy(dtype=float)[keep]
                expected = np.interp(y, reference.position_y, reference[velocity])
                difference = actual - expected
                errors[f"{label.lower().replace(' ', '_')}_x{x_position}_{velocity}"] = {
                    "rms": float(np.sqrt(trapezoid(difference**2, y) / (y[-1] - y[0]))),
                    "maximum": float(np.abs(difference).max()),
                }
            axis.grid(alpha=0.22)
            if row == 1:
                axis.set_xlabel(rf"$y/D$,\quad x/D={x_position}$")
        axes[0, 0].set_ylabel(r"$u/U_\infty$")
        axes[1, 0].set_ylabel(r"$v/U_\infty$")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.025),
        frameon=False,
    )
    data.write_json(
        "reference_profile_errors.json",
        {"reference": reference.name, "time": time, "errors": errors},
    )
    centered_subplots_adjust(
        figure,
        outer=0.16,
        bottom=0.18,
        top=0.95,
        hspace=0.18,
        wspace=0.16,
    )
    fit_thesis_y_label_margins(figure, axes.flat)
    data.save_figure(figure, axes.flat, "reference_profiles", arguments.format)


if __name__ == "__main__":
    main()
