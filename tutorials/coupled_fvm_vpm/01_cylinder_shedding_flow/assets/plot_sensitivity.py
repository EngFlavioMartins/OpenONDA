#!/usr/bin/env python3
"""Force errors of the coupling sensitivity study against the FVM reference."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from openonda.plotting import BACKGROUND_STRONG, CM, COLORS, LINE_WIDTH, MARKER_SIZE
from openonda.plotting import MAX_FIGURE_WIDTH_CM, REFERENCE_STYLE
from openonda.plotting import centered_subplots_adjust, save_validation_figure, set_thesis_style

from assets.plot_forces import DEVELOPED_SHEDDING, read_forces, shedding_statistics
from setup import BLEND_WIDTH, BOX_MARGIN, CELL_SIZE, DIAMETER, FVM_SUBSTEPS, TRANSFER_INSET
from setup import VPM_ONLY_WIDTH

CASE_DIR = Path(__file__).resolve().parents[1]
ERROR_BAND = 2.0  # %, accuracy targeted for the coupled forces
STATISTICS_LENGTH = 10.0  # s, about two shedding periods of developed shedding at least

# Study parameters with their baselines, and the force statistics compared
PARAMETERS = {
    "box_margin": ("Box margin [$D$]", BOX_MARGIN),
    "fvm_substeps": ("FVM substeps", FVM_SUBSTEPS),
    "transfer_inset": ("Inset [cells]", TRANSFER_INSET),
}
STATISTICS = {
    "mean drag": r"$\bar C_D$ error [\%]",
    "lift amplitude": r"$\hat C_L$ error [\%]",
    "Strouhal number": r"$St$ error [\%]",
}


def read_runs():
    """Parameters and absolute relative force errors of every run in sensitivity/."""
    reference = read_forces(CASE_DIR / "reference_flow/samples/forces_history.csv")
    runs = []
    for path in sorted(CASE_DIR.glob("sensitivity/*/samples/forces_history.csv")):
        forces = read_forces(path)
        end_time = forces["time"][-1]
        if end_time < DEVELOPED_SHEDDING + STATISTICS_LENGTH:
            continue  # still developing

        # Both statistics over the same developed window, up to the coupled run's end
        expected = shedding_statistics(reference[reference["time"] <= end_time])
        measured = shedding_statistics(forces)
        box, substeps, inset = re.fullmatch(
            r"box(.+)_sub(.+)_inset(.+)", path.parents[1].name
        ).groups()
        runs.append(
            {
                "name": path.parents[1].name,
                "box_margin": float(box),
                "fvm_substeps": int(substeps),
                "transfer_inset": float(inset),
                "end time": end_time,
                **{name: 100.0 * abs(measured[name] / expected[name] - 1.0) for name in STATISTICS},
            }
        )
    return runs


def main():
    runs = read_runs()
    print(f"Absolute relative errors [%] against the reference, t >= {DEVELOPED_SHEDDING:g} s")
    print(
        f"  {'run':28s} {'end [s]':>8s} {'FVM core [D]':>13s}"
        + "".join(f" {name:>16s}" for name in STATISTICS)
    )
    for run in runs:
        # Half-width of the region where FVM vorticity alone replaces the particles
        core = (
            run["box_margin"]
            - (run["transfer_inset"] + BLEND_WIDTH + VPM_ONLY_WIDTH) * CELL_SIZE / DIAMETER
        )
        errors = "".join(f" {run[name]:16.2f}" for name in STATISTICS)
        print(f"  {run['name']:28s} {run['end time']:8.1f} {core:13.2f}{errors}")
    print("  An FVM core below 0.5 D lies inside the cylinder: the wall vorticity is blended.")

    # One column per parameter, varied while the other two keep their baselines
    set_thesis_style()
    figure, axes = plt.subplots(
        len(STATISTICS),
        len(PARAMETERS),
        sharex="col",
        sharey="row",
        figsize=(MAX_FIGURE_WIDTH_CM * CM, 11.0 * CM),
    )
    centered_subplots_adjust(figure, outer=0.12, bottom=0.11, top=0.97, wspace=0.15, hspace=0.12)
    for column, (parameter, (label, baseline)) in enumerate(PARAMETERS.items()):
        others = [name for name in PARAMETERS if name != parameter]
        varied = sorted(
            (run for run in runs if all(run[name] == PARAMETERS[name][1] for name in others)),
            key=lambda run: run[parameter],
        )
        for row, (statistic, error_label) in enumerate(STATISTICS.items()):
            axis = axes[row, column]
            axis.axhline(ERROR_BAND, **REFERENCE_STYLE)
            axis.axvline(baseline, color=BACKGROUND_STRONG, linewidth=LINE_WIDTH, linestyle=":")
            axis.plot(
                [run[parameter] for run in varied],
                [run[statistic] for run in varied],
                color=COLORS["hybrid"],
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=MARKER_SIZE,
            )
            if column == 0:
                axis.set_ylabel(error_label)
        axes[-1, column].set_xlabel(label)

    axes[-1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))  # whole FVM steps

    # Each row spans zero to its largest error and always shows the target error
    for row, statistic in enumerate(STATISTICS):
        largest = max(ERROR_BAND, *(run[statistic] for run in runs))
        axes[row, 0].set_ylim(0.0, 1.15 * largest)
    save_validation_figure(figure, CASE_DIR / "figures/sensitivity")


if __name__ == "__main__":
    main()
