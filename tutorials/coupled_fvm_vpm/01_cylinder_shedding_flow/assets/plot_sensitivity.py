#!/usr/bin/env python3
"""Result quality and cost of the coupling sensitivity study against the FVM reference.

Force errors, the wake velocity difference at equal shedding phase and the wall time per second
of flow time, as each coupling parameter is varied about its baseline.
"""

import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

from openonda.plotting import BACKGROUND_STRONG, CM, COLORS, LINE_WIDTH, MARKER_SIZE
from openonda.plotting import MAX_FIGURE_WIDTH_CM, REFERENCE_STYLE
from openonda.plotting import centered_subplots_adjust, save_validation_figure, set_thesis_style

from assets.plot_forces import DEVELOPED_SHEDDING, read_forces, shedding_statistics
from assets.shedding_phase import phase_averaged_difference, shedding_phase
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
WAKE_STATIONS = (2, 4)  # x/D of the transverse velocity probes
ROWS = {
    **STATISTICS,
    "wake velocity": r"$\|\Delta\mathbf{u}\|/U_\infty$ [\%]",
    "cost": "Wall time per\nflow time [s/s]",
}


def wall_time_per_flow_time(path):
    """Mean wall time of the developed exchanges per second of flow time."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    exchange_step = records[1]["time"] - records[0]["time"]
    developed = [
        record["timing_seconds"]["total"]
        for record in records
        if record["time"] >= DEVELOPED_SHEDDING
    ]
    return np.mean(developed) / exchange_step


def wake_velocity_difference(run_dir, reference_phase):
    """Rms velocity difference from the reference at both probe stations, at equal phase, in %."""
    run_phase = shedding_phase(run_dir / "samples/forces_history.csv")
    differences = [
        phase_averaged_difference(
            np.genfromtxt(
                run_dir / f"samples/vpm_transverse_x{station}.csv", delimiter=",", names=True
            ),
            run_phase,
            np.genfromtxt(
                CASE_DIR / f"reference_flow/samples/transverse_x{station}.csv",
                delimiter=",",
                names=True,
            ),
            reference_phase,
            DEVELOPED_SHEDDING,
        )[1]
        for station in WAKE_STATIONS
    ]
    return 100.0 * np.sqrt(np.mean(np.concatenate([d.ravel() for d in differences]) ** 2))


def read_runs():
    """Parameters and absolute relative force errors of every run in sensitivity/."""
    reference = read_forces(CASE_DIR / "reference_flow/samples/forces_history.csv")
    reference_phase = shedding_phase(CASE_DIR / "reference_flow/samples/forces_history.csv")
    runs = []
    for path in sorted(CASE_DIR.glob("sensitivity/*/samples/forces_history.csv")):
        forces = np.atleast_1d(read_forces(path))
        end_time = forces["time"][-1]
        if end_time < DEVELOPED_SHEDDING + STATISTICS_LENGTH:
            continue  # still developing, or stopped early

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
                "wake velocity": wake_velocity_difference(path.parents[1], reference_phase),
                "cost": wall_time_per_flow_time(
                    path.parents[1] / "solution/coupler_diagnostics.jsonl"
                ),
            }
        )
    return runs


def main():
    runs = read_runs()
    steps = [
        json.loads(line)
        for line in (CASE_DIR / "reference_flow/solution/performance.jsonl")
        .read_text()
        .splitlines()
    ]
    reference_cost = (
        np.mean([step["step_seconds"]["max"] for step in steps]) / steps[0]["time_step_size"]
    )

    print(f"Errors [%] against the reference and cost, t >= {DEVELOPED_SHEDDING:g} s")
    print(
        f"  {'run':28s} {'end [s]':>8s} {'FVM core [D]':>13s}"
        + "".join(f" {name:>16s}" for name in ROWS)
    )
    for run in runs:
        # Half-width of the region where FVM vorticity alone replaces the particles
        core = (
            run["box_margin"]
            - (run["transfer_inset"] + BLEND_WIDTH + VPM_ONLY_WIDTH) * CELL_SIZE / DIAMETER
        )
        values = "".join(f" {run[name]:16.2f}" for name in ROWS)
        print(f"  {run['name']:28s} {run['end time']:8.1f} {core:13.2f}{values}")
    print(f"  Reference FVM wall time per flow time: {reference_cost:.1f} s/s")
    print("  An FVM core below 0.5 D lies inside the cylinder: the wall vorticity is blended.")

    # One column per parameter, varied while the other two keep their baselines
    set_thesis_style()
    figure, axes = plt.subplots(
        len(ROWS),
        len(PARAMETERS),
        sharex="col",
        sharey="row",
        figsize=(MAX_FIGURE_WIDTH_CM * CM, 17.0 * CM),
    )
    centered_subplots_adjust(figure, outer=0.14, bottom=0.08, top=0.98, wspace=0.15, hspace=0.12)
    for column, (parameter, (label, baseline)) in enumerate(PARAMETERS.items()):
        others = [name for name in PARAMETERS if name != parameter]
        varied = sorted(
            (run for run in runs if all(run[name] == PARAMETERS[name][1] for name in others)),
            key=lambda run: run[parameter],
        )
        for row, (quantity, row_label) in enumerate(ROWS.items()):
            axis = axes[row, column]
            if quantity in STATISTICS:
                axis.axhline(ERROR_BAND, **REFERENCE_STYLE)  # accuracy target of the forces
            if quantity == "cost":
                axis.axhline(reference_cost, **REFERENCE_STYLE)  # the reference FVM
            axis.axvline(baseline, color=BACKGROUND_STRONG, linewidth=LINE_WIDTH, linestyle=":")
            axis.plot(
                [run[parameter] for run in varied],
                [run[quantity] for run in varied],
                color=COLORS["hybrid"],
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=MARKER_SIZE,
            )
            if column == 0:
                axis.set_ylabel(row_label)
        axes[-1, column].set_xlabel(label)
        # Every studied value and the baseline, also before any run has finished
        values = [baseline, *(run[parameter] for run in varied)]
        low, high = min(values), max(values)
        margin = 0.08 * (high - low) if high > low else 0.5 * baseline
        axes[-1, column].set_xlim(low - margin, high + margin)

    axes[-1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))  # whole FVM steps
    axes[-1, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))  # no offset

    # Each row spans zero to its largest value and always shows its reference line
    for row, quantity in enumerate(ROWS):
        floor = reference_cost if quantity == "cost" else ERROR_BAND
        largest = max([floor, *(run[quantity] for run in runs)])
        axes[row, 0].set_ylim(0.0, 1.15 * largest)
    save_validation_figure(figure, CASE_DIR / "figures/sensitivity")


if __name__ == "__main__":
    main()
