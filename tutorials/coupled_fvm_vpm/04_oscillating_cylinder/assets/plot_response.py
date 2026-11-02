#!/usr/bin/env python3
"""Reduced amplitudes of the elastically mounted cylinder against the reduced velocity.

The cross-flow amplitude is overlaid on the medium-Reynolds, 0.5 % damping
response of Martins and Avila (2019): points B1 and B2 of their table 5,
V* = 3 Re / 2500. The streamwise amplitude and the frequency ratio have no
tabulated counterpart.
"""

import csv
from pathlib import Path

import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, MARKER_SIZE, REFERENCE_STYLE
from openonda.plotting import save_validation_figure, validation_legend, validation_subplots

from assets.response import CASE_DIR, completed_runs, paper_points, statistics, variant

FIGURES = CASE_DIR / "figures"
RUNS = {
    "Coupled FVM and VPM": (CASE_DIR / "samples", COLORS["hybrid"], "D"),
    "FVM only": (CASE_DIR / "reference_flow/samples", COLORS["fvm"], "s"),
}


def main():
    rows = []
    results = {}
    for label, (root, _, _) in RUNS.items():
        results[label] = [statistics(root / variant(*run), *run) for run in completed_runs(root)]
        for item in results[label]:
            rows.append({"run": label, **item})
            print(
                f"{label}, V* = {item['reduced_velocity']:g} (Re = {item['reynolds_number']:g}), "
                f"window {item['window_start']:.2f}-{item['window_end']:.2f} s "
                f"({item['window_periods']:.1f} natural periods): A*_y = "
                f"{item['cross_flow_amplitude']:.3f}, A*_x = {item['streamwise_amplitude']:.3f}, "
                f"f_y/f_n = {item['frequency_ratio']:.3f}, C_D = {item['mean_drag']:.3f}, "
                f"C_L amplitude = {item['lift_amplitude']:.3f}"
            )

    paper_velocity, paper_amplitude, paper_source = paper_points()
    figure, axes = validation_subplots(3, sharex=True)
    axes[0].plot(
        paper_velocity,
        paper_amplitude,
        linestyle="none",
        color=COLORS["reference"],
        marker="o",
        markersize=MARKER_SIZE,
        label="Martins and Avila (2019), table 5",
    )
    for label, (_, colour, marker) in RUNS.items():
        items = results[label]
        if not items:
            continue
        v = [item["reduced_velocity"] for item in items]
        for axis, key in zip(
            axes, ("cross_flow_amplitude", "streamwise_amplitude", "frequency_ratio")
        ):
            axis.plot(
                v,
                [item[key] for item in items],
                linestyle="none",
                color=colour,
                marker=marker,
                markersize=MARKER_SIZE + 1,
                linewidth=LINE_WIDTH,
                label=label if axis is axes[0] else None,
            )
    axes[0].set_ylabel("$A_y^*$")
    axes[1].set_ylabel("$A_x^*$")
    axes[2].set_ylabel("$f_y/f_{n,w}$")
    axes[2].set_xlabel("$V^* = U_\\infty/(f_{n,w} D)$")
    axes[2].set_xlim(6.0, 12.0)
    axes[0].set_ylim(0.0, 1.5)
    validation_legend(figure, axes[0], ncol=1).set_loc("upper right")
    save_validation_figure(figure, FIGURES / "response")

    (FIGURES / "auxiliary").mkdir(parents=True, exist_ok=True)
    if rows:
        with (FIGURES / "auxiliary/response.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
