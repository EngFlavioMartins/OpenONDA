#!/usr/bin/env python3
"""Thrust and power coefficients of the meshed-blade rotor against the VLM rotor."""

import csv

import numpy as np

from openonda.plotting import COLORS, LINE_WIDTH, REFERENCE_STYLE, save_validation_figure
from openonda.plotting import validation_legend, validation_subplots

from assets.rotor_data import FIGURES, ROTATION_PERIOD, fvm_blades, rotor_totals, vlm_blades

MEAN_REVOLUTIONS = 5  # the VLM case averages its last five nominal revolutions


def window_mean(time, values, revolutions):
    """Mean over the last ``revolutions`` nominal revolutions, or over all the data if shorter."""
    start = max(time.min(), time.max() - revolutions * ROTATION_PERIOD)
    window = time >= start - 1e-9
    return float(values[window].mean()), float(start), float(time.max())


def main():
    rotors = {"FVM and VPM": (fvm_blades(), {"color": COLORS["hybrid"], "linewidth": LINE_WIDTH})}
    vlm = vlm_blades()
    if vlm:
        rotors["VLM and VPM"] = (vlm, REFERENCE_STYLE)

    rows = []
    figure, axes = validation_subplots(2, sharex=True)
    for label, (blades, style) in rotors.items():
        time, thrust, power = rotor_totals(blades)
        revolutions = time / ROTATION_PERIOD
        axes[0].plot(revolutions, thrust, label=label, **style)
        axes[1].plot(revolutions, power, **style)
        for name, values in (("CT", thrust), ("CP", power)):
            mean, start, end = window_mean(time, values, MEAN_REVOLUTIONS)
            rows.append(
                {
                    "rotor": label,
                    "coefficient": name,
                    "mean": mean,
                    "window_start": start,
                    "window_end": end,
                }
            )
            print(
                f"{label}: {name} = {mean:.4f} over {start:.3f}-{end:.3f} s "
                f"({(end - start) / ROTATION_PERIOD:.2f} nominal revolutions)"
            )
    axes[0].set_ylabel("$C_T$")
    axes[1].set_ylabel("$C_P$")
    axes[1].set_xlabel("Nominal revolutions")
    validation_legend(figure, axes[0])
    save_validation_figure(figure, FIGURES / "rotor_performance")

    (FIGURES / "auxiliary").mkdir(parents=True, exist_ok=True)
    with (FIGURES / "auxiliary/rotor_performance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
