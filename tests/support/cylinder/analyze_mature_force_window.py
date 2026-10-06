"""Verify and plot one frozen common physical-time cylinder force window.

Only immutable CSV snapshots identified by the source comparison are consumed.
The full raw histories remain unchanged, and complete-cycle variations and
pressure/viscous contributions are reported at the same force extrema.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import io
import json
from pathlib import Path

import numpy as np

from .analyze_force_cycles import FIELDS
from .compare_boundary_input_replays import (
    QUANTITIES,
    independent_extrema_and_trend,
    measure,
)
from .replay_boundary_inputs import digest, write_json


def cycle_variation(record):
    values = np.asarray([cycle["peak_to_peak_raw"] for cycle in record["cycles"]])
    drift = np.asarray([cycle["peak_to_peak_drift_corrected"] for cycle in record["cycles"]])
    if len(values) < 2:
        raise ValueError("Cycle variation requires at least two complete cycles")
    return {
        "raw_minimum": float(values.min()),
        "raw_maximum": float(values.max()),
        "raw_cycle_standard_deviation": float(np.std(values, ddof=1)),
        "raw_relative_standard_deviation": float(np.std(values, ddof=1) / values.mean()),
        "drift_minimum": float(drift.min()),
        "drift_maximum": float(drift.max()),
        "drift_relative_standard_deviation": float(np.std(drift, ddof=1) / drift.mean()),
        "mean_component_fraction_at_raw_extrema": {
            component: float(
                np.mean(
                    [
                        cycle[f"{component}_contribution_at_raw_extrema"]
                        for cycle in record["cycles"]
                    ]
                )
                / values.mean()
            )
            for component in ("pressure", "viscous")
        },
        "mean_component_fraction_after_drift_correction": {
            component: float(
                np.mean(
                    [
                        cycle[f"{component}_contribution_drift_corrected"]
                        for cycle in record["cycles"]
                    ]
                )
                / drift.mean()
            )
            for component in ("pressure", "viscous")
        },
    }


def render(histories, statistics, ratios, output, window):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.8))
    figure.subplots_adjust(left=0.085, right=0.975, top=0.82, bottom=0.25, wspace=0.29, hspace=0.40)
    names = {"reference": "Reference flow", "coupled": "Coupled FVM–VPM"}
    colours = {"reference": "#272727", "coupled": "#326fa8"}
    for row, field in enumerate(FIELDS):
        symbol = "Cd" if row == 0 else "Cl"
        for label in ("reference", "coupled"):
            values = histories[label]
            selected = (values["time"] >= window[0] - 1e-8) & (values["time"] <= window[1] + 1e-8)
            axes[row, 0].plot(
                values["time"][selected],
                values[field][selected],
                color=colours[label],
                linewidth=1.35,
                label=names[label],
            )
            records = statistics[label][field]["cycles"]
            axes[row, 1].plot(
                [cycle["peak_time"] for cycle in records],
                [cycle["peak_to_peak_raw"] for cycle in records],
                "o-",
                markersize=4,
                linewidth=1.15,
                color=colours[label],
                label=names[label],
            )
            axes[row, 1].plot(
                [cycle["peak_time"] for cycle in records],
                [cycle["peak_to_peak_drift_corrected"] for cycle in records],
                "x--",
                markersize=4,
                linewidth=0.9,
                color=colours[label],
            )
        axes[row, 0].set_ylabel(symbol)
        axes[row, 1].set_ylabel(symbol + " complete-cycle peak-to-peak")
        axes[row, 1].set_ylim(bottom=0)
        for axis in axes[row]:
            axis.set_xlim(*window)
            axis.set_xlabel("Accepted physical flow time (s)")
            axis.grid(color="#e5e6e8", linewidth=0.65)
            axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.077, 0.855),
        ncol=2,
        fontsize=9,
    )
    axes[0, 1].set_title("Raw circles · drift corrected crosses", fontsize=10)
    figure.suptitle(
        "Planar cylinder force deficit remains in developed shedding",
        x=0.085,
        y=0.97,
        ha="left",
        fontsize=17,
        weight="bold",
    )
    cd, cl = (ratios[field]["median_peak_to_peak_raw"] for field in FIELDS)
    figure.text(
        0.085,
        0.924,
        f"Same physical-time window {window[0]:g}–{window[1]:g} s: coupled Cd amplitude {100 * cd:.1f}% and Cl {100 * cl:.1f}% of reference",
        fontsize=11,
    )
    figure.text(
        0.085,
        0.89,
        f"Complete cycles: Cd {statistics['coupled']['drag_coefficient']['complete_cycle_count']} coupled / {statistics['reference']['drag_coefficient']['complete_cycle_count']} reference; Cl {statistics['coupled']['lift_coefficient']['complete_cycle_count']} / {statistics['reference']['lift_coefficient']['complete_cycle_count']}. Raw force data are preserved.",
        fontsize=10,
    )
    for index, field in enumerate(FIELDS):
        first, second = (statistics[label][field] for label in ("coupled", "reference"))
        symbol = "Cd" if index == 0 else "Cl"
        figure.text(
            0.085 + 0.465 * index,
            0.116,
            f"{symbol} median raw: {first['median_peak_to_peak_raw']:.6f} / {second['median_peak_to_peak_raw']:.6f}\nDrift corrected ratio: {ratios[field]['median_peak_to_peak_drift_corrected']:.4f}",
            fontsize=10,
            linespacing=1.6,
        )
    figure.text(
        0.085,
        0.06,
        "Complete cycles require a peak between two troughs wholly within the common window. No phase or amplitude rescaling.",
        fontsize=9.5,
    )
    figure.text(
        0.085,
        0.034,
        "This production comparison establishes the persistent deficit; boundary-input causality is assessed by separate forced controls.",
        fontsize=9.5,
    )
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output / f"mature_force_comparison.{extension}", dpi=175)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-comparison", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    source_path = args.source_comparison.resolve()
    source_hash = digest(source_path)
    source = json.loads(source_path.read_text())
    output = args.output_directory or source_path.parent / (
        "independent_review_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    window = tuple(source["window"])
    histories, statistics, checks = {}, {}, {}
    for label in ("reference", "coupled"):
        path = Path(source["sources"][label]["snapshot"])
        if digest(path) != source["sources"][label]["immutable_snapshot_sha256"]:
            raise ValueError("Frozen force CSV changed after the original measurement")
        histories[label] = np.atleast_1d(
            np.genfromtxt(io.BytesIO(path.read_bytes()), names=True, delimiter=",")
        )
        statistics[label] = measure(histories[label], window)
        for field in FIELDS:
            for quantity in QUANTITIES:
                if (
                    abs(
                        statistics[label][field][quantity]
                        - source["statistics"][label][field][quantity]
                    )
                    > 1e-11
                ):
                    raise ValueError("Recomputed amplitudes differ from the frozen comparison")
        checks[label] = independent_extrema_and_trend(histories[label], statistics[label])
    ratios = {
        field: {
            quantity: statistics["coupled"][field][quantity]
            / statistics["reference"][field][quantity]
            for quantity in QUANTITIES
        }
        for field in FIELDS
    }
    report = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "source_comparison": str(source_path),
        "source_comparison_sha256": source_hash,
        "window": list(window),
        "frozen_force_sources": source["sources"],
        "complete_cycle_statistics": statistics,
        "coupled_to_reference_amplitude_ratios": ratios,
        "independent_direct_extrema_and_cycle_trend": checks,
        "cycle_variation_and_force_components": {
            label: {field: cycle_variation(record) for field, record in values.items()}
            for label, values in statistics.items()
        },
        "force_normalization": "Direct CSV verification: Cd/Cl=2F, density=reference_speed=diameter=span=1; pressure plus viscous equals total force.",
        "limitations": "Identical physical-time windows, no phase or amplitude shifts. This comparison establishes a persistent force deficit, not its causal origin. No mean-lift ratio is interpreted.",
        "analysis_source_sha256": {
            str(Path(__file__).resolve()): digest(Path(__file__)),
            str(Path(__file__).with_name("compare_boundary_input_replays.py")): digest(
                Path(__file__).with_name("compare_boundary_input_replays.py")
            ),
        },
    }
    if digest(source_path) != source_hash:
        raise ValueError("Frozen force comparison changed during review")
    write_json(output / "independent_force_review.json", report)
    render(histories, statistics, ratios, output, window)
    print(
        json.dumps(
            {
                "directory": str(output),
                "ratios": ratios,
                "cycle_variation": report["cycle_variation_and_force_components"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
