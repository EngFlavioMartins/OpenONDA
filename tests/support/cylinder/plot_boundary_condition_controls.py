"""Freeze completed boundary controls and plot their measured force amplitudes.

The externally forced small-domain FVM controls and the actual coupled case are
shown in explicitly separate accepted-time windows. Complete-cycle statistics
are recomputed from copied raw histories and checked against the native study's
canonical comparison. Production samples and tutorial figures are untouched.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analyze_force_cycles import FIELDS, complete_cycles, freeze_csv, window_statistics

STUDY = Path(__file__).resolve().parent / "force_boundary_study/20261005T214514Z"
PRIVATE = Path("/tmp/openonda-cylinder-boundary-study-20261005")
LABELS = {
    "reference": "Reference flow",
    "mixed": "Mixed boundary · NumPy · Δt = 0.008 s",
    "full_velocity": "Full velocity · NumPy · Δt = 0.008 s",
    "mixed_subcycled_numba": "Mixed boundary · Numba · ΔTex = 0.04 s",
}
COLOURS = {
    "reference": "#252525",
    "mixed": "#27649b",
    "full_velocity": "#d18324",
    "mixed_subcycled_numba": "#7c538b",
}
LINESTYLES = {
    "reference": "-",
    "mixed": (0, (5, 2)),
    "full_velocity": (0, (2, 2)),
    "mixed_subcycled_numba": (0, (7, 2, 1, 2)),
}
SHORT_LABELS = {
    "reference": "Reference flow",
    "mixed": "Mixed · NumPy",
    "full_velocity": "Full velocity · NumPy",
    "mixed_subcycled_numba": "Mixed · Numba + subcycling",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_file(source, destination):
    expected = digest(source)
    shutil.copyfile(source, destination)
    if digest(source) != expected or digest(destination) != expected:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Changing input during figure preparation: {source}")
    return {"source": str(source), "snapshot": str(destination), "sha256": expected}


def cycle_statistics(values, window):
    if (
        np.max(abs(2 * values["total_force_x"] - values["drag_coefficient"])) > 1e-11
        or np.max(abs(2 * values["total_force_y"] - values["lift_coefficient"])) > 1e-11
    ):
        raise ValueError("Coefficient normalisation differs from rho=Uref=area=1")
    cycles = {field: complete_cycles(values, field, 2.0) for field in FIELDS}
    return window_statistics(values, cycles, *window, 2.0)


def amplitude_rows(statistics, labels):
    return [
        [
            labels[label],
            f"{record['drag_coefficient']['median_peak_to_peak_raw']:.6f}",
            f"{record['drag_coefficient']['median_peak_to_peak_drift_corrected']:.6f}",
            f"{record['lift_coefficient']['median_peak_to_peak_raw']:.6f}",
            f"{record['lift_coefficient']['median_peak_to_peak_drift_corrected']:.6f}",
        ]
        for label, record in statistics.items()
    ]


def draw_table(axis, rows, title, *, ratio_row=False):
    axis.axis("off")
    axis.set_title(title, loc="left", fontsize=11, weight="bold", pad=9)
    table = axis.table(
        cellText=rows,
        colLabels=["Case", "Cd raw", "Cd drift-corrected", "Cl raw", "Cl drift-corrected"],
        colWidths=[0.32, 0.14, 0.20, 0.14, 0.20],
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.86],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d5d7da")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#f0f1f3")
            cell.set_text_props(weight="bold")
        elif ratio_row and row == len(rows):
            cell.set_facecolor("#f5f5f5")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")
        if column == 0:
            cell.set_text_props(ha="left")
    return table


def draw_ratio_bars(axis, statistics, field, symbol):
    labels = [label for label in statistics if label != "reference"]
    reference = statistics["reference"][field]
    positions = np.arange(len(labels))
    for index, quantity in enumerate(
        ("median_peak_to_peak_raw", "median_peak_to_peak_drift_corrected")
    ):
        values = [statistics[label][field][quantity] / reference[quantity] for label in labels]
        offset = -0.16 if index == 0 else 0.16
        bars = axis.barh(
            positions + offset,
            values,
            height=0.27,
            color=[COLOURS[label] if index == 0 else "white" for label in labels],
            edgecolor=[COLOURS[label] for label in labels],
            linewidth=1.1,
            hatch=None if index == 0 else "///",
        )
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                value + 0.025,
                bar.get_y() + bar.get_height() / 2,
                f"{100 * value:.1f}%",
                va="center",
                fontsize=8.5,
            )
    axis.set_yticks(
        positions, ["Mixed\nNumPy", "Full velocity\nNumPy", "Mixed + subcycling\nNumba"]
    )
    axis.invert_yaxis()
    axis.set_xlim(0, 1.18)
    axis.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1.0"])
    axis.axvline(1, color="#666666", linestyle=(0, (3, 2)), linewidth=0.8, zorder=0)
    axis.set_title(f"{symbol} amplitude / reference", fontsize=11, loc="left", pad=9)
    axis.set_xlabel("Complete-cycle median ratio", fontsize=9)
    axis.grid(axis="x", color="#e8e9eb", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="y", labelsize=8.5, length=0)


def render(histories, controls, actual, output, window):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#252525",
            "text.color": "#252525",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure = plt.figure(figsize=(13.3, 11.0))
    grid = figure.add_gridspec(
        4,
        3,
        height_ratios=[1.2, 1.2, 0.90, 0.86],
        width_ratios=[1, 1, 1.05],
        left=0.07,
        right=0.97,
        top=0.85,
        bottom=0.13,
        hspace=0.60,
        wspace=0.70,
    )
    figure.suptitle(
        "Cylinder boundary-condition controls",
        x=0.07,
        y=0.975,
        ha="left",
        fontsize=18,
        weight="bold",
    )
    figure.text(
        0.07,
        0.941,
        "Native reference advanced from 100 to 112 s; three isolated FVM controls replay its outer fields",
        fontsize=11,
    )
    figure.text(
        0.07,
        0.919,
        "Shown and measured: 102–112 s, after 2 s settling. All retain fixedFluxPressure.",
        fontsize=10,
        color="#505050",
    )
    axes = []
    for row, field in enumerate(FIELDS):
        axis = figure.add_subplot(grid[row, :2])
        axes.append(axis)
        symbol = "Cd" if row == 0 else "Cl"
        for label, values in histories.items():
            selected = (values["time"] >= window[0] - 1e-8) & (values["time"] <= window[1] + 1e-8)
            axis.plot(
                values["time"][selected],
                values[field][selected],
                label=LABELS[label],
                color=COLOURS[label],
                linestyle=LINESTYLES[label],
                linewidth=1.35 if label == "reference" else 1.15,
            )
        axis.set_xlim(*window)
        axis.set_xticks([102, 104, 106, 108, 110, 112])
        axis.set_ylabel(symbol, fontsize=11)
        axis.set_xlabel("Accepted flow time (s)")
        axis.grid(color="#e5e7eb", linewidth=0.6)
        axis.set_title(
            f"({'a' if row == 0 else 'b'}) {symbol}: reference-driven controls",
            loc="left",
            fontsize=11,
            pad=9,
        )
        ratio_axis = figure.add_subplot(grid[row, 2])
        draw_ratio_bars(ratio_axis, controls, field, symbol)
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower left",
        bbox_to_anchor=(0.07, 0.875),
        ncol=2,
        frameon=False,
        fontsize=8.6,
        columnspacing=1.4,
        handlelength=3.0,
        borderaxespad=0,
    )
    control_axis = figure.add_subplot(grid[2, :])
    draw_table(
        control_axis,
        amplitude_rows(controls, SHORT_LABELS),
        "(c) Forced controls · 102–112 s · complete-cycle medians · 3 Cd cycles / 1 Cl cycle per run",
    )
    actual_axis = figure.add_subplot(grid[3, :])
    actual_rows = amplitude_rows(
        actual, {"reference": "Reference flow", "coupled": "Actual coupled FVM–VPM"}
    )
    ratios = []
    for field in FIELDS:
        for quantity in ("median_peak_to_peak_raw", "median_peak_to_peak_drift_corrected"):
            ratios.append(
                f"{100 * actual['coupled'][field][quantity] / actual['reference'][field][quantity]:.1f}%"
            )
    actual_rows.append(["Coupled / reference", *ratios])
    draw_table(
        actual_axis,
        actual_rows,
        "(d) Separate production comparison · 30–44 s · 4 Cd cycles / 2 Cl cycles per run",
        ratio_row=True,
    )
    figure.text(
        0.07,
        0.084,
        "Raw = actual min–max within a complete trough-bracketed cycle; drift-corrected = peak minus interpolated trough baseline.",
        fontsize=9,
        color="#505050",
    )
    figure.text(
        0.07,
        0.064,
        "Bars use control/reference median ratios. Hatched bars are drift-corrected; solid bars are raw. No confidence intervals inferred.",
        fontsize=9,
        color="#505050",
    )
    figure.text(
        0.07,
        0.039,
        "These externally forced controls demonstrate boundary-model capability; they do not demonstrate self-sustained coupled amplitude recovery.",
        fontsize=9.5,
        weight="bold",
    )
    for extension in ("png", "pdf", "svg"):
        figure.savefig(
            output / f"boundary_condition_controls.{extension}", dpi=180, facecolor="white"
        )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, default=PRIVATE)
    parser.add_argument("--study", type=Path, default=STUDY)
    args = parser.parse_args()
    output = (
        args.study.resolve()
        / f"boundary_control_figure_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output.mkdir()
    snapshots = output / "snapshots"
    snapshots.mkdir()
    sources = {
        "canonical_comparison": freeze_file(
            args.private / "comparison.json", output / "comparison.json"
        )
    }
    comparison = json.loads((output / "comparison.json").read_text())
    if set(comparison["series"]) != set(LABELS) or any(
        comparison["runs"][label]["status"] != "complete" for label in LABELS
    ):
        raise ValueError("Require completed reference and all three controls")
    window = comparison["statistics_window"]
    histories = {}
    controls = {}
    for label in LABELS:
        histories[label], sources[label] = freeze_csv(
            Path(comparison["series"][label]["source"]), snapshots / f"{label}_forces_history.csv"
        )
        if sources[label]["sha256"] != comparison["series"][label]["force_history_sha256"]:
            raise ValueError(f"Raw force history differs from canonical comparison: {label}")
        controls[label] = cycle_statistics(histories[label], window)
        for field, expected_count in zip(FIELDS, (3, 1), strict=True):
            record = controls[label][field]
            if record["complete_cycle_count"] != expected_count:
                raise ValueError(f"Unexpected complete-cycle count in {label}/{field}")
            expected = comparison["complete_cycle_statistics"][label]["statistics"][field]
            for quantity in ("median_peak_to_peak_raw", "median_peak_to_peak_drift_corrected"):
                if abs(record[quantity] - expected[quantity]) > 1e-13:
                    raise ValueError(f"Recomputed amplitude differs from canonical {label}/{field}")
    actual = {}
    previous_statistics = json.loads((args.study / "force_cycle_statistics.json").read_text())
    sources["actual_statistics"] = freeze_file(
        args.study / "force_cycle_statistics.json", snapshots / "actual_force_cycle_statistics.json"
    )
    for label in ("reference", "coupled"):
        values, sources[f"actual_{label}"] = freeze_csv(
            args.study / f"snapshots/{label}_forces_history.csv",
            snapshots / f"actual_{label}_forces_history.csv",
        )
        actual[label] = cycle_statistics(values, (30, 44))
        for field, expected_count in zip(FIELDS, (4, 2), strict=True):
            if actual[label][field]["complete_cycle_count"] != expected_count:
                raise ValueError("Unexpected actual-run complete-cycle count")
            for quantity in ("median_peak_to_peak_raw", "median_peak_to_peak_drift_corrected"):
                if (
                    abs(
                        actual[label][field][quantity]
                        - previous_statistics["windows"][label]["30-44"][field][quantity]
                    )
                    > 1e-13
                ):
                    raise ValueError("Actual-run amplitude changed from its frozen study")
    render(histories, controls, actual, output, window)
    data = {
        "schema": "openonda-cylinder-boundary-control-figure/1",
        "created_at": datetime.now(UTC).isoformat(),
        "output": str(output),
        "source_files": sources,
        "analysis_source": str(Path(__file__).resolve()),
        "analysis_source_sha256": digest(Path(__file__)),
        "controls_window": window,
        "controls_complete_cycle_statistics": controls,
        "actual_window": [30, 44],
        "actual_complete_cycle_statistics": actual,
        "figure_files": {
            name: digest(output / name)
            for name in (
                "boundary_condition_controls.png",
                "boundary_condition_controls.pdf",
                "boundary_condition_controls.svg",
            )
        },
        "verification": {
            "all_four_control_csv_hashes_match_canonical": True,
            "all_recomputed_medians_match_canonical": True,
            "control_cycles_per_run": {"Cd": 3, "Cl": 1},
            "actual_cycles_per_run": {"Cd": 4, "Cl": 2},
            "production_sources_or_figures_modified": False,
        },
        "limitations": [
            "All three small-domain controls are externally forced with reference boundary fields and start from a restricted mature reference state; successful amplitudes are not autonomous coupled recovery.",
            "Actual production 30–44 s is visually separated from the control 102–112 s window; these are different trajectories and accepted-time windows.",
            "Only one complete lift cycle is available in the post-settling control window; no confidence interval or saturation claim is inferred.",
            "Native reference reconstruction and state-history remapping limitations are retained in the unmodified copied canonical comparison.",
        ],
    }
    (output / "figure_data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "control_complete_cycle_ratios": {
                    label: comparison[f"{label}_complete_cycle_relative_to_reference"]
                    for label in LABELS
                    if label != "reference"
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
