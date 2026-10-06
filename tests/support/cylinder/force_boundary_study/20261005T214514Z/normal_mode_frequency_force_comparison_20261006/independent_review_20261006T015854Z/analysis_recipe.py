"""Read-only independent complete-cycle analysis and scientific figure recipe."""

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

STUDY = Path("/home/flavio-martins/Projects/OpenONDA/tests/support/cylinder/force_boundary_study/20261005T214514Z")
COMPARISON = STUDY / "normal_mode_frequency_force_comparison_20261006/comparison.json"
MODELS = STUDY / "normal_frequency_inputs_20261006T014123Z/normal_frequency_model_report.json"
AUDIT = STUDY / "normal_frequency_replay_independent_review_20261006.json"
FIELDS = ("drag_coefficient", "lift_coefficient")
LABELS = (
    ("coupled_harmonic", "actual_normal_mode_reference_frequency"),
    ("reference_normal_mode_actual_frequency", "reference_normal_oscillation"),
)
NAMES = {
    "coupled_harmonic": "Actual mode · actual frequency",
    "actual_normal_mode_reference_frequency": "Actual mode · reference frequency",
    "reference_normal_mode_actual_frequency": "Reference mode · actual frequency",
    "reference_normal_oscillation": "Reference mode · reference frequency",
}
COLOURS = {
    "coupled_harmonic": "#ce7131",
    "actual_normal_mode_reference_frequency": "#a84b23",
    "reference_normal_mode_actual_frequency": "#3b7a76",
    "reference_normal_oscillation": "#4c659b",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independently_measure(values, start=102, end=112):
    """Pair successive troughs and directly measure their intervening maximum."""
    time = values["time"]
    result = {}
    for field, minimum_period, maximum_period in (
        (FIELDS[0], 1.5, 5), (FIELDS[1], 4, 8)
    ):
        coefficient = values[field]
        separation = round(minimum_period / np.median(np.diff(time)))
        troughs, _ = find_peaks(-coefficient, distance=separation, prominence=1e-4)
        cycles = []
        for left, right in zip(troughs[:-1], troughs[1:]):
            if time[left] < start - 1e-8 or time[right] > end + 1e-8:
                continue
            period = time[right] - time[left]
            if not minimum_period <= period <= maximum_period:
                continue
            peak = left + int(np.argmax(coefficient[left:right+1]))
            if peak in (left, right):
                continue
            slope = (coefficient[right] - coefficient[left]) / period
            drift = coefficient[peak] - coefficient[left] - slope * (time[peak] - time[left])
            if drift <= 0:
                continue
            cycles.append({
                "left_trough_time": float(time[left]),
                "peak_time": float(time[peak]),
                "right_trough_time": float(time[right]),
                "raw_peak_to_peak": float(np.ptp(coefficient[left:right+1])),
                "drift_corrected_peak_to_peak": float(drift),
                "trough_baseline_slope_per_second": float(slope),
            })
        if not cycles:
            raise ValueError("No complete physical force cycle")
        result[field] = {
            "complete_cycle_count": len(cycles),
            "median_raw": float(np.median([c["raw_peak_to_peak"] for c in cycles])),
            "median_drift": float(np.median([c["drift_corrected_peak_to_peak"] for c in cycles])),
            "first_two_median_raw": float(np.median([c["raw_peak_to_peak"] for c in cycles[:2]])),
            "first_two_median_drift": float(np.median([c["drift_corrected_peak_to_peak"] for c in cycles[:2]])),
            "last_to_first_raw_amplitude": cycles[-1]["raw_peak_to_peak"] / cycles[0]["raw_peak_to_peak"] if len(cycles) > 1 else None,
            "last_to_first_drift_amplitude": cycles[-1]["drift_corrected_peak_to_peak"] / cycles[0]["drift_corrected_peak_to_peak"] if len(cycles) > 1 else None,
            "cycles": cycles,
        }
    return result


original = json.loads(COMPARISON.read_text())
models = json.loads(MODELS.read_text())
audit = json.loads(AUDIT.read_text())
if audit["status"] != "all input, initialization and completed native finite-state checks passed":
    raise ValueError("The normal mode/frequency native audit did not pass")
source_hashes = {str(path): sha256(path) for path in (COMPARISON, MODELS, AUDIT)}
histories, measurements, differences = {}, {}, {}
for label in ("reference_flow", *(label for row in LABELS for label in row)):
    source = original["sources"][label]
    path = Path(source["snapshot"])
    source_hashes[str(path)] = sha256(path)
    if source_hashes[str(path)] != source["sha256"]:
        raise ValueError("Frozen force CSV differs from its canonical receipt")
    values = np.genfromtxt(path, delimiter=",", names=True)
    if not all(np.isfinite(values[field]).all() for field in ("time", *FIELDS)) or np.any(np.diff(values["time"]) <= 0):
        raise ValueError("Force data have invalid clocks or coefficients")
    for axis, field in zip(("x", "y"), FIELDS):
        if np.max(abs(2 * values[f"total_force_{axis}"] - values[field])) > 1e-11:
            raise ValueError("Native force normalization differs")
    histories[label] = values
    measured = independently_measure(values)
    measurements[label] = measured
    expected = original["complete_cycle_statistics"][label]
    differences[label] = {}
    for field in FIELDS:
        if measured[field]["complete_cycle_count"] != expected[field]["complete_cycle_count"]:
            raise ValueError("Independent trough pairing differs from canonical cycle count")
        error = max(
            abs(measured[field]["median_raw"] - expected[field]["median_peak_to_peak_raw"]),
            abs(measured[field]["median_drift"] - expected[field]["median_peak_to_peak_drift_corrected"]),
        )
        for actual, canonical in zip(measured[field]["cycles"], expected[field]["cycles"]):
            error = max(error, *[abs(actual[k] - canonical[k]) for k in ("left_trough_time", "peak_time", "right_trough_time")])
        if error > 1e-11:
            raise ValueError("Independent force extrema differ from canonical comparison")
        differences[label][field] = error

ratios = {
    label: {
        field: {
            kind: values[field][f"median_{kind}"] / measurements["reference_flow"][field][f"median_{kind}"]
            for kind in ("raw", "drift")
        }
        for field in FIELDS
    }
    for label, values in measurements.items()
}
paired_changes = {}
for kind, pairs in {
    "frequency_at_fixed_mode": {"actual_mode": LABELS[0], "reference_mode": LABELS[1]},
    "mode_at_fixed_frequency": {"actual_frequency": (LABELS[0][0], LABELS[1][0]), "reference_frequency": (LABELS[0][1], LABELS[1][1])},
}.items():
    paired_changes[kind] = {
        name: {
            field: {quantity: ratios[b][field][quantity] / ratios[a][field][quantity] for quantity in ("raw", "drift")}
            for field in FIELDS
        }
        for name, (a, b) in pairs.items()
    }

first_two_ratios = {
    label: {
        field: {
            quantity: values[field][f"first_two_median_{quantity}"] / measurements["reference_flow"][field][f"first_two_median_{quantity}"]
            for quantity in ("raw", "drift")
        }
        for field in FIELDS
    }
    for label, values in measurements.items()
}

stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
output = COMPARISON.parent / f"independent_review_{stamp}"
output.mkdir(exist_ok=False)
recipe = output / "analysis_recipe.py"
recipe.write_bytes(Path(__file__).read_bytes())
review = {
    "schema": "openonda-independent-normal-mode-frequency-force-review/1",
    "captured_at_utc": datetime.now(UTC).isoformat(),
    "window": [102, 112],
    "source_sha256": source_hashes,
    "analysis_recipe_sha256": sha256(recipe),
    "independent_complete_cycle_statistics": measurements,
    "maximum_differences_from_canonical_extrema": differences,
    "relative_to_reference_flow": ratios,
    "paired_changes": paired_changes,
    "first_two_complete_cycles_relative_to_reference_first_two": first_two_ratios,
    "input_control_matrix": LABELS,
    "input_frequencies_hz": models["original_frequency_hz"],
    "phase_policy": "Each mode retains its measured per-face complex coefficients and initial force-relative phase. Only the normal oscillator clock rate changes in a frequency substitution; actual Ut and Gt remain on their original clocks.",
    "mode_definition": "Spatial coefficients are the fundamental and second-harmonic cosine/sine arrays for Un on all356 faces, including amplitudes and side-to-side phase. They also retain the original global initial phase relative to the source lift. Mean/trend are actual in all four controls.",
    "original_global_phase_translation": models["original_global_phase_translation"],
    "frequency_substitution_models": models["normal_oscillation_models"],
    "native_admission_receipt": str(AUDIT),
    "scope": "Finite102–112s forced native FVM responses, common unchanged100s reference primary/BDF initialization, fixed mixed Un/Gt with fixedFluxPressure, .04 endpoint/.008 native schedule; no autonomous coupled evolution or recovered mature production claim.",
    "method": "Independent successive-trough pairing with1.5–5s Cd and4–8s Cl periods, direct native-row max-minus-min and peak minus linear bracketing-trough baseline. CSV normalization checked independently. No history phase or amplitude rescaling.",
    "limitations": [
        "Each control has only one complete lift cycle; drag has two cycles except reference mode/reference frequency, which has three and grows10.16%. Unequal counts and finite settling prevent interpreting the four median ratios as saturated amplitudes.",
        "The first-two-cycle sensitivity is reported separately to show the impact of the extra late cycle. Those cycles have slightly different physical intervals and are not phase-shifted or made temporally identical.",
        "Changing normal frequency deliberately changes its subsequent temporal phase relative to unchanged Ut/Gt. It is not a frequency change of all flow inputs together.",
        "A spatial-mode substitution includes the measured initial force-relative phase, harmonic amplitude, spatial phase and waveform. This comparison does not isolate those mode properties individually.",
        "All inputs are low-mode reconstructions; downstream omitted higher modes, source-state differences, startup settling and feedback remain outside this control.",
    ],
}
(output / "independent_review.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n")

figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))
figure.subplots_adjust(left=0.11, right=0.975, bottom=0.225, top=0.76, wspace=0.30, hspace=0.62)
for row in range(2):
    for column in range(2):
        axis, label = axes[row, column], LABELS[row][column]
        for index, field in enumerate(FIELDS):
            axis.barh(index - 0.13, ratios[label][field]["raw"], height=0.22, color=COLOURS[label])
            axis.barh(index + 0.13, ratios[label][field]["drift"], height=0.22, color="white", edgecolor=COLOURS[label], hatch="///")
            axis.text(1.23, index, f"{ratios[label][field]['raw']:.3f} raw\n{ratios[label][field]['drift']:.3f} drift", va="center", ha="left", fontsize=10)
        axis.set_yticks([0, 1], ["Cd", "Cl"])
        axis.invert_yaxis()
        axis.set_xlim(0, 1.5)
        axis.axvline(1, color="#777777", linestyle=(0, (3, 2)), linewidth=0.8)
        axis.set_xticks([0, 0.5, 1.0])
        axis.grid(axis="x", color="#e4e6e8", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("Complete-cycle amplitude / native reference", fontsize=9)
        trend = measurements[label][FIELDS[0]]["last_to_first_raw_amplitude"]
        count = measurements[label][FIELDS[0]]["complete_cycle_count"]
        axis.text(0.0, -0.37, f"{count} Cd cycles · 1 Cl cycle; last/first Cd amplitude {trend:.3f}", transform=axis.transAxes, fontsize=9)
        if row == 0:
            source = "coupled" if column == 0 else "reference"
            name = "Actual" if column == 0 else "Reference"
            axis.set_title(f"{name} normal frequency: {models['original_frequency_hz'][source]:.6f} Hz", loc="left", fontsize=12, pad=15)
figure.text(0.028, 0.66, "Actual Un modes", rotation=90, ha="center", va="center", fontsize=12, weight="bold")
figure.text(0.028, 0.36, "Reference Un modes", rotation=90, ha="center", va="center", fontsize=12, weight="bold")
figure.suptitle("Normal-velocity spatial modes and frequency", x=0.085, y=0.973, ha="left", fontsize=18, weight="bold")
figure.text(0.085, 0.923, "2 × 2 forced FVM control; complete force cycles wholly within 102–112 s", fontsize=11)
figure.text(0.085, 0.885, "Actual DC/trend, tangential velocity and Gt are fixed; each mode keeps its initial force-relative phase.", fontsize=10)
figure.text(0.085, 0.848, "Mode = measured spatial fundamental/second-harmonic coefficients. Filled: raw; hatched: drift corrected.", fontsize=10)
figure.text(0.085, 0.112, "Frequency substitutions change only Un oscillator rate; its subsequent phase relative to unchanged Ut/Gt evolves.", fontsize=10)
figure.text(0.085, 0.077, "Two drag cycles versus three in the last cell; growth and one lift cycle limit saturation claims.", fontsize=10)
figure.text(0.085, 0.042, "Common native mapped initialization and numerical settings. These responses do not demonstrate autonomous recovery.", fontsize=10)
for extension in ("png", "pdf", "svg"):
    figure.savefig(output / f"normal_mode_frequency_matrix.{extension}", dpi=180)
plt.close(figure)

figure, axes = plt.subplots(2, 1, figsize=(13.4, 8.4))
figure.subplots_adjust(left=0.08, right=0.975, top=0.77, bottom=0.17, hspace=0.35)
for row, field in enumerate(FIELDS):
    for index, label in enumerate(("reference_flow", *(label for pair in LABELS for label in pair))):
        values = histories[label]
        selected = (values["time"] >= 102 - 1e-8) & (values["time"] <= 112 + 1e-8)
        axes[row].plot(values["time"][selected], values[field][selected], color="#272727" if label == "reference_flow" else COLOURS[label], linewidth=1.4, linestyle="-" if index in (0, 1, 3) else (0, (5, 2)), label="Reference flow" if index == 0 else NAMES[label])
    axes[row].set_xlim(102, 112)
    axes[row].set_ylabel("Cd" if row == 0 else "Cl")
    axes[row].set_xlabel("Accepted research replay time (s)")
    axes[row].grid(color="#e4e6e8", linewidth=0.6)
    axes[row].spines[["top", "right"]].set_visible(False)
handles, legend_labels = axes[0].get_legend_handles_labels()
figure.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(0.073, 0.874), ncol=3, frameon=False, fontsize=9, handlelength=3)
figure.suptitle("Normal-mode and frequency force responses", x=0.08, y=0.97, ha="left", fontsize=18, weight="bold")
figure.text(0.08, 0.924, "Actual normal frequency 0.181507 Hz; reference 0.187330 Hz. Actual mean/trend and Ut/Gt are retained.", fontsize=10)
figure.text(0.08, 0.09, "Measured 102–112 s after 2 s initial settling; curves retain native times with no amplitude or phase rescaling.", fontsize=10)
figure.text(0.08, 0.045, "Mode changes spatial coefficients and initial force-relative phase. Finite forced controls are distinct from mature production.", fontsize=10)
for extension in ("png", "pdf", "svg"):
    figure.savefig(output / f"normal_mode_frequency_response.{extension}", dpi=180)
plt.close(figure)

table_rows = []
for row in LABELS:
    for label in row:
        values = measurements[label]
        table_rows.append([NAMES[label], *[f"{ratios[label][field][quantity]:.6f}" for field in FIELDS for quantity in ("raw", "drift")], f"{values[FIELDS[0]]['complete_cycle_count']}/1", f"{values[FIELDS[0]]['last_to_first_raw_amplitude']:.6f}"])
table = ",".join(["control", "Cd_raw_ratio", "Cd_drift_ratio", "Cl_raw_ratio", "Cl_drift_ratio", "Cd_Cl_cycles", "Cd_last_first_raw"]) + "\n"
table += "\n".join(",".join(str(x) for x in row) for row in table_rows) + "\n"
(output / "normal_mode_frequency_table.csv").write_text(table)
if any(sha256(path) != expected for path, expected in source_hashes.items()):
    raise ValueError("Frozen analysis evidence changed while plotting")
print(json.dumps({"output": str(output), "ratios": ratios, "paired_changes": paired_changes, "first_two_cycle_ratios": first_two_ratios, "maximum_extrema_error": max(value for record in differences.values() for value in record.values())}, indent=2))
