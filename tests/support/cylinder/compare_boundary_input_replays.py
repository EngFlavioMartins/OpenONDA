"""Compare complete force cycles from explicitly identified boundary inputs.

Histories are frozen without editing sources. The 102–112 s default window
excludes the first 2 s from the unchanged 100 s mapped initialization. Filtering
admissibility is measured against the completed exact-fed Numba/.04 control;
no amplitude, frequency or phase is rescaled for a comparison.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import re

import numpy as np

from .analyze_force_cycles import FIELDS, complete_cycles, freeze_csv, window_statistics
from .replay_boundary_inputs import check_equation_sources, digest, write_json

STUDY = Path(__file__).resolve().parent / "force_boundary_study/20261005T214514Z"
EXACT = Path("/tmp/openonda-cylinder-boundary-study-20261005")
QUANTITIES = ("median_peak_to_peak_raw", "median_peak_to_peak_drift_corrected")
SOURCE_EVIDENCE = STUDY / "verified_replay_sources"


def read_replay_argument(value):
    label, separator, directory = value.partition("=")
    if not separator or not re.fullmatch("[a-zA-Z][a-zA-Z0-9_]*", label) or not directory:
        raise argparse.ArgumentTypeError("Replay must be LABEL=/absolute/private/directory")
    if label in ("reference_flow", "exact_mixed"):
        raise argparse.ArgumentTypeError("Replay label conflicts with a fixed comparison source")
    return label, Path(directory).resolve()


def replay_result_directory(directory):
    """Admit one completed velocity-form result or an explicitly named result."""
    names = ("mixed_subcycled_numba", "full_velocity_subcycled_numba")
    if directory.name in names:
        if not (directory / "report.json").is_file():
            raise ValueError(f"Explicit replay result is incomplete: {directory}")
        return directory, directory.parent / "boundary_trace.json"
    candidates = [
        directory / name for name in names if (directory / name / "report.json").is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Specify one completed replay result directory when a root has zero or multiple completed forms"
        )
    return candidates[0], directory / "boundary_trace.json"


def measure(values, window):
    for axis, coefficient in (("x", "drag_coefficient"), ("y", "lift_coefficient")):
        if np.max(abs(2 * values[f"total_force_{axis}"] - values[coefficient])) > 1e-11:
            raise ValueError("Force normalization differs from the commonrho=Uref=area=1")
        if (
            np.max(
                abs(
                    values[f"pressure_force_{axis}"]
                    + values[f"viscous_force_{axis}"]
                    - values[f"total_force_{axis}"]
                )
            )
            > 1e-11
        ):
            raise ValueError("Pressure/viscous force decomposition is inconsistent")
    cycles = {field: complete_cycles(values, field, 2.0) for field in FIELDS}
    return window_statistics(values, cycles, *window, 2.0)


def relative_statistics(statistics, denominator):
    return {
        label: {
            field: {
                quantity: None
                if record[field][quantity] is None
                or statistics[denominator][field][quantity] is None
                else record[field][quantity] / statistics[denominator][field][quantity]
                for quantity in QUANTITIES
            }
            for field in FIELDS
        }
        for label, record in statistics.items()
    }


def independent_extrema_and_trend(values, statistics):
    """Recompute extrema and linear trough baselines directly from CSV rows."""
    result = {}
    time = values["time"]
    maximum_difference = 0.0
    for field in FIELDS:
        coefficient = values[field]
        cycles = []
        for cycle in statistics[field]["cycles"]:
            left, peak, right = [
                int(np.argmin(abs(time - cycle[name])))
                for name in ("left_trough_time", "peak_time", "right_trough_time")
            ]
            if (
                max(
                    abs(time[index] - cycle[name])
                    for index, name in zip(
                        (left, peak, right),
                        ("left_trough_time", "peak_time", "right_trough_time"),
                        strict=True,
                    )
                )
                > 1e-10
            ):
                raise ValueError("Recorded cycle boundaries do not match actual native CSV rows")
            raw = float(max(coefficient[left : right + 1]) - min(coefficient[left : right + 1]))
            slope = float((coefficient[right] - coefficient[left]) / (time[right] - time[left]))
            drift = float(coefficient[peak] - coefficient[left] - slope * (time[peak] - time[left]))
            maximum_difference = max(
                maximum_difference,
                abs(raw - cycle["peak_to_peak_raw"]),
                abs(drift - cycle["peak_to_peak_drift_corrected"]),
            )
            cycles.append(
                {
                    "interval": [float(time[left]), float(time[right])],
                    "peak_time": float(time[peak]),
                    "raw_peak_to_peak": raw,
                    "drift_corrected_peak_to_peak": drift,
                    "linear_trough_baseline_slope_per_second": slope,
                }
            )
        result[field] = {
            "mean_coefficient": statistics[field]["mean"],
            "complete_cycle_count": len(cycles),
            "cycles": cycles,
            "last_to_first_raw_amplitude": cycles[-1]["raw_peak_to_peak"]
            / cycles[0]["raw_peak_to_peak"]
            if len(cycles) > 1
            else None,
            "last_to_first_drift_corrected_amplitude": cycles[-1]["drift_corrected_peak_to_peak"]
            / cycles[0]["drift_corrected_peak_to_peak"]
            if len(cycles) > 1
            else None,
        }
    if maximum_difference > 1e-11:
        raise ValueError("Direct CSV extrema differ from complete-cycle calculation")
    return {"maximum_independent_extrema_difference": maximum_difference, "statistics": result}


def filtering_admissibility(statistics, ratios, label, tolerance):
    if label not in statistics:
        return {"status": "filtered reference result unavailable", "passed": None}
    differences = {
        field: {
            quantity: None
            if ratios[label][field][quantity] is None
            else abs(ratios[label][field][quantity] - 1)
            for quantity in QUANTITIES
        }
        for field in FIELDS
    }
    insufficient = any(
        value is None for record in differences.values() for value in record.values()
    )
    if insufficient:
        return {
            "status": "insufficient complete cycles",
            "passed": False,
            "relative_amplitude_differences": differences,
        }
    return {
        "status": "threshold not specified; report the measured filtering bias"
        if tolerance is None
        else "pass"
        if all(value <= tolerance for record in differences.values() for value in record.values())
        else "fail",
        "passed": None
        if tolerance is None
        else all(
            value <= tolerance for record in differences.values() for value in record.values()
        ),
        "label": label,
        "denominator": "exact_mixed",
        "relative_amplitude_tolerance": tolerance,
        "relative_amplitude_differences": differences,
        "scope": "Paired filtered-reference response on unchanged native mapped initialization/settings. A pass only admits interpreting the filtered actual-input experiment; it does not prove autonomous force recovery.",
    }


def audit_native_result(directory, report, information):
    from source.solvers.fvm.io.backup import FORMAT_VERSION, decode_state

    path = directory / "final_backup.npz"
    if digest(path) != report["final_checkpoint_sha256"]:
        raise ValueError("Completed replay native checkpoint differs from its receipt")
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"]))
        if metadata["format_version"] != FORMAT_VERSION:
            raise ValueError("Completed replay checkpoint does not use the current native format")
        state = decode_state({name: stored[name].copy() for name in stored.files})
    names = (
        "velocity",
        "velocity_old",
        "velocity_older",
        "kinematic_pressure",
        "volumetric_face_flux",
        "volumetric_face_flux_old",
        "volumetric_face_flux_older",
    )
    if not all(name in state and np.isfinite(state[name]).all() for name in names):
        raise ValueError("Completed replay checkpoint lacks finite primary/BDF/flux fields")
    if abs(float(state["time"]) - report["accepted_time"]) > 1e-9:
        raise ValueError("Completed replay checkpoint clock differs from its receipt")
    if (
        int(state["step"]) != information["start_step"] + information["steps"]
        or report["accepted_new_steps"] != information["steps"]
        or abs(float(state["time"]) - information["end_time"]) > 1e-9
    ):
        raise ValueError("Completed replay lacks the full native accepted-step horizon")
    if not np.allclose(
        [
            state[name]
            for name in ("time_step_size", "accepted_time_step_size", "previous_time_step_size")
        ],
        information["step_size"],
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("Completed replay changed native temporal step sizes")
    return {
        "checkpoint_sha256": digest(path),
        "accepted_time": float(state["time"]),
        "accepted_step": int(state["step"]),
        "finite_complete_primary_BDF_and_flux_fields": True,
        "configuration_hash": metadata["config_hash"],
        "decoded_field_shapes": {name: list(state[name].shape) for name in names},
    }


def recorded_helper_evidence(path, expected):
    """Preserve exact completed-control code without changing its old receipt."""
    snapshot = SOURCE_EVIDENCE / f"{path.stem}_{expected}.txt"
    if path.is_file() and digest(path) == expected:
        SOURCE_EVIDENCE.mkdir(exist_ok=True)
        if not snapshot.exists():
            snapshot.write_bytes(path.read_bytes())
    if not snapshot.is_file() or digest(snapshot) != expected:
        raise ValueError("Completed control lacks its exact recorded helper source evidence")
    return {"recorded_sha256": expected, "exact_source_snapshot": str(snapshot)}


def audit_frozen_replay_inputs(directory, result_directory, receipt):
    """Check completed inputs independently from live replay admission rules."""
    information = json.loads((directory / "inputs/inputs.json").read_text())
    check_equation_sources(information)
    expected_files = {
        directory / "boundary_trace.h5": receipt["boundary_trace_sha256"],
        directory / "initial_state.npz": receipt["initial_state_sha256"],
        directory / "inputs/original_capture_report.json": receipt[
            "original_capture_receipt_sha256"
        ],
        **{
            directory / "inputs" / name: value
            for name, value in receipt["native_input_sha256"].items()
        },
    }
    if receipt["source_receipt_sha256"]:
        expected_files[directory / "inputs/boundary_model_report.json"] = receipt[
            "source_receipt_sha256"
        ]
    for path, expected in expected_files.items():
        if digest(path) != expected:
            raise ValueError(f"Completed replay input changed: {path}")
    adapter = recorded_helper_evidence(
        Path(__file__).with_name("replay_boundary_inputs.py"), receipt["adapter_source_sha256"]
    )
    initialization = recorded_helper_evidence(
        Path(__file__).with_name("run_boundary_condition_study.py"),
        receipt["native_replay_helper_sha256"],
    )
    from .run_boundary_condition_study import validate_saved_settings

    actual = json.loads((result_directory / "solution/fvm_metadata.json").read_text())
    frozen = json.loads((directory / "inputs/small_metadata.json").read_text())
    settings = validate_saved_settings(actual, frozen)
    return {
        "strict_live_native_equation_sources_match": True,
        "all_frozen_input_hashes_match": True,
        "adapter_source": adapter,
        "native_initialization_source": initialization,
        "native_saved_settings_comparison": settings,
    }


def audit_same_actual_input_forms(series, provenance):
    """Verify a velocity-form comparison uses identical measured input arrays."""
    labels = ("coupled_harmonic", "coupled_full_velocity")
    if not all(label in series for label in labels):
        return {"status": "paired measured-input velocity forms unavailable"}
    import h5py

    roots = [series[label][1].parent for label in labels]
    names = (
        "time",
        "face_centre",
        "face_normal",
        "face_area",
        "velocity",
        "normal_velocity",
        "tangential_gradient",
    )
    differences = {}
    with (
        h5py.File(roots[0] / "boundary_trace.h5", "r") as mixed,
        h5py.File(roots[1] / "boundary_trace.h5", "r") as full,
    ):
        for name in names:
            a, b = mixed[name][...], full[name][...]
            if a.shape != b.shape or not np.array_equal(a, b):
                raise ValueError(
                    f"Measured-input velocity forms prescribe different arrays: {name}"
                )
            differences[name] = 0.0
        if mixed["time"].shape != (1501,):
            raise ValueError("Paired velocity forms lack the complete common native input schedule")
    reports = [provenance[label]["result"] for label in labels]
    if reports[0]["initial_state_sha256"] != reports[1]["initial_state_sha256"] or any(
        report["pressure_condition"] != "fixedFluxPressure" or report["operator_backend"] != "numba"
        for report in reports
    ):
        raise ValueError(
            "Measured-input velocity forms changed initialization, pressure or backend"
        )
    if reports[0]["mode"] != "mixed_subcycled" or reports[1]["mode"] != "full_velocity_subcycled":
        raise ValueError("Paired results do not represent the intended two velocity forms")
    return {
        "status": "identical measured actual VPM arrays and initialization verified",
        "compared_labels": list(labels),
        "bitwise_array_differences": differences,
        "native_schedule": "1501 rows at .008 s, from301 accepted .04 s endpoints",
        "shared_pressure_condition": "fixedFluxPressure",
        "scope": "Only the imposed velocity form differs: mixed Un/Gt versus all components of the same actual VPM velocity. No amplitude or source-phase modification.",
    }


def particle_image_source_quality(directory, receipt):
    """Report every qualified cold frame separately from its force response."""
    schema = receipt["source_attributes"].get("schema", "")
    if not schema.startswith("openonda-cylinder-reference-particle-image/"):
        return None
    if schema != "openonda-cylinder-reference-particle-image/2":
        raise ValueError(
            "Retired warm image controls are excluded; retain their existing failure evidence"
        )
    import h5py

    source = Path(receipt["source_hdf5"])
    expected = receipt["source_hdf5_sha256"]
    admission = receipt.get("particle_image_admission", {})
    if (
        digest(source) != expected
        or admission.get("status") != "qualified independent cold schema-2 merged image"
    ):
        raise ValueError("Cold image lacks its frozen complete native admission")
    producer = json.loads((directory / "inputs/boundary_model_report.json").read_text())
    if (
        producer.get("projection_history") != "cold"
        or producer.get("frames") != 301
        or len(producer["frame_reports"]) != 301
    ):
        raise ValueError("Cold image producer does not report all independent 301 frames")
    frequency_path = STUDY / "boundary_input_replay_20261005T235543Z/model_report.json"
    frequency_report = json.loads(frequency_path.read_text())
    frequency = frequency_report["force_frequency_hz"]["reference"]
    frame_quality = []
    fields, mode_quality = {}, {}
    with h5py.File(source, "r") as stored:
        time = stored["time"][:]
        area = stored["face_area"][:]
        if time.shape != (301,) or not np.isfinite(area).all() or np.any(area <= 0):
            raise ValueError("Cold source clock or face-area weights are invalid")
        for index, frame in enumerate(producer["frame_reports"]):
            if frame["index"] != index or frame["time"] != time[index]:
                raise ValueError("Cold frame reports and HDF5 source clocks differ")
            image = frame["image"]
            inverse = image["auxiliary_renewals"]
            target = image["auxiliary_target"]
            rows = [
                inverse[-1],
                frame["production_renewals"][-1],
                frame["reference_support_renewals"][-1],
            ]
            budgets = [row["native_replacement_budget"] for row in rows]
            frame_quality.append(
                {
                    "time": float(time[index]),
                    "maximum_auxiliary_absolute_strength": max(
                        row["maximum_stored_axial_strength"] for row in inverse
                    ),
                    "maximum_auxiliary_strength_l1": max(
                        row["stored_strength_l1"] for row in inverse
                    ),
                    "maximum_coefficient_to_target_ratio": max(
                        row["maximum_stored_axial_strength"] for row in inverse
                    )
                    / target["maximum_absolute_axial_strength"],
                    "maximum_coefficient_l1_to_target_ratio": max(
                        row["stored_strength_l1"] for row in inverse
                    )
                    / target["axial_strength_l1"],
                    "maximum_final_native_moment_closure_fraction": max(
                        budget["closure_correction_fraction"] for budget in budgets
                    ),
                    "maximum_final_native_strength_error_fraction_of_tolerance": max(
                        budget["vortex_strength_error"] / budget["vortex_strength_tolerance"]
                        for budget in budgets
                    ),
                    "maximum_final_native_linear_impulse_error_fraction_of_tolerance": max(
                        budget["linear_impulse_error"] / budget["linear_impulse_tolerance"]
                        for budget in budgets
                    ),
                    "final_represented_vorticity_relative_residual": inverse[-1][
                        "representation_residual_after_prune"
                    ],
                }
            )
        elapsed = time - float(np.mean(time))
        phase = 2 * np.pi * frequency * elapsed
        design = np.column_stack(
            [
                np.ones(len(time)),
                elapsed,
                np.cos(phase),
                np.sin(phase),
                np.cos(2 * phase),
                np.sin(2 * phase),
            ]
        )
        for name in ("normal_velocity", "tangential_gradient"):
            fields[name] = {}
            measured = {
                group: stored[group][name][:]
                for group in ("exact_reference", "reference_support", "production")
            }
            reference = measured["exact_reference"]
            ref_fit = np.linalg.lstsq(design, reference.reshape(301, -1), rcond=None)[0].reshape(
                (6, *reference.shape[1:])
            )
            for group in ("reference_support", "production"):
                difference = measured[group] - reference
                squared = difference**2 if difference.ndim == 2 else np.sum(difference**2, axis=2)
                rms = np.sqrt(np.sum(area[None] * squared, axis=1) / np.sum(area))
                fields[name][group] = rms.tolist()
                fitted = np.linalg.lstsq(design, measured[group].reshape(301, -1), rcond=None)[
                    0
                ].reshape(ref_fit.shape)
                quality = {}
                for harmonic in (1, 2):
                    offset = 2 * harmonic
                    actual_mode = fitted[offset] - 1j * fitted[offset + 1]
                    reference_mode = ref_fit[offset] - 1j * ref_fit[offset + 1]
                    weights = area if actual_mode.ndim == 1 else area[:, None]
                    reference_norm = float(np.sqrt(np.sum(weights * abs(reference_mode) ** 2)))
                    quality[str(harmonic)] = {
                        "area_weighted_amplitude_ratio_to_exact_reference": float(
                            np.sqrt(np.sum(weights * abs(actual_mode) ** 2)) / reference_norm
                        ),
                        "area_weighted_complex_mode_relative_error": float(
                            np.sqrt(np.sum(weights * abs(actual_mode - reference_mode) ** 2))
                            / reference_norm
                        ),
                        "common_clock_weighted_phase_difference_degrees": float(
                            np.angle(
                                np.sum(weights * actual_mode * np.conj(reference_mode)), deg=True
                            )
                        ),
                    }
                mode_quality.setdefault(group, {})[name] = quality
    if digest(source) != expected:
        raise ValueError("Cold particle-image source changed during its quality audit")
    bounded = {
        name: {
            "minimum": min(row[name] for row in frame_quality),
            "maximum": max(row[name] for row in frame_quality),
        }
        for name in frame_quality[0]
        if name != "time"
    }
    samples = []
    for target in (100, 104, 108, 112):
        index = int(np.argmin(abs(time - target)))
        samples.append(
            {
                **frame_quality[index],
                "image_error_to_exact_reference": {
                    group: {
                        name + "_area_weighted_rms": fields[name][group][index] for name in fields
                    }
                    for group in ("reference_support", "production")
                },
            }
        )
    return {
        "source_hdf5": str(source),
        "source_hdf5_sha256": expected,
        "projection_history": "cold",
        "frame_count": 301,
        "native_image_admission": admission,
        "all301_frame_quality": frame_quality,
        "all301_frame_quality_range": bounded,
        "all301_trace_area_weighted_rms_errors": fields,
        "sampled_source_quality": samples,
        "common_clock_first_and_second_harmonic_quality": mode_quality,
        "harmonic_frequency_hz": frequency,
        "force_frequency_receipt_sha256": digest(frequency_path),
        "source_model_admissible": True,
        "finite_wake_force_attribution_admissible": None,
        "interpretation_status": "Qualified independent cold input; paired field and force effects must still be assessed",
        "reason": "Every frame starts at zero auxiliary coefficients and satisfies the recorded native closure/strength/impulse and target-relative guards. The .08 limit is atomic moment closure, not represented-vorticity residual. Cold input validity alone does not prove autonomous VPM behavior or a force-deficit mechanism.",
        "error_definition": "Area-weighted RMS trace errors on identical faces. Harmonic modes use mean, trend, fundamental and second harmonic on one unchanged 100–112 s physical source clock; no phase or amplitude alignment.",
    }


CONTROL_NAMES = {
    "reference_flow": "Reference flow",
    "exact_mixed": "Exact reference traces",
    "reference_harmonic": "Reference harmonics",
    "coupled_harmonic": "Actual VPM harmonics",
    "reference_mean_actual_oscillation": "Reference mean · actual oscillation",
    "actual_mean_reference_oscillation": "Actual mean · reference oscillation",
    "image_reference_support": "Cold image · reference support",
    "image_production_support": "Cold image · production support",
    "coupled_full_velocity": "Actual VPM harmonics · full velocity",
    "reference_normal_oscillation": "Reference Un oscillation · actual Gt",
    "reference_gradient_oscillation": "Reference Gt oscillation · actual Un",
}
DEFAULT_CURVES = (
    "reference_flow",
    "reference_harmonic",
    "coupled_harmonic",
    "reference_normal_oscillation",
    "reference_gradient_oscillation",
    "image_production_support",
)


def render(histories, statistics, ratios, output, window, curve_labels):
    """Keep eligible response curves separate from the full amplitude table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(histories)
    palette = [
        "#242424",
        "#707070",
        "#326fa8",
        "#cf6d28",
        "#8b5798",
        "#408367",
        "#927940",
        "#596fab",
        "#b45b63",
        "#96507c",
        "#33978d",
        "#709157",
    ]
    colours = {label: palette[index % len(palette)] for index, label in enumerate(labels)}
    names = {label: CONTROL_NAMES.get(label, label.replace("_", " ")) for label in labels}
    styles = [
        "-",
        (0, (6, 2)),
        (0, (2, 2)),
        (0, (7, 2, 1, 2)),
        (0, (5, 2, 1, 2, 1, 2)),
        (0, (3, 2)),
    ]
    figure, axes = plt.subplots(2, 1, figsize=(13.8, 8.4))
    figure.subplots_adjust(left=0.08, right=0.975, top=0.77, bottom=0.20, hspace=0.35)
    for row, field in enumerate(FIELDS):
        for index, label in enumerate(curve_labels):
            values = histories[label]
            selected = (values["time"] >= window[0] - 1e-8) & (values["time"] <= window[1] + 1e-8)
            axes[row].plot(
                values["time"][selected],
                values[field][selected],
                color=colours[label],
                linestyle=styles[index % len(styles)],
                linewidth=1.5,
                label=names[label],
            )
        axes[row].set_xlim(*window)
        axes[row].set_xlabel("Accepted research replay time (s)")
        axes[row].set_ylabel("Cd" if row == 0 else "Cl")
        axes[row].grid(color="#e5e6e8", linewidth=0.65)
        axes[row].spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.072, 0.885),
        ncol=3,
        fontsize=9,
        frameon=False,
        handlelength=3.4,
    )
    figure.suptitle(
        "Cylinder boundary-input response controls",
        x=0.08,
        y=0.97,
        ha="left",
        fontsize=18,
        weight="bold",
    )
    figure.text(
        0.08,
        0.926,
        f"Common mapped reference initialization; measured {window[0]:g}–{window[1]:g} s after 2 s settling",
        fontsize=11,
    )
    figure.text(
        0.08,
        0.116,
        "Forced FVM responses. Only eligible completed inputs are shown; full amplitudes and cycle counts are in the separate table.",
        fontsize=10,
    )
    figure.text(
        0.08,
        0.078,
        "Source clocks are recorded separately. Finite response trends and one complete lift cycle do not establish autonomous recovery.",
        fontsize=10,
    )
    figure.text(
        0.08,
        0.040,
        "No amplitude, frequency or phase rescaling is applied to the plotted force histories.",
        fontsize=10,
    )
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output / f"boundary_input_response_curves.{extension}", dpi=175)
    plt.close(figure)

    figure = plt.figure(figsize=(16, max(11.2, 0.58 * len(labels) + 5.3)))
    measured = [label for label in labels if label != "reference_flow"]
    positions = np.arange(len(measured))
    for row, field in enumerate(FIELDS):
        axis = figure.add_axes((0.31 + row * 0.345, 0.48, 0.28, 0.36))
        for index, quantity in enumerate(QUANTITIES):
            values = [ratios[label][field][quantity] for label in measured]
            if any(value is None for value in values):
                raise ValueError("An amplitude-table series lacks a complete cycle")
            axis.barh(
                positions + (-0.17 if index == 0 else 0.17),
                values,
                height=0.29,
                color=[colours[label] if index == 0 else "white" for label in measured],
                edgecolor=[colours[label] for label in measured],
                hatch=None if index == 0 else "///",
                linewidth=0.9,
            )
        axis.set_yticks(
            positions,
            [names[label].replace(" · ", "\n") for label in measured]
            if row == 0
            else [""] * len(measured),
            fontsize=9,
        )
        axis.invert_yaxis()
        axis.axvline(1, color="#777777", linestyle=(0, (3, 2)), linewidth=0.8)
        maximum = max(
            ratios[label][field][quantity] for label in measured for quantity in QUANTITIES
        )
        axis.set_xlim(0, max(1.08, maximum * 1.05))
        axis.set_title(
            ("Cd" if row == 0 else "Cl") + " amplitude / native reference", loc="left", fontsize=12
        )
        axis.set_xlabel("Raw filled · drift corrected hatched", fontsize=10)
        axis.grid(axis="x", color="#e5e6e8", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    table_axis = figure.add_axes((0.05, 0.15, 0.90, 0.26))
    table_axis.axis("off")
    rows = [
        [
            names[label],
            f"{record['drag_coefficient']['median_peak_to_peak_raw']:.6f}",
            f"{record['drag_coefficient']['median_peak_to_peak_drift_corrected']:.6f}",
            f"{record['lift_coefficient']['median_peak_to_peak_raw']:.6f}",
            f"{record['lift_coefficient']['median_peak_to_peak_drift_corrected']:.6f}",
            f"{record['drag_coefficient']['complete_cycle_count']}/{record['lift_coefficient']['complete_cycle_count']}",
        ]
        for label, record in statistics.items()
    ]
    table = table_axis.table(
        cellText=rows,
        colLabels=["Input/control", "Cd raw", "Cd drift", "Cl raw", "Cl drift", "Cd/Cl cycles"],
        colWidths=[0.39, 0.122, 0.122, 0.122, 0.122, 0.122],
        cellLoc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d5d7da")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#edf0f3")
            cell.set_text_props(weight="bold")
        if column == 0:
            cell.set_text_props(ha="left")
    figure.suptitle(
        "Complete-cycle boundary-control amplitudes",
        x=0.05,
        y=0.97,
        ha="left",
        fontsize=19,
        weight="bold",
    )
    figure.text(
        0.05,
        0.928,
        f"Measured {window[0]:g}–{window[1]:g} s on the common research replay clock; no amplitude rescaling",
        fontsize=11,
    )
    figure.text(
        0.05,
        0.084,
        "Each cycle is wholly bracketed by troughs inside the measured window. Pressure/viscous contributions and transient trends are in the JSON report.",
        fontsize=10,
    )
    figure.text(
        0.05,
        0.050,
        "Forced boundary-input controls do not demonstrate a self-sustained production repair. Retired warm-image experiments are excluded.",
        fontsize=10,
    )
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output / f"boundary_input_amplitude_controls.{extension}", dpi=175)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-study-directory", type=Path, default=EXACT)
    parser.add_argument("--replay", type=read_replay_argument, action="append", default=[])
    parser.add_argument("--start", type=float, default=102.0)
    parser.add_argument("--end", type=float, default=112.0)
    parser.add_argument("--filtered-reference-label", default="reference_harmonic")
    parser.add_argument("--filtering-tolerance", type=float)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--curve-label", action="append")
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    if not args.start < args.end or not np.isfinite((args.start, args.end)).all():
        raise ValueError("A finite, increasing accepted-time window is required")
    if args.filtering_tolerance is not None and not 0 <= args.filtering_tolerance < 1:
        raise ValueError("Explicit filtering tolerance must lie in[0,1)")
    if len({label for label, _directory in args.replay}) != len(args.replay):
        raise ValueError("Replay labels must be unique")
    exact = args.exact_study_directory.resolve()
    output = args.output_directory or STUDY / (
        "boundary_input_force_comparison_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / "snapshots"
    snapshots.mkdir()
    series = {
        "reference_flow": (exact / "reference", None),
        "exact_mixed": (exact / "mixed_subcycled_numba", None),
    }
    for label, directory in args.replay:
        series[label] = replay_result_directory(directory)
    window = (args.start, args.end)
    statistics, sources, provenance, histories = {}, {}, {}, {}
    initial_hashes = set()
    original_information = json.loads((exact / "inputs/inputs.json").read_text())
    for label, (directory, receipt_path) in series.items():
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text())
        if report["status"] != "complete" or report["accepted_time"] < args.end - 1e-8:
            raise ValueError(
                f"Control has not completed the requested accepted-time window: {label}"
            )
        if report.get("finite_primary_and_temporal_fields") is False:
            raise ValueError(f"Control has nonfinite native fields: {label}")
        values, sources[label] = freeze_csv(
            directory / "samples/forces_history.csv", snapshots / f"{label}_forces_history.csv"
        )
        statistics[label] = measure(values, window)
        provenance[label] = {
            "result_report": str(report_path),
            "result_report_sha256": digest(report_path),
            "result": report,
        }
        histories[label] = values
        provenance[label]["native_checkpoint_audit"] = audit_native_result(
            directory, report, original_information
        )
        if receipt_path:
            receipt = json.loads(receipt_path.read_text())
            provenance[label]["independent_input_audit"] = audit_frozen_replay_inputs(
                receipt_path.parent, directory, receipt
            )
            if (
                report["boundary_trace_sha256"] != receipt["boundary_trace_sha256"]
                or report["initial_state_sha256"] != receipt["initial_state_sha256"]
            ):
                raise ValueError(f"Replay result differs from its explicit input receipt: {label}")
            initial_hashes.add(receipt["initial_state_sha256"])
            provenance[label]["trace_receipt"] = receipt
            provenance[label]["trace_receipt_sha256"] = digest(receipt_path)
            if not report["frozen_settings_comparison"]["matched"]:
                raise ValueError("Explicit control native equation settings did not match")
        elif label == "exact_mixed":
            initial_hashes.add(report["initial_state_sha256"])
    if len(initial_hashes) != 1:
        raise ValueError(
            "Exact and explicit controls do not share the unchanged mapped initial state"
        )
    canonical_path = exact / "comparison.json"
    canonical = json.loads(canonical_path.read_text())
    for label, native_label in (
        ("reference_flow", "reference"),
        ("exact_mixed", "mixed_subcycled_numba"),
    ):
        if sources[label]["sha256"] != canonical["series"][native_label]["force_history_sha256"]:
            raise ValueError("Exact baseline force history differs from its canonical source hash")
    canonical_difference = {}
    if np.allclose(canonical["statistics_window"], window, atol=1e-8, rtol=0):
        for label, native_label in (
            ("reference_flow", "reference"),
            ("exact_mixed", "mixed_subcycled_numba"),
        ):
            for field in FIELDS:
                for quantity in QUANTITIES:
                    difference = abs(
                        statistics[label][field][quantity]
                        - canonical["complete_cycle_statistics"][native_label]["statistics"][field][
                            quantity
                        ]
                    )
                    canonical_difference[f"{label}.{field}.{quantity}"] = difference
        if max(canonical_difference.values()) > 1e-11:
            raise ValueError(
                "Independent complete-cycle calculation differs from the frozen native comparison"
            )
    relative_to_reference = relative_statistics(statistics, "reference_flow")
    relative_to_exact = relative_statistics(statistics, "exact_mixed")
    gate = filtering_admissibility(
        statistics, relative_to_exact, args.filtered_reference_label, args.filtering_tolerance
    )
    source_quality = {
        label: particle_image_source_quality(
            receipt_path.parent, provenance[label]["trace_receipt"]
        )
        for label, (_directory, receipt_path) in series.items()
        if receipt_path
    }
    curve_labels = args.curve_label or [label for label in DEFAULT_CURVES if label in histories]
    if (
        not curve_labels
        or len(curve_labels) > 6
        or len(set(curve_labels)) != len(curve_labels)
        or any(label not in histories for label in curve_labels)
    ):
        raise ValueError("Response plots require one to six unique measured curve labels")
    result = {
        "schema": "openonda-explicit-boundary-input-force-comparison/1",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "window": list(window),
        "settling_seconds_from100s_reference_initialization": args.start - 100,
        "sources": sources,
        "provenance": provenance,
        "canonical_comparison_sha256": digest(canonical_path),
        "independent_baseline_statistics_maximum_differences": canonical_difference,
        "complete_cycle_statistics": statistics,
        "independent_direct_extrema_and_cycle_trend": {
            label: independent_extrema_and_trend(histories[label], record)
            for label, record in statistics.items()
        },
        "relative_to_native_reference_flow": relative_to_reference,
        "relative_to_exact_fed_mixed_numba_control": relative_to_exact,
        "reference_harmonic_filtering_admissibility": gate,
        "measured_input_velocity_form_audit": audit_same_actual_input_forms(series, provenance),
        "particle_image_source_quality_and_admissibility": source_quality,
        "eligible_response_curve_labels": curve_labels,
        "normalization": "Cd/Cl=2F/(rho Uref^2 D span),rho=Uref=D=span=1; verified directly from native force CSVs. No amplitude scaling applied to comparisons.",
        "method": "Complete trough-bracketed cycles wholly inside the accepted-time window; rawmax-min and peak minus linearly interpolated bracketing troughs. Pressure/viscous contributions use the same extrema. Curves and cycle times retain the native/research replay clock without phase shifts.",
        "limitations": [
            "The102–112s window generally contains only3complete drag cycles and1complete lift cycle. The10s measured response does not establish fully settled saturation.",
            "Actual/hybrid/image boundary inputs and reference initialization are distinct flow states. Source-clock mappings and reconstruction/filtering limitations are retained in their explicit input receipts.",
            "Filtering admissibility is assessed against exact-fed mixedNumba/.04control, not inferred merely because a model has finite fields or passed an analytic test.",
            "No source history, production force, native solver or user figure is modified by this utility.",
        ],
    }
    write_json(output / "comparison.json", result)
    if args.plot:
        render(histories, statistics, relative_to_reference, output, window, curve_labels)
    print(
        json.dumps(
            {
                "directory": str(output),
                "amplitude_ratios_to_reference": relative_to_reference,
                "filtering_admissibility": gate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
