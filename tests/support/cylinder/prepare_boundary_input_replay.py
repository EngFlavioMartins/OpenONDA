"""Prepare paired low-mode boundary-input controls without advancing a solver.

The reference model checks whether temporal filtering preserves the reference
force response. The measured coupled model retains its own frequency, mean,
linear trend and field-relative phases, with one global initial lift-phase
translation onto the unchanged mapped reference100s primary/BDF state.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

import h5py
import numpy as np

from .analyze_boundary_harmonics import gaussian_velocity_normal_derivative
from .replay_boundary_inputs import digest, interpolate_endpoint_values, prepare_input, write_json

STUDY = Path(__file__).resolve().parent / "force_boundary_study/20261005T214514Z"
HARMONICS = STUDY / "boundary_harmonics_20261005T222923Z"
REFERENCE = Path("/tmp/openonda-cylinder-boundary-study-20261005")
FIELDS = ("velocity", "normal_velocity", "tangential_gradient")


def design(time, frequency, midpoint):
    elapsed = np.asarray(time) - midpoint
    phase = 2 * np.pi * frequency * elapsed
    return np.column_stack(
        (
            np.ones(len(elapsed)),
            elapsed,
            np.cos(phase),
            np.sin(phase),
            np.cos(2 * phase),
            np.sin(2 * phase),
        )
    )


def coefficient(time, values, frequency, midpoint):
    fitted = np.linalg.lstsq(
        design(time, frequency, midpoint), values.reshape(len(time), -1), rcond=None
    )[0]
    return fitted.reshape((6, *values.shape[1:]))


def evaluate(time, fitted, frequency, midpoint):
    return (design(time, frequency, midpoint) @ fitted.reshape(6, -1)).reshape(
        (len(time), *fitted.shape[1:])
    )


def weighted_rms(values, area):
    squared = values**2
    if values.ndim == 3:
        squared = np.sum(squared, axis=-1)
    return float(np.sqrt(np.mean(np.average(squared, weights=area, axis=-1))))


def side_groups(centre):
    sides = {
        "upstream": np.isclose(centre[:, 0], centre[:, 0].min(), atol=1e-10),
        "downstream": np.isclose(centre[:, 0], centre[:, 0].max(), atol=1e-10),
        "lower": np.isclose(centre[:, 1], centre[:, 1].min(), atol=1e-10),
        "upper": np.isclose(centre[:, 1], centre[:, 1].max(), atol=1e-10),
    }
    sides["corners"] = ~np.logical_or.reduce(list(sides.values()))
    if sum(np.count_nonzero(value) for value in sides.values()) != len(centre):
        raise ValueError("Boundary side groups overlap or omit faces")
    sides["all"] = np.ones(len(centre), dtype=bool)
    return sides


def fit_statistics(time, values, fitted, frequency, midpoint, area, sides):
    predicted = evaluate(time, fitted, frequency, midpoint)
    baseline = (design(time, frequency, midpoint)[:, :2] @ fitted[:2].reshape(2, -1)).reshape(
        values.shape
    )
    residual = values - predicted
    result = {}
    for side, selected in sides.items():
        error = weighted_rms(residual[:, selected], area[selected])
        fluctuation = weighted_rms((values - baseline)[:, selected], area[selected])
        maximum = np.abs(residual[:, selected])
        if values.ndim == 3:
            maximum = np.linalg.norm(residual[:, selected], axis=-1)
        result[side] = {
            "residual_rms": error,
            "measured_fluctuation_rms_after_mean_and_trend": fluctuation,
            "residual_fraction_of_fluctuation_rms": error / fluctuation,
            "captured_fluctuation_energy_fraction": 1 - (error / fluctuation) ** 2,
            "maximum_residual_magnitude": float(np.max(maximum)),
            "linear_slope_rms_per_second": weighted_rms(fitted[1][None, selected], area[selected]),
        }
    return result


def coupled_velocity(harmonics, centre, normal, area, expected_normal):
    result = []
    records = []
    times = []
    for path in sorted((harmonics / "snapshots").glob("vpm_*.h5")):
        before = digest(path)
        with h5py.File(path, "r") as stored:
            attributes = dict(stored["solver"].attrs)
            position = stored["particles/position"][:].astype(np.float64)
            strength = stored["particles/vortex_strength"][:].astype(np.float64)
            radius = stored["particles/core_radius"][:].astype(np.float64)
        configuration = json.loads(attributes["numerical_configuration"])
        induction = configuration["induction"]
        if induction["method"] != "PLANAR" or induction["kernel"] != "GAUSSIAN":
            raise ValueError("Measured source does not use the native Gaussian planar kernel")
        velocity, _derivative = gaussian_velocity_normal_derivative(
            centre, normal, position, strength, radius, induction["planar_span"]
        )
        velocity += np.asarray(attributes["freestream_velocity"], dtype=np.float64)
        correction = float(np.dot(np.einsum("fi,fi->f", velocity, normal), area) / area.sum())
        velocity -= correction * normal
        result.append(velocity)
        times.append(float(attributes["time"]))
        records.append(
            {
                "path": str(path),
                "sha256": before,
                "time": float(attributes["time"]),
                "step": int(attributes["step"]),
                "particle_count": len(position),
                "span": induction["planar_span"],
                "configuration_sha256": attributes["numerical_configuration_sha256"],
                "normal_flux_correction": correction,
            }
        )
        if digest(path) != before:
            raise RuntimeError("Frozen native particle snapshot changed during reconstruction")
    result = np.asarray(result)
    if len(result) != len(expected_normal):
        raise ValueError("Frozen particle snapshots differ from the21 measured boundary times")
    mismatch = float(np.max(abs(np.einsum("tfi,fi->tf", result, normal) - expected_normal)))
    if mismatch > 1e-10:
        raise ValueError("Native reconstructed velocity differs from frozen measured Un")
    return result, np.asarray(times), records, mismatch


def phase_mapping(
    coupled_midpoint,
    coupled_frequency,
    coupled_phase,
    reference_midpoint,
    reference_frequency,
    reference_phase,
    replay_start,
    duration,
    measured_window,
):
    initial_reference_phase = reference_phase + 2 * np.pi * reference_frequency * (
        replay_start - reference_midpoint
    )
    first_solution = coupled_midpoint + (initial_reference_phase - coupled_phase) / (
        2 * np.pi * coupled_frequency
    )
    low, high = measured_window[0], measured_window[1] - duration
    candidates = [
        first_solution + integer / coupled_frequency
        for integer in range(-20, 21)
        if low - 1e-9 <= first_solution + integer / coupled_frequency <= high + 1e-9
    ]
    if not candidates:
        raise ValueError(
            "No lift-phase translation keeps the model inside its measured source window"
        )
    target_start = coupled_midpoint - duration / 2
    mapped_start = min(candidates, key=lambda value: abs(value - target_start))
    difference = (
        coupled_phase
        + 2 * np.pi * coupled_frequency * (mapped_start - coupled_midpoint)
        - initial_reference_phase
    )
    phase_error = float(np.angle(np.exp(1j * difference)))
    if abs(phase_error) > 1e-12:
        raise ValueError("Initial lift-phase translation failed")
    return {
        "replay_clock_start": replay_start,
        "source_clock_start": mapped_start,
        "source_clock_end": mapped_start + duration,
        "source_minus_replay_clock": mapped_start - replay_start,
        "source_clock_rate": 1.0,
        "initial_lift_phase_error_radians_modulo_2pi": phase_error,
        "equivalent_source_start_candidates_inside_measured_window": candidates,
        "selection": "Equivalent lift-phase start nearest the measured window midpoint after accounting for replay duration",
    }


def write_endpoint_model(
    path, *, clock, source_clock, geometry, values, frequency, midpoint, origin
):
    with h5py.File(path, "w") as stored:
        stored.attrs["complete"] = False
        stored.attrs["schema"] = "openonda-harmonic-boundary-endpoint-model/1"
        stored.attrs["trace_origin"] = origin
        stored.attrs["fundamental_frequency_hz"] = frequency
        stored.attrs["source_fit_midpoint_time"] = midpoint
        stored.attrs["model"] = (
            "mean + linear trend + fundamental + second harmonic; no amplitude rescaling"
        )
        stored.attrs["phase_policy"] = (
            "One global initial lift-phase translation only; reference model keeps its native clock"
        )
        stored.create_dataset("time", data=clock)
        stored.create_dataset("source_time", data=source_clock)
        for name, value in geometry.items():
            stored.create_dataset(name, data=value)
        for name, value in values.items():
            stored.create_dataset(name, data=value, compression="gzip", shuffle=True)
        stored.attrs["complete"] = True


def main():
    execution_sources = (
        Path(__file__).resolve(),
        Path(__file__).with_name("replay_boundary_inputs.py"),
        Path(__file__).with_name("analyze_boundary_harmonics.py"),
        Path(__file__).with_name("analyze_boundary_harmonic_phase.py"),
    )
    execution_source_hashes = {str(path): digest(path) for path in execution_sources}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmonic-directory", type=Path, default=HARMONICS)
    parser.add_argument("--reference-directory", type=Path, default=REFERENCE)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    harmonic = args.harmonic_directory.resolve()
    reference = args.reference_directory.resolve()
    output = args.output_directory or STUDY / (
        "boundary_input_replay_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    measurements = json.loads((harmonic / "boundary_harmonic_statistics.json").read_text())
    phases = json.loads((harmonic / "boundary_harmonic_phase_statistics.json").read_text())
    with np.load(harmonic / "boundary_harmonic_fields.npz", allow_pickle=False) as stored:
        fields = {name: stored[name].copy() for name in stored.files}
    with np.load(harmonic / "boundary_harmonic_complex_fields.npz", allow_pickle=False) as stored:
        saved_coefficients = {
            name: stored[name].copy() for name in stored.files if name.endswith("_coefficient")
        }
    reference_trace = harmonic / "snapshots/reference_traces.h5"
    with h5py.File(reference_trace, "r") as stored:
        if not stored.attrs["complete"]:
            raise ValueError("Original reference trace is incomplete")
        geometry = {name: stored[name][:] for name in ("face_centre", "face_normal", "face_area")}
        native_clock = stored["time"][:]
        reference_velocity = stored["velocity"][:]
    if any(not np.array_equal(value, fields[name]) for name, value in geometry.items()):
        raise ValueError("Frozen harmonic and native reference face order or geometry differ")
    coupled_full_velocity, coupled_clock, particle_records, velocity_normal_error = (
        coupled_velocity(
            harmonic,
            geometry["face_centre"],
            geometry["face_normal"],
            geometry["face_area"],
            fields["coupled_normal_velocity"],
        )
    )
    if not np.allclose(coupled_clock, fields["coupled_time"], atol=1e-10, rtol=0):
        raise ValueError("Frozen reconstructed particle and fitted boundary clocks differ")
    frequency = {
        label: measurements["force_frequency_fits"][label]["fundamental_frequency"]
        for label in ("coupled", "reference")
    }
    midpoint = {
        label: (fields[f"{label}_time"][0] + fields[f"{label}_time"][-1]) / 2 for label in frequency
    }
    values = {
        label: {
            field: fields[f"{label}_{field}"]
            for field in ("normal_velocity", "tangential_gradient")
        }
        for label in frequency
    }
    values["coupled"]["velocity"] = coupled_full_velocity
    values["reference"]["velocity"] = reference_velocity
    fitted = {
        label: {
            field: coefficient(fields[f"{label}_time"], value, frequency[label], midpoint[label])
            for field, value in series.items()
        }
        for label, series in values.items()
    }
    coefficient_error = max(
        float(
            np.max(abs(fitted[label][field] - saved_coefficients[f"{label}_{field}_coefficient"]))
        )
        for label in frequency
        for field in ("normal_velocity", "tangential_gradient")
    )
    if coefficient_error > 1e-10:
        raise ValueError("Fresh fit differs from the frozen complex harmonic coefficients")
    sides = side_groups(geometry["face_centre"])
    fit_quality = {
        label: {
            field: fit_statistics(
                fields[f"{label}_time"],
                value,
                fitted[label][field],
                frequency[label],
                midpoint[label],
                geometry["face_area"],
                sides,
            )
            for field, value in series.items()
        }
        for label, series in values.items()
    }
    duration = float(native_clock[-1] - native_clock[0])
    force_phase = {
        label: np.deg2rad(phases["force_phase_references"][label]["lift_fundamental_phase_degrees"])
        for label in frequency
    }
    mapping = phase_mapping(
        midpoint["coupled"],
        frequency["coupled"],
        force_phase["coupled"],
        midpoint["reference"],
        frequency["reference"],
        force_phase["reference"],
        float(native_clock[0]),
        duration,
        fields["coupled_time"][[0, -1]],
    )
    substeps = 5
    endpoint_clock = native_clock[::substeps]
    source_clock = {
        "reference": endpoint_clock.copy(),
        "coupled": mapping["source_clock_start"] + endpoint_clock - endpoint_clock[0],
    }
    interpolation_error = {}
    initialization_difference = {}
    model_paths = {}
    for label in frequency:
        endpoint_values = {
            field: evaluate(
                source_clock[label], fitted[label][field], frequency[label], midpoint[label]
            )
            for field in FIELDS
        }
        model_path = output / f"{label}_harmonic_endpoints.h5"
        model_paths[label] = model_path
        origin = (
            "Measured actual coupled VPM Gaussian field harmonic model"
            if label == "coupled"
            else "Native reference FVM field harmonic model"
        )
        write_endpoint_model(
            model_path,
            clock=endpoint_clock,
            source_clock=source_clock[label],
            geometry=geometry,
            values=endpoint_values,
            frequency=frequency[label],
            midpoint=midpoint[label],
            origin=origin,
        )
        continuous_source_clock = (
            native_clock
            if label == "reference"
            else mapping["source_clock_start"] + native_clock - native_clock[0]
        )
        interpolation_error[label] = {}
        initialization_difference[label] = {}
        for field in FIELDS:
            continuous = evaluate(
                continuous_source_clock, fitted[label][field], frequency[label], midpoint[label]
            )
            difference = interpolate_endpoint_values(endpoint_values[field], substeps) - continuous
            interpolation_error[label][field] = {
                "native_endpoint_linear_interpolation_rms_error_vs_continuous_low_mode_model": weighted_rms(
                    difference, geometry["face_area"]
                ),
                "maximum_component_error": float(np.max(abs(difference))),
            }
            original = (
                reference_velocity[0] if field == "velocity" else fields[f"reference_{field}"][0]
            )
            initialization_difference[label][field] = {
                "rms_difference_from_exact_reference_start": weighted_rms(
                    (endpoint_values[field][0] - original)[None], geometry["face_area"]
                ),
                "maximum_component_difference_from_exact_reference_start": float(
                    np.max(abs(endpoint_values[field][0] - original))
                ),
            }
    sampling = {}
    reference_time = fields["reference_time"]
    decimated = np.arange(0, len(reference_time), 125)
    for field in ("normal_velocity", "tangential_gradient"):
        dense = fitted["reference"][field]
        sparse = coefficient(
            reference_time[decimated],
            values["reference"][field][decimated],
            frequency["reference"],
            midpoint["reference"],
        )
        harmonic_difference = {}
        for order in (1, 2):
            complex_dense = dense[2 * order] - 1j * dense[2 * order + 1]
            complex_sparse = sparse[2 * order] - 1j * sparse[2 * order + 1]
            harmonic_difference[f"harmonic_{order}"] = {
                "amplitude_rms_ratio_one_second_to_dense": weighted_rms(
                    abs(complex_sparse)[None], geometry["face_area"]
                )
                / weighted_rms(abs(complex_dense)[None], geometry["face_area"]),
                "complex_coefficient_rms_difference_fraction": weighted_rms(
                    abs(complex_sparse - complex_dense)[None], geometry["face_area"]
                )
                / weighted_rms(abs(complex_dense)[None], geometry["face_area"]),
            }
        sampling[field] = {
            "reference_one_second_sample_count": len(decimated),
            "dense_vs_one_second_complex_coefficients": harmonic_difference,
            "dense_time_rms_difference_between_low_mode_models": weighted_rms(
                evaluate(reference_time, sparse, frequency["reference"], midpoint["reference"])
                - evaluate(reference_time, dense, frequency["reference"], midpoint["reference"]),
                geometry["face_area"],
            ),
        }
    coefficient_path = output / "harmonic_coefficients.npz"
    np.savez_compressed(
        coefficient_path,
        **{
            f"{label}_{field}": value
            for label, series in fitted.items()
            for field, value in series.items()
        },
    )
    source_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("replay_boundary_inputs.py"),
        Path(__file__).with_name("analyze_boundary_harmonics.py"),
        Path(__file__).with_name("analyze_boundary_harmonic_phase.py"),
        harmonic / "boundary_harmonic_fields.npz",
        harmonic / "boundary_harmonic_complex_fields.npz",
        harmonic / "boundary_harmonic_statistics.json",
        harmonic / "boundary_harmonic_phase_statistics.json",
        reference_trace,
        reference / "reference/report.json",
        reference / "initial_state.npz",
        harmonic / "snapshots/coupled_forces.csv",
        harmonic / "snapshots/reference_forces.csv",
    ]
    report = {
        "schema": "openonda-paired-harmonic-boundary-input-models/1",
        "status": "prepared; neither model has been advanced by a solver",
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "sources_sha256": {str(path): digest(path) for path in source_paths},
        "execution_source_sha256": execution_source_hashes,
        "particle_sources": particle_records,
        "force_frequency_hz": frequency,
        "fit_midpoint_clock": midpoint,
        "source_windows": {label: fields[f"{label}_time"][[0, -1]].tolist() for label in frequency},
        "measured_samples": {label: len(fields[f"{label}_time"]) for label in frequency},
        "model_columns": [
            "mean",
            "linear_slope",
            "cos_fundamental",
            "sin_fundamental",
            "cos_second_harmonic",
            "sin_second_harmonic",
        ],
        "design_matrix_condition_number": {
            label: float(
                np.linalg.cond(design(fields[f"{label}_time"], frequency[label], midpoint[label]))
            )
            for label in frequency
        },
        "coupled_phase_mapping": mapping,
        "phase_policy": "The coupled source clock is translated at unit rate so that its fitted lift fundamental matches the reference force phase at the unchanged100s initial state. No individual field/harmonic phase is aligned, no frequency is rescaled, and no amplitude is rescaled. The coupled drag-second minus twice-lift phase remains its measured value.",
        "force_relative_phase_degrees": {
            label: phases["force_phase_references"][label] for label in frequency
        },
        "fit_quality": fit_quality,
        "reference_one_second_sampling_sensitivity": sampling,
        "coupling_endpoint_interpolation_error": interpolation_error,
        "initial_input_difference_from_reference_state_trace": initialization_difference,
        "verification": {
            "reconstructed_full_velocity_vs_frozen_normal_trace_maximum_error": velocity_normal_error,
            "fresh_vs_frozen_harmonic_coefficient_maximum_error": coefficient_error,
            "exact_original_face_geometry": True,
            "solver_or_device_initialization": False,
        },
        "harmonic_coefficients_sha256": digest(coefficient_path),
        "endpoint_models": {
            label: {
                "path": str(path),
                "sha256": digest(path),
                "accepted_endpoint_count": len(endpoint_clock),
            }
            for label, path in model_paths.items()
        },
        "units": {"velocity": "m/s", "normal_velocity": "m/s", "tangential_gradient": "1/s"},
        "kernel_and_weights": "Exact Gaussian planar induction using each stored core radius and circulation=stored_strength_z/span; volume h_xy^2*span is not multiplied again. Full velocity uses the same uniform normal-flux correction as frozen Un.",
        "interpretation_gate": "Run the reference-harmonic model first against the exact reference-fed Numba/.04 control. Interpret the actual coupled harmonic control only if this filtering model retains the reference force amplitudes after settling; quantify any filtering bias instead of assuming zero.",
        "limitations": [
            "The measured VPM field has21 one-second snapshots and a .5Hz Nyquist frequency. Fundamental .1815Hz and second .3630Hz are resolved; higher modes, including the third .5445Hz, are not identifiable without aliasing.",
            "Residual energy is concentrated at the downstream boundary. The models remove around5%coupled and6%reference Gt fluctuation energy there; this control cannot exclude a causal role for omitted higher modes.",
            "The actual data originate at30–50s; native reference traces and the mapped initialization originate at100–112s. The clock translation is a research control, not a physical-time-identical coupled restart.",
            "A100s reference interior/BDF state is initialized with a different harmonic boundary input. Initial lift-phase alignment reduces one phase discontinuity but does not make the complete state consistent with the coupled field. Exclude initial settling and compare complete cycles;12s may still be insufficient for a fully equilibrated response.",
            "One global lift-phase translation preserves the measured drag/lift relative phase, side-to-side phase and frequency differences. It does not enforce the reference's phase relationships.",
            "This forced-FVM comparison probes measured boundary inputs separately from closed-loop feedback. It does not uniquely isolate particle kernel, transfer, diffusion, tail truncation or a source-state change.",
        ],
    }
    if any(digest(path) != expected for path, expected in execution_source_hashes.items()):
        raise RuntimeError("A preparation source changed during the host-only analysis")
    model_report = output / "model_report.json"
    write_json(model_report, report)
    prepared = {}
    for label, model_path in model_paths.items():
        prepared[label] = prepare_input(
            output / f"{label}_harmonic_replay",
            source=model_path,
            group="",
            reference=reference,
            origin=report["endpoint_models"][label]["path"],
            source_receipt=model_report,
        )
    write_json(
        output / "prepared_replays.json",
        {
            label: {
                "directory": str(output / f"{label}_harmonic_replay"),
                "boundary_trace_sha256": value["boundary_trace_sha256"],
                "initial_state_sha256": value["initial_state_sha256"],
                "status": value["status"],
            }
            for label, value in prepared.items()
        },
    )
    print(
        json.dumps(
            {
                "directory": str(output),
                "phase_mapping": mapping,
                "fit_quality": {
                    label: {
                        field: fit_quality[label][field]["all"]
                        for field in ("normal_velocity", "tangential_gradient")
                    }
                    for label in frequency
                },
                "status": report["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
