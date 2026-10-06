"""Separate normal-velocity spatial modes from their shedding frequency.

This host-only diagnostic changes the fundamental and second-harmonic clock
rate of normal velocity. It retains the coupled mean, linear trend, tangential
velocity and tangential gradient. Original phase translations and harmonic
coefficients remain unchanged. No replay is prepared or solver advanced.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .prepare_boundary_component_inputs import FIELDS, GEOMETRY, NORMAL_CONTROL
from .prepare_boundary_hybrid_inputs import components
from .prepare_boundary_input_replay import side_groups, weighted_rms

STUDY = Path(__file__).resolve().parent / "force_boundary_study/20261005T214514Z"
MODELS = STUDY / "boundary_input_replay_20261005T235543Z"
COMPONENTS = STUDY / "boundary_component_inputs_20261006T010013Z"
ACTUAL_MODE_REFERENCE_FREQUENCY = "actual_normal_mode_reference_frequency"
REFERENCE_MODE_ACTUAL_FREQUENCY = "reference_normal_mode_actual_frequency"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def oscillator_clock(replay_clock, source_clock, original_frequency, imposed_frequency):
    """Retain the first source phase and change only its subsequent clock rate."""
    replay = np.asarray(replay_clock, dtype=np.float64)
    source = np.asarray(source_clock, dtype=np.float64)
    if (
        replay.ndim != 1
        or source.shape != replay.shape
        or len(replay) < 2
        or not np.isfinite(replay).all()
        or not np.isfinite(source).all()
        or not np.isfinite([original_frequency, imposed_frequency]).all()
        or original_frequency <= 0
        or imposed_frequency <= 0
        or np.any(np.diff(replay) <= 0)
        or not np.allclose(np.diff(source), np.diff(replay), atol=1e-9, rtol=0)
    ):
        raise ValueError(
            "Positive frequencies and matching unit-rate source/replay clocks required"
        )
    if original_frequency == imposed_frequency:
        return source.copy()
    return source[0] + (imposed_frequency / original_frequency) * (replay - replay[0])


def replace_normal_oscillation(
    actual, original_normal, original_oscillation, altered_oscillation, normal
):
    """Change normal velocity while retaining actual velocity tangent and gradient."""
    changed_normal = original_normal + (altered_oscillation - original_oscillation)
    return {
        "velocity": actual["velocity"]
        + normal[None] * (changed_normal - actual["normal_velocity"])[..., None],
        "normal_velocity": changed_normal,
        "tangential_gradient": actual["tangential_gradient"].copy(),
    }


def prepare_models(models_directory, component_directory, output_directory):
    """Prepare all 301 endpoints, checking original fits and existing normal control."""
    models = Path(models_directory).resolve()
    previous = Path(component_directory).resolve()
    output = Path(output_directory).resolve()
    report_path = models / "model_report.json"
    original = json.loads(report_path.read_text())
    coefficient_path = models / "harmonic_coefficients.npz"
    if digest(coefficient_path) != original["harmonic_coefficients_sha256"]:
        raise ValueError("Frozen harmonic coefficients changed")
    source_hashes = {str(path): digest(path) for path in (report_path, coefficient_path)}
    with np.load(coefficient_path, allow_pickle=False) as stored:
        fitted = {
            label: {field: stored[f"{label}_{field}"].copy() for field in FIELDS}
            for label in ("coupled", "reference")
        }
    source_clocks, full, geometry, replay_clock = {}, {}, None, None
    for label in fitted:
        path = models / f"{label}_harmonic_endpoints.h5"
        source_hashes[str(path)] = digest(path)
        if source_hashes[str(path)] != original["endpoint_models"][label]["sha256"]:
            raise ValueError(f"Frozen endpoint model changed: {label}")
        with h5py.File(path, "r") as stored:
            if not stored.attrs.get("complete", False):
                raise ValueError("Frozen endpoint model is incomplete")
            candidate = {name: stored[name][:] for name in GEOMETRY}
            current_clock = stored["time"][:]
            if geometry is None:
                geometry, replay_clock = candidate, current_clock
            elif not np.array_equal(current_clock, replay_clock) or any(
                not np.array_equal(value, geometry[name]) for name, value in candidate.items()
            ):
                raise ValueError("Frozen models have different geometry or replay clocks")
            source_clocks[label] = stored["source_time"][:]
            full[label] = {field: stored[field][:] for field in FIELDS}
    if len(replay_clock) != 301 or not np.allclose(np.diff(replay_clock), 0.04, atol=1e-9, rtol=0):
        raise ValueError("Frequency controls require all 301 accepted 0.04 s endpoints")
    component_report_path = previous / "component_model_report.json"
    component_report = json.loads(component_report_path.read_text())
    source_hashes[str(component_report_path)] = digest(component_report_path)
    if any(
        component_report["source_sha256"].get(path) != expected
        for path, expected in source_hashes.items()
        if path != str(component_report_path)
    ):
        raise ValueError("Existing normal control consumed different frozen models")
    previous_path = previous / f"{NORMAL_CONTROL}_endpoints.h5"
    source_hashes[str(previous_path)] = digest(previous_path)
    if (
        source_hashes[str(previous_path)]
        != component_report["endpoint_models"][NORMAL_CONTROL]["sha256"]
    ):
        raise ValueError("Existing reference-normal endpoint control changed")
    with h5py.File(previous_path, "r") as stored:
        if (
            not stored.attrs.get("complete", False)
            or not np.array_equal(stored["time"][:], replay_clock)
            or any(not np.array_equal(stored[name][:], geometry[name]) for name in GEOMETRY)
        ):
            raise ValueError(
                "Existing reference-normal control is incomplete or geometrically different"
            )
        previous_normal = stored["normal_velocity"][:]
    frequencies = original["force_frequency_hz"]
    midpoints = original["fit_midpoint_clock"]
    separated = {
        label: {
            name: components(source_clocks[label], value, frequencies[label], midpoints[label])
            for name, value in fields.items()
        }
        for label, fields in fitted.items()
    }
    reconstruction_error = max(
        float(np.max(abs(dc + oscillation - full[label][name])))
        for label, fields in separated.items()
        for name, (dc, oscillation) in fields.items()
    )
    reference_control_error = float(
        np.max(
            abs(
                previous_normal
                - (
                    separated["coupled"]["normal_velocity"][0]
                    + separated["reference"]["normal_velocity"][1]
                )
            )
        )
    )
    if max(reconstruction_error, reference_control_error) > 1e-12:
        raise ValueError("Frozen coefficients do not reconstruct original endpoint controls")
    specifications = {
        ACTUAL_MODE_REFERENCE_FREQUENCY: (
            "coupled",
            "reference",
            full["coupled"]["normal_velocity"],
        ),
        REFERENCE_MODE_ACTUAL_FREQUENCY: ("reference", "coupled", previous_normal),
    }
    normal, area = geometry["face_normal"], geometry["face_area"]
    if not np.allclose(np.sum(normal**2, axis=1), 1.0, atol=1e-12, rtol=0):
        raise ValueError("Boundary normals are not unit vectors")
    controls, altered_clocks, checks, changes, oscillation_reports = {}, {}, {}, {}, {}
    sides = side_groups(geometry["face_centre"])
    for label, (mode, frequency_source, original_normal) in specifications.items():
        altered_clock = oscillator_clock(
            replay_clock, source_clocks[mode], frequencies[mode], frequencies[frequency_source]
        )
        measured_window = original["source_windows"][mode]
        if (
            altered_clock[0] < measured_window[0] - 1e-9
            or altered_clock[-1] > measured_window[-1] + 1e-9
        ):
            raise ValueError("Altered oscillator leaves the measured source window")
        unchanged_coefficients = fitted[mode]["normal_velocity"][2:].copy()
        _, altered_oscillation = components(
            altered_clock, fitted[mode]["normal_velocity"], frequencies[mode], midpoints[mode]
        )
        baseline_oscillation = separated[mode]["normal_velocity"][1]
        values = replace_normal_oscillation(
            full["coupled"], original_normal, baseline_oscillation, altered_oscillation, normal
        )
        unchanged_dc = values["normal_velocity"] - altered_oscillation
        actual_dc = separated["coupled"]["normal_velocity"][0]
        checks[label] = {
            "finite_fields": all(bool(np.isfinite(value).all()) for value in values.values()),
            "first_normal_endpoint_bitwise_equal_to_original_control": bool(
                np.array_equal(values["normal_velocity"][0], original_normal[0])
            ),
            "first_oscillator_source_clock_bitwise_equal": bool(
                altered_clock[0] == source_clocks[mode][0]
            ),
            "normal_coefficients_bitwise_unchanged": bool(
                np.array_equal(unchanged_coefficients, fitted[mode]["normal_velocity"][2:])
            ),
            "tangential_gradient_bitwise_unchanged": bool(
                np.array_equal(
                    values["tangential_gradient"], full["coupled"]["tangential_gradient"]
                )
            ),
            "normal_projection_maximum_difference": float(
                np.max(
                    abs(
                        np.einsum("tfi,fi->tf", values["velocity"], normal)
                        - values["normal_velocity"]
                    )
                )
            ),
            "gradient_normal_component_maximum": float(
                np.max(abs(np.einsum("tfi,fi->tf", values["tangential_gradient"], normal)))
            ),
            "maximum_signed_normal_flux": float(np.max(abs(values["normal_velocity"] @ area))),
            "actual_normal_dc_trend_maximum_difference": float(
                np.max(abs(unchanged_dc - actual_dc))
            ),
            "actual_velocity_tangent_maximum_difference": float(
                np.max(
                    abs(
                        (values["velocity"] - full["coupled"]["velocity"])
                        - normal[None]
                        * (values["normal_velocity"] - full["coupled"]["normal_velocity"])[
                            ..., None
                        ]
                    )
                )
            ),
        }
        if any(
            not value if isinstance(value, bool) else value > 1e-9
            for value in checks[label].values()
        ):
            raise ValueError(
                "A normal-frequency control failed its field or unchanged-input checks"
            )
        harmonic_amplitudes = np.hypot(unchanged_coefficients[::2], unchanged_coefficients[1::2])
        oscillation_reports[label] = {
            "normal_mode_source": mode,
            "imposed_frequency_source": frequency_source,
            "original_mode_frequency_hz": frequencies[mode],
            "imposed_normal_frequency_hz": frequencies[frequency_source],
            "oscillator_clock_rate": frequencies[frequency_source] / frequencies[mode],
            "original_source_clock_window": source_clocks[mode][[0, -1]].tolist(),
            "altered_oscillator_source_clock_window": altered_clock[[0, -1]].tolist(),
            "measured_source_clock_window": measured_window,
            "initial_fundamental_phase_radians": float(
                2 * np.pi * frequencies[mode] * (source_clocks[mode][0] - midpoints[mode])
            ),
            "harmonic_amplitude_area_weighted_rms": [
                float(np.sqrt(np.average(value**2, weights=area))) for value in harmonic_amplitudes
            ],
            "harmonic_amplitude_maximum_change": 0.0,
            "normal_coefficient_sha256": hashlib.sha256(
                unchanged_coefficients.tobytes()
            ).hexdigest(),
        }
        controls[label], altered_clocks[label] = values, altered_clock
        changes[label] = {
            side: weighted_rms(
                values["normal_velocity"][:, selected]
                - full["coupled"]["normal_velocity"][:, selected],
                area[selected],
            )
            for side, selected in sides.items()
        }
    output.mkdir(parents=True, exist_ok=False)
    endpoint_models = {}
    for label, values in controls.items():
        mode, frequency_source, _ = specifications[label]
        path = output / f"{label}_endpoints.h5"
        with h5py.File(path, "w") as stored:
            stored.attrs["complete"] = False
            stored.attrs["schema"] = "openonda-normal-frequency-boundary-endpoint-model/1"
            stored.attrs["trace_origin"] = label
            stored.attrs["dc_trend_source"] = "coupled"
            stored.attrs["normal_mode_source"] = mode
            stored.attrs["normal_frequency_hz"] = frequencies[frequency_source]
            stored.attrs["normal_oscillator_clock_rate"] = (
                frequencies[frequency_source] / frequencies[mode]
            )
            stored.attrs["tangential_input_source"] = "coupled with original source clock"
            stored.attrs["phase_policy"] = (
                "Original normal-mode initial phase and spatial phases; only oscillator clock rate changes"
            )
            stored.attrs["velocity_policy"] = (
                "Actual velocity plus normal*(newUn-actualUn); actual tangent retained"
            )
            stored.attrs["source_sha256"] = json.dumps(source_hashes, sort_keys=True)
            stored.create_dataset("time", data=replay_clock)
            stored.create_dataset("source_time", data=source_clocks["coupled"])
            for source, source_clock in source_clocks.items():
                stored.create_dataset(f"{source}_original_source_time", data=source_clock)
            stored.create_dataset("normal_oscillator_source_time", data=altered_clocks[label])
            stored.create_dataset(
                "normal_oscillation_coefficients", data=fitted[mode]["normal_velocity"][2:]
            )
            for name, value in geometry.items():
                stored.create_dataset(name, data=value)
            for name, value in values.items():
                stored.create_dataset(name, data=value, compression="gzip", shuffle=True)
            stored.attrs["complete"] = True
        endpoint_models[label] = {
            "path": str(path),
            "sha256": digest(path),
            "accepted_endpoint_count": len(replay_clock),
        }
    report = {
        "schema": "openonda-normal-frequency-boundary-input-controls/1",
        "status": "prepared endpoint models only; no replay prepared or solver advanced",
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": source_hashes,
        "helper_sha256": digest(Path(__file__).resolve()),
        "decomposition_helper_sha256": digest(
            Path(__file__).with_name("prepare_boundary_hybrid_inputs.py")
        ),
        "design_helper_sha256": digest(
            Path(__file__).with_name("prepare_boundary_input_replay.py")
        ),
        "adapter_source_sha256": digest(Path(__file__).with_name("replay_boundary_inputs.py")),
        "original_frequency_hz": frequencies,
        "fit_midpoint_clock": midpoints,
        "original_global_phase_translation": original["coupled_phase_mapping"],
        "replay_clock_window": replay_clock[[0, -1]].tolist(),
        "endpoint_models": endpoint_models,
        "normal_oscillation_models": oscillation_reports,
        "field_checks": checks,
        "normal_difference_from_actual_rms_by_boundary": changes,
        "original_model_reconstruction_maximum_difference": reconstruction_error,
        "existing_reference_normal_reconstruction_maximum_difference": reference_control_error,
        "scope": "Two missing normal-input controls in a 2x2 comparison of spatial harmonic coefficients and frequency. Both retain actual DC/trend, actual tangential velocity and gradient on their original clocks. Only normal oscillator rate changes; harmonic coefficients and initial normal phase do not change. No amplitude rescaling or solver change.",
        "limitations": original["limitations"]
        + [
            "Changing normal frequency changes its subsequent temporal phase relative to the unchanged tangential inputs. This is deliberate and must not be interpreted as changing all boundary components coherently.",
            "No force result exists at preparation. Compare complete force cycles after settling; a finite replay cannot establish mature autonomous coupled recovery.",
        ],
        "preparation_recipe": {
            label: [
                "/home/flavio-martins/anaconda3/envs/OpenONDA/bin/python",
                "-m",
                "tests.support.cylinder.replay_boundary_inputs",
                "prepare",
                "--directory",
                f"/tmp/openonda-{label.replace('_', '-')}-20261006",
                "--source",
                record["path"],
                "--source-receipt",
                str(output / "normal_frequency_model_report.json"),
                "--reference-directory",
                "/tmp/openonda-cylinder-boundary-study-20261005",
                "--origin",
                label,
            ]
            for label, record in endpoint_models.items()
        },
    }
    if any(digest(path) != expected for path, expected in source_hashes.items()):
        raise RuntimeError("Frozen inputs changed during normal-frequency preparation")
    (output / "normal_frequency_model_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-directory", type=Path, default=MODELS)
    parser.add_argument("--component-directory", type=Path, default=COMPONENTS)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    output = args.output_directory or STUDY / (
        "normal_frequency_inputs_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    report = prepare_models(args.models_directory, args.component_directory, output)
    print(
        json.dumps(
            {
                "directory": str(output.resolve()),
                "status": report["status"],
                "field_checks": report["field_checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
