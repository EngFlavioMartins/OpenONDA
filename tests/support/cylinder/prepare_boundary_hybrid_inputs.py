"""Swap frozen boundary mean/trend and oscillatory components in test inputs.

Both component clocks are preserved from the existing paired harmonic models.
The helper performs host-only algebra and prepares private-replay inputs; it
never advances a solver or edits the adapter used by an active control.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

import h5py
import numpy as np

from .prepare_boundary_input_replay import design, side_groups, weighted_rms
from .replay_boundary_inputs import digest, prepare_input, write_json

STUDY = Path(__file__).resolve().parent / "force_boundary_study/20261005T214514Z"
MODELS = STUDY / "boundary_input_replay_20261005T235543Z"
REFERENCE = Path("/tmp/openonda-cylinder-boundary-study-20261005")
FIELDS = ("velocity", "normal_velocity", "tangential_gradient")


def components(source_clock, fitted, frequency, midpoint):
    matrix = design(source_clock, frequency, midpoint)
    shape = (len(source_clock), *fitted.shape[1:])
    dc_trend = (matrix[:, :2] @ fitted[:2].reshape(2, -1)).reshape(shape)
    oscillation = (matrix[:, 2:] @ fitted[2:].reshape(4, -1)).reshape(shape)
    return dc_trend, oscillation


def write_model(path, clock, source_clocks, geometry, values, dc_label, oscillation_label, report):
    with h5py.File(path, "w") as stored:
        stored.attrs["complete"] = False
        stored.attrs["schema"] = "openonda-hybrid-harmonic-boundary-endpoint-model/1"
        stored.attrs["trace_origin"] = (
            f"{dc_label} mean/linear trend with {oscillation_label} oscillatory boundary modes"
        )
        stored.attrs["dc_trend_source"] = dc_label
        stored.attrs["oscillation_source"] = oscillation_label
        stored.attrs["oscillation_frequency_hz"] = report["force_frequency_hz"][oscillation_label]
        stored.attrs["phase_policy"] = (
            "Identical component clocks to the original paired models; no further phase alignment or amplitude rescaling"
        )
        stored.attrs["model"] = (
            "DC mean + linear trend from one source, fundamental and second harmonic from the other source"
        )
        stored.create_dataset("time", data=clock)
        stored.create_dataset("source_time", data=source_clocks[oscillation_label])
        stored.create_dataset("dc_trend_source_time", data=source_clocks[dc_label])
        stored.create_dataset("oscillation_source_time", data=source_clocks[oscillation_label])
        for name, value in geometry.items():
            stored.create_dataset(name, data=value)
        for name, value in values.items():
            stored.create_dataset(name, data=value, compression="gzip", shuffle=True)
        stored.attrs["complete"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-directory", type=Path, default=MODELS)
    parser.add_argument("--reference-directory", type=Path, default=REFERENCE)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    models = args.models_directory.resolve()
    reference = args.reference_directory.resolve()
    output = args.output_directory or STUDY / (
        "hybrid_boundary_inputs_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report_path = models / "model_report.json"
    original = json.loads(report_path.read_text())
    coefficient_path = models / "harmonic_coefficients.npz"
    if digest(coefficient_path) != original["harmonic_coefficients_sha256"]:
        raise ValueError("Frozen harmonic coefficients changed")
    with np.load(coefficient_path, allow_pickle=False) as stored:
        fitted = {
            label: {field: stored[f"{label}_{field}"].copy() for field in FIELDS}
            for label in ("reference", "coupled")
        }
    source_clock, full, geometry = {}, {}, None
    source_hashes = {
        str(report_path): digest(report_path),
        str(coefficient_path): digest(coefficient_path),
    }
    for label in fitted:
        path = models / f"{label}_harmonic_endpoints.h5"
        if digest(path) != original["endpoint_models"][label]["sha256"]:
            raise ValueError(f"Frozen endpoint model changed: {label}")
        source_hashes[str(path)] = digest(path)
        with h5py.File(path, "r") as stored:
            if not stored.attrs["complete"]:
                raise ValueError("Frozen endpoint model is incomplete")
            candidate_geometry = {
                name: stored[name][:] for name in ("face_centre", "face_normal", "face_area")
            }
            current_clock = stored["time"][:]
            if geometry is None:
                geometry = candidate_geometry
                clock = current_clock
            elif any(
                not np.array_equal(value, geometry[name])
                for name, value in candidate_geometry.items()
            ) or not np.array_equal(current_clock, clock):
                raise ValueError("Frozen component geometry or replay clocks differ")
            source_clock[label] = stored["source_time"][:]
            full[label] = {field: stored[field][:] for field in FIELDS}
    if len(clock) != 301:
        raise ValueError("Hybrid models require the complete301 accepted endpoint schedule")
    separated = {
        label: {
            field: components(
                source_clock[label],
                value,
                original["force_frequency_hz"][label],
                original["fit_midpoint_clock"][label],
            )
            for field, value in series.items()
        }
        for label, series in fitted.items()
    }
    reconstruction_error = max(
        float(np.max(abs(dc + oscillation - full[label][field])))
        for label, series in separated.items()
        for field, (dc, oscillation) in series.items()
    )
    if reconstruction_error > 1e-12:
        raise ValueError("Component decomposition differs from the original paired endpoints")
    combinations = {
        "reference_mean_actual_oscillation": ("reference", "coupled"),
        "actual_mean_reference_oscillation": ("coupled", "reference"),
    }
    hybrid = {
        label: {
            field: separated[dc_label][field][0] + separated[oscillation_label][field][1]
            for field in FIELDS
        }
        for label, (dc_label, oscillation_label) in combinations.items()
    }
    normal, area = geometry["face_normal"], geometry["face_area"]
    sides = side_groups(geometry["face_centre"])
    projection, component_difference = {}, {}
    for label, values in hybrid.items():
        projection[label] = {
            "velocity_normal_projection_maximum_difference": float(
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
        }
        if max(projection[label].values()) > 1e-9 or not all(
            np.isfinite(value).all() for value in values.values()
        ):
            raise ValueError("Hybrid field projection, flux or finite-state check failed")
    for field in FIELDS:
        component_difference[field] = {}
        for side, selected in sides.items():
            dc_difference = (
                separated["reference"][field][0][:, selected]
                - separated["coupled"][field][0][:, selected]
            )
            oscillation_difference = (
                separated["reference"][field][1][:, selected]
                - separated["coupled"][field][1][:, selected]
            )
            component_difference[field][side] = {
                "reference_minus_actual_dc_trend_rms": weighted_rms(dc_difference, area[selected]),
                "reference_minus_actual_oscillation_rms": weighted_rms(
                    oscillation_difference, area[selected]
                ),
                "reference_dc_trend_time_and_area_mean": np.average(
                    separated["reference"][field][0][:, selected], weights=area[selected], axis=1
                )
                .mean(axis=0)
                .tolist(),
                "actual_dc_trend_time_and_area_mean": np.average(
                    separated["coupled"][field][0][:, selected], weights=area[selected], axis=1
                )
                .mean(axis=0)
                .tolist(),
            }
    sum_error = max(
        float(
            np.max(
                abs(
                    hybrid["reference_mean_actual_oscillation"][field]
                    + hybrid["actual_mean_reference_oscillation"][field]
                    - full["reference"][field]
                    - full["coupled"][field]
                )
            )
        )
        for field in FIELDS
    )
    model_paths = {}
    for label, (dc_label, oscillation_label) in combinations.items():
        path = output / f"{label}_endpoints.h5"
        write_model(
            path,
            clock,
            source_clock,
            geometry,
            hybrid[label],
            dc_label,
            oscillation_label,
            original,
        )
        model_paths[label] = path
    report = {
        "schema": "openonda-paired-hybrid-boundary-input-models/1",
        "status": "prepared; no solver constructed or advanced",
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": source_hashes,
        "helper_sha256": digest(Path(__file__).resolve()),
        "adapter_source_sha256": digest(Path(__file__).with_name("replay_boundary_inputs.py")),
        "component_models": original["endpoint_models"],
        "component_frequency_hz": original["force_frequency_hz"],
        "component_fit_midpoint_clock": original["fit_midpoint_clock"],
        "component_source_time_windows": {
            label: values[[0, -1]].tolist() for label, values in source_clock.items()
        },
        "replay_clock": clock[[0, -1]].tolist(),
        "coupled_global_phase_translation": original["coupled_phase_mapping"],
        "model_components": {
            label: {"dc_trend_source": dc_label, "oscillation_source": oscillation_label}
            for label, (dc_label, oscillation_label) in combinations.items()
        },
        "endpoint_models": {
            label: {
                "path": str(path),
                "sha256": digest(path),
                "accepted_endpoint_count": len(clock),
            }
            for label, path in model_paths.items()
        },
        "projection_checks": projection,
        "component_differences": component_difference,
        "component_recombination_maximum_difference_from_original_endpoints": reconstruction_error,
        "hybrid_sum_minus_original_sum_maximum_component_difference": sum_error,
        "interpretation_gate": "Do not interpret these hybrid controls unless the reference harmonic replay retains the exact-fed force amplitudes and the actual harmonic replay reproduces the material force deficit. There are no hybrid force results yet.",
        "scope": "Test-only forcing decomposition. DC/trend and first/second-harmonic components use exactly the original source fits and research-clock translation. No amplitude rescaling or additional phase alignment; no new native or tutorial algorithm.",
        "limitations": original["limitations"],
    }
    if any(digest(path) != expected for path, expected in source_hashes.items()):
        raise RuntimeError("A frozen source changed during hybrid preparation")
    hybrid_report = output / "hybrid_model_report.json"
    write_json(hybrid_report, report)
    prepared = {}
    for label, path in model_paths.items():
        prepared[label] = prepare_input(
            output / f"{label}_replay",
            source=path,
            group="",
            reference=reference,
            origin=label,
            source_receipt=hybrid_report,
        )
    write_json(
        output / "prepared_replays.json",
        {
            label: {
                "directory": str(output / f"{label}_replay"),
                "boundary_trace_sha256": receipt["boundary_trace_sha256"],
                "initial_state_sha256": receipt["initial_state_sha256"],
                "status": receipt["status"],
            }
            for label, receipt in prepared.items()
        },
    )
    print(
        json.dumps(
            {
                "directory": str(output),
                "status": report["status"],
                "projection_checks": projection,
                "component_recombination_error": reconstruction_error,
                "hybrid_sum_error": sum_error,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
