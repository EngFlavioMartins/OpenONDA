"""Prepare separate normal-velocity and tangential-gradient forcing controls.

Both retain the actual coupled mean and linear trend. Each replaces only one
oscillatory mixed-boundary component with the frozen reference oscillation,
using the original source clocks and global phase translation. Preparation
writes endpoint models only; it never prepares replay inputs or starts a solver.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .prepare_boundary_hybrid_inputs import components
from .prepare_boundary_input_replay import side_groups, weighted_rms

STUDY = Path(__file__).resolve().parent / "force_boundary_study/20261005T214514Z"
MODELS = STUDY / "boundary_input_replay_20261005T235543Z"
FIELDS = ("velocity", "normal_velocity", "tangential_gradient")
GEOMETRY = ("face_centre", "face_normal", "face_area")
NORMAL_CONTROL = "actual_mean_reference_normal_oscillation"
GRADIENT_CONTROL = "actual_mean_reference_gradient_oscillation"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def component_endpoint_fields(actual_dc, actual_oscillation, reference_oscillation, normal):
    """Keep actual DC/trend while replacing exactly one mixed oscillation."""
    actual = {name: actual_dc[name] + actual_oscillation[name] for name in FIELDS}
    normal_velocity = actual_dc["normal_velocity"] + reference_oscillation["normal_velocity"]
    normal_model = {
        "velocity": actual["velocity"]
        + normal[None, :, :] * (normal_velocity - actual["normal_velocity"])[..., None],
        "normal_velocity": normal_velocity,
        "tangential_gradient": actual["tangential_gradient"].copy(),
    }
    gradient_model = {
        "velocity": actual["velocity"].copy(),
        "normal_velocity": actual["normal_velocity"].copy(),
        "tangential_gradient": actual_dc["tangential_gradient"]
        + reference_oscillation["tangential_gradient"],
    }
    return {NORMAL_CONTROL: normal_model, GRADIENT_CONTROL: gradient_model}


def prepare_models(models_directory, output_directory):
    """Hash-check frozen fits and write the two complete endpoint models."""
    models = Path(models_directory).resolve()
    output = Path(output_directory).resolve()
    report_path = models / "model_report.json"
    original = json.loads(report_path.read_text())
    coefficient_path = models / "harmonic_coefficients.npz"
    if digest(coefficient_path) != original["harmonic_coefficients_sha256"]:
        raise ValueError("Frozen harmonic coefficients changed")
    source_hashes = {str(path): digest(path) for path in (report_path, coefficient_path)}
    with np.load(coefficient_path, allow_pickle=False) as stored:
        fitted = {
            label: {name: stored[f"{label}_{name}"].copy() for name in FIELDS}
            for label in ("reference", "coupled")
        }
    source_clock, full, geometry, clock = {}, {}, None, None
    for label in fitted:
        path = models / f"{label}_harmonic_endpoints.h5"
        source_hashes[str(path)] = digest(path)
        if source_hashes[str(path)] != original["endpoint_models"][label]["sha256"]:
            raise ValueError(f"Frozen endpoint model changed: {label}")
        with h5py.File(path, "r") as stored:
            if not stored.attrs.get("complete", False):
                raise ValueError("Frozen endpoint model is incomplete")
            current_geometry = {name: stored[name][:] for name in GEOMETRY}
            current_clock = stored["time"][:]
            if geometry is None:
                geometry, clock = current_geometry, current_clock
            elif any(
                not np.array_equal(value, geometry[name])
                for name, value in current_geometry.items()
            ) or not np.array_equal(current_clock, clock):
                raise ValueError("Frozen geometry or replay clock differs between sources")
            source_clock[label] = stored["source_time"][:]
            full[label] = {name: stored[name][:] for name in FIELDS}
    if len(clock) != 301 or not np.allclose(np.diff(clock), 0.04, rtol=0, atol=1e-9):
        raise ValueError("Component controls require all301 accepted .04s endpoints")
    separated = {
        label: {
            name: components(
                source_clock[label],
                value,
                original["force_frequency_hz"][label],
                original["fit_midpoint_clock"][label],
            )
            for name, value in fields.items()
        }
        for label, fields in fitted.items()
    }
    reconstruction_error = max(
        float(np.max(abs(dc + oscillation - full[label][name])))
        for label, fields in separated.items()
        for name, (dc, oscillation) in fields.items()
    )
    if reconstruction_error > 1e-12:
        raise ValueError("Component algebra does not reconstruct the frozen endpoints")
    actual_dc = {name: value[0] for name, value in separated["coupled"].items()}
    actual_oscillation = {name: value[1] for name, value in separated["coupled"].items()}
    reference_oscillation = {name: value[1] for name, value in separated["reference"].items()}
    normal, area = geometry["face_normal"], geometry["face_area"]
    if not np.allclose(np.sum(normal**2, axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Frozen boundary normals are not unit vectors")
    controls = component_endpoint_fields(
        actual_dc, actual_oscillation, reference_oscillation, normal
    )
    checks, differences = {}, {}
    sides = side_groups(geometry["face_centre"])
    for label, values in controls.items():
        checks[label] = {
            "finite_fields": all(bool(np.isfinite(value).all()) for value in values.values()),
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
            "velocity_tangential_component_change_maximum": float(
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
        if not checks[label]["finite_fields"] or any(
            value > 1e-9 for name, value in checks[label].items() if name != "finite_fields"
        ):
            raise ValueError("Component field projection, flux or finite-state check failed")
        differences[label] = {
            name: {
                side: weighted_rms(
                    values[name][:, selected] - full["coupled"][name][:, selected], area[selected]
                )
                for side, selected in sides.items()
            }
            for name in FIELDS
        }
    specifications = {
        NORMAL_CONTROL: {"normal_velocity": "reference", "tangential_gradient": "coupled"},
        GRADIENT_CONTROL: {"normal_velocity": "coupled", "tangential_gradient": "reference"},
    }
    output.mkdir(parents=True, exist_ok=False)
    endpoint_models = {}
    for label, values in controls.items():
        path = output / f"{label}_endpoints.h5"
        with h5py.File(path, "w") as stored:
            stored.attrs["complete"] = False
            stored.attrs["schema"] = "openonda-component-harmonic-boundary-endpoint-model/1"
            stored.attrs["trace_origin"] = label
            stored.attrs["dc_trend_source"] = "coupled"
            stored.attrs["oscillation_sources"] = json.dumps(specifications[label], sort_keys=True)
            stored.attrs["phase_policy"] = (
                "Original component clocks and global phase; no additional phase or amplitude rescaling"
            )
            stored.attrs["velocity_policy"] = (
                "Actual full velocity plus normal*(newUn-actualUn); actual tangent retained"
            )
            stored.attrs["component_frequency_hz"] = json.dumps(
                original["force_frequency_hz"], sort_keys=True
            )
            stored.attrs["source_sha256"] = json.dumps(source_hashes, sort_keys=True)
            stored.create_dataset("time", data=clock)
            stored.create_dataset("source_time", data=source_clock["coupled"])
            for source, values_clock in source_clock.items():
                stored.create_dataset(f"{source}_source_time", data=values_clock)
            for name, value in geometry.items():
                stored.create_dataset(name, data=value)
            for name, value in values.items():
                stored.create_dataset(name, data=value, compression="gzip", shuffle=True)
            stored.attrs["complete"] = True
        endpoint_models[label] = {
            "path": str(path),
            "sha256": digest(path),
            "accepted_endpoint_count": len(clock),
        }
    report = {
        "schema": "openonda-boundary-component-input-models/1",
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
        "component_models": original["endpoint_models"],
        "component_frequency_hz": original["force_frequency_hz"],
        "component_fit_midpoint_clock": original["fit_midpoint_clock"],
        "component_source_time_windows": {
            label: values[[0, -1]].tolist() for label, values in source_clock.items()
        },
        "replay_clock": clock[[0, -1]].tolist(),
        "coupled_global_phase_translation": original["coupled_phase_mapping"],
        "model_components": {
            label: {"dc_trend_source": "coupled", "oscillation_sources": value}
            for label, value in specifications.items()
        },
        "endpoint_models": endpoint_models,
        "projection_checks": checks,
        "difference_from_actual_harmonic_inputs": differences,
        "component_reconstruction_maximum_difference": reconstruction_error,
        "scope": "Only one oscillatory mixed-boundary component changes per model. Actual DC/trend coefficients remain unchanged; this does not claim the window average is unchanged when frequencies differ. Actual velocity tangent remains unchanged. No further phase alignment, scaling or native algorithm change.",
        "interpretation_gate": "Reference harmonic forcing must preserve reference force response, and actual harmonic forcing must reproduce the measured deficit. Finite settling and complete force-cycle counts remain required. No component force result exists yet.",
        "limitations": original["limitations"],
    }
    if any(digest(path) != expected for path, expected in source_hashes.items()):
        raise RuntimeError("Frozen inputs changed during component preparation")
    destination = output / "component_model_report.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-directory", type=Path, default=MODELS)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    output = args.output_directory or STUDY / (
        "boundary_component_inputs_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    report = prepare_models(args.models_directory, output)
    print(
        json.dumps(
            {
                "directory": str(output.resolve()),
                "status": report["status"],
                "projection_checks": report["projection_checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
