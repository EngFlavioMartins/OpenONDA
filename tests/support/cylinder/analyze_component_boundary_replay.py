"""Independently verify a completed mixed-boundary component forcing control.

The algebra uses the frozen fit coefficients directly rather than the endpoint
preparation helper. It checks both the 301 endpoint rows and the 1501 native
FVM rows, then preserves the already audited complete-cycle force comparison.
No solver is initialized or advanced.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

import h5py
import numpy as np

from .compare_boundary_input_replays import STUDY, digest, write_json

FIELDS = ("velocity", "normal_velocity", "tangential_gradient")
GEOMETRY = ("face_centre", "face_normal", "face_area")
LABELS = ("reference_normal_oscillation", "reference_gradient_oscillation")


def separate_coefficients(coefficients, clock, frequency, midpoint):
    """Evaluate DC/trend and both measured oscillatory modes directly."""
    elapsed = clock - midpoint
    phase = 2 * np.pi * frequency * elapsed
    mean_design = np.column_stack((np.ones(len(clock)), elapsed))
    mode_design = np.column_stack(
        (np.cos(phase), np.sin(phase), np.cos(2 * phase), np.sin(2 * phase))
    )
    dc_trend = np.tensordot(mean_design, coefficients[:2], axes=(1, 0))
    oscillation = np.tensordot(mode_design, coefficients[2:], axes=(1, 0))
    return dc_trend, oscillation


def audit_component(comparison_path, label):
    comparison_path = Path(comparison_path).resolve()
    comparison = json.loads(comparison_path.read_text())
    provenance = comparison["provenance"][label]
    receipt = provenance["trace_receipt"]
    source = Path(receipt["source_hdf5"])
    directory = Path(provenance["result_report"]).parent.parent
    models = STUDY / "boundary_input_replay_20261005T235543Z"
    model_report_path = models / "model_report.json"
    coefficient_path = models / "harmonic_coefficients.npz"
    model_report = json.loads(model_report_path.read_text())
    source_hashes = {
        str(path): digest(path)
        for path in (
            source,
            coefficient_path,
            model_report_path,
            models / "coupled_harmonic_endpoints.h5",
            models / "reference_harmonic_endpoints.h5",
            directory / "boundary_trace.h5",
            directory / "boundary_trace.json",
            comparison_path,
        )
    }
    if (
        source_hashes[str(source)] != receipt["source_hdf5_sha256"]
        or source_hashes[str(coefficient_path)] != model_report["harmonic_coefficients_sha256"]
        or source_hashes[str(directory / "boundary_trace.h5")] != receipt["boundary_trace_sha256"]
    ):
        raise ValueError("Component control source differs from frozen evidence")
    with np.load(coefficient_path, allow_pickle=False) as stored:
        coefficients = {
            fluid: {field: stored[f"{fluid}_{field}"].copy() for field in FIELDS}
            for fluid in ("reference", "coupled")
        }
    source_clocks, full = {}, {}
    for fluid in coefficients:
        path = models / f"{fluid}_harmonic_endpoints.h5"
        if source_hashes[str(path)] != model_report["endpoint_models"][fluid]["sha256"]:
            raise ValueError("Original harmonic endpoints differ from frozen coefficients")
        with h5py.File(path, "r") as stored:
            source_clocks[fluid] = stored["source_time"][:]
            full[fluid] = {field: stored[field][:] for field in FIELDS}
            if fluid == "coupled":
                geometry = {name: stored[name][:] for name in GEOMETRY}
                clock = stored["time"][:]
    components = {
        fluid: {
            field: separate_coefficients(
                value,
                source_clocks[fluid],
                model_report["force_frequency_hz"][fluid],
                model_report["fit_midpoint_clock"][fluid],
            )
            for field, value in fields.items()
        }
        for fluid, fields in coefficients.items()
    }
    actual = {field: dc + modes for field, (dc, modes) in components["coupled"].items()}
    expected = {field: value.copy() for field, value in actual.items()}
    normals = geometry["face_normal"]
    if label == "reference_normal_oscillation":
        expected["normal_velocity"] = (
            components["coupled"]["normal_velocity"][0]
            + components["reference"]["normal_velocity"][1]
        )
        expected["velocity"] += (
            normals[None] * (expected["normal_velocity"] - actual["normal_velocity"])[..., None]
        )
    else:
        expected["tangential_gradient"] = (
            components["coupled"]["tangential_gradient"][0]
            + components["reference"]["tangential_gradient"][1]
        )
    with h5py.File(source, "r") as stored:
        endpoint = {field: stored[field][:] for field in FIELDS}
        clocks_match = bool(
            np.array_equal(stored["time"][:], clock)
            and np.array_equal(stored["source_time"][:], source_clocks["coupled"])
            and all(
                np.array_equal(stored[f"{fluid}_source_time"][:], source_clocks[fluid])
                for fluid in source_clocks
            )
        )
        geometry_match = all(
            np.array_equal(stored[name][:], value) for name, value in geometry.items()
        )
    reconstruction_errors = {
        field: float(np.max(abs(value - expected[field]))) for field, value in endpoint.items()
    }
    actual_reconstruction_error = max(
        float(np.max(abs(actual[field] - full["coupled"][field]))) for field in FIELDS
    )
    actual_tangent = actual["velocity"] - normals[None] * actual["normal_velocity"][..., None]
    model_tangent = endpoint["velocity"] - normals[None] * endpoint["normal_velocity"][..., None]
    native_steps = np.arange(1501)
    lower = np.maximum((native_steps - 1) // 5, 0)
    fraction = (native_steps - 5 * lower) / 5
    native_errors = {}
    with h5py.File(directory / "boundary_trace.h5", "r") as stored:
        for field, values in endpoint.items():
            weights = fraction.reshape((-1,) + (1,) * (values.ndim - 1))
            interpolated = (1 - weights) * values[lower] + weights * values[lower + 1]
            native_errors[field] = float(np.max(abs(interpolated - stored[field][:])))
    checks = {
        "301_endpoint_count": len(clock) == 301,
        "source_and_research_clocks_bitwise_equal": clocks_match,
        "outer_face_geometry_bitwise_equal": geometry_match,
        "endpoint_algebra_maximum_errors": reconstruction_errors,
        "actual_coefficients_reconstruction_maximum_error": actual_reconstruction_error,
        "actual_velocity_tangent_retained_maximum_error": float(
            np.max(abs(model_tangent - actual_tangent))
        ),
        "maximum_signed_normal_flux": float(
            np.max(abs(endpoint["normal_velocity"] @ geometry["face_area"]))
        ),
        "1501_native_row_interpolation_maximum_errors": native_errors,
    }
    if (
        not clocks_match
        or not geometry_match
        or len(clock) != 301
        or max(reconstruction_errors.values()) > 1e-12
        or max(native_errors.values()) > 1e-12
        or actual_reconstruction_error > 1e-12
        or checks["actual_velocity_tangent_retained_maximum_error"] > 1e-12
        or checks["maximum_signed_normal_flux"] > 1e-12
    ):
        raise ValueError("Independent component/channel or interpolation identity failed")
    if any(digest(path) != expected_hash for path, expected_hash in source_hashes.items()):
        raise ValueError("Frozen component evidence changed during its independent audit")
    ratios = comparison["relative_to_native_reference_flow"]
    ratio_to_actual = {
        field: {
            quantity: ratios[label][field][quantity] / ratios["coupled_harmonic"][field][quantity]
            for quantity in ratios[label][field]
        }
        for field in ratios[label]
    }
    return {
        "schema": "openonda-independent-boundary-component-review/1",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "label": label,
        "analysis_source_sha256": digest(Path(__file__).resolve()),
        "source_sha256": source_hashes,
        "independent_component_checks": checks,
        "native_force_and_input_audit": provenance,
        "complete_cycle_statistics": comparison["complete_cycle_statistics"][label],
        "independent_direct_extrema_and_cycle_trend": comparison[
            "independent_direct_extrema_and_cycle_trend"
        ][label],
        "relative_to_native_reference_flow": ratios[label],
        "relative_to_actual_harmonic_mixed_response": ratio_to_actual,
        "interpretation_scope": (
            "Only one oscillatory boundary channel is substituted at fixed actual DC/trend, "
            "native initialization, pressure condition and temporal schedule. Normal substitution "
            "changes the measured spatial waveform, spatial phase and frequency together. These "
            "finite forced responses identify channel sensitivity, not a unique VPM defect or "
            "self-sustained production recovery."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--label", required=True, choices=LABELS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_component(args.comparison, args.label)
    output = args.output or args.comparison.parent / f"independent_{args.label}_receipt.json"
    if output.exists():
        raise ValueError("Preserve the existing independent component review")
    write_json(output, result)
    print(
        json.dumps(
            {"output": str(output), "checks": result["independent_component_checks"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
