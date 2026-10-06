"""Align frozen outer-boundary harmonics to each run's own force phase.

The complex convention is u(t)=Re(C exp(i omega (t-t_mid))). First harmonics
are aligned to the lift fundamental; second harmonics to drag's second harmonic.
This removes the arbitrary difference between coupled and reference time origins.
All calculations use the previously frozen boundary fields and force CSVs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .analyze_boundary_harmonics import design_matrix


def fit_coefficient(time, values, frequency):
    return np.linalg.lstsq(
        design_matrix(time, frequency), values.reshape(len(time), -1), rcond=None
    )[0].reshape((6, *values.shape[1:]))


def complex_coefficient(coefficient, order):
    first = 2 * order
    return coefficient[first] - 1j * coefficient[first + 1]


def weighted_product(first, second, weights):
    product = first * np.conjugate(second)
    if product.ndim > 1:
        product = np.sum(product, axis=-1)
    return np.dot(product, weights) / np.sum(weights)


def weighted_rms(values, weights):
    return float(np.sqrt(weighted_product(values, values, weights).real))


def wrapped_degrees(value):
    return float((np.rad2deg(value) + 180) % 360 - 180)


def phase_statistics(coupled, reference, weights):
    norm_coupled = weighted_rms(coupled, weights)
    norm_reference = weighted_rms(reference, weights)
    product = weighted_product(coupled, reference, weights)
    gain = product / norm_reference**2
    scale = norm_reference / norm_coupled
    return {
        "coupled_harmonic_rms": norm_coupled,
        "reference_harmonic_rms": norm_reference,
        "amplitude_ratio": norm_coupled / norm_reference,
        "complex_projection_gain_real": float(gain.real),
        "complex_projection_gain_imaginary": float(gain.imag),
        "complex_projection_gain_magnitude": float(abs(gain)),
        "force_aligned_projection_phase_difference_degrees": wrapped_degrees(np.angle(gain)),
        "complex_shape_coherence": float(abs(product) / (norm_coupled * norm_reference)),
        "normalised_complex_rms_difference": weighted_rms(coupled - reference, weights)
        / norm_reference,
        "amplitude_rescaled_complex_rms_difference": weighted_rms(
            scale * coupled - reference, weights
        )
        / norm_reference,
        "normalised_residual_after_optimal_complex_projection": weighted_rms(
            coupled - gain * reference, weights
        )
        / norm_coupled,
    }


def mean_statistics(coupled, reference, weights):
    difference = coupled - reference
    mean = np.average(difference, axis=0, weights=weights)
    magnitude = abs(difference) if difference.ndim == 1 else np.linalg.norm(difference, axis=-1)
    reference_norm = weighted_rms(reference, weights)
    return {
        "coupled_signed_area_mean": np.average(coupled, axis=0, weights=weights).tolist(),
        "reference_signed_area_mean": np.average(reference, axis=0, weights=weights).tolist(),
        "signed_area_mean_difference": mean.tolist(),
        "rms_mean_difference": weighted_rms(difference, weights),
        "normalised_rms_mean_difference": weighted_rms(difference, weights) / reference_norm,
        "maximum_mean_difference_magnitude": float(np.max(magnitude)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    report_path = directory / "boundary_harmonic_statistics.json"
    report = json.loads(report_path.read_text())
    with np.load(directory / "boundary_harmonic_fields.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    centre = arrays["face_centre"]
    area = arrays["face_area"]
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
    coefficient = {}
    force_phase = {}
    phase_records = {}
    for label in ("coupled", "reference"):
        time = arrays[f"{label}_time"]
        frequency = report["force_frequency_fits"][label]["fundamental_frequency"]
        forces = np.atleast_1d(
            np.genfromtxt(
                io.BytesIO((directory / f"snapshots/{label}_forces.csv").read_bytes()),
                delimiter=",",
                names=True,
            )
        )
        selected = (forces["time"] >= time[0] - 1e-8) & (forces["time"] <= time[-1] + 1e-8)
        force_time = forces["time"][selected]
        # Use the boundary's midpoint for both fits; dense force samples do not
        # necessarily include the boundary's first timestamp.
        force_design = design_matrix(force_time, frequency)
        shift = (force_time[0] + force_time[-1] - time[0] - time[-1]) / 2
        phases = {}
        for field, order in (("lift_coefficient", 1), ("drag_coefficient", 2)):
            fitted = np.linalg.lstsq(force_design, forces[field][selected], rcond=None)[0]
            value = complex_coefficient(fitted, order) * np.exp(
                -1j * 2 * np.pi * frequency * order * shift
            )
            phases[order] = np.angle(value)
        force_phase[label] = phases
        phase_records[label] = {
            "boundary_midpoint_time": float((time[0] + time[-1]) / 2),
            "lift_fundamental_phase_degrees": wrapped_degrees(phases[1]),
            "drag_second_harmonic_phase_degrees": wrapped_degrees(phases[2]),
            "drag_second_minus_twice_lift_fundamental_degrees": wrapped_degrees(
                phases[2] - 2 * phases[1]
            ),
        }
        coefficient[label] = {
            field: fit_coefficient(time, arrays[f"{label}_{field}"], frequency)
            for field in ("normal_velocity", "tangential_gradient")
        }
    aligned = {
        label: {
            field: {
                order: complex_coefficient(value, order) * np.exp(-1j * force_phase[label][order])
                for order in (1, 2)
            }
            for field, value in values.items()
        }
        for label, values in coefficient.items()
    }
    phases = {
        side: {
            field: {
                f"harmonic_{order}": phase_statistics(
                    aligned["coupled"][field][order][selected],
                    aligned["reference"][field][order][selected],
                    area[selected],
                )
                for order in (1, 2)
            }
            for field in ("normal_velocity", "tangential_gradient")
        }
        for side, selected in sides.items()
    }
    means = {
        side: {
            field: mean_statistics(
                coefficient["coupled"][field][0][selected],
                coefficient["reference"][field][0][selected],
                area[selected],
            )
            for field in ("normal_velocity", "tangential_gradient")
        }
        for side, selected in sides.items()
    }
    output = {
        "schema": "openonda-planar-boundary-harmonic-phase/1",
        "source_boundary_fields": str(directory / "boundary_harmonic_fields.npz"),
        "source_boundary_fields_sha256": hashlib.sha256(
            (directory / "boundary_harmonic_fields.npz").read_bytes()
        ).hexdigest(),
        "analysis_source": str(Path(__file__).resolve()),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "force_phase_references": phase_records,
        "boundary_phase_relative_to_forces": phases,
        "boundary_mean_differences": means,
        "side_geometry": {
            side: {
                "face_count": int(np.count_nonzero(selected)),
                "area": float(np.sum(area[selected])),
            }
            for side, selected in sides.items()
        },
        "method": {
            "complex_convention": "u=Re(C exp(i omega (t-t_mid))); C=cosine_coefficient-i*sine_coefficient.",
            "alignment": "First harmonic divided by unit phase of each run's lift fundamental; second harmonic by each run's drag second harmonic. Force fits are transformed to the matching boundary fit midpoint.",
            "complex_projection": "Gain=sum(area*Cb*conjugate(Cr))/sum(area*abs(Cr)^2), with vector components summed for gt. Phase is arg(gain); coherence is the magnitude of the normalised complex inner product.",
            "amplitude_rescaled_error": "RMS((norm(Cr)/norm(Cb))*Cb-Cr)/norm(Cr); reveals force-relative phase and spatial shape differences after matching global amplitudes.",
            "means": "Signed area-average and RMS of the local linear-trend fit intercept, evaluated at each window midpoint; units m/s for un and 1/s for gt.",
        },
        "limitations": [
            "Force-aligned phase differences measure a coupled relationship, not whether a boundary trace creates positive or negative feedback; that requires a controlled evolution.",
            "The two time windows, different shedding frequency and possible output aliasing retain the limitations of the parent harmonic measurement.",
            "Three short oblique corner facets are reported separately instead of being silently assigned to an axis-aligned side.",
        ],
    }
    (directory / "boundary_harmonic_phase_statistics.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    report["force_phase_analysis"] = output
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    phase_arrays = {
        f"{label}_{field}_coefficient": value
        for label, values in coefficient.items()
        for field, value in values.items()
    }
    for label, fields in aligned.items():
        phase_arrays.update(
            {
                f"{label}_{field}_harmonic_{order}_force_aligned": value
                for field, orders in fields.items()
                for order, value in orders.items()
            }
        )
    np.savez_compressed(directory / "boundary_harmonic_complex_fields.npz", **phase_arrays)
    print(
        json.dumps(
            {
                "output": str(directory),
                "force_phase_references": phase_records,
                "all_phases": phases["all"],
                "mean_differences": means,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
