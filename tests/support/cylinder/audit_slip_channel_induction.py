"""Measure the slip-channel image velocity omitted by free-space induction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from source._numba import cacheable_njit as njit

from .audit_saved_wall_circulation import digest, velocity_statistics


@njit(cache=True)
def channel_image_velocity_gradient(points, position, circulation, half_width):
    """Sum the harmonic images for slip walls at y=+/-half_width.

    The infinite image families have period 4H. Their cotangent sum is
    evaluated as coth, with the singular free-space vortex subtracted.
    The image correction is smooth even at a vortex source and has no curl.
    """
    velocity = np.zeros((len(points), 3))
    gradient = np.zeros((len(points), 3, 3))
    wave_number = np.pi / (4 * half_width)
    for target in range(len(points)):
        value = 0j
        derivative = 0j
        for source in range(len(position)):
            delta = complex(
                points[target, 0] - position[source, 0], points[target, 1] - position[source, 1]
            )
            reflected = complex(
                delta.real, points[target, 1] - 2 * half_width + position[source, 1]
            )
            scaled = wave_number * delta
            if abs(scaled) < 1e-3:
                positive = wave_number * (scaled / 3 - scaled**3 / 45 + 2 * scaled**5 / 945)
                positive_derivative = wave_number**2 * (
                    1 / 3 - scaled**2 / 15 + 2 * scaled**4 / 189
                )
            else:
                coth = 1 / np.tanh(scaled)
                positive = wave_number * coth - 1 / delta
                positive_derivative = wave_number**2 * (1 - coth**2) + 1 / delta**2
            mirror_coth = 1 / np.tanh(wave_number * reflected)
            correction = positive - wave_number * mirror_coth
            correction_derivative = positive_derivative - wave_number**2 * (1 - mirror_coth**2)
            factor = circulation[source] / (2 * np.pi * 1j)
            value += factor * correction
            derivative += factor * correction_derivative
        velocity[target, 0], velocity[target, 1] = value.real, -value.imag
        gradient[target, 0, 0] = derivative.real
        gradient[target, 0, 1] = -derivative.imag
        gradient[target, 1, 0] = -derivative.imag
        gradient[target, 1, 1] = -derivative.real
    return velocity, gradient


def audit(snapshot, output):
    metadata_path = snapshot / "checkpoint/checkpoint_info.json"
    metadata = json.loads(metadata_path.read_text())
    particle_path = metadata_path.parent / metadata["checkpoint_files"]["vpm"]
    field_path = snapshot / "wall_trace_fields.npz"
    hashes = {str(path): digest(path) for path in (metadata_path, particle_path, field_path)}
    with h5py.File(particle_path) as saved:
        position = np.asarray(saved["particles/position"], dtype=float)
        circulation = np.asarray(saved["particles/vortex_strength"], dtype=float)[:, 2]
    span = metadata["config"]["vpm"]["induction"]["planar_span"]
    circulation /= span
    rows = {}
    with np.load(field_path) as saved:
        for patch in ("cylinder", "numericalBoundary"):
            points, normal, area = (
                saved[patch + "_" + key] for key in ("centre", "normal", "area")
            )
            rows[patch] = {}
            for half_width in (10.0, 20.0, 40.0):
                velocity, gradient = channel_image_velocity_gradient(
                    points, position, circulation, half_width
                )
                rows[patch][str(half_width)] = {
                    "velocity": velocity_statistics(velocity, normal, area),
                    "mean_streamwise_velocity": float(np.average(velocity[:, 0], weights=area)),
                    "mean_transverse_velocity": float(np.average(velocity[:, 1], weights=area)),
                    "maximum_gradient": float(np.max(np.abs(gradient))),
                }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "physical_time": metadata["time"],
                "scope": "Frozen particle wake; exact harmonic slip-wall images only. No force recovery claimed.",
                "input_sha256": hashes,
                "measurements": rows,
            },
            indent=2,
        )
        + "\n"
    )
    if any(digest(Path(path)) != expected for path, expected in hashes.items()):
        raise RuntimeError("Frozen scientific input changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    audit(arguments.snapshot, arguments.output)
