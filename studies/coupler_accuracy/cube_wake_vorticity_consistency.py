"""Compare Gaussian particle vorticity with its actual Biot-Savart velocity curl.

The body source potential and uniform freestream have zero curl. Thus this
comparison uses saved particles directly, without an FVM reconstruction or a
tree approximation. It measures representation inconsistency, not its cause.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from cube_wake_particle_probe import direct_gaussian, rms, validate_direct_formula
import h5py
from numba import njit, prange, set_num_threads
import numpy as np


@njit(parallel=True, cache=True)
def gaussian_vorticity_and_divergence(points, position, strength, sigma):
    vorticity = np.zeros((len(points), 3))
    divergence = np.zeros(len(points))
    for i in prange(len(points)):
        for j in range(len(position)):
            dx = points[i, 0] - position[j, 0]
            dy = points[i, 1] - position[j, 1]
            dz = points[i, 2] - position[j, 2]
            variance = sigma[j] ** 2
            kernel = math.exp(-(dx * dx + dy * dy + dz * dz) / variance)
            kernel /= math.pi**1.5 * sigma[j] ** 3
            for axis in range(3):
                vorticity[i, axis] += kernel * strength[j, axis]
            divergence[i] -= (
                2
                * kernel
                * (dx * strength[j, 0] + dy * strength[j, 1] + dz * strength[j, 2])
                / variance
            )
    return vorticity, divergence


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    axes = (np.linspace(-1.2, 3.6, 25), np.linspace(-1.2, 1.2, 9), np.linspace(-1.2, 1.2, 9))
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    points = points[~np.all(np.abs(points) <= 0.5 + 1e-10, axis=1)]
    masks = {
        "inner": np.all(np.abs(points) < 0.89, axis=1),
        "outflow_belt": (points[:, 0] >= 0.89) & (points[:, 0] <= 1.5),
        "outer_wake": points[:, 0] > 1.5,
    }
    report = {
        "description": __doc__,
        "particle_directory": str(args.solution.resolve()),
        "query_count": len(points),
        "spacing": [0.2, 0.3, 0.3],
        "vorticity_scale": "U_infinity/D=1 s^-1; relative metrics are null below 1e-10 s^-1 local RMS",
        "direct_derivative_relative_error": validate_direct_formula(),
        "times": [],
    }
    for time in args.times:
        filename = args.solution / f"vpm_{round(time * 100):06d}.h5"
        with h5py.File(filename) as h5:
            p = h5["particles"]
            position, strength, sigma = (
                np.asarray(p[k], dtype=float)
                for k in ("position", "vortex_strength", "core_radius")
            )
            saved_time = float(h5["solver"].attrs["time"])
        if not np.isclose(saved_time, time, rtol=0, atol=1e-8):
            raise ValueError(f"Unexpected saved physical time in {filename}")
        _velocity, gradient = direct_gaussian(points, position, strength, sigma)
        curl = np.column_stack(
            (
                gradient[:, 2, 1] - gradient[:, 1, 2],
                gradient[:, 0, 2] - gradient[:, 2, 0],
                gradient[:, 1, 0] - gradient[:, 0, 1],
            )
        )
        omega, divergence = gaussian_vorticity_and_divergence(points, position, strength, sigma)
        row = {"time": time, "particles": len(position), "regions": {}}
        for name, mask in masks.items():
            scale = rms(omega[mask])
            row["regions"][name] = {
                "points": int(mask.sum()),
                "gaussian_vorticity_rms": scale,
                "velocity_curl_rms": rms(curl[mask]),
                "vorticity_minus_curl_rms": rms(omega[mask] - curl[mask]),
                "vorticity_minus_curl_relative_rms": rms(omega[mask] - curl[mask]) / scale
                if scale > 1e-10
                else None,
                "spacing_times_divergence_relative_rms": (
                    0.06 * float(np.sqrt(np.mean(divergence[mask] ** 2))) / scale
                    if scale > 1e-10
                    else None
                ),
            }
        report["times"].append(row)
        np.savez_compressed(
            args.output / f"fields_t{time:g}.npz",
            points=points,
            vorticity=omega,
            velocity_curl=curl,
            divergence=divergence,
        )
        (args.output / "consistency.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    args = parser.parse_args()
    set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
