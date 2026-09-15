"""Independently qualify the compact flux operator on a 3D analytic potential.

The affine background is u=(a*x-Omega*y, Omega*x-a*y, 0). The compact scalar
potential is phi0*product[(1-(x_j/L)^2)^4_+]. Its exact rate is -2 J.T grad(phi),
with zero net source and a known nonzero impulse. Gaussian velocity-rate
comparisons keep core radius fixed while refining source quadrature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cube_covector_flux_control import potential_flux_source
from cube_wake_particle_probe import direct_gaussian, rms
from numba import set_num_threads
import numpy as np
from threadpoolctl import threadpool_limits


def analytic(position):
    """Return compact phi and its independent exact Cartesian gradient."""
    scaled = position / 0.5
    base = np.maximum(1 - scaled * scaled, 0)
    bump = base**4
    derivative = -8 * scaled * base**3 / 0.5
    phi = 0.8 * np.prod(bump, axis=1)
    gradient = np.empty_like(position)
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        gradient[:, i] = 0.8 * derivative[:, i] * bump[:, j] * bump[:, k]
    return phi, gradient


def grid(spacing):
    """Build the same physical box and one ghost cell on every side."""
    count = round(1.5 / spacing)
    axis = -0.75 + (np.arange(count) + 0.5) * spacing
    position = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    index = np.indices((count, count, count)).reshape(3, -1).T + 1
    identifiers = np.full((count + 2,) * 3, -1, dtype=int)
    identifiers[tuple(index.T)] = np.arange(len(position))
    return position, index, identifiers


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    jacobian = np.array([[0.3, -0.7, 0], [0.7, -0.3, 0], [0, 0, 0]])
    exact_impulse = np.array([0, 0, -1.4 * 0.8 * 0.5**3 * (256 / 315) ** 3])
    points = np.random.default_rng(20260915).uniform(-0.7, 0.7, (32, 3))
    core_radius = 0.11
    reference_velocity = []
    for spacing in (0.03125, 0.015625):
        position, _, _ = grid(spacing)
        _, gradient = analytic(position)
        source = -2 * spacing**3 * gradient @ jacobian
        nonzero = np.linalg.norm(source, axis=1) > 0
        velocity, _ = direct_gaussian(
            points, position[nonzero], source[nonzero], np.full(nonzero.sum(), core_radius)
        )
        reference_velocity.append(velocity)
    rows = []
    for spacing in (0.125, 0.0625, 0.03125):
        position, index, identifiers = grid(spacing)
        phi, gradient = analytic(position)
        source, _, _, _ = potential_flux_source(
            phi, np.broadcast_to(jacobian, (len(position), 3, 3)), index, identifiers, spacing
        )
        exact_density = -2 * gradient @ jacobian
        nonzero = np.linalg.norm(source, axis=1) > 0
        velocity, _ = direct_gaussian(
            points, position[nonzero], source[nonzero], np.full(nonzero.sum(), core_radius)
        )
        rows.append(
            {
                "spacing": spacing,
                "density_error_rms": rms(source / spacing**3 - exact_density),
                "net_strength_rate_norm": float(np.linalg.norm(source.sum(axis=0))),
                "impulse_error_norm": float(
                    np.linalg.norm(0.5 * np.cross(position, source).sum(axis=0) - exact_impulse)
                ),
                "gaussian_velocity_rate_error_rms": rms(velocity - reference_velocity[-1]),
            }
        )
    for i in (1, 2):
        rows[i]["density_convergence_order"] = float(
            np.log2(rows[i - 1]["density_error_rms"] / rows[i]["density_error_rms"])
        )
        rows[i]["gaussian_velocity_convergence_order"] = float(
            np.log2(
                rows[i - 1]["gaussian_velocity_rate_error_rms"]
                / rows[i]["gaussian_velocity_rate_error_rms"]
            )
        )
    quadrature_error = rms(reference_velocity[0] - reference_velocity[1])
    checks = {
        "net_conservation": max(row["net_strength_rate_norm"] for row in rows) < 1e-12,
        "nonzero_impulse_target": rows[-1]["impulse_error_norm"] < 1e-6,
        "spatial_consistency": rows[-1]["density_convergence_order"] > 1.8,
        "fixed_core_velocity_consistency": rows[-1]["gaussian_velocity_convergence_order"] > 1.8,
        "reference_quadrature_resolved": quadrature_error
        < 0.01 * rows[-1]["gaussian_velocity_rate_error_rms"],
    }
    report = {
        "scope": __doc__,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("cube_covector_flux_control.py"))
        },
        "exact_impulse_source": exact_impulse.tolist(),
        "fixed_core_radius": core_radius,
        "analytic_reference_quadrature_difference_rms": quadrature_error,
        "rows": rows,
        "checks": checks,
        "passed": all(checks.values()),
    }
    (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise AssertionError("Compact potential flux failed its analytic qualification")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.output)
