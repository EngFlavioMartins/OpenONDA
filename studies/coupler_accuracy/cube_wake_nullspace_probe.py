"""Test whether an invisible vorticity component produces a velocity rate.

A compactly truncated gradient of a 3D Gaussian is curl-free. Its Biot-Savart
velocity vanishes in the continuum, also after Gaussian core convolution.
The saved cube velocity transports this added field. Positive-transpose
stretching is contrasted with negative-transpose covector transport, for
which a gradient remains a gradient. Two quadrature spacings check that a
nonzero velocity rate is not a poorly resolved gradient-particle sum.
No physical solver state is changed and no FVM boundary condition is used.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

from cube_wake_particle_probe import direct_gaussian, load_case, rms
from numba import njit, prange, set_num_threads
import numpy as np
from threadpoolctl import threadpool_limits


@njit(parallel=True, cache=True)
def direct_velocity_rate(points, position, strength, radius, velocity, strength_rate):
    """Differentiate the complete Gaussian Biot-Savart sum in time."""
    result = np.zeros((len(points), 3))
    for i in prange(len(points)):
        for j in range(len(position)):
            r = points[i] - position[j]
            r2 = np.dot(r, r)
            q2 = r2 / radius[j] ** 2
            if q2 < 1e-6:
                c = 1 / (3 * math.pi**1.5 * radius[j] ** 3)
                factor = c * (1 - 0.6 * q2 + 3 / 14 * q2**2)
                derivative = c / radius[j] ** 2 * (-1.2 + 6 / 7 * q2 - q2**2 / 3)
            else:
                q = math.sqrt(q2)
                g = math.erf(q) - 2 * q / math.sqrt(math.pi) * math.exp(-q2)
                factor = g / (4 * math.pi * r2**1.5)
                derivative = (4 * q**3 / math.sqrt(math.pi) * math.exp(-q2) - 3 * g) / (
                    4 * math.pi * r2**2.5
                )
            result[i] += factor * (
                np.cross(strength_rate[j], r) - np.cross(strength[j], velocity[j])
            )
            result[i] -= derivative * np.dot(r, velocity[j]) * np.cross(strength[j], r)
    return result


def verify_time_derivative():
    rng = np.random.default_rng(782)
    p, g, v, rate = (rng.normal(size=(9, 3)) for _ in range(4))
    radius = np.full(9, 0.3)
    points = rng.normal(size=(11, 3))
    actual = direct_velocity_rate(points, p, g, radius, v, rate)
    dt = 1e-6
    plus, _ = direct_gaussian(points, p + dt * v, g + dt * rate, radius)
    minus, _ = direct_gaussian(points, p - dt * v, g - dt * rate, radius)
    error = rms(actual - (plus - minus) / (2 * dt)) / rms(actual)
    assert error < 1e-8, error
    return error


def run_manufactured(output):
    """Repeat with exact, irrotational 3D strain and no body or recirculation."""
    output.mkdir(parents=True, exist_ok=False)
    axes = (np.linspace(1.08, 1.92, 8), np.linspace(-0.36, 0.36, 5), np.linspace(-0.36, 0.36, 5))
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    strain = np.diag([0.5, -0.25, -0.25])
    report = {
        "description": "Exact affine incompressible potential strain; zero physical vorticity; no body, boundary, recirculation, diffusion, LES, treecode, or time integrator.",
        "velocity": "u=[1,0,0]+diag(0.5,-0.25,-0.25)@(x-[1.5,0,0])",
        "time_derivative_relative_check": verify_time_derivative(),
        "quadratures": [],
    }
    for spacing in (0.06, 0.03):
        width = 0.12
        axis = np.arange(-round(7 * width / spacing), round(7 * width / spacing) + 1) * spacing
        local = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        position = local + [1.5, 0, 0]
        potential = width * np.exp(0.5) * np.exp(-np.sum(local**2, axis=1) / (2 * width**2))
        strength = -local * (potential / width**2)[:, None] * spacing**3
        radius = np.full(len(position), 0.066)
        velocity = local @ strain.T + [1, 0, 0]
        rate = strength @ strain
        induced, _ = direct_gaussian(points, position, strength, radius)
        positive = direct_velocity_rate(points, position, strength, radius, velocity, rate)
        negative = direct_velocity_rate(points, position, strength, radius, velocity, -rate)
        identity_velocity, _ = direct_gaussian(points, position, rate, radius)
        residual = rms(positive - negative - 2 * identity_velocity)
        assert residual < 1e-12, residual
        row = {
            "spacing": spacing,
            "particles": len(position),
            "minimum_streamwise_background_velocity": float(velocity[:, 0].min()),
            "initial_induced_velocity_rms": rms(induced),
            "positive_transpose_velocity_rate_rms": rms(positive),
            "negative_transpose_velocity_rate_rms": rms(negative),
            "positive_minus_negative_identity_residual": residual,
        }
        report["quadratures"].append(row)
        np.savez_compressed(
            output / f"fields_h{spacing:g}.npz",
            points=points,
            initial_velocity=induced,
            positive_rate=positive,
            negative_rate=negative,
        )
        (output / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    case = load_case(args.source_tree)
    from source.solvers.vpm import VPMSolver

    policy = replace(
        case.VPM_CASE,
        directory=args.output / "runtime",
        samplers=case.vpm.Samplers(),
        numerics=replace(
            case.VPM_CASE.numerics, max_n_particles=300000, max_evaluation_points=300000
        ),
    )
    x, yz = np.linspace(1.08, 1.92, 8), np.linspace(-0.36, 0.36, 5)
    points = np.stack(np.meshgrid(x, yz, yz, indexing="ij"), axis=-1).reshape(-1, 3)
    report = {
        "description": __doc__,
        "time_derivative_relative_check": verify_time_derivative(),
        "potential_centre": [1.5, 0, 0],
        "potential_width": 0.12,
        "maximum_added_vorticity_before_core_filter": 1.0,
        "quadratures": [],
    }
    solver = VPMSolver(policy)
    try:
        solver.load_backup(args.checkpoint)
        solver.refresh_boundary_element_solution()
        source_p, source_g, source_sigma = (
            getattr(solver, k).astype(float)
            for k in ("particle_position", "particle_vortex_strength", "particle_core_radius")
        )
        for spacing in (0.06, 0.03):
            width = 0.12
            # Seven Gaussian standard deviations leave negligible truncation.
            axis = np.arange(-round(7 * width / spacing), round(7 * width / spacing) + 1) * spacing
            local = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
            position = local + [1.5, 0, 0]
            potential = width * np.exp(0.5) * np.exp(-np.sum(local**2, axis=1) / (2 * width**2))
            strength = -local * (potential / width**2)[:, None] * spacing**3
            radius = np.full(len(position), 0.066)
            assert np.min(position[:, 0]) > 0.5
            induced, gradient = direct_gaussian(points, position, strength, radius)
            velocity, jacobian = solver.compute_velocity_and_gradient_at_points(
                position, particle_spacing=0.06
            )
            velocity = np.asarray(velocity, dtype=np.float64)
            jacobian = np.asarray(jacobian, dtype=np.float64).reshape(-1, 3, 3)
            rate = np.einsum("nji,nj->ni", jacobian, strength)
            positive = direct_velocity_rate(points, position, strength, radius, velocity, rate)
            negative = direct_velocity_rate(points, position, strength, radius, velocity, -rate)
            selected = np.random.default_rng(891).choice(len(position), 128, replace=False)
            exact_u, exact_j = direct_gaussian(position[selected], source_p, source_g, source_sigma)
            exact_u += solver.panel_solver.compute_induced_velocity(position[selected]) + [1, 0, 0]
            exact_j += solver.panel_solver.compute_induced_velocity_gradient(position[selected])
            row = {
                "spacing": spacing,
                "source_count": len(position),
                "initial_induced_velocity_rms": rms(induced),
                "initial_induced_velocity_max": float(np.max(np.linalg.norm(induced, axis=1))),
                "initial_velocity_gradient_rms": rms(gradient),
                "positive_transpose_velocity_rate_rms": rms(positive),
                "negative_transpose_velocity_rate_rms": rms(negative),
                "positive_transpose_transverse_rate_rms": rms(positive[:, 1:]),
                "advecting_velocity_error_rms_at_128_targets": rms(velocity[selected] - exact_u),
                "advecting_gradient_relative_error_at_128_targets": rms(
                    jacobian[selected] - exact_j
                )
                / rms(exact_j),
            }
            report["quadratures"].append(row)
            np.savez_compressed(
                args.output / f"fields_h{spacing:g}.npz",
                points=points,
                initial_velocity=induced,
                positive_rate=positive,
                negative_rate=negative,
            )
            (args.output / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    finally:
        solver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manufactured-only", action="store_true")
    args = parser.parse_args()
    if not args.manufactured_only and (args.source_tree is None or args.checkpoint is None):
        parser.error("the saved-state probe requires --source-tree and --checkpoint")
    set_num_threads(2)
    with threadpool_limits(limits=2):
        if args.manufactured_only:
            run_manufactured(args.output)
        else:
            run(args)


if __name__ == "__main__":
    main()
