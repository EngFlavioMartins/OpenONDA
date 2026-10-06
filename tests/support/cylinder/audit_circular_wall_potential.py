"""Frozen circular harmonic correction of measured discrete particle wall penetration.

This prototype fits exterior, single-valued Fourier potentials on the actual
cylinder facet normals. It changes neither particle circulation nor Gaussian
cores. It tests correcting measured discrete wall-normal slip, not a missing
continuum potential: the complete continuum no-slip zero extension already
has the required solenoidal velocity. No solver or production model is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from scipy.linalg import null_space

from tests.support.cylinder.audit_saved_wall_circulation import (
    digest,
    gaussian_velocity_and_gradient,
    load_donor_geometry,
    velocity_statistics,
)


def circular_potential_basis(points, radius=0.5, modes=16):
    """Return unit-velocity potential bases and their analytic U/J in SI units.

    Potential bases are R/m*(R/r)^m*cos(mθ) and its sine partner, for m>=1.
    J[i,j]=du_i/dx_j. All bases are curl-free/divergence-free outside r=0;
    their circular flux and circulation are zero, and velocity decays r^(-m-1).
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("Finite circular potential points of shape(N,3) are required")
    if not np.isfinite(radius) or radius <= 0 or not isinstance(modes, int) or not 1 <= modes <= 24:
        raise ValueError("Positive radius and 1 to 24 bounded Fourier modes are required")
    z = points[:, 0] + 1j * points[:, 1]
    if np.any(np.abs(z) <= 0):
        raise ValueError("Exterior circular potential is undefined at the body centre")
    phi = np.zeros((len(points), 2 * modes))
    velocity = np.zeros((len(points), 3, 2 * modes))
    jacobian = np.zeros((len(points), 3, 3, 2 * modes))
    for m in range(1, modes + 1):
        factor = radius ** (m + 1)
        potential = factor / m * z ** (-m)
        first = -factor * z ** (-m - 1)
        second = (m + 1) * factor * z ** (-m - 2)
        c, s = 2 * (m - 1), 2 * (m - 1) + 1
        phi[:, c], phi[:, s] = potential.real, -potential.imag
        velocity[:, 0, c], velocity[:, 1, c] = first.real, -first.imag
        velocity[:, 0, s], velocity[:, 1, s] = -first.imag, -first.real
        jacobian[:, 0, 0, c], jacobian[:, 1, 1, c] = second.real, -second.real
        jacobian[:, 0, 1, c], jacobian[:, 1, 0, c] = -second.imag, -second.imag
        jacobian[:, 0, 0, s], jacobian[:, 1, 1, s] = -second.imag, second.imag
        jacobian[:, 0, 1, s], jacobian[:, 1, 0, s] = -second.real, -second.real
    return phi, velocity, jacobian


def fit_circular_wall_potential(points, normals, areas, particle_velocity, *, modes=16, radius=0.5):
    """Fit bounded exterior modes with zero discrete area-weighted facet flux."""
    points, normals, areas = np.asarray(points), np.asarray(normals), np.asarray(areas)
    velocity = np.asarray(particle_velocity)
    if (
        normals.shape != points.shape
        or velocity.shape != points.shape
        or areas.shape != (len(points),)
        or len(points) < 2 * modes
        or not np.isfinite(areas).all()
        or np.any(areas <= 0)
    ):
        raise ValueError("Wall fit requires enough finite facet normals/areas/velocities")
    _, basis, _ = circular_potential_basis(points, radius, modes)
    normal_basis = np.einsum("fib,fi->fb", basis, normals)
    original_normal = np.einsum("fi,fi->f", velocity, normals)
    target = -original_normal
    target_flux = float(np.sum(areas * target))
    target -= target_flux / float(np.sum(areas))
    flux_row = np.sum(areas[:, None] * normal_basis, axis=0) / np.sum(areas)
    # The continuum modes already have zero flux. Restrict the small facet
    # quadrature departure explicitly, without introducing a logarithmic mode.
    flux_norm = float(np.linalg.norm(flux_row))
    admissible = null_space(flux_row[None]) if flux_norm > 1e-13 else np.eye(2 * modes)
    weighted = np.sqrt(areas / np.sum(areas))[:, None] * normal_basis @ admissible
    singular = np.linalg.svd(weighted, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    if not np.isfinite(condition) or condition > 100:
        raise ValueError("Circular wall-normal fit exceeds the condition bound 100")
    solution, _residual, rank, _singular = np.linalg.lstsq(
        weighted, np.sqrt(areas / np.sum(areas)) * target, rcond=1e-12
    )
    if rank != admissible.shape[1]:
        raise ValueError("Circular wall-normal fit lost full bounded-mode rank")
    coefficient = admissible @ solution
    corrected_normal = original_normal + normal_basis @ coefficient
    return coefficient, {
        "mode_count": modes,
        "coefficient_count": len(coefficient),
        "condition": condition,
        "condition_bound": 100,
        "rank": int(rank),
        "discrete_flux_constraint_applied": flux_norm > 1e-13,
        "original_wall_flux": -target_flux,
        "correction_wall_flux": float(np.sum(areas * (normal_basis @ coefficient))),
        "remaining_wall_flux": float(np.sum(areas * corrected_normal)),
        "remaining_wall_normal_rms": float(np.sqrt(np.average(corrected_normal**2, weights=areas))),
        "remaining_wall_normal_maximum": float(np.max(np.abs(corrected_normal))),
        "target_flux_mean_removed": target_flux / float(np.sum(areas)),
        "maximum_coefficient": float(np.max(np.abs(coefficient))),
    }


def potential_field(points, coefficients, radius=0.5):
    phi, velocity, jacobian = circular_potential_basis(points, radius, len(coefficients) // 2)
    return phi @ coefficients, velocity @ coefficients, jacobian @ coefficients


def tangential_normal_gradient(jacobian, normals):
    derivative = np.einsum("fij,fj->fi", jacobian, normals)
    return derivative - np.einsum("fi,fi->f", derivative, normals)[:, None] * normals


def validate_potential(coefficients, points, radius=0.5):
    """Independent finite differences, flux/circulation quadrature and decay."""
    _, velocity, jacobian = potential_field(points, coefficients, radius)
    finite_difference = np.zeros_like(jacobian)
    step = radius * 1e-5
    for axis in (0, 1):
        offset = np.zeros_like(points)
        offset[:, axis] = step
        finite_difference[:, :, axis] = (
            potential_field(points + offset, coefficients, radius)[1]
            - potential_field(points - offset, coefficients, radius)[1]
        ) / (2 * step)
    derivative_error = float(np.max(np.abs(finite_difference - jacobian)))
    derivative_scale = float(np.max(np.abs(jacobian), initial=0))
    if derivative_error > 1e-6 * max(1, derivative_scale):
        raise ValueError(
            "Analytic circular potential Jacobian fails independent finite differences"
        )
    theta = np.arange(4096) * (2 * np.pi / 4096)
    radial = np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
    tangent = np.column_stack((-np.sin(theta), np.cos(theta), np.zeros_like(theta)))
    surface = radius * radial
    _, circle_velocity, _ = potential_field(surface, coefficients, radius)
    circle_flux = float(
        np.sum(np.einsum("fi,fi->f", circle_velocity, radial)) * radius * 2 * np.pi / len(theta)
    )
    circle_circulation = float(
        np.sum(np.einsum("fi,fi->f", circle_velocity, tangent)) * radius * 2 * np.pi / len(theta)
    )
    sample_decay = []
    for multiplier in (1, 2, 4, 16):
        _, decayed, _ = potential_field(surface * multiplier, coefficients, radius)
        sample_decay.append(
            {
                "radius": radius * multiplier,
                "velocity_rms": float(np.sqrt(np.mean(np.sum(decayed**2, axis=1)))),
            }
        )
    return {
        "maximum_divergence": float(np.max(np.abs(np.trace(jacobian, axis1=1, axis2=2)))),
        "maximum_curl": float(np.max(np.abs(jacobian[:, 1, 0] - jacobian[:, 0, 1]))),
        "maximum_jacobian_finite_difference_error": derivative_error,
        "finite_difference_step": step,
        "circular_flux": circle_flux,
        "circular_circulation": circle_circulation,
        "radial_decay": sample_decay,
        "evaluation_velocity_maximum": float(np.max(np.linalg.norm(velocity, axis=1))),
    }


def audit(image_path, checkpoint_directory, output_directory, modes=(16, 24)):
    started = perf_counter()
    image_path, checkpoint_directory = Path(image_path), Path(checkpoint_directory)
    fields_path = checkpoint_directory / "wall_trace_fields.npz"
    metadata_path = checkpoint_directory / "checkpoint/checkpoint_info.json"
    metadata = json.loads(metadata_path.read_text())
    vpm_path = metadata_path.parent / metadata["checkpoint_files"]["vpm"]
    fvm_path = metadata_path.parent / metadata["checkpoint_files"]["fvm"]
    paths = [
        image_path,
        image_path.with_suffix(".json"),
        fields_path,
        metadata_path,
        vpm_path,
        fvm_path,
        checkpoint_directory / "coupled_mesh.npz",
        Path(__file__),
    ]
    hashes = {str(path): digest(path) for path in paths}
    with np.load(fields_path) as saved:
        fields = {key: saved[key].copy() for key in saved.files}
    wall_points, wall_normals, wall_areas = (
        fields[f"cylinder_{name}"] for name in ("centre", "normal", "area")
    )
    with np.load(image_path) as saved:
        image = {key: saved[key].copy() for key in saved.files}
    image_report = json.loads(image_path.with_suffix(".json").read_text())
    outer_points, normals, areas = (image[f"face_{name}"] for name in ("centre", "normal", "area"))
    if not np.array_equal(outer_points, fields["numericalBoundary_centre"]):
        raise ValueError("Mature image and actual checkpoint use different outer face ordering")
    cases = {}
    for name in ("native_masked", "full_reference_native_masked"):
        position, strength = image[name + "_position"], image[name + "_strength"]
        wall_velocity, _gradient = gaussian_velocity_and_gradient(
            wall_points,
            position.astype(float),
            strength[:, 2].astype(float),
            image_report["core_radius"],
        )
        wall_velocity += np.array([1.0, 0.0, 0.0])
        cases[name] = {
            "time": image_report["reference_time"],
            "wall_velocity": wall_velocity,
            "outer_velocity": image[name + "_velocity"],
            "outer_jacobian": image[name + "_jacobian"],
            "reference_velocity": image["exact_velocity"],
            "reference_jacobian": image["exact_jacobian"],
            "comparison": "Simultaneous mature fully meshed reference 100 s trace",
        }
    with h5py.File(vpm_path) as saved:
        position, strength, radius = (
            np.asarray(saved[f"particles/{key}"])
            for key in ("position", "vortex_strength", "core_radius")
        )
    if not np.all(radius == radius[0]):
        raise ValueError("This independent frozen kernel requires uniform stored core radius")
    wall_velocity, _gradient = gaussian_velocity_and_gradient(
        wall_points, position.astype(float), strength[:, 2].astype(float), float(radius[0])
    )
    wall_velocity += np.array([1.0, 0.0, 0.0])
    outer_velocity, outer_jacobian = gaussian_velocity_and_gradient(
        outer_points, position.astype(float), strength[:, 2].astype(float), float(radius[0])
    )
    outer_velocity += np.array([1.0, 0.0, 0.0])
    mesh, _geometry, state, gradient, _boundary, _wall, trace = load_donor_geometry(
        checkpoint_directory, fvm_path
    )
    plan = trace.prepare(outer_points)
    fvm_velocity = plan.sample(state["velocity"][: mesh["n_cells"]], gradient)
    fvm_jacobian = np.einsum("fk,fkij->fij", plan.weights, gradient[plan.indices]).swapaxes(1, 2)
    cases["actual_coupled_particles"] = {
        "time": float(state["time"]),
        "wall_velocity": wall_velocity,
        "outer_velocity": outer_velocity,
        "outer_jacobian": outer_jacobian,
        "reference_velocity": fvm_velocity,
        "reference_jacobian": fvm_jacobian,
        "comparison": "Same-time accepted coupled FVM real-cell affine trace, not an independent fully meshed reference",
    }
    measurements, arrays = (
        {},
        {
            "wall_points": wall_points,
            "wall_normals": wall_normals,
            "wall_areas": wall_areas,
            "outer_points": outer_points,
            "outer_normals": normals,
            "outer_areas": areas,
        },
    )
    for name, case in cases.items():
        measurements[name] = {
            "time": case["time"],
            "comparison": case["comparison"],
            "uncorrected_wall_velocity": velocity_statistics(
                case["wall_velocity"], wall_normals, wall_areas
            ),
            "modes": {},
        }
        exact_gt = tangential_normal_gradient(case["reference_jacobian"], normals)
        for mode_count in modes:
            coefficients, fit = fit_circular_wall_potential(
                wall_points, wall_normals, wall_areas, case["wall_velocity"], modes=mode_count
            )
            _, wall_delta, _wall_jacobian = potential_field(wall_points, coefficients)
            _, outer_delta, jacobian_delta = potential_field(outer_points, coefficients)
            corrected_wall = case["wall_velocity"] + wall_delta
            corrected_outer = case["outer_velocity"] + outer_delta
            corrected_gt = tangential_normal_gradient(
                case["outer_jacobian"] + jacobian_delta, normals
            )
            before_gt = tangential_normal_gradient(case["outer_jacobian"], normals)
            row = {
                "fit": fit,
                "validation": validate_potential(
                    coefficients, np.concatenate((wall_points, outer_points))
                ),
                "corrected_wall_velocity": velocity_statistics(
                    corrected_wall, wall_normals, wall_areas
                ),
                "uncorrected_outer_velocity_error": velocity_statistics(
                    case["outer_velocity"] - case["reference_velocity"], normals, areas
                ),
                "corrected_outer_velocity_error": velocity_statistics(
                    corrected_outer - case["reference_velocity"], normals, areas
                ),
                "uncorrected_outer_tangential_gradient_error_rms": float(
                    np.sqrt(np.average(np.sum((before_gt - exact_gt) ** 2, axis=1), weights=areas))
                ),
                "corrected_outer_tangential_gradient_error_rms": float(
                    np.sqrt(
                        np.average(np.sum((corrected_gt - exact_gt) ** 2, axis=1), weights=areas)
                    )
                ),
                "outer_correction": velocity_statistics(outer_delta, normals, areas),
                "outer_tangential_gradient_correction_rms": float(
                    np.sqrt(
                        np.average(
                            np.sum(
                                tangential_normal_gradient(jacobian_delta, normals) ** 2, axis=1
                            ),
                            weights=areas,
                        )
                    )
                ),
            }
            measurements[name]["modes"][str(mode_count)] = row
            arrays.update(
                {
                    f"{name}_{mode_count}_coefficient": coefficients,
                    f"{name}_{mode_count}_wall_corrected_velocity": corrected_wall,
                    f"{name}_{mode_count}_outer_corrected_velocity": corrected_outer,
                    f"{name}_{mode_count}_outer_corrected_jacobian": case["outer_jacobian"]
                    + jacobian_delta,
                }
            )
    if {str(path): digest(path) for path in paths} != hashes:
        raise ValueError("Frozen input or harmonic diagnostic source changed during audit")
    report = {
        "scope": "Frozen circular harmonic correction of measured discrete particle wall penetration only; no solver/native model change or force-causality claim",
        "particle_circulation_or_core_changed": False,
        "input_and_helper_sha256": hashes,
        "measurements": measurements,
        "wall_seconds": perf_counter() - started,
        "limitations": [
            "Circular single-valued exterior Fourier prototype, not a production or general-body model.",
            "Complete continuum no-slip zero extension needs no added potential term; this diagnoses finite-resolution particle wall penetration.",
            "Mature image outer errors use simultaneous100s reference fields; actual 46 s particle errors compare its own accepted coupled FVM field, not phase-inconsistent 100 s reference.",
            "No evolved force or interface convergence evidence is provided by this frozen correction.",
        ],
    }
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_directory / "circular_wall_potential_fields.npz", **arrays)
    (output_directory / "circular_wall_potential.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.image, args.checkpoint_directory, args.output_directory)
    print(
        json.dumps(
            {"measurements": result["measurements"], "wall_seconds": result["wall_seconds"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
