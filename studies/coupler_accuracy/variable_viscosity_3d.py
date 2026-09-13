#!/usr/bin/env python3
"""Compare the continuum variable-viscosity operators used by FVM and GBD.

This fully 3D periodic manufactured-field audit has no body or coupled time
integration. Spectral differentiation isolates model equivalence from mesh
interpolation, particle representation and boundary errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file


def derivative(field, axis):
    n = field.shape[axis]
    shape = [1, 1, 1]
    shape[axis] = n
    wave = np.fft.fftfreq(n, 1 / n).reshape(shape)
    return np.fft.ifftn(1j * wave * np.fft.fftn(field)).real


def curl(vector):
    return np.stack([
        derivative(vector[..., 2], 1) - derivative(vector[..., 1], 2),
        derivative(vector[..., 0], 2) - derivative(vector[..., 2], 0),
        derivative(vector[..., 1], 0) - derivative(vector[..., 0], 1),
    ], axis=-1)


def evaluate(n, viscosity_model):
    axis = 2 * np.pi * np.arange(n) / n
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    velocity = np.stack((np.sin(z) + np.cos(y), np.sin(x) + np.cos(z), np.sin(y) + np.cos(x)), axis=-1)
    gradient = np.stack([
        np.stack([derivative(velocity[..., i], j) for j in range(3)], axis=-1)
        for i in range(3)
    ], axis=-2)
    strain = 0.5 * (gradient + gradient.swapaxes(-1, -2))
    if viscosity_model == "constant":
        viscosity = np.full(x.shape, 0.001)
    elif viscosity_model == "smooth_variable":
        viscosity = 0.001 * (1 + 0.3 * np.cos(x) * np.cos(y) * np.cos(z))
    else:
        # Same equilibrium constants as the cube, with one fixed filter width.
        # Refining the differentiation grid does not alter the physical model.
        viscosity = 0.001 + (0.094**1.5 / np.sqrt(1.048)) * 0.125**2 * np.sqrt(
            2 * np.sum(strain**2, axis=(-1, -2))
        )
    stress = 2 * viscosity[..., None, None] * strain
    force = np.stack([
        sum(derivative(stress[..., i, j], j) for j in range(3))
        for i in range(3)
    ], axis=-1)
    fvm_curl_of_stress = curl(force)
    omega = curl(velocity)
    gbd_vorticity_diffusion = np.stack([
        sum(derivative(viscosity * derivative(omega[..., i], j), j) for j in range(3))
        for i in range(3)
    ], axis=-1)
    error = gbd_vorticity_diffusion - fvm_curl_of_stress
    # Even an exact solenoidal projection cannot, in general, replace the
    # missing variable-stress terms. Check that separately from raw div(omega).
    wave = np.stack(np.meshgrid(*([np.fft.fftfreq(n, 1 / n)] * 3), indexing="ij"), axis=-1)
    wave_sq = np.sum(wave**2, axis=-1)
    source_hat = np.fft.fftn(gbd_vorticity_diffusion, axes=(0, 1, 2))
    longitudinal = np.divide(
        np.sum(wave * source_hat, axis=-1), wave_sq,
        out=np.zeros_like(source_hat[..., 0]), where=wave_sq > 0,
    )
    projected_source = np.fft.ifftn(
        source_hat - wave * longitudinal[..., None], axes=(0, 1, 2)
    ).real
    analytical_difference_error = None
    if viscosity_model == "smooth_variable":
        amplitude = 0.0003
        viscosity_gradient = -amplitude * np.stack((
            np.sin(x) * np.cos(y) * np.cos(z),
            np.cos(x) * np.sin(y) * np.cos(z),
            np.cos(x) * np.cos(y) * np.sin(z),
        ), axis=-1)
        hessian = np.zeros((*x.shape, 3, 3))
        for i in range(3):
            hessian[..., i, i] = -amplitude * np.cos(x) * np.cos(y) * np.cos(z)
        hessian[..., 0, 1] = hessian[..., 1, 0] = amplitude * np.sin(x) * np.sin(y) * np.cos(z)
        hessian[..., 0, 2] = hessian[..., 2, 0] = amplitude * np.sin(x) * np.cos(y) * np.sin(z)
        hessian[..., 1, 2] = hessian[..., 2, 1] = amplitude * np.cos(x) * np.sin(y) * np.sin(z)
        product = np.einsum("...lj,...ij->...li", hessian, 2 * strain)
        hessian_term = np.stack((
            product[..., 1, 2] - product[..., 2, 1],
            product[..., 2, 0] - product[..., 0, 2],
            product[..., 0, 1] - product[..., 1, 0],
        ), axis=-1)
        # For div(u)=0, curl(div(2 nu S))-div(nu grad(omega))
        # = grad(nu) × laplacian(u) + epsilon_kli nu_,lj (2 S_ij).
        # The ABC field has laplacian(u)=-u exactly.
        analytical_difference = -np.cross(viscosity_gradient, -velocity) - hessian_term
        np.testing.assert_allclose(error, analytical_difference, rtol=0, atol=5e-13)
        analytical_difference_error = float(np.max(np.abs(error - analytical_difference)))
    divergence = sum(gradient[..., i, i] for i in range(3))
    assert np.max(np.abs(divergence)) < 1e-12
    np.testing.assert_allclose(omega, velocity, rtol=0, atol=5e-13)
    if viscosity_model == "constant":
        np.testing.assert_allclose(fvm_curl_of_stress, -0.001 * omega, rtol=0, atol=5e-13)
        np.testing.assert_allclose(error, 0, rtol=0, atol=5e-13)
    return {
        "points_per_axis": n,
        "viscosity_model": viscosity_model,
        "viscosity_min": float(viscosity.min()),
        "viscosity_max": float(viscosity.max()),
        "fvm_curl_of_stress_rms": float(np.sqrt(np.mean(np.sum(fvm_curl_of_stress**2, axis=-1)))),
        "operator_difference_rms": float(np.sqrt(np.mean(np.sum(error**2, axis=-1)))),
        "relative_operator_difference": float(np.linalg.norm(error) / np.linalg.norm(fvm_curl_of_stress)),
        "relative_difference_after_exact_solenoidal_projection": float(
            np.linalg.norm(projected_source - fvm_curl_of_stress) / np.linalg.norm(fvm_curl_of_stress)
        ),
        "maximum_velocity_divergence": float(np.max(np.abs(divergence))),
        "analytical_commutator_max_error": analytical_difference_error,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The output file must be new")
    rows = [evaluate(n, model) for n in (16, 32, 64) for model in ("constant", "smooth_variable", "equilibrium_smagorinsky")]
    report = {
        "schema": "openonda-3d-variable-viscosity-model-audit/1",
        "spatial_dimensions": 3,
        "domain": "triply periodic [0, 2π)^3",
        "field": "ABC: (sin(z)+cos(y), sin(x)+cos(z), sin(y)+cos(x)); curl(u)=u",
        "fvm_operator": "curl(div(2 nu_eff S))",
        "gbd_operator": "div(nu_eff grad(omega))",
        "shared_filter_width": 0.125,
        "results": rows,
        "sources": [hash_file(p) for p in (
            Path(__file__), ROOT / "source/solvers/fvm/assemble/momentum.py",
            ROOT / "source/solvers/vpm/physics/diffusion/grid.py",
        )],
        "limitations": [
            "Continuum-operator audit, not a cube simulation or a direct measurement of cube error.",
            "Both models receive exactly the same viscosity and velocity; actual solver discretizations are not executed.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
