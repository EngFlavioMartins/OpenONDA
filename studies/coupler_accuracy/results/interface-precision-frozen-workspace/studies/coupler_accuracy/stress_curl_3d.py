#!/usr/bin/env python3
"""Qualify a complete viscous-stress source on a fully 3D periodic grid.

This is an isolated candidate, not a selectable production VPM model. A
centered derivative and its negative adjoint form the strain/divergence pair.
The curl uses that same derivative. Consequently the periodic velocity work
is exactly minus the deviatoric-strain dissipation, and the vorticity source
has zero discrete divergence. Spectral differentiation of the manufactured
fields supplies an independent continuum reference.

A separate, explicitly band-limited Gaussian representation experiment checks
how a physical source must be mapped back to particle coefficients. It does
not replace free-space Biot--Savart, include bodies, or model particle scatter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from studies.coupler_accuracy.cube_boundary_oracle import ROOT, hash_file
from studies.coupler_accuracy.variable_viscosity_3d import derivative as spectral_derivative


def centered_derivative(field, axis, spacing):
    return (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (2 * spacing)


def vector_curl(vector, differentiate):
    return np.stack((
        differentiate(vector[..., 2], 1) - differentiate(vector[..., 1], 2),
        differentiate(vector[..., 0], 2) - differentiate(vector[..., 2], 0),
        differentiate(vector[..., 1], 0) - differentiate(vector[..., 0], 1),
    ), axis=-1)


def stress_source(velocity, viscosity, differentiate):
    """Return acceleration, curl(acceleration), and nonnegative dissipation.

    Tensor convention is gradient[..., i, j] = d(u_i)/d(x_j). Including the
    trace removal makes the work identity valid even for a velocity with a
    nonzero discrete divergence; incompressible fields have zero trace.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    viscosity = np.asarray(viscosity, dtype=np.float64)
    if velocity.ndim != 4 or velocity.shape[-1] != 3:
        raise ValueError("Expected a full 3D vector grid with shape (nx, ny, nz, 3)")
    if viscosity.shape != velocity.shape[:3]:
        raise ValueError("Expected one viscosity per 3D grid node")
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(viscosity)):
        raise ValueError("Velocity and viscosity must be finite")
    if np.any(viscosity < 0):
        raise ValueError("Viscosity must be nonnegative")
    gradient = np.stack([
        np.stack([differentiate(velocity[..., i], j) for j in range(3)], axis=-1)
        for i in range(3)
    ], axis=-2)
    strain = (gradient + gradient.swapaxes(-1, -2)) / 2
    strain -= np.trace(strain, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
    stress = 2 * viscosity[..., None, None] * strain
    acceleration = np.stack([
        sum(differentiate(stress[..., i, j], j) for j in range(3))
        for i in range(3)
    ], axis=-1)
    source = vector_curl(acceleration, differentiate)
    dissipation_density = 2 * viscosity * np.sum(strain**2, axis=(-1, -2))
    return acceleration, source, dissipation_density


def manufactured(n, model):
    axis = 2 * np.pi * np.arange(n) / n
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    velocity = np.stack((np.sin(z) + np.cos(y), np.sin(x) + np.cos(z), np.sin(y) + np.cos(x)), axis=-1)
    if model == "constant":
        viscosity = np.full(x.shape, 0.001)
    elif model == "smooth_variable":
        viscosity = 0.001 * (1 + 0.3 * np.cos(x) * np.cos(y) * np.cos(z))
    elif model == "equilibrium_smagorinsky":
        # Analytical strain of the ABC field. Filter width stays fixed while
        # the differentiation grid is refined.
        strain_sq = ((np.cos(x) - np.sin(y))**2 + (np.cos(y) - np.sin(z))**2
                     + (np.cos(z) - np.sin(x))**2) / 2
        viscosity = 0.001 + (0.094**1.5 / np.sqrt(1.048)) * 0.125**2 * np.sqrt(2 * strain_sq)
    else:
        raise ValueError(f"Unknown viscosity model: {model}")
    return velocity, viscosity


def heat_source(omega, viscosity, spacing):
    """Independent arithmetic-face discretization of the existing GBD PDE."""
    source = np.zeros_like(omega)
    for axis in range(3):
        for sign in (-1, 1):
            face_viscosity = (viscosity + np.roll(viscosity, sign, axis=axis)) / 2
            source += face_viscosity[..., None] * (np.roll(omega, sign, axis=axis) - omega) / spacing**2
    return source


def production_gbd_source(omega, viscosity, spacing):
    """Run the real f32 GBD variable-viscosity kernel with a periodic halo.

    Only the interior is sampled. Its six neighbors are all present, so the
    production zero-flux treatment at the allocated outer edge cannot enter
    this one-step stencil. The measured increment includes f32 cancellation.
    """
    import taichi as ti

    from source.solvers.vpm.physics.diffusion.grid import _GridDiffusionMixin

    if ti.lang.impl.get_runtime().prog is None:
        ti.init(arch=ti.cpu)
    padded_omega = np.pad(omega, ((1, 1), (1, 1), (1, 1), (0, 0)), mode="wrap").astype(np.float32)
    padded_nu = np.pad(viscosity, ((1, 1),) * 3, mode="wrap").astype(np.float32)
    shape = padded_nu.shape
    src, dst = (ti.Vector.field(3, ti.f32, shape=shape) for _ in range(2))
    nu = ti.field(ti.f32, shape=shape)
    body = ti.field(ti.i32, shape=shape)
    src.from_numpy(padded_omega)
    nu.from_numpy(padded_nu)
    body.fill(0)
    # Use a resolved increment below the production stable stage limit.
    dt = float(np.float32(min(0.25, spacing**2 / (16 * float(viscosity.max())))))
    _GridDiffusionMixin()._laplacian_step_variable_gpu_kernel(
        src, dst, nu, body, dt, spacing, *shape
    )
    increment = (dst.to_numpy().astype(np.float64) - padded_omega.astype(np.float64)) / dt
    return increment[1:-1, 1:-1, 1:-1], dt


def relative(error, reference):
    return float(np.linalg.norm(error) / np.linalg.norm(reference))


def evaluate(n, model, *, run_production=False):
    velocity, viscosity = manufactured(n, model)
    h = 2 * np.pi / n
    def central(field, axis):
        return centered_derivative(field, axis, h)
    acceleration, source, dissipation = stress_source(velocity, viscosity, central)
    _, reference, _ = stress_source(velocity, viscosity, spectral_derivative)
    omega = vector_curl(velocity, spectral_derivative)
    gbd = heat_source(omega, viscosity, h)
    divergence = sum(central(source[..., i], i) for i in range(3))
    work = np.mean(np.sum(velocity * acceleration, axis=-1))
    result = {
        "points_per_axis": n,
        "viscosity_model": model,
        "candidate_relative_source_error": relative(source - reference, reference),
        "gbd_pde_relative_source_error": relative(gbd - reference, reference),
        "candidate_max_source_divergence": float(np.max(np.abs(divergence))),
        "candidate_net_source": np.mean(source, axis=(0, 1, 2)).tolist(),
        "velocity_work_per_volume": float(work),
        "strain_dissipation_per_volume": float(dissipation.mean()),
        "relative_energy_identity_error": float(abs(work + dissipation.mean()) / dissipation.mean()),
    }
    if run_production:
        actual, dt = production_gbd_source(omega, viscosity, h)
        result.update({
            "production_gbd_relative_source_error": relative(actual - reference, reference),
            "production_gbd_relative_increment_roundoff": relative(actual - gbd, gbd),
            "production_gbd_step_size": dt,
        })
    return result


def periodic_projection(vector):
    """Orthogonal Helmholtz projection, used only by the periodic experiment."""
    n = vector.shape[0]
    wave = np.stack(np.meshgrid(*([np.fft.fftfreq(n, 1 / n)] * 3), indexing="ij"), axis=-1)
    k2 = np.sum(wave**2, axis=-1)
    value = np.fft.fftn(vector, axes=(0, 1, 2))
    longitudinal = np.divide(np.sum(wave * value, axis=-1), k2,
                             out=np.zeros_like(value[..., 0]), where=k2 > 0)
    return np.fft.ifftn(value - wave * longitudinal[..., None], axes=(0, 1, 2)).real


def gaussian_source_experiment(n=31, core_ratio=1):
    """Show the coefficient/source mapping with a band-limited 3D Gaussian.

    The continuous Fourier multiplier exp(-sigma² |k|²/4) is used on an odd
    grid, avoiding a Nyquist ambiguity. This is an exact finite-dimensional
    periodic representation test, not a claim of exact particle quadrature.
    """
    h, sigma = 2 * np.pi / n, core_ratio * 2 * np.pi / n
    wave = np.meshgrid(*([np.fft.fftfreq(n, 1 / n)] * 3), indexing="ij")
    multiplier = np.exp(-sigma**2 * sum(k**2 for k in wave) / 4)[..., None]

    def gaussian(field, *, inverse=False):
        transformed = np.fft.fftn(field, axes=(0, 1, 2))
        transformed = transformed / multiplier if inverse else transformed * multiplier
        return np.fft.ifftn(transformed, axes=(0, 1, 2)).real

    velocity, viscosity = manufactured(n, "smooth_variable")
    coefficient = gaussian(vector_curl(velocity, spectral_derivative), inverse=True)
    force, physical_source, _ = stress_source(velocity, viscosity, spectral_derivative)
    coefficient_source = gaussian(physical_source, inverse=True)
    # Directly adding a physical source to coefficients applies the Gaussian
    # to that source a second time. The corrected mapping must undo it.
    direct_source = gaussian(physical_source)
    corrected_source = gaussian(coefficient_source)
    dt = min(0.05, h**2 / (24 * viscosity.max()))
    next_coefficient = coefficient + dt * coefficient_source
    expected_omega = vector_curl(velocity + dt * force, spectral_derivative)
    result = {
        "points_per_axis": n,
        "core_to_spacing_ratio": core_ratio,
        "representation": "band-limited continuous periodic Gaussian multiplier",
        "gaussian_condition_number": float(multiplier.max() / multiplier.min()),
        "direct_physical_to_coefficient_relative_error": relative(direct_source - physical_source, physical_source),
        "mapped_source_relative_error": relative(corrected_source - physical_source, physical_source),
        "velocity_and_coefficient_step_relative_difference": relative(gaussian(next_coefficient) - expected_omega, expected_omega),
        "coefficient_source_l2_over_physical_source_l2": float(np.linalg.norm(coefficient_source) / np.linalg.norm(physical_source)),
    }
    # Explicit dissipative velocity evolution checks the intended complete
    # operator. Projection is orthogonal and preserves curl. This does not
    # execute the production particle evolution or a no-slip wall closure.
    energies, worst_identity = [], 0.0
    for _ in range(100):
        energies.append(float(np.mean(np.sum(velocity**2, axis=-1)) / 2))
        acceleration, _, dissipation = stress_source(velocity, viscosity, spectral_derivative)
        work = np.mean(np.sum(velocity * acceleration, axis=-1))
        worst_identity = max(worst_identity, abs(work + dissipation.mean()) / dissipation.mean())
        velocity = periodic_projection(velocity + dt * acceleration)
    energies.append(float(np.mean(np.sum(velocity**2, axis=-1)) / 2))
    result.update({
        "velocity_diffusion_steps": 100,
        "velocity_diffusion_dt": float(dt),
        "initial_kinetic_energy_per_volume": energies[0],
        "final_kinetic_energy_per_volume": energies[-1],
        "maximum_energy_step_increase": float(np.max(np.diff(energies))),
        "worst_relative_work_identity_error": float(worst_identity),
    })
    return result


def run(output):
    if output.exists():
        raise FileExistsError("The output must be new")
    sources = [hash_file(p) for p in (
        Path(__file__), ROOT / "studies/coupler_accuracy/variable_viscosity_3d.py",
        ROOT / "source/solvers/vpm/physics/diffusion/grid.py",
        ROOT / "source/solvers/fvm/assemble/momentum.py",
    )]
    rows = []
    for n in (16, 32, 64):
        for model in ("constant", "smooth_variable", "equilibrium_smagorinsky"):
            row = evaluate(n, model, run_production=True)
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "schema": "openonda-3d-stress-curl-candidate/1",
        "spatial_dimensions": 3,
        "domain": "triply periodic [0, 2π)^3",
        "status": "component_experiment_only",
        "results": rows,
        "gaussian_representation": gaussian_source_experiment(),
        "sources": sources,
        "limitations": [
            "The candidate is not wired into production GBD or the cube solver.",
            "Centered differences have checkerboard null modes; this is not a final grid layout.",
            "The Gaussian test is band limited and periodic, with no free-space boundaries, particle scatter, pruning, or body mask.",
            "A body-complete velocity, compatible coefficient mapping, and wall treatment remain required for a production implementation.",
            "The actual f32 GBD kernel is measured only through its one-step interior stencil, with a periodic halo.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["gaussian_representation"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
