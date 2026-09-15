"""Falsify a fixed-basis projection on a compact, genuinely helical 3D field.

Construct a solenoidal flow from the curl of a compact Beltrami-type vector
potential on a regular 3D lattice. Add a sampled gradient field that should
induce no physical velocity or helicity. Fit on unchanged Gaussian particle
centres, then compare the represented helicity and longitudinal content with
the clean helical reference. This is a body-free numerical operator test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from threadpoolctl import threadpool_limits

from source.solvers.vpm.numerics.fourier_integrals import gaussian_fourier_integrals
from source.solvers.vpm.stabilization.divergence_relaxation import (
    GaussianParticleGridOperator,
    _MomentNullspace,
    gaussian_invariant_rows,
)
from source.solvers.vpm.stabilization.filament_refinement import gaussian_particle_moments


def _curl(field: np.ndarray, wave: tuple[np.ndarray, ...]) -> np.ndarray:
    transformed = [np.fft.fftn(field[..., axis]) for axis in range(3)]
    return np.stack(
        [
            np.fft.ifftn(
                1j
                * (
                    wave[(axis + 1) % 3] * transformed[(axis + 2) % 3]
                    - wave[(axis + 2) % 3] * transformed[(axis + 1) % 3]
                )
            ).real
            for axis in range(3)
        ],
        axis=-1,
    )


def _gradient(field: np.ndarray, wave: tuple[np.ndarray, ...]) -> np.ndarray:
    transformed = np.fft.fftn(field)
    return np.stack(
        [np.fft.ifftn(1j * wave[axis] * transformed).real for axis in range(3)], axis=-1
    )


def _grid_metric(operator: GaussianParticleGridOperator, strength: np.ndarray) -> dict[str, float]:
    represented = operator.smooth(operator.scatter(strength))
    solenoidal, divergence, _ = operator.helmholtz_project(represented)
    cell_volume = operator.spacing**3
    return {
        "represented_divergence_relative": divergence,
        "longitudinal_enstrophy_m3_s2": float(
            np.sum((represented - solenoidal) ** 2, dtype=np.float64) * cell_volume
        ),
        "solenoidal_enstrophy_m3_s2": float(np.sum(solenoidal**2, dtype=np.float64) * cell_volume),
    }


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    n, h = 33, 0.06
    coordinates = (np.arange(n) - n // 2) * h
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    position = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    length = n * h
    k = 2.0 * np.pi / length
    frequencies = 2.0 * np.pi * np.fft.fftfreq(n, d=h)
    wave = tuple(
        np.reshape(frequencies, tuple(n if j == axis else 1 for j in range(3))) for axis in range(3)
    )
    radius = np.sqrt(x * x + y * y + z * z)
    support = np.clip(1.0 - (radius / 0.75) ** 2, 0.0, None) ** 4
    beltrami = np.stack(
        (
            np.sin(k * z) + np.cos(k * y),
            np.sin(k * x) + np.cos(k * z),
            np.sin(k * y) + np.cos(k * x),
        ),
        axis=-1,
    )
    vector_potential = support[..., None] * beltrami
    physical_velocity = _curl(vector_potential, wave)
    clean_vorticity = _curl(physical_velocity, wave)
    scalar_potential = support * np.cos(k * x) * np.sin(k * y)
    longitudinal = _gradient(scalar_potential, wave)
    longitudinal_scale = 0.25 * np.linalg.norm(clean_vorticity) / np.linalg.norm(longitudinal)
    longitudinal *= longitudinal_scale
    physical_helicity = float(np.sum(physical_velocity * clean_vorticity, dtype=np.float64) * h**3)
    gradient_orthogonality = float(
        np.sum(physical_velocity * longitudinal, dtype=np.float64) * h**3
    )
    clean = (h**3 * clean_vorticity).reshape(-1, 3)
    contaminated = (h**3 * (clean_vorticity + longitudinal)).reshape(-1, 3)
    core_radius = np.full(len(position), 1.1 * h)
    volume = np.full(len(position), h**3)
    started = perf_counter()
    operator = GaussianParticleGridOperator(
        position, core_radius, np.linalg.norm(contaminated, axis=1), spacing=h
    )
    residual, _, _ = operator.relaxation_residual(contaminated)
    sqrt_volume = np.sqrt(volume)
    nullspace = _MomentNullspace(gaussian_invariant_rows(position, core_radius), volume)
    iterations = 0

    def matrix_vector(vector: np.ndarray) -> np.ndarray:
        values = nullspace.project_transformed(vector.reshape(-1, 3))
        transformed = sqrt_volume[:, None] * operator.apply(sqrt_volume[:, None] * values)
        return (nullspace.project_transformed(transformed) + 0.1 * values).ravel()

    def count_iteration(_vector: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    system = LinearOperator(
        (3 * len(position), 3 * len(position)), matvec=matrix_vector, dtype=np.float64
    )
    rhs = nullspace.project_transformed(sqrt_volume[:, None] * residual)
    solution, info = cg(
        system, rhs.ravel(), rtol=1e-5, atol=0.0, maxiter=30, callback=count_iteration
    )
    if info != 0:
        raise RuntimeError(f"Helical manufactured projection failed to converge: {info}")
    correction = nullspace.to_correction(solution.reshape(-1, 3))
    candidate = contaminated + correction
    fit_wall_seconds = perf_counter() - started
    states = {
        "clean": clean,
        "contaminated": contaminated,
        "candidate": candidate,
    }
    metrics = {}
    for name, strength in states.items():
        fourier = gaussian_fourier_integrals(position, strength, core_radius, volume, spacing=h)
        metrics[name] = {
            **_grid_metric(operator, strength),
            "fourier_helicity_m4_s2": fourier.total_helicity,
            "fourier_energy_m5_s2": fourier.total_kinetic_energy,
            "fourier_raw_enstrophy_m3_s2": fourier.total_enstrophy,
        }
    before_moments = gaussian_particle_moments(position, contaminated, core_radius)
    after_moments = gaussian_particle_moments(position, candidate, core_radius)
    report = {
        "scope": __doc__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "grid_spacing_m": h,
        "core_radius_m": 1.1 * h,
        "particle_count": len(position),
        "compact_support_radius_m": 0.75,
        "helical_wavenumber_m_inv": k,
        "contamination_rms_ratio": 0.25,
        "physical_reference_helicity_m4_s2": physical_helicity,
        "physical_gradient_velocity_orthogonality_m4_s2": gradient_orthogonality,
        "cg_iterations": iterations,
        "fit_wall_seconds": fit_wall_seconds,
        "correction_strength_relative": float(np.linalg.norm(correction))
        / float(np.linalg.norm(contaminated)),
        "moment_change_net_m3_s": (after_moments[0] - before_moments[0]).tolist(),
        "moment_change_linear_impulse_m4_s": (after_moments[2] - before_moments[2]).tolist(),
        "moment_change_angular_impulse_m5_s": (after_moments[3] - before_moments[3]).tolist(),
        "metrics": metrics,
        "limit": "Spectral manufactured flow is periodic, while the particle fit and Fourier-integral audit use padded finite-support grids; quadrature and Gaussian smoothing are part of the test.",
    }
    args.output.mkdir(parents=True)
    (args.output / "manufactured.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args)


if __name__ == "__main__":
    main()
