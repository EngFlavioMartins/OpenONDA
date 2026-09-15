"""Separate raw from solenoidal enstrophy for frozen cube projection screens.

The existing Gaussian Fourier-integral gate counts all reconstructed particle
vorticity, including the longitudinal component that induces no velocity.
This diagnostic uses the SAME padded grid for all three strength fields and
reports the Helmholtz-orthogonal solenoidal and longitudinal energies. It is
not a change to any production acceptance criterion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from threadpoolctl import threadpool_limits

from source.solvers.vpm.numerics.fourier_integrals import gaussian_fourier_integrals
from source.solvers.vpm.stabilization.divergence_relaxation import GaussianParticleGridOperator


def _integrals(
    operator: GaussianParticleGridOperator,
    position: np.ndarray,
    strength: np.ndarray,
    core_radius: np.ndarray,
    volume: np.ndarray,
) -> dict[str, float]:
    represented = operator.smooth(operator.scatter(strength))
    solenoidal, divergence_before, _ = operator.helmholtz_project(represented)
    cell_volume = operator.spacing**3
    raw_grid_enstrophy = float(np.sum(represented**2, dtype=np.float64) * cell_volume)
    solenoidal_grid_enstrophy = float(np.sum(solenoidal**2, dtype=np.float64) * cell_volume)
    longitudinal_grid_enstrophy = float(
        np.sum((represented - solenoidal) ** 2, dtype=np.float64) * cell_volume
    )
    fourier = gaussian_fourier_integrals(
        position, strength, core_radius, volume, spacing=operator.spacing
    )
    return {
        "gaussian_fourier_energy_m5_s2": fourier.total_kinetic_energy,
        "gaussian_fourier_raw_enstrophy_m3_s2": fourier.total_enstrophy,
        "gaussian_fourier_helicity_m4_s2": fourier.total_helicity,
        "grid_raw_enstrophy_m3_s2": raw_grid_enstrophy,
        "grid_solenoidal_enstrophy_m3_s2": solenoidal_grid_enstrophy,
        "grid_longitudinal_enstrophy_m3_s2": longitudinal_grid_enstrophy,
        "grid_divergence_relative": divergence_before,
        "grid_helmholtz_orthogonality_error": abs(
            raw_grid_enstrophy - solenoidal_grid_enstrophy - longitudinal_grid_enstrophy
        )
        / max(raw_grid_enstrophy, np.finfo(float).tiny),
    }


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    with h5py.File(args.checkpoint) as saved:
        particles = saved["particles"]
        position, strength, core_radius, volume = (
            np.asarray(particles[key], dtype=np.float64)
            for key in ("position", "vortex_strength", "core_radius", "particle_volume")
        )
    candidates = {}
    for label, path in (("global", args.global_candidate), ("outer_only", args.outer_candidate)):
        with np.load(path) as stored:
            candidate = stored["vortex_strength"].astype(np.float64)
        if candidate.shape != strength.shape or not np.all(np.isfinite(candidate)):
            raise ValueError(f"{label} candidate does not match the saved cloud")
        fit = json.loads((path.parent / "screen.json").read_text())
        if fit["checkpoint_sha256"] != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
            raise ValueError(f"{label} candidate uses a different checkpoint")
        if fit["candidate_strength_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"{label} candidate checksum does not match its fitting report")
        candidates[label] = candidate
    operator = GaussianParticleGridOperator(
        position, core_radius, np.linalg.norm(strength, axis=1), spacing=0.06
    )
    results = {
        "native": _integrals(operator, position, strength, core_radius, volume),
        **{
            label: _integrals(operator, position, candidate, core_radius, volume)
            for label, candidate in candidates.items()
        },
    }
    report = {
        "scope": __doc__,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "global_candidate_sha256": hashlib.sha256(args.global_candidate.read_bytes()).hexdigest(),
        "outer_candidate_sha256": hashlib.sha256(args.outer_candidate.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "grid_shape": operator.shape,
        "grid_spacing_m": operator.spacing,
        "results": results,
        "limit": "Solenoidal/longitudinal split is a padded periodic-grid diagnostic, not an exact body-aware free-space norm. Raw Fourier values use a separate doubled grid.",
    }
    args.output.mkdir(parents=True)
    (args.output / "integrals.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--global-candidate", type=Path, required=True)
    parser.add_argument("--outer-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args)


if __name__ == "__main__":
    main()
