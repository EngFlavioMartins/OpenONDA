"""Audit when an outer-only Gaussian fit becomes eligible in the cube wake.

The original accepted particle checkpoints and physical renewal bounds are
unchanged. This study applies the fixed-basis operator independently at each
saved time and compares its correction with the current production physics
gates. It does not advance either solver or apply a correction to a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cube_fixed_basis_runtime_fit import fit_outer_strength
import h5py
import numpy as np
from threadpoolctl import threadpool_limits

from source.solvers.vpm.numerics.fourier_integrals import gaussian_fourier_integrals
from source.solvers.vpm.stabilization.divergence_relaxation import (
    GaussianParticleGridOperator,
    _MomentNullspace,
    gaussian_invariant_rows,
)
from source.solvers.vpm.stabilization.filament_refinement import gaussian_particle_moments


def screen_one(checkpoint: Path) -> dict:
    """Measure one accepted 3D cloud against unmodified divergence-fit gates."""
    with h5py.File(checkpoint) as saved:
        time_s = float(saved["solver"].attrs["time"])
        particles = saved["particles"]
        position, strength, radius, volume = (
            np.asarray(particles[name], dtype=np.float64)
            for name in ("position", "vortex_strength", "core_radius", "particle_volume")
        )
    if not np.allclose(radius, 0.066, atol=1e-7, rtol=0):
        raise ValueError(f"Checkpoint {checkpoint} changed the Gaussian core radius")
    if not np.allclose(volume, 0.06**3, atol=1e-9, rtol=0):
        raise ValueError(f"Checkpoint {checkpoint} changed the particle volume")
    bounds = np.array([-1.385, 1.385] * 3)
    outer = np.any((position < bounds[::2]) | (position > bounds[1::2]), axis=1)
    record = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "time_s": time_s,
        "particles": len(position),
        "vpm_owned_outer_particles": int(outer.sum()),
    }
    if not outer.any():
        record["status"] = "no_outer_particles"
        return record
    candidate, fit = fit_outer_strength(
        position,
        strength,
        radius,
        volume,
        operator_type=GaussianParticleGridOperator,
        moment_nullspace_type=_MomentNullspace,
        gaussian_moment_rows=gaussian_invariant_rows,
        renewal_bounds=bounds,
    )
    before = gaussian_fourier_integrals(position, strength, radius, volume, spacing=0.06)
    after = gaussian_fourier_integrals(position, candidate, radius, volume, spacing=0.06)
    before_moments = gaussian_particle_moments(position, strength, radius)
    after_moments = gaussian_particle_moments(position, candidate, radius)
    energy_error = (
        after.total_kinetic_energy - before.total_kinetic_energy
    ) / before.total_kinetic_energy
    enstrophy_error = (after.total_enstrophy - before.total_enstrophy) / before.total_enstrophy
    helicity_error = (after.total_helicity - before.total_helicity) / np.sqrt(
        before.total_kinetic_energy * before.total_enstrophy
    )
    variation_error = (after_moments[1] - before_moments[1]) / before_moments[1]
    record.update(
        status="fitted_without_mutation",
        fit=fit,
        gaussian_fourier_energy_before_after=[
            before.total_kinetic_energy,
            after.total_kinetic_energy,
        ],
        gaussian_fourier_raw_enstrophy_before_after=[before.total_enstrophy, after.total_enstrophy],
        gaussian_fourier_helicity_before_after=[before.total_helicity, after.total_helicity],
        energy_relative_change=float(energy_error),
        raw_enstrophy_relative_change=float(enstrophy_error),
        helicity_change_normalized=float(helicity_error),
        total_variation_relative_change=float(variation_error),
        raw_moment_errors=[
            float(np.linalg.norm(after_moments[index] - before_moments[index]))
            for index in (0, 2, 3)
        ],
        unrestored_fit_within_gate_numbers={
            "correction_norm": bool(fit["correction_strength_relative"] <= 0.02),
            "kinetic_energy": bool(abs(energy_error) <= 1e-6),
            "raw_enstrophy": bool(abs(enstrophy_error) <= 1e-4),
            "helicity": bool(abs(helicity_error) <= 1e-4),
            "total_variation": bool(abs(variation_error) <= 1e-3),
            "residual_ratio": bool(fit["final_residual_ratio"] <= 0.9),
        },
    )
    return record


def main() -> None:
    """Write a checksum-bearing multi-checkpoint operator screen as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with threadpool_limits(limits=4):
        records = [screen_one(path) for path in args.checkpoint]
    if any(
        records[index]["time_s"] >= records[index + 1]["time_s"]
        for index in range(len(records) - 1)
    ):
        raise ValueError("Accepted cube checkpoints must be in strictly increasing time order")
    report = {
        "scope": __doc__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fit_source_sha256": hashlib.sha256(
            Path(__file__).with_name("cube_fixed_basis_runtime_fit.py").read_bytes()
        ).hexdigest(),
        "renewal_bounds_m": [-1.385, 1.385] * 3,
        "single_sweep_gate_limits": {
            "correction_norm": 0.02,
            "kinetic_energy": 1e-6,
            "raw_enstrophy": 1e-4,
            "helicity": 1e-4,
            "total_variation": 1e-3,
            "residual_ratio": 0.9,
        },
        "records": records,
        "limit": "Independent frozen float64 fits, not a clean-start correction history or a production gate decision; the current solver also restores quadratic invariants and searches correction amplitudes.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
