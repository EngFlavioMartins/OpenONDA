"""Compare physical local helicity at 3D cube probes with the fine FVM state.

The velocity-curl product uses the VPM body's complete target velocity and
the independently differentiated Gaussian particle curl. The harmonic body
velocity has zero analytic curl. Fine FVM vorticity is interpolated from its
native cell field with the same affine 12-cell probe used for fine velocity.
This is a local field test, not a global helicity-conservation assertion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cube_wake_drift_audit import comparison_geometry, frame, ordered_fields
from cube_wake_operator_audit import curl
from cube_wake_particle_probe import direct_gaussian, rms
import h5py
from numba import set_num_threads
import numpy as np
from threadpoolctl import threadpool_limits

from source.solvers.fvm.sampling.fields import _PointProbe


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    with np.load(args.body_fields) as stored:
        points = stored["points"].astype(np.float64)
        baseline_velocity = stored["baseline_complete"].astype(np.float64)
        candidate_velocity = stored["candidate_complete"].astype(np.float64)
        cached_reference = stored["reference"].astype(np.float64)
    with h5py.File(args.checkpoint) as saved:
        particles = saved["particles"]
        position, baseline_strength, radius = (
            np.asarray(particles[key], dtype=np.float64)
            for key in ("position", "vortex_strength", "core_radius")
        )
    with np.load(args.candidate) as stored:
        candidate_strength = stored["vortex_strength"].astype(np.float64)
    if candidate_strength.shape != baseline_strength.shape:
        raise ValueError("Candidate does not match the saved particle cloud")
    mesh_path = args.fine_solution / "mesh.npz"
    geometry = comparison_geometry(mesh_path, args.output / "geometry-cache")
    cell_centre = geometry["cell_centre"]
    native_fields = ordered_fields(frame(args.fine_solution, "fine", 6.0), len(cell_centre))
    probe = _PointProbe(points, k=12, reconstruction="affine")
    reference_velocity = probe._interpolate(native_fields["velocity"], cell_centre)
    reference_vorticity = probe._interpolate(native_fields["vorticity"], cell_centre)
    if not np.allclose(reference_velocity, cached_reference, rtol=0, atol=1e-9):
        raise ValueError("Fresh fine FVM velocity does not match the saved probe cache")
    _, baseline_jacobian = direct_gaussian(points, position, baseline_strength, radius)
    _, candidate_jacobian = direct_gaussian(points, position, candidate_strength, radius)
    reference_helicity = np.sum(reference_velocity * reference_vorticity, axis=1)
    baseline_helicity = np.sum(baseline_velocity * curl(baseline_jacobian), axis=1)
    candidate_helicity = np.sum(candidate_velocity * curl(candidate_jacobian), axis=1)
    masks = {
        "authority_ramp": points[:, 0] < 1.25,
        "renewal_seam": (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62),
        "outer_wake": points[:, 0] > 1.62,
    }
    report = {
        "scope": __doc__,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "fine_pvtu_sha256": hashlib.sha256(
            frame(args.fine_solution, "fine", 6.0).read_bytes()
        ).hexdigest(),
        "fine_mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "body_fields_sha256": hashlib.sha256(args.body_fields.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fresh_reference_velocity_match_max_m_s": float(
            np.max(np.abs(reference_velocity - cached_reference))
        ),
        "regions": {
            name: {
                "points": int(mask.sum()),
                "reference_helicity_mean_m_s2": float(np.mean(reference_helicity[mask])),
                "baseline_helicity_mean_m_s2": float(np.mean(baseline_helicity[mask])),
                "candidate_helicity_mean_m_s2": float(np.mean(candidate_helicity[mask])),
                "baseline_local_helicity_error_rms_m_s2": float(
                    np.sqrt(np.mean((baseline_helicity[mask] - reference_helicity[mask]) ** 2))
                ),
                "candidate_local_helicity_error_rms_m_s2": float(
                    np.sqrt(np.mean((candidate_helicity[mask] - reference_helicity[mask]) ** 2))
                ),
                "reference_vorticity_rms_s_inv": rms(reference_vorticity[mask]),
            }
            for name, mask in masks.items()
        },
        "limit": "Fine FVM cell vorticity and VPM analytic curl are different reconstructions; local helicity is a validation field, not an exact shared invariant.",
    }
    (args.output / "local_helicity.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output / "local_helicity_fields.npz",
        points=points,
        reference_vorticity=reference_vorticity,
        reference_helicity=reference_helicity,
        baseline_helicity=baseline_helicity,
        candidate_helicity=candidate_helicity,
    )
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--body-fields", type=Path, required=True)
    parser.add_argument("--fine-solution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_num_threads(4)
    with threadpool_limits(limits=4):
        run(args)


if __name__ == "__main__":
    main()
