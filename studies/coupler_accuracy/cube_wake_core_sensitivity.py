"""Isolate kernel-width sensitivity of the remaining saved VPM plane error.

Particle positions and strengths are fixed. The saved nonparticle contribution
and tree error are held fixed by adding direct-sum velocity differences to the
saved velocity. This is an instantaneous diagnostic, not an advancing candidate
or a body-panel re-equilibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cube_wake_particle_probe import direct_gaussian, rms
import h5py
from numba import set_num_threads
import numpy as np
import pandas as pd


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    comparison = json.loads(args.comparison.read_text())
    rows = [r for r in comparison["times"] if abs(r["time"] - args.time) < 1e-8]
    if len(rows) != 1:
        raise ValueError("Expected one matched comparison state")
    plane = args.comparison.parent / f"wake_plane_z0_t{args.time:g}.csv"
    data = pd.read_csv(plane)
    points = data[["position_" + a for a in "xyz"]].to_numpy()
    reference = data[["reference_velocity_" + a for a in "xyz"]].to_numpy()
    saved = data[["candidate_velocity_" + a for a in "xyz"]].to_numpy()
    fluid = np.isfinite(reference).all(axis=1) & np.isfinite(saved).all(axis=1)
    points, reference, saved = points[fluid], reference[fluid], saved[fluid]
    state = Path(comparison["trial"]) / "solution" / f"vpm_{round(args.time * 100):06d}.h5"
    with h5py.File(state) as h5:
        if abs(float(h5["solver"].attrs["time"]) - args.time) > 1e-8:
            raise ValueError("Particle time does not match the comparison")
        position, strength, radius = (
            np.asarray(h5["particles"][key], dtype=float)
            for key in ("position", "vortex_strength", "core_radius")
        )
    original, _ = direct_gaussian(points, position, strength, radius)
    masks = {
        "wake": points[:, 0] > 0.5,
        "near_xmax": (points[:, 0] >= 1.25) & (points[:, 0] <= 1.75),
        "outer_wake": points[:, 0] > 1.5,
    }
    report = {
        "description": __doc__,
        "time": args.time,
        "comparison": str(args.comparison),
        "comparison_sha256": hashlib.sha256(args.comparison.read_bytes()).hexdigest(),
        "plane_sha256": hashlib.sha256(plane.read_bytes()).hexdigest(),
        "particle_sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
        "particles": len(position),
        "scales": [],
    }
    for scale in args.scales:
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Core scale must be positive and finite")
        altered = (
            original
            if scale == 1
            else direct_gaussian(points, position, strength, radius * scale)[0]
        )
        complete = saved + (altered - original)
        errors = {name: rms(complete[mask] - reference[mask]) for name, mask in masks.items()}
        if scale == 1:
            for name, value in errors.items():
                np.testing.assert_allclose(
                    value, rows[0]["sampled_plane"]["candidate"][name]["rms"], rtol=1e-12
                )
        row = {"core_scale": scale, "velocity_error_rms": errors}
        report["scales"].append(row)
        (args.output / "sensitivity.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.8, 1.2])
    args = parser.parse_args()
    set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
