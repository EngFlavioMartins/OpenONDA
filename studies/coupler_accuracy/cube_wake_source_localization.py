"""Locate the sources of the saved cube's transverse symmetry defect.

The plane is an observation surface of the full 3D Gaussian particle field.
Linear Biot-Savart source partitions add exactly; they are not separate flows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cube_wake_particle_probe import direct_gaussian, rms
import h5py
from numba import set_num_threads
import numpy as np
import pandas as pd


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"description": __doc__, "times": []}
    for time in args.times:
        table = pd.read_csv(args.comparison / f"wake_plane_z0_t{time:g}.csv")
        points = table[["position_" + a for a in "xyz"]].to_numpy()
        mask = (points[:, 0] >= 1.25) & (points[:, 0] <= 1.75)
        points = points[mask]
        reference = table.loc[mask, ["reference_velocity_" + a for a in "xyz"]].to_numpy()
        saved = table.loc[mask, ["candidate_velocity_" + a for a in "xyz"]].to_numpy()
        error = saved - reference
        peak = int(np.argmax(np.linalg.norm(error, axis=1)))
        with h5py.File(args.solution / f"vpm_{round(100 * time):06d}.h5") as file:
            particle = file["particles"]
            p, g, sigma = (
                np.asarray(particle[k], dtype=float)
                for k in ("position", "vortex_strength", "core_radius")
            )
        distance = np.max(np.abs(p), axis=1) - 0.5
        labels = np.full(len(p), "upstream_and_lateral", dtype=object)
        labels[p[:, 0] >= 0.89] = "authority_ramp"
        labels[p[:, 0] >= 1.25] = "renewal_seam"
        labels[p[:, 0] > 1.65] = "outer_wake"
        labels[np.max(np.abs(p), axis=1) < 0.75] = "near_body"
        contributions = {}
        for name in sorted(set(labels)):
            m = labels == name
            contributions[name], _ = direct_gaussian(points, p[m], g[m], sigma[m])
        total = sum(contributions.values())
        row = {
            "time": time,
            "minimum_particle_cube_clearance": float(distance.min()),
            "particles_within_0_12D_of_cube": int((distance < 0.12).sum()),
            "peak_error_position": points[peak].tolist(),
            "peak_error": error[peak].tolist(),
            "saved_plane_uz_rms": rms(saved[:, 2:]),
            "reference_plane_uz_rms": rms(reference[:, 2:]),
            "reverse_flow_probe_count": int((saved[:, 0] < 0).sum()),
            "source_regions": {},
        }
        for name, velocity in contributions.items():
            row["source_regions"][name] = {
                "particles": int((labels == name).sum()),
                "uz_rms": rms(velocity[:, 2:]),
                "peak_velocity": velocity[peak].tolist(),
                "signed_fraction_of_induced_uz_energy": float(
                    np.sum(velocity[:, 2] * total[:, 2]) / np.sum(total[:, 2] ** 2)
                ),
            }
        fractions = [
            v["signed_fraction_of_induced_uz_energy"] for v in row["source_regions"].values()
        ]
        np.testing.assert_allclose(sum(fractions), 1.0, rtol=0, atol=1e-12)
        np.savez_compressed(
            args.output / f"fields_t{time:g}.npz",
            points=points,
            reference=reference,
            saved=saved,
            **contributions,
        )
        report["times"].append(row)
        (args.output / "localization.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", default=[1, 3, 5, 6, 7, 8])
    args = parser.parse_args()
    set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
