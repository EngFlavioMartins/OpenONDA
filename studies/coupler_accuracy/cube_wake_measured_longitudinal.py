"""Measure candidate longitudinal-rate contributions on actual saved cube particles.

This is a frozen-stage diagnostic, not a qualified particle correction. In
particular, multiplying a smooth density by particle volume and applying the
Gaussian induction kernel filters that density again. The report preserves
this limitation rather than identifying a nodal substitution with an exact
Helmholtz projection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from cube_wake_nullspace_probe import direct_velocity_rate
from cube_wake_operator_audit import curl, reflection_asymmetry
from cube_wake_particle_probe import direct_gaussian, rms
from cube_wake_vorticity_consistency import gaussian_vorticity_and_divergence
import h5py
from numba import njit, prange, set_num_threads
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits


@njit(cache=True, parallel=True)
def local_gaussian(points, position, strength, sigma, indices, offsets):
    """Sum the Gaussian density over independently selected local neighbours."""
    result = np.zeros((len(points), 3))
    for i in prange(len(points)):
        value_x = 0.0
        value_y = 0.0
        value_z = 0.0
        for slot in range(offsets[i], offsets[i + 1]):
            j = indices[slot]
            dx = points[i, 0] - position[j, 0]
            dy = points[i, 1] - position[j, 1]
            dz = points[i, 2] - position[j, 2]
            squared = dx * dx + dy * dy + dz * dz
            weight = np.exp(-squared / sigma[j] ** 2) / (np.pi**1.5 * sigma[j] ** 3)
            value_x += weight * strength[j, 0]
            value_y += weight * strength[j, 1]
            value_z += weight * strength[j, 2]
        result[i, 0] = value_x
        result[i, 1] = value_y
        result[i, 2] = value_z
    return result


def density_at(points, position, strength, sigma, cutoff=4.5):
    """Evaluate a Gaussian sum in bounded batches; validate truncation separately."""
    tree = cKDTree(position)
    output = np.empty((len(points), 3))
    for start in range(0, len(points), 1024):
        block = points[start : start + 1024]
        neighbours = tree.query_ball_point(block, cutoff * np.max(sigma), workers=2)
        offsets = np.r_[0, np.cumsum([len(row) for row in neighbours])]
        indices = np.concatenate(neighbours).astype(np.int64)
        output[start : start + len(block)] = local_gaussian(
            block, position, strength, sigma, indices, offsets
        )
    return output


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "description": __doc__,
        "input_directory": str(args.audit.resolve()),
        "times": [],
    }
    for physical_time in (6, 8):
        started = time.perf_counter()
        with np.load(args.audit / f"fields_t{physical_time}.npz") as data:
            fields = {key: data[key].astype(float) for key in data.files}
        with np.load(args.audit / f"particles_accepted_t{physical_time}.npz") as data:
            sigma = data["core_radius"].copy()
        with h5py.File(args.solution / f"vpm_{100 * physical_time:06d}.h5") as h5:
            volume = np.asarray(h5["particles/particle_volume"], dtype=float)
            np.testing.assert_array_equal(h5["particles/position"][:], fields["particle_position"])
        p, g, u, j = (
            fields[key]
            for key in (
                "particle_position",
                "particle_strength",
                "stage_velocity",
                "stage_gradient",
            )
        )
        points = fields["points"]
        omega = density_at(p, p, g, sigma)
        chosen = np.linspace(0, len(p) - 1, 24, dtype=int)
        exact_omega, _ = gaussian_vorticity_and_divergence(p[chosen], p, g, sigma)
        density_error = rms(omega[chosen] - exact_omega)
        assert density_error < 1e-6, density_error
        q = curl(j)
        longitudinal = omega - q
        rate_from_longitudinal = 2 * np.einsum("nji,nj->ni", j, longitudinal) * volume[:, None]
        native_rate = fields["stage_strength_rate"]
        native_udot = direct_velocity_rate(points, p, g, sigma, u, native_rate)
        correction_velocity, _ = direct_gaussian(points, p, rate_from_longitudinal, sigma)
        direct_rate = np.einsum("nij,nj->ni", j, g)
        direct_difference, _ = direct_gaussian(points, p, direct_rate - native_rate, sigma)
        asymmetry = reflection_asymmetry(points, fields["accepted_velocity"])
        variants = {
            "native": native_udot,
            "nodal_longitudinal_control": native_udot - correction_velocity,
            "direct_stretching_control": native_udot + direct_difference,
        }
        row = {
            "time": physical_time,
            "particles": len(p),
            "density_truncation_absolute_rms_at_24_particles": density_error,
            "regions": {},
        }
        for name, mask in {
            "renewal_seam": (points[:, 0] >= 1.25) & (points[:, 0] <= 1.62),
            "outer_wake": points[:, 0] > 1.62,
        }.items():
            metrics = {}
            for variant, rate in variants.items():
                odd = reflection_asymmetry(points, rate)
                metrics[variant] = {
                    "velocity_rate_rms": rms(rate[mask]),
                    "reflection_energy_derivative": float(
                        2 * np.mean(np.sum(asymmetry[mask] * odd[mask], axis=1))
                    ),
                    "reflection_rms_after_linear_dt_001": rms((asymmetry + 0.01 * odd)[mask]),
                }
            row["regions"][name] = metrics
        row["wall_seconds_including_diagnostics"] = time.perf_counter() - started
        report["times"].append(row)
        np.savez_compressed(
            args.output / f"fields_t{physical_time}.npz",
            points=points,
            particle_position=p,
            gaussian_vorticity=omega,
            velocity_curl=q,
            longitudinal=longitudinal,
            longitudinal_strength_rate=rate_from_longitudinal,
            correction_velocity=correction_velocity,
            **variants,
        )
        (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args)
