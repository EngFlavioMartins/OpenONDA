"""Check physical agreement preservation in the current vorticity blend.

Both solvers describe zero velocity and zero physical curl. VPM additionally
has a Gaussian-gradient density invisible to Biot-Savart. Applying the actual
cube authority ramp must not manufacture velocity from this representation.
The refined quadrature keeps the particle core and continuum input fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cube_wake_particle_probe import direct_gaussian, rms
from numba import set_num_threads
import numpy as np
from threadpoolctl import threadpool_limits

from source.coupler.stable_renewal import blend_represented_state, inward_cosine_authority


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    records = []
    centre = np.array([1.08, 0, 0])
    axes = (np.linspace(0.66, 1.5, 8), np.linspace(-0.36, 0.36, 5), np.linspace(-0.36, 0.36, 5))
    targets = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    for h in (0.06, 0.03):
        width = 0.12
        axis = np.arange(-round(7 * width / h), round(7 * width / h) + 1) * h
        local = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
        position = local + centre
        potential = width * np.exp(0.5) * np.exp(-np.sum(local**2, axis=1) / (2 * width**2))
        strength = -local * (potential / width**2)[:, None] * h**3
        sigma = np.full(len(position), 0.066)
        before, _ = direct_gaussian(targets, position, strength, sigma)
        authority = inward_cosine_authority(position, np.array([-1.25, 1.25] * 3), 0.36)
        variants = {}
        for name, weights in {
            "cube_authority": authority,
            "constant_authority_control": np.full(len(position), 0.5),
        }.items():
            result = blend_represented_state(
                strength,
                np.zeros_like(strength),
                weights,
                (len(axis),) * 3,
                h,
                core_radius=0.066,
                amplification_cap=1.8,
            )
            after, _ = direct_gaussian(targets, position, result.vortex_strength, sigma)
            variants[name] = {
                "velocity_after_rms": rms(after),
                "velocity_change_rms": rms(after - before),
                "velocity_after_max": float(np.linalg.norm(after, axis=1).max()),
            }
            np.savez_compressed(
                output / f"{name}_h{h:g}.npz", points=targets, before=before, after=after
            )
        record = {
            "quadrature_spacing": h,
            "core_radius": 0.066,
            "particle_count": len(position),
            "initial_velocity_rms": rms(before),
            "fvm_velocity": [0, 0, 0],
            "fvm_physical_vorticity": [0, 0, 0],
            "variants": variants,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        (output / "probe.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_num_threads(2)
    with threadpool_limits(limits=2):
        run(args.output)
