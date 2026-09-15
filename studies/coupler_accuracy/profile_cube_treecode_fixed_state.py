"""Measure fixed-state GPU treecode accuracy and traversal cost on a native VPM backup.

This script does not advance either solver. It compares induced particle velocity
and its Cartesian Jacobian while holding centres, strengths and cores fixed.

Usage:
    python studies/coupler_accuracy/profile_cube_treecode_fixed_state.py \
        --checkpoint tutorials/coupled_fvm_vpm/02_cube_flow/solution/backups/vpm_001550.h5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import h5py
import numpy as np
import taichi as ti

from source.solvers.vpm.physics.induction.treecode.lbvh import TaichiTreecode
from source.solvers.vpm.runtime.backend import initialize_taichi_backend


def _relative_rms(actual: np.ndarray, baseline: np.ndarray) -> float:
    """Return the float64 L2 relative difference of complete fixed-state fields."""
    difference = actual.astype(np.float64) - baseline.astype(np.float64)
    return float(np.linalg.norm(difference) / np.linalg.norm(baseline))


def _evaluate(tree: TaichiTreecode, count: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Synchronize one all-particle velocity/Jacobian traversal and copy its result."""
    started = perf_counter()
    tree.compute_velocity_and_gradient_gpu()
    ti.sync()
    seconds = perf_counter() - started
    return (
        seconds,
        tree.velocity.to_numpy()[:count].copy(),
        tree.velocity_gradient.to_numpy()[:count].copy(),
    )


def run(checkpoint: Path) -> dict[str, object]:
    """Compare opening angles and target schedules for one saved 3D cloud.

    Parameters
    ----------
    checkpoint : pathlib.Path
        Native VPM HDF5 backup containing particle centres [m], vector
        strengths [m³/s] and Gaussian core radii [m]. The file is read only.

    Returns
    -------
    dict
        Fixed-state Metal timings [s] and field differences against the
        production order-one opening angle 0.1. Timings include traversal
        synchronization but exclude initial Taichi compilation, tree build,
        host result downloads and any coupled-solver work.
    """
    with h5py.File(checkpoint, "r") as file:
        position = np.asarray(file["particles/position"], dtype=np.float32)
        strength = np.asarray(file["particles/vortex_strength"], dtype=np.float32)
        radius = np.asarray(file["particles/core_radius"], dtype=np.float32)
    count = len(position)
    if (
        count == 0
        or strength.shape != (count, 3)
        or radius.shape != (count,)
        or position.shape != (count, 3)
        or not np.isfinite(position).all()
        or not np.isfinite(strength).all()
        or not np.isfinite(radius).all()
        or np.any(radius <= 0)
    ):
        raise ValueError(
            "checkpoint requires finite, nonempty 3D particle fields and positive cores"
        )
    near_body = np.all(np.abs(position) <= 1.5, axis=1)
    if not near_body.any():
        raise ValueError("checkpoint has no particle targets in the near-body mask")
    with TemporaryDirectory(prefix="openonda-treecode-cache-") as cache_directory:
        os.environ["TI_OFFLINE_CACHE_FILE_PATH"] = cache_directory
        backend = initialize_taichi_backend("METAL", precision="f32")
        tree = TaichiTreecode(
            max_n_particles=count,
            max_nodes=2 * count,
            theta=0.1,
            kernel_type="GAUSSIAN",
            multipole_order=1,
        )
        tree.build(position, strength, radius)
        timings: dict[str, object] = {}
        reference = None
        for angle in (0.1, 0.15, 0.2, 0.3):
            tree.theta = angle
            seconds, velocity, gradient = _evaluate(tree, count)
            if reference is None:
                reference = (velocity, gradient)
            baseline_velocity, baseline_gradient = reference
            timings[str(angle)] = {
                "evaluation_seconds": seconds,
                "all_velocity_relative_rms": _relative_rms(velocity, baseline_velocity),
                "all_gradient_relative_rms": _relative_rms(gradient, baseline_gradient),
                "near_body_velocity_relative_rms": _relative_rms(
                    velocity[near_body], baseline_velocity[near_body]
                ),
                "near_body_gradient_relative_rms": _relative_rms(
                    gradient[near_body], baseline_gradient[near_body]
                ),
            }
        tree.theta = 0.1
        tree.set_sort_particle_targets(True)
        sorted_seconds, sorted_velocity, sorted_gradient = _evaluate(tree, count)
        timings["morton_sorted_targets"] = {
            "evaluation_seconds": sorted_seconds,
            "maximum_velocity_difference": float(np.abs(sorted_velocity - baseline_velocity).max()),
            "maximum_gradient_difference": float(np.abs(sorted_gradient - baseline_gradient).max()),
        }
    return {
        "checkpoint": str(checkpoint),
        "backend": backend,
        "particles": count,
        "near_body_targets": int(near_body.sum()),
        "opening_angle_baseline": 0.1,
        "multipole_order": 1,
        "measurements": timings,
    }


def main() -> None:
    """Print one read-only, fixed-state cost and field-difference JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
