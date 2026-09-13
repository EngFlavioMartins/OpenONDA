"""Read-only mature-wake GPU traversal benchmark, separate from physical evolution."""

import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from source.solvers.vpm.physics.base import PhysicsBase
from source.solvers.vpm.physics.induction.treecode.evaluator import TreecodeInduction


def main():
    """Measure identical fixed sources and save fields for permutation/error checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sort", action="store_true")
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--tag", help="Fresh evidence label; defaults to the scheduling parameters")
    args = parser.parse_args()
    tag = args.tag or f"mature_induction_sort{int(args.sort)}_block{args.block}"
    if Path(tag).name != tag:
        raise ValueError("tag must be a simple filename")
    base = Path(__file__).with_name(tag)
    if base.with_suffix(".json").exists():
        raise FileExistsError(base)
    with h5py.File(ROOT / "tutorials/vpm/06_rotor_flow_PENDING/solution/vpm_001152.h5") as archive:
        x, gamma, core = [archive[f"particles/{name}"][:] for name in ("position", "vortex_strength", "core_radius")]
    count = len(x)
    ti.init(arch=ti.metal, default_fp=ti.f32, offline_cache=False)
    try:
        physics = PhysicsBase(particle_kernel="GAUSSIAN", max_n_particles=count, accumulator_dtype=ti.f32)
        backend = TreecodeInduction._for_testing(theta=.3, multipole_order=3, sort_particle_targets=args.sort, traversal_block_dim=args.block).bind(physics)
        position, strength, velocity, rate = [ti.Vector.field(3, ti.f32, shape=count) for _ in range(4)]
        radius = ti.field(ti.f32, shape=count)
        gradient = ti.Matrix.field(3, 3, ti.f32, shape=count)
        position.from_numpy(x)
        strength.from_numpy(gamma)
        radius.from_numpy(core)
        durations = []
        for _ in range(3):
            ti.sync()
            started = time.perf_counter()
            backend.evaluate_stage(position=position, vortex_strength=strength, core_radius=radius,
                                   count=count, velocity_out=velocity, vortex_strength_rate_out=rate,
                                   velocity_gradient_out=gradient)
            ti.sync()
            durations.append(time.perf_counter() - started)
        np.savez_compressed(base.with_suffix(".npz"), velocity=velocity.to_numpy(), gradient=gradient.to_numpy(), rate=rate.to_numpy())
        result = {"particles": count, "sort": args.sort, "block": args.block, "seconds": durations,
                  "tree_depth": int(physics._treecode._max_depth[None]),
                  "warm_mean_seconds": float(np.mean(durations[1:]))}
        base.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        ti.reset()


if __name__ == "__main__":
    main()
