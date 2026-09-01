#!/usr/bin/env python3
"""
Tier-0 measurement harness (OpenONDA_VPM_GPU_plan_v2.md §2 Tier 0).

Times the self-induced velocity evaluation by **DIRECT** (naive O(N²)) vs the
**LBVH treecode** across N, and breaks the treecode into *build* vs *traverse*.
The deliverable is the **crossover N** — below it a (future, tiled) direct kernel
likely wins outright; above it the treecode earns its complexity. The 2A-vs-2B
strategy decision in the plan is meant to be made from these numbers, on the GPU
you actually run, **not** from preference.

This is intentionally standalone (no full Solver) so it boots anywhere. Run it on
your real backend, e.g.::

    python scripts/benchmarks/bench_velocity_methods.py --arch metal  --n 5000 49000 200000
    python scripts/benchmarks/bench_velocity_methods.py --arch vulkan --n 5000 49000 200000
    python scripts/benchmarks/bench_velocity_methods.py --arch cuda   --n 5000 49000 200000

The direct kernel here mirrors the production Winckelmans ``q`` kernel and the
treecode's leaf sum, so the timing is apples-to-apples (and the two are sanity-
checked against each other at the smallest N).

Interpreting the result (plan §0, §2):
  * If wall-clock at your working N is dominated by the treecode *build* (Karras
    NSL/NSR serial passes + per-build CPU argsort + ~8 ti.sync barriers), you are
    launch/sync-bound — kernel fusion / sync removal / a parallel build beats any
    accuracy tweak.
  * If DIRECT is competitive at your N, Tier 1 (tile the direct kernel with
    block-local shared memory — CUDA only; Vulkan has no portable shared-mem
    path) may be the whole job.
"""

import argparse
import time

import numpy as np
import taichi as ti

ONE_OVER_FOUR_PI = 0.07957747154594767


@ti.data_oriented
class DirectEvaluator:
    """Naive O(N²) Winckelmans self-induced velocity (the current DIRECT path)."""

    def __init__(self, max_n: int):
        self.position = ti.Vector.field(3, ti.f32, shape=max_n)
        self.vortex_strength = ti.Vector.field(3, ti.f32, shape=max_n)
        self.core_radius = ti.field(ti.f32, shape=max_n)
        self.out = ti.Vector.field(3, ti.f32, shape=max_n)

    def load(self, position, vortex_strength, core_radius):
        n = len(position)
        mx = self.position.shape[0]
        self.position.from_numpy(np.pad(position, ((0, mx - n), (0, 0))).astype(np.float32))
        self.vortex_strength.from_numpy(
            np.pad(vortex_strength, ((0, mx - n), (0, 0))).astype(np.float32)
        )
        self.core_radius.from_numpy(np.pad(core_radius, ((0, mx - n),)).astype(np.float32))

    @ti.kernel
    def velocity(self, N: ti.i32):
        for i in range(N):
            vel = ti.Vector([0.0, 0.0, 0.0])
            pi = self.position[i]
            ri = self.core_radius[i]
            for j in range(N):
                rij = pi - self.position[j]
                r2 = rij.dot(rij)
                rm = ti.sqrt(r2)
                if rm > 1e-10:
                    sigma = 0.5 * (ri + self.core_radius[j])
                    rs = rm / sigma
                    rs2 = rs * rs
                    q = rs * rs * rs * (rs2 + 2.5) / ti.pow(rs2 + 1.0, 2.5) * ONE_OVER_FOUR_PI
                    vel -= q * rij.cross(self.vortex_strength[j]) / (r2 * rm)
            self.out[i] = vel


def _seed(N, seed=1):
    rng = np.random.default_rng(seed)
    position = (rng.random((N, 3)) - 0.5).astype(np.float32)
    vortex_strength = (rng.normal(size=(N, 3)) * 0.1).astype(np.float32)
    core_radius = np.full(N, 0.05, dtype=np.float32)
    return position, vortex_strength, core_radius


def _time(fn, repeats):
    fn()  # warm-up / JIT
    ti.sync()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ti.sync()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples)) * 1e3  # ms


def run(arch_str, sizes, theta, repeats):
    # Initialise via the project backend (config/backend.py), exactly as the
    # solver does — a raw ``ti.init(arch=ti.vulkan)`` triggers Taichi's adaptive
    # arch probe, which can hang on some Vulkan setups.
    from source.solvers.vpm.runtime.backend import initialize_taichi_backend

    backend = {
        "auto": "AUTO",
        "cpu": "CPU",
        "metal": "METAL",
        "cuda": "CUDA",
        "vulkan": "VULKAN",
    }[arch_str]
    chosen = initialize_taichi_backend(preferred_backend=backend, debug_mode=False, precision="f32")
    print(f"[backend] requested={backend} → using {chosen}")
    from source.solvers.vpm.physics.induction.treecode.lbvh import TaichiTreecode

    sizes = sorted(sizes)
    direct = DirectEvaluator(max(sizes) + 8)
    bg = np.zeros(3, dtype=np.float32)

    print(f"\narch={arch_str}  theta={theta}  repeats={repeats}")
    print(
        f"{'N':>8} | {'direct ms':>10} | {'tc build ms':>11} | {'tc eval ms':>10} | "
        f"{'tc total ms':>11} | {'direct/tc':>9}"
    )
    print("-" * 78)

    crossover = None
    prev_winner = None
    for N in sizes:
        position, vortex_strength, core_radius = _seed(N)
        direct.load(position, vortex_strength, core_radius)
        t_direct = _time(lambda particle_count=N: direct.velocity(particle_count), repeats)

        tree = TaichiTreecode(
            max_n_particles=N + 8, max_nodes=2 * (N + 8), theta=theta, kernel_type="WINCKELMANS"
        )
        t_build = _time(
            lambda evaluator=tree, position=position, vortex_strength=vortex_strength, core_radii=core_radius: (
                evaluator.build(position, vortex_strength, core_radii, force=True)
            ),
            repeats,
        )
        tree.build(position, vortex_strength, core_radius, force=True)
        t_eval = _time(lambda evaluator=tree: evaluator.compute_velocities_gpu(bg), repeats)
        t_tc = t_build + t_eval

        if sizes[0] == N:  # one-time apples-to-apples sanity
            direct.velocity(N)
            vd = direct.out.to_numpy()[:N]
            vt = tree.compute_velocities(bg)
            rel = np.linalg.norm(vt - vd) / (np.linalg.norm(vd) + 1e-30)
            print(f"   [sanity] treecode vs direct relL2 @ N={N}, theta={theta}: {rel:.2e}")

        winner = "direct" if t_direct < t_tc else "treecode"
        if prev_winner == "direct" and winner == "treecode" and crossover is None:
            crossover = N
        prev_winner = winner
        print(
            f"{N:>8} | {t_direct:>10.2f} | {t_build:>11.2f} | {t_eval:>10.2f} | "
            f"{t_tc:>11.2f} | {t_direct / t_tc:>9.2f}"
        )

    print("-" * 78)
    if crossover:
        i = sizes.index(crossover)
        print(f"Crossover (direct→treecode) between N={sizes[i - 1]} and N={crossover}.")
    else:
        print("No crossover in the tested range (one method won throughout).")
    print(
        "Build-dominated treecode at your N => launch/sync-bound (fuse/parallelise the "
        "build) rather than FLOP-bound."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--arch",
        default="cpu",
        choices=["auto", "cpu", "metal", "cuda", "vulkan"],
    )
    ap.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=[2000, 8000, 32000],
        help="particle counts (use 5000 49000 200000 on a GPU)",
    )
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    run(args.arch, args.n, args.theta, args.repeats)
