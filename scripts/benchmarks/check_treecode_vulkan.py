#!/usr/bin/env python3
"""
Vulkan correctness check for the on-GPU LBVH treecode (the lambOseen freeze).

Background: the lambOseen run uses ``dns_simulation`` (default velocity =
treecode) on GPU_VULKAN, and the vortex came out *frozen* — the treecode build
produced ~0 velocities on Vulkan while being correct on CPU.  The build was made
Vulkan-portable (parallel Karras: no ``serialize=True``, no shared ``_stack``, no
mid-build device→host scalar read).  This script confirms the fix **on your GPU**,
which the CPU sandbox cannot.

It seeds a random cloud, builds the treecode + evaluates velocity on the chosen
backend, and compares to the exact DIRECT Biot–Savart on the same data.  Before
the fix the treecode |v| was ~5e-6× direct; after it should match to the
Barnes–Hut band (~few % at theta=0.5, tightening as theta→0).

    python scripts/benchmarks/check_treecode_vulkan.py --arch vulkan
    python scripts/benchmarks/check_treecode_vulkan.py --arch cuda
    python scripts/benchmarks/check_treecode_vulkan.py --arch cpu     # sanity

Exit code 0 = pass (treecode tracks direct), 1 = fail (still degenerate).
"""

import argparse
import sys

import numpy as np

ONE_OVER_FOUR_PI = 0.07957747154594767


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="vulkan", choices=["cpu", "cuda", "vulkan", "gpu"])
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--kernel", default="WINCKELMANS", choices=["WINCKELMANS", "GAUSSIAN"])
    args = ap.parse_args()

    # Initialise Taichi exactly the way the solver does (config/backend.py).
    # A raw ``ti.init(arch=ti.vulkan)`` triggers Taichi's adaptive arch probe,
    # which can hang on some Vulkan setups; the project initializer avoids it and
    # applies the correct device-memory kwargs.
    from source.solvers.VPM.config.backend import initialize_taichi_backend

    backend = {"cpu": "CPU", "cuda": "CUDA", "vulkan": "GPU_VULKAN", "gpu": "GPU"}[args.arch]
    chosen = initialize_taichi_backend(preferred_backend=backend, debug_mode=False, precision="f32")
    print(f"[backend] requested={backend} → using {chosen}")
    from source.solvers.VPM.acceleration.treecode_gpu import TaichiTreecode

    N = args.n
    rng = np.random.default_rng(0)
    pos = (rng.random((N, 3)) - 0.5).astype(np.float32)
    circ = (rng.normal(size=(N, 3)) * 0.1).astype(np.float32)
    rad = np.full(N, 0.05, dtype=np.float32)

    tree = TaichiTreecode(
        max_particles=N + 8, max_nodes=2 * (N + 8), theta=args.theta, kernel_type=args.kernel
    )
    tree.build(pos, circ, rad, force=True)
    v_tree = tree.compute_velocities(np.zeros(3, dtype=np.float32))

    # Exact direct reference on the same data (Winckelmans q), chunked.
    def direct(pos, circ, rad, chunk=512):
        p = pos.astype(np.float64)
        c = circ.astype(np.float64)
        r = rad.astype(np.float64)
        out = np.zeros((len(p), 3))
        for s in range(0, len(p), chunk):
            e = min(s + chunk, len(p))
            d = p[s:e, None, :] - p[None, :, :]
            rm = np.linalg.norm(d, axis=2)
            sig = 0.5 * (r[s:e, None] + r[None, :])
            with np.errstate(divide="ignore", invalid="ignore"):
                rs = rm / sig
                r2 = rs * rs
                if args.kernel == "WINCKELMANS":
                    q = rs**3 * (r2 + 2.5) / (r2 + 1.0) ** 2.5 * ONE_OVER_FOUR_PI
                else:
                    from scipy import special

                    q = (special.erf(rs) - (2 / np.sqrt(np.pi)) * rs * np.exp(-r2)) / (4 * np.pi)
                contrib = -q[..., None] * np.cross(d, c[None, :, :]) / (rm[..., None] ** 3)
            contrib = np.where((rm > 1e-10)[..., None], contrib, 0.0)
            for ii in range(s, e):
                contrib[ii - s, ii] = 0.0
            out[s:e] = contrib.sum(axis=1)
        return out

    v_dir = direct(pos, circ, rad)
    relL2 = np.linalg.norm(v_tree - v_dir) / (np.linalg.norm(v_dir) + 1e-30)
    print(f"arch={args.arch} N={N} theta={args.theta} kernel={args.kernel}")
    print(f"  treecode |v| mean={np.linalg.norm(v_tree, axis=1).mean():.4f}")
    print(f"  direct   |v| mean={np.linalg.norm(v_dir, axis=1).mean():.4f}")
    print(f"  treecode-vs-direct relL2 = {relL2:.3e}")
    ok = relL2 < 0.20  # Barnes-Hut band at theta=0.5; the bug gave ~1.0
    print(
        "  RESULT:",
        "PASS — treecode is correct on this backend"
        if ok
        else "FAIL — treecode still degenerate (≈0 / wrong) on this backend",
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
