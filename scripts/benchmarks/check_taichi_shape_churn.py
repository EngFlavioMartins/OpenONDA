#!/usr/bin/env python3
"""
Probe Taichi external-array staging behavior under ndarray shape churn.

The VPM DVH/GBD path can create many host/device transfers whose NumPy array
shape changes as the regeneration grid grows.  Taichi 1.7.x Vulkan has been
observed to retain one staging allocation per distinct external-array shape.
This script isolates that behavior from OpenONDA physics by repeatedly copying
NumPy arrays into/out of fixed Taichi fields.

Examples
--------
    python scripts/benchmarks/check_taichi_shape_churn.py --arch vulkan
    python scripts/benchmarks/check_taichi_shape_churn.py --arch cuda
    python scripts/benchmarks/check_taichi_shape_churn.py --arch cpu
    python scripts/benchmarks/check_taichi_shape_churn.py --arch vulkan --mode fixed

Interpretation
--------------
If ``--mode variable`` grows GPU memory roughly with the number of distinct
shapes, but ``--mode fixed`` plateaus, the leak is in backend external-array
staging/cache lifetime rather than in the VPM solver's numerical kernels.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import taichi as ti


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm", encoding="utf-8") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1 << 20)
    except (OSError, ValueError, IndexError):
        return float("nan")


def _nvidia_gpu_mb() -> float | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None

    pid = str(os.getpid())
    total = 0.0
    found = False
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or parts[0] != pid:
            continue
        try:
            total += float(parts[1])
            found = True
        except ValueError:
            pass
    return total if found else None


def _vulkan_device_memory_mb() -> float | None:
    """Best-effort process GPU memory from amdgpu fdinfo, when available."""
    total = 0
    fd_root = f"/proc/{os.getpid()}/fdinfo"
    try:
        fd_names = os.listdir(fd_root)
    except OSError:
        return None

    for name in fd_names:
        try:
            with open(os.path.join(fd_root, name), encoding="utf-8") as f:
                for line in f:
                    if line.startswith("drm-memory-vram:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            total += int(parts[1])
                    elif line.startswith("drm-memory-gtt:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            total += int(parts[1])
        except (OSError, ValueError):
            continue
    return total / (1 << 20) if total else None


def _device_mb() -> float | None:
    nvidia = _nvidia_gpu_mb()
    if nvidia is not None:
        return nvidia
    return _vulkan_device_memory_mb()


@ti.kernel
def _upload(src: ti.types.ndarray(dtype=ti.f32, ndim=2), dst: ti.template(), n: ti.i32):  # type: ignore
    for i in range(n):
        for k in ti.static(range(3)):
            dst[i][k] = src[i, k]


@ti.kernel
def _download(src: ti.template(), dst: ti.types.ndarray(dtype=ti.f32, ndim=2), n: ti.i32):  # type: ignore
    for i in range(n):
        for k in ti.static(range(3)):
            dst[i, k] = src[i][k]


def _shape_schedule(min_n: int, max_n: int, distinct: int, repeats: int) -> list[int]:
    if distinct <= 1:
        return [max_n] * repeats
    raw = np.linspace(min_n, max_n, distinct)
    shapes = [max(1, int(v)) for v in raw]
    return shapes * repeats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="vulkan", choices=["cpu", "cuda", "vulkan", "gpu"])
    ap.add_argument("--mode", default="variable", choices=["variable", "fixed"])
    ap.add_argument("--min-n", type=int, default=4096)
    ap.add_argument("--max-n", type=int, default=262144)
    ap.add_argument("--distinct", type=int, default=96)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--sample-every", type=int, default=24)
    ap.add_argument("--download", action="store_true", help="Also copy device field back to NumPy.")
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from source.solvers.VPM.config.backend import initialize_taichi_backend

    backend = {"cpu": "CPU", "cuda": "CUDA", "vulkan": "GPU_VULKAN", "gpu": "GPU"}[args.arch]
    chosen = initialize_taichi_backend(preferred_backend=backend, debug_mode=False, precision="f32")
    print(f"[backend] requested={backend} -> using {chosen}")

    max_n = int(args.max_n)
    field = ti.Vector.field(3, dtype=ti.f32, shape=max_n)
    fixed_in = np.empty((max_n, 3), dtype=np.float32)
    fixed_out = np.empty((max_n, 3), dtype=np.float32) if args.download else None
    schedule = _shape_schedule(args.min_n, max_n, args.distinct, args.repeats)

    print(
        "mode={mode} iterations={iters} distinct_shapes={distinct} "
        "min_n={min_n} max_n={max_n} download={download}".format(
            mode=args.mode,
            iters=len(schedule),
            distinct=args.distinct,
            min_n=args.min_n,
            max_n=max_n,
            download=args.download,
        )
    )
    print("iter,n,rss_mb,device_mb,elapsed_s")

    if args.mode == "fixed":
        fixed_in.fill(0.0)
        _upload(fixed_in, field, max_n)
        if args.download:
            fixed_out.fill(0.0)
            _download(field, fixed_out, max_n)
        ti.sync()

    t0 = time.perf_counter()
    for it, n in enumerate(schedule, start=1):
        if args.mode == "variable":
            src = np.ones((n, 3), dtype=np.float32)
            _upload(src, field, n)
            if args.download:
                dst = np.empty((n, 3), dtype=np.float32)
                _download(field, dst, n)
        else:
            fixed_in[:n] = 1.0
            _upload(fixed_in, field, n)
            if args.download:
                _download(field, fixed_out, n)

        if it == 1 or it % args.sample_every == 0 or it == len(schedule):
            ti.sync()
            dev = _device_mb()
            dev_s = "" if dev is None else f"{dev:.1f}"
            print(f"{it},{n},{_rss_mb():.1f},{dev_s},{time.perf_counter() - t0:.2f}")

    ti.sync()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
