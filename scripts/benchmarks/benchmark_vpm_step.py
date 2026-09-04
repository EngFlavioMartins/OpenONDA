"""Deterministic per-stage VPM timestep benchmark.

Small, fixed, seeded case designed for before/after comparison of solver changes
— not a scientific case.  Reports the ``RuntimeProfiler`` breakdown per stage
plus peak RSS, so a change can be attributed to a stage rather than guessed at.

Usage::

    python scripts/benchmarks/benchmark_vpm_step.py                 # direct default sweep
    python scripts/benchmarks/benchmark_vpm_step.py --n 1000 8000  # explicit N
    python scripts/benchmarks/benchmark_vpm_step.py --induction treecode --backend CPU
    python scripts/benchmarks/benchmark_vpm_step.py --induction fmm --backend CPU

The particle field is a seeded random cloud in a unit box with random vortex
vortex_strength; identical for a given N across runs, so timings are comparable.
"""

import argparse
from dataclasses import asdict
import json
import platform
import resource
import sys
import time

import numpy as np


def _rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak / 1024**2 if platform.system() == "Darwin" else peak / 1024)


def _make_solver(
    backend: str,
    n: int,
    tmpdir: str,
    induction_name: str,
):
    from openonda.vpm import (
        Backup,
        DirectInduction,
        FMMInduction,
        Numerics,
        TreecodeInduction,
        VPMCase,
        VPMSolver,
    )

    induction_factories = {
        "direct": DirectInduction,
        "treecode": TreecodeInduction,
        "fmm": FMMInduction,
    }
    try:
        induction = induction_factories[induction_name.lower()]()
    except KeyError as exc:
        raise ValueError(f"unsupported induction method: {induction_name}") from exc

    solver = VPMSolver(
        VPMCase(
            directory=tmpdir,
            backup=Backup(0, tmpdir, tmpdir),
            numerics=Numerics(
                compute_device=backend,
                induction=induction,
                max_n_particles=max(n, 4096),
            ),
        )
    )

    rng = np.random.default_rng(20260807)
    h = n ** (-1.0 / 3.0)
    solver.add_vortex_particles(
        position=rng.random((n, 3)).astype(np.float32),
        velocity=np.zeros((n, 3), np.float32),
        vortex_strength=(rng.normal(size=(n, 3)) * 1e-3).astype(np.float32),
        core_radius=np.full(n, 1.5 * h, np.float32),
        particle_volume=np.full(n, h**3, np.float32),
        kinematic_viscosity=np.full(n, 1e-3, np.float32),
    )
    return solver


def run_case(
    backend: str,
    n: int,
    steps: int,
    tmpdir: str,
    induction_name: str,
) -> dict:
    solver = _make_solver(backend, n, tmpdir, induction_name)
    solver.advance()  # warm-up: pays the Taichi JIT
    solver.profiler.reset()

    t0 = time.perf_counter()
    for _ in range(steps):
        solver.advance()
    wall = time.perf_counter() - t0

    stages = {name: total / steps * 1e3 for name, total in solver.profiler._cumulative.items()}

    result = {
        "backend": backend,
        "induction": induction_name,
        "particles": n,
        "steps": steps,
        "ms_per_step": wall / steps * 1e3,
        "peak_rss_mb": _rss_mb(),
        "stages_ms": dict(sorted(stages.items(), key=lambda kv: -kv[1])),
    }

    diagnostics = getattr(solver.induction, "diagnostics", None)
    if diagnostics is not None:
        result["induction_diagnostics"] = (
            diagnostics if isinstance(diagnostics, dict) else asdict(diagnostics)
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[1000, 4000])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--backend", default="CPU")
    parser.add_argument("--induction", choices=("direct", "treecode", "fmm"), default="direct")
    parser.add_argument("--json", type=str, default=None, help="write results to this path")
    parser.add_argument("--tmpdir", type=str, default="/tmp/vpm_bench")
    args = parser.parse_args()

    results = []
    for n in args.n:
        res = run_case(
            args.backend,
            n,
            args.steps,
            args.tmpdir,
            args.induction,
        )
        results.append(res)
        print(
            f"N={res['particles']:>7}  {res['ms_per_step']:9.1f} ms/step  "
            f"RSS {res['peak_rss_mb']:7.1f} MB",
            file=sys.stderr,
        )
        for stage, ms in list(res["stages_ms"].items())[:8]:
            print(f"            {stage:<28} {ms:8.2f} ms", file=sys.stderr)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
