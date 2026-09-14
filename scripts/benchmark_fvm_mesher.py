"""Time uncached native mesh generation without starting a flow solver.

Example (from the repository root)::

    python scripts/benchmark_fvm_mesher.py --output /tmp/cube-mesher-benchmark

The default is the coupled cube tutorial's actual FVM_MESH specification.
Each run calls build() directly, includes final mesh validation, and records
stage timings and hashes. Numba's compilation cache is retained unless the
caller selects a fresh NUMBA_CACHE_DIR. Case solution/cache files are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import runpy
import sys
import tempfile
from time import perf_counter


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        type=Path,
        default=root / "tutorials/coupled_fvm_vpm/02_cube_flow/setup.py",
    )
    parser.add_argument("--mesh-name", default="FVM_MESH")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--save-mesh", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.threads < 1:
        parser.error("runs and threads must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    os.environ["NUMBA_NUM_THREADS"] = str(args.threads)
    sys.path.insert(0, str(root))

    import numba
    import numpy as np

    from source.solvers.fvm.io.mesh_storage import save_native_mesh
    from source.solvers.fvm.mesh.cache import CachedMesh
    from source.solvers.fvm.mesh.progress import mesher_log_session

    output = (args.output or Path(tempfile.mkdtemp(prefix="openonda-mesher-"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for run in range(1, args.runs + 1):
        mesher = runpy.run_path(str(args.case.resolve()))[args.mesh_name]
        identity = CachedMesh(mesher, output / "cache.npz").identity()
        log = output / f"run-{run}.log"
        # A new file keeps repeated invocations from mixing stage timings.
        log.write_text("")
        start = perf_counter()
        with mesher_log_session(log):
            mesh = mesher.build()
        elapsed = perf_counter() - start
        stages = [
            {"stage": match[0], "seconds": float(match[1])}
            for match in re.findall(r"DONE\s+(.+?) \| seconds=([\d.]+)", log.read_text())
        ]
        hashes = {}
        for name in ("vertex_position", "faces", "owners", "neighbours"):
            values = mesh[name]
            if name == "faces":
                values = np.concatenate(values).astype(np.int64)
            hashes[name] = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        result = {
            "run": run,
            "build_seconds": elapsed,
            "cells": int(mesh["n_cells"]),
            "faces": int(mesh["n_faces"]),
            "points": int(mesh["n_points"]),
            "stages": stages,
            "sha256": hashes,
            "surface_constraint": mesh["mesh_generation"]["surface_constraint"],
            "quality": mesher.report.as_dict()["diagnostics"]["quality"],
            "cartesian_cache_identity": identity,
        }
        results.append(result)
        if args.save_mesh:
            mesh["cartesian_cache_identity"] = identity
            save_native_mesh(mesh, output / f"run-{run}.npz")
        (output / "benchmark.json").write_text(
            json.dumps(
                {
                    "case": str(args.case.resolve()),
                    "threads": args.threads,
                    "mesh_cache": "bypassed",
                    "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
                    "environment": {
                        "platform": platform.platform(),
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                        "numba": numba.__version__,
                    },
                    "runs": results,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Run {run}: {elapsed:.3f}s, {mesh['n_cells']:,} cells; {log}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
