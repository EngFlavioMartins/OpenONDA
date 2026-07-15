#!/usr/bin/env python3
"""Reproducible end-to-end benchmark for the serial FVM reference backend."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from source.solvers.FVM import (  # noqa: E402
    BoundaryConfig,
    FVMConfig,
    Solver,
    SolverParams,
    TimeConfig,
    TransportConfig,
)
from tutorials.FVM.taylorGreen.assets.mesh_periodic import periodic_square_mesh  # noqa: E402


def _velocity(centres: np.ndarray) -> np.ndarray:
    x = centres[:, 0]
    y = centres[:, 1]
    return np.column_stack((np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y), np.zeros_like(x)))


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _git_identity() -> dict[str, str | bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _run(target_cells: int, verbose: bool) -> dict[str, float | int]:
    n = max(4, int(round(math.sqrt(target_cells))))
    mesh = periodic_square_mesh(n)
    params = SolverParams.pimple(
        n_correctors=2,
        n_outer=1,
        linear_solver="spsolve",
        convection_scheme="central",
    )
    config = FVMConfig(
        case_name=f"benchmark_{n}",
        time=TimeConfig(delta_t=0.001, end_time=0.001, write_interval=2),
        solver=params,
        transport=TransportConfig(density=1.0, nu=0.1),
        boundaries=[
            BoundaryConfig.cyclic("xmin", "xmax"),
            BoundaryConfig.cyclic("xmax", "xmin"),
            BoundaryConfig.cyclic("ymin", "ymax"),
            BoundaryConfig.cyclic("ymax", "ymin"),
            BoundaryConfig.empty("zmin"),
            BoundaryConfig.empty("zmax"),
        ],
    )

    output = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with tempfile.TemporaryDirectory(prefix="openonda-fvm-benchmark-") as case_dir, output:
        start = time.perf_counter()
        solver = Solver(config, case_dir=case_dir, mesh_data=mesh)
        initialization = time.perf_counter() - start
        solver.auto_write = False

        start = time.perf_counter()
        solver.set_initial_velocity(_velocity(solver.geo_data["element_centroids"]))
        field_initialization = time.perf_counter() - start

        start = time.perf_counter()
        solver.evolve()
        step = time.perf_counter() - start
        linear_results = solver.last_diagnostics.linear_solves

    linear_setup = float(sum(result.setup_seconds for result in linear_results))
    linear_solve = float(sum(result.solve_seconds for result in linear_results))
    return {
        "target_cells": target_cells,
        "cells": mesh["n_elements"],
        "faces": mesh["n_faces"],
        "initialization_seconds": initialization,
        "field_initialization_seconds": field_initialization,
        "step_seconds": step,
        "linear_setup_seconds": linear_setup,
        "linear_solve_seconds": linear_solve,
        "operators_and_diagnostics_seconds": max(step - linear_setup - linear_solve, 0.0),
        "peak_rss_bytes": _peak_rss_bytes(),
        "continuity_max": solver.last_diagnostics.continuity_max,
        "cfl_max": solver.last_diagnostics.cfl_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if any(size < 16 for size in args.sizes):
        raise ValueError("Each benchmark size must contain at least 16 cells")

    report = {
        "schema_version": 1,
        "backend": "numpy-scipy-float64",
        "source": _git_identity(),
        "dependencies": {"numpy": np.__version__, "scipy": scipy.__version__},
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
        },
        "cases": [_run(size, args.verbose) for size in args.sizes],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
