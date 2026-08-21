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

from openonda.fvm import (
    BoundaryConfig,
    ComputeConfig,
    DiscretizationConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    PimpleControl,
    TimeConfig,
    TransportConfig,
    periodic_square_mesh,
)


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


def _linear_telemetry(results):
    """Summarize backend-neutral linear telemetry for one solved step."""
    return {
        "linear_setup_seconds": float(sum(result.setup_seconds for result in results)),
        "linear_solve_seconds": float(sum(result.solve_seconds for result in results)),
        "linear_iterations": int(sum(result.iterations for result in results)),
        "preconditioner_rebuilds": int(
            sum(bool(result.preconditioner_rebuilt) for result in results)
        ),
    }


def _summary(samples: list[float]) -> dict[str, float | list[float]]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "samples": [float(value) for value in values],
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _run(
    target_cells: int,
    verbose: bool,
    operator_backend: str,
    linear_solver: str,
    warmup_steps: int,
    measured_steps: int,
    cold_one_step: bool,
) -> dict:
    n = max(4, int(round(math.sqrt(target_cells))))
    mesh = periodic_square_mesh(n)
    selected_solver = (
        "spsolve" if linear_solver == "auto" and target_cells <= 100_000 else linear_solver
    )
    if selected_solver == "auto":
        selected_solver = "bicgstab"
    params_schemes = DiscretizationConfig(convection_scheme="central")
    params_linear = LinearSolverConfig(
        linear_solver=selected_solver,
        momentum_solver=selected_solver,
        pressure_solver="amg" if selected_solver != "spsolve" else "spsolve",
        momentum_tolerance=1e-8,
        pressure_tolerance=1e-9,
    )
    params_pimple = PimpleControl(n_correctors=2, n_outer_correctors=1)
    config = FVMSetup(
        case_name=f"benchmark_{n}",
        execution=ComputeConfig(operator_backend=operator_backend),
        time=TimeConfig(time_step_size=0.001, end_time=0.001, output_interval_steps=2),
        schemes=params_schemes,
        linear=params_linear,
        pimple=params_pimple,
        transport=TransportConfig(density=1.0, kinematic_viscosity=0.1),
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
        solver = FVMSolver(config, case_dir=case_dir, mesh_data=mesh)
        initialization = time.perf_counter() - start
        solver.auto_write = False

        start = time.perf_counter()
        solver.set_initial_velocity(_velocity(solver.geo_data["element_centroids"]))
        field_initialization = time.perf_counter() - start

        step_samples = []
        telemetry_samples = []
        first_step = None
        total_steps = 1 if cold_one_step else warmup_steps + measured_steps
        for index in range(total_steps):
            start = time.perf_counter()
            solver.advance()
            elapsed = time.perf_counter() - start
            telemetry = _linear_telemetry(solver.last_diagnostics.linear_solves)
            if index == 0:
                first_step = {"seconds": elapsed, **telemetry}
            if not cold_one_step and index >= warmup_steps:
                step_samples.append(elapsed)
                telemetry_samples.append(telemetry)

    warmed = _summary(step_samples) if step_samples else None
    linear_setup = (
        float(np.median([sample["linear_setup_seconds"] for sample in telemetry_samples]))
        if telemetry_samples
        else float(first_step["linear_setup_seconds"])
    )
    linear_solve = (
        float(np.median([sample["linear_solve_seconds"] for sample in telemetry_samples]))
        if telemetry_samples
        else float(first_step["linear_solve_seconds"])
    )
    step = warmed["median"] if warmed is not None else first_step["seconds"]
    return {
        "target_cells": target_cells,
        "cells": mesh["n_elements"],
        "faces": mesh["n_faces"],
        "momentum_solver": params_linear.momentum_solver,
        "pressure_solver": params_linear.pressure_solver,
        "initialization_seconds": initialization,
        "field_initialization_seconds": field_initialization,
        "cold_one_step": bool(cold_one_step),
        "warmup_steps": 0 if cold_one_step else warmup_steps,
        "measured_steps": 1 if cold_one_step else measured_steps,
        "first_step": first_step,
        "warmed_step": warmed,
        # Compatibility aliases retain the former one-number consumer API.
        "step_seconds": step,
        "linear_setup_seconds": linear_setup,
        "linear_solve_seconds": linear_solve,
        "operators_and_diagnostics_seconds": max(float(step) - linear_setup - linear_solve, 0.0),
        "peak_rss_bytes": _peak_rss_bytes(),
        "continuity_max": solver.last_diagnostics.continuity_max,
        "cfl_max": solver.last_diagnostics.cfl_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator-backend", choices=("numpy", "numba", "taichi"), default="numpy")
    parser.add_argument(
        "--linear-solver",
        choices=("auto", "spsolve", "bicgstab"),
        default="auto",
        help="auto keeps the frozen direct baseline through 100k cells and uses iterative "
        "solves for larger memory qualifications",
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-regression", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument(
        "--cold-one-step",
        action="store_true",
        help="measure one first-use step only; incompatible with warmed throughput claims",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit a development-only report from a dirty source tree",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if any(size < 16 for size in args.sizes):
        raise ValueError("Each benchmark size must contain at least 16 cells")
    if args.warmup_steps < 0 or args.measured_steps < 1:
        raise ValueError("warmup-steps must be >= 0 and measured-steps must be >= 1")
    source = _git_identity()
    if source["dirty"] and not args.allow_dirty:
        raise SystemExit(
            "Refusing official benchmark from dirty source; pass --allow-dirty for development"
        )

    report = {
        "schema_version": 2,
        "official": not bool(source["dirty"]),
        "backend": f"{args.operator_backend}-scipy-float64",
        "source": source,
        "dependencies": {"numpy": np.__version__, "scipy": scipy.__version__},
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
        },
        "cases": [
            _run(
                size,
                args.verbose,
                args.operator_backend,
                args.linear_solver,
                args.warmup_steps,
                args.measured_steps,
                args.cold_one_step,
            )
            for size in args.sizes
        ],
    }
    regressions = []
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text())
        baseline_cases = {case["target_cells"]: case for case in baseline["cases"]}
        for case in report["cases"]:
            previous = baseline_cases.get(case["target_cells"])
            if previous is None:
                continue
            ratio = case["step_seconds"] / previous["step_seconds"] - 1.0
            if ratio > args.max_regression:
                regressions.append(
                    {
                        "target_cells": case["target_cells"],
                        "step_regression": ratio,
                        "allowed": args.max_regression,
                    }
                )
    report["regressions"] = regressions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if regressions:
        raise SystemExit("FVM benchmark exceeded the configured regression threshold")


if __name__ == "__main__":
    main()
