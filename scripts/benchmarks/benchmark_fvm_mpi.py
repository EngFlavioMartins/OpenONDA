#!/usr/bin/env python3
"""Weak-scaling benchmark for partitioned PETSc PIMPLE execution."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import platform
import resource
import tempfile
import time

from mpi4py import MPI
import numpy as np

from source.solvers.FVM import (
    BoundaryConfig,
    ComputeConfig,
    DiscretizationConfig,
    FVMSetup,
    FVMSolver,
    LinearSolverConfig,
    PimpleControl,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.mesh.cartesian import structured_box


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-per-rank", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=3)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if args.cells_per_rank < 64 or args.cells_per_rank % 64:
        raise ValueError("cells-per-rank must be a multiple of 64 and at least 64")
    if args.warmup_steps < 0 or args.measured_steps < 1:
        raise ValueError("warmup-steps must be >= 0 and measured-steps must be >= 1")
    nx_per_rank = args.cells_per_rank // 64
    mesh = structured_box(nx_per_rank * size, 8, 8) if rank == 0 else None
    execution = ComputeConfig.petsc_partitioned()
    config = FVMSetup(
        case_name="partitioned-weak-scaling",
        execution=execution,
        time=TimeConfig.transient(time_step_size=0.01, duration=0.01, write_interval=10**9),
        schemes=DiscretizationConfig(convection_scheme="upwind", gradient_scheme="gauss"),
        linear=LinearSolverConfig(
            momentum_solver="bicgstab",
            pressure_solver="cg",
            momentum_tolerance=1e-8,
            pressure_tolerance=1e-8,
        ),
        pimple=PimpleControl(n_correctors=2),
        transport=TransportConfig(density=1.0, kinematic_viscosity=0.02),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", 0.0),
            BoundaryConfig.slip("ymin"),
            BoundaryConfig.slip("ymax"),
            BoundaryConfig.slip("zmin"),
            BoundaryConfig.slip("zmax"),
        ],
        # A zero field with a nonzero inlet gives the pressure and momentum
        # solvers meaningful work; the previous uniform stream was an exact
        # state and mostly measured launcher overhead.
        initial_velocity=[0.0, 0.0, 0.0],
        initial_kinematic_pressure=0.0,
    )
    with tempfile.TemporaryDirectory(prefix=f"openonda-mpi-r{rank}-") as case_dir:
        comm.Barrier()
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            solver = FVMSolver(config, case_dir, mesh_data=mesh)
            solver.auto_write = False
        initialization = time.perf_counter() - started
        for _ in range(args.warmup_steps):
            comm.Barrier()
            with contextlib.redirect_stdout(io.StringIO()):
                solver.solve_pimple(0.01)
        samples = []
        linear_samples = []
        for _ in range(args.measured_steps):
            comm.Barrier()
            started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                solver.solve_pimple(0.01)
            samples.append(comm.allreduce(time.perf_counter() - started, op=MPI.MAX))
            linear = solver.last_diagnostics.linear_solves
            linear_samples.append(
                {
                    "setup_seconds_max": comm.allreduce(
                        sum(result.setup_seconds for result in linear), op=MPI.MAX
                    ),
                    "solve_seconds_max": comm.allreduce(
                        sum(result.solve_seconds for result in linear), op=MPI.MAX
                    ),
                    "iterations_max": comm.allreduce(
                        sum(result.iterations for result in linear), op=MPI.MAX
                    ),
                }
            )

    report = {
        "schema_version": 2,
        "backend": "numpy-petsc-partitioned-float64",
        "ranks": size,
        "cells_per_rank": args.cells_per_rank,
        "global_cells": args.cells_per_rank * size,
        "initialization_seconds_max": comm.allreduce(initialization, op=MPI.MAX),
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "step_seconds_max_samples": samples,
        "step_seconds_max_median": float(np.median(samples)),
        "step_seconds_max_minimum": float(np.min(samples)),
        "step_seconds_max_maximum": float(np.max(samples)),
        "linear_samples": linear_samples,
        "peak_rss_bytes_max": comm.allreduce(_peak_rss_bytes(), op=MPI.MAX),
        "continuity_max": solver.last_diagnostics.continuity_max,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
