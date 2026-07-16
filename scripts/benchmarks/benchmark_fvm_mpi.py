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

from source.solvers.FVM import (
    BoundaryConfig,
    ExecutionConfig,
    FVMConfig,
    Solver,
    SolverParams,
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
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if args.cells_per_rank < 64 or args.cells_per_rank % 64:
        raise ValueError("cells-per-rank must be a multiple of 64 and at least 64")
    nx_per_rank = args.cells_per_rank // 64
    mesh = structured_box(nx_per_rank * size, 8, 8) if rank == 0 else None
    execution = ExecutionConfig.petsc_partitioned()
    config = FVMConfig(
        case_name="partitioned-weak-scaling",
        execution=execution,
        time=TimeConfig.transient(dt=0.01, duration=0.01, write_interval=10**9),
        solver=SolverParams.pimple(
            n_correctors=2,
            momentum_solver="bicgstab",
            pressure_solver="cg",
            convection_scheme="upwind",
            gradient_scheme="gauss",
            momentum_tol=1e-8,
            pressure_tol=1e-8,
        ),
        transport=TransportConfig(density=1.0, nu=0.02),
        boundaries=[
            BoundaryConfig.inlet("xmin", [1.0, 0.0, 0.0]),
            BoundaryConfig.outlet("xmax", 0.0),
            BoundaryConfig.slip("ymin"),
            BoundaryConfig.slip("ymax"),
            BoundaryConfig.slip("zmin"),
            BoundaryConfig.slip("zmax"),
        ],
        initial_U=[1.0, 0.0, 0.0],
        initial_p=0.0,
    )
    with tempfile.TemporaryDirectory(prefix=f"openonda-mpi-r{rank}-") as case_dir:
        comm.Barrier()
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            solver = Solver(config, case_dir, mesh_data=mesh)
            solver.auto_write = False
        initialization = time.perf_counter() - started
        comm.Barrier()
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            solver.solve_pimple(0.01)
        step = time.perf_counter() - started

    report = {
        "schema_version": 1,
        "backend": "numpy-petsc-partitioned-float64",
        "ranks": size,
        "cells_per_rank": args.cells_per_rank,
        "global_cells": args.cells_per_rank * size,
        "initialization_seconds_max": comm.allreduce(initialization, op=MPI.MAX),
        "step_seconds_max": comm.allreduce(step, op=MPI.MAX),
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
