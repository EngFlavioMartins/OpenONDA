"""Reproduce the PETSc warm-guess workspace component benchmark.

The baseline is loaded directly from a Git revision, rather than from a
working-tree or ``/tmp`` file.  The recorded baseline blob must hash to
``618ee67cd88bb19254b698707bac162e9513f13b4692f5bd6aff9e2cf03d1b6a``.

Examples:
    python studies/petsc_partitioned_workspace_reuse_benchmark.py
    mpirun -n 2 python studies/petsc_partitioned_workspace_reuse_benchmark.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace

from mpi4py import MPI
import numpy as np
from scipy.sparse import diags

from source.solvers.fvm.solve.petsc_partitioned import (
    OwnedRowsCSR,
    PartitionedLinearWorkspace,
)

BASELINE_PATH = "source/solvers/fvm/solve/petsc_partitioned.py"
BASELINE_SHA256 = "618ee67cd88bb19254b698707bac162e9513f13b4692f5bd6aff9e2cf03d1b6a"


def _baseline_workspace(root: Path, reference: str):
    source = subprocess.check_output(("git", "show", f"{reference}:{BASELINE_PATH}"), cwd=root)
    digest = hashlib.sha256(source).hexdigest()
    if digest != BASELINE_SHA256:
        raise RuntimeError(
            f"Baseline {reference}:{BASELINE_PATH} has sha256={digest}, expected {BASELINE_SHA256}"
        )
    name = "source.solvers.fvm.solve._workspace_reuse_baseline"
    module = ModuleType(name)
    module.__file__ = f"<git {reference}:{BASELINE_PATH}>"
    module.__package__ = "source.solvers.fvm.solve"
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module.PartitionedLinearWorkspace


def _context():
    return SimpleNamespace(
        size=MPI.COMM_WORLD.Get_size(),
        global_sum=lambda value: MPI.COMM_WORLD.allreduce(value, op=MPI.SUM),
        global_max=lambda value: MPI.COMM_WORLD.allreduce(value, op=MPI.MAX),
    )


def _measure_interleaved(workspace_types, system, guess, expected, repeats):
    """Alternate implementations so order and thermal drift affect both paths."""
    workspaces = {
        name: workspace_type(_context()) for name, workspace_type in workspace_types.items()
    }
    kwargs = {
        "method": "bicgstab",
        "tolerance": 1e-11,
        "max_iterations": 200,
        "constant_nullspace": False,
        "initial_guess": guess,
        "matrix_values_unchanged": True,
    }
    samples = {name: [] for name in workspaces}
    setup = {name: [] for name in workspaces}
    solutions = {}
    try:
        for workspace in workspaces.values():
            workspace.solve(system, **kwargs)
        MPI.COMM_WORLD.Barrier()
        for repeat in range(repeats):
            names = ("current", "baseline") if repeat % 2 == 0 else ("baseline", "current")
            for name in names:
                MPI.COMM_WORLD.Barrier()
                started = perf_counter()
                solution, result = workspaces[name].solve(system, **kwargs)
                samples[name].append(MPI.COMM_WORLD.allreduce(perf_counter() - started, op=MPI.MAX))
                setup[name].append(MPI.COMM_WORLD.allreduce(result.setup_seconds, op=MPI.MAX))
                solutions[name] = solution
        return {
            name: {
                "wall_median_seconds": float(np.median(samples[name])),
                "setup_median_seconds": float(np.median(setup[name])),
                "solution_max_error": float(np.max(np.abs(solutions[name] - expected))),
            }
            for name in workspace_types
        }
    finally:
        for workspace in workspaces.values():
            workspace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--size", type=int, default=40_000)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.size < 3 or args.repeats < 1:
        raise ValueError("size must be at least three and repeats must be positive")
    root = Path(__file__).resolve().parents[1]
    matrix = diags(
        (-np.ones(args.size - 1), 4.0 * np.ones(args.size), -np.ones(args.size - 1)),
        (-1, 0, 1),
    ).tocsr()
    expected = np.linspace(-0.5, 1.5, args.size)
    rhs = matrix @ expected
    guess = expected + 1.0e-2 * np.sin(np.arange(args.size))
    rank, size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    system = OwnedRowsCSR.from_global(matrix, rhs, rank, size)
    result = {
        "ranks": size,
        "repeats": args.repeats,
        "size": args.size,
        "baseline_ref": args.baseline_ref,
        "baseline_sha256": BASELINE_SHA256,
        "current": _measure_interleaved(
            {
                "current": PartitionedLinearWorkspace,
                "baseline": _baseline_workspace(root, args.baseline_ref),
            },
            system,
            guess[system.row_start : system.row_end],
            expected[system.row_start : system.row_end],
            args.repeats,
        ),
    }
    if rank == 0:
        result.update(result.pop("current"))
        encoded = json.dumps(result, sort_keys=True)
        if args.output is None:
            print(encoded)
        else:
            args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
