"""Repeated replicated pressure solves, with RSS and PETSc object diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
from types import SimpleNamespace

from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from scipy import sparse

from source.solvers.fvm.solve.linear_interface import _solve_petsc


def run(directory, calls, maximum_growth_mib=None):
    comm = MPI.COMM_WORLD
    n = 2048
    # A Neumann graph Laplacian with the real pressure null space.
    A = sparse.diags((-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)), (-1, 0, 1), format="csr")
    A[0, 0] = A[-1, -1] = 1.0
    exact = np.cos(np.linspace(0, 4 * np.pi, n))
    exact -= exact.mean()
    b = A @ exact
    context = SimpleNamespace(size=comm.size)
    directory.mkdir(exist_ok=True, parents=True)
    records = []
    for index in range(calls):
        solution, info = _solve_petsc(
            A, b, "amg", "kinematic_pressure", 1e-7, 0.005, 1000, None, context, "constant"
        )
        assert info.converged and np.isfinite(solution).all()
        if index % 20 == 0 or index == calls - 1:
            rss = next(
                line
                for line in Path("/proc/self/status").read_text().splitlines()
                if line.startswith("VmRSS:")
            )
            records.append(
                {
                    "calls": index + 1,
                    "rss_bytes": int(rss.split()[1]) * 1024,
                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    "residual": info.final_residual,
                    "iterations": info.iterations,
                }
            )
            (directory / f"rank-{comm.rank}.json").write_text(json.dumps(records, indent=2) + "\n")
    PETSc.garbage_view()
    PETSc.garbage_cleanup()
    if maximum_growth_mib is not None:
        growth = records[-1]["rss_bytes"] - records[0]["rss_bytes"]
        assert growth <= maximum_growth_mib * (1 << 20), records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--calls", type=int, default=300)
    parser.add_argument("--maximum-growth-mib", type=float)
    options = parser.parse_args()
    run(options.directory, options.calls, options.maximum_growth_mib)
