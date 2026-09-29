"""Replay one bounded CPU IBM step's sparse systems with and without exact LU reuse."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

import numpy as np

from source.solvers.fvm.solve import linear_interface as linear
from tests.fvm.test_freestream_projection_stability import make_solver


def main():
    systems = []
    factorizations = []
    actual_factorize = linear.splu
    actual_solve = linear._solve_direct_with_workspace

    def record_factorization(matrix):
        factorizations.append(matrix.shape)
        return actual_factorize(matrix)

    def record_system(matrix, rhs, workspace):
        systems.append((matrix.copy(), np.asarray(rhs).copy()))
        return actual_solve(matrix, rhs, workspace)

    with TemporaryDirectory(prefix="openonda-direct-replay-") as work:
        linear.splu = record_factorization
        linear._solve_direct_with_workspace = record_system
        try:
            with make_solver(Path(work), immersed=True) as solver:
                solver.advance()
                n_cells = solver.mesh_data["n_cells"]
        finally:
            linear.splu = actual_factorize
            linear._solve_direct_with_workspace = actual_solve

    if len(systems) < 3:
        raise RuntimeError("IBM replay did not capture both momentum and pressure systems")

    def replay(cached):
        workspace = SimpleNamespace(direct_factorization=None) if cached else None
        started = time.perf_counter()
        solutions = []
        for matrix, rhs in systems:
            solution = linear.solve_linear_system(
                matrix,
                rhs,
                method="spsolve",
                tol=1e-9,
                direct_workspace=workspace,
                return_info=False,
            )
            solutions.append(solution)
        return time.perf_counter() - started, solutions

    # Warm both SciPy paths before timing, then alternate to reduce drift.
    replay(False)
    replay(True)
    plain_times = []
    cached_times = []
    for _ in range(6):
        plain, plain_solutions = replay(False)
        cached, cached_solutions = replay(True)
        for left, right in zip(plain_solutions, cached_solutions, strict=True):
            np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-10)
        plain_times.append(plain)
        cached_times.append(cached)
    report = {
        "cells": n_cells,
        "systems_per_step": len(systems),
        "native_factorizations": len(factorizations),
        "plain_median_seconds": median(plain_times),
        "cached_median_seconds": median(cached_times),
        "speedup": median(plain_times) / median(cached_times),
        "scope": "CPU IBM one accepted step, matrix/RHS replay; excludes assembly and whole-case runtime",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
