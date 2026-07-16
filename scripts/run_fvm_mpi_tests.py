#!/usr/bin/env python3
"""Run the collective FVM test module with one JUnit report per MPI rank."""

from pathlib import Path

from mpi4py import MPI
import pytest


def main() -> int:
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    rank = MPI.COMM_WORLD.Get_rank()
    return pytest.main(
        [
            "tests/fvm/test_petsc_parallel.py",
            "-q",
            f"--junitxml={output / f'petsc-mpi-rank-{rank}.xml'}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
