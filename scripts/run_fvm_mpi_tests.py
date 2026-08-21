"""Run the collective FVM verification suite under an MPI launcher.

Example::

    mpiexec -n 2 python scripts/run_fvm_mpi_tests.py

Every rank must execute the same pytest process because the tests perform MPI
collectives.  Only rank zero writes the compact CI result artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from mpi4py import MPI
import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    ROOT / "tests/fvm/test_petsc_parallel.py",
    ROOT / "tests/fvm/test_surface_sampler_mpi.py",
)


def main() -> int:
    """Execute the collective tests and return a communicator-wide status."""
    comm = MPI.COMM_WORLD
    local_status = int(
        pytest.main(
            [
                "-q",
                "--tb=short",
                "-m",
                "mpi",
                *(str(path) for path in TESTS),
            ]
        )
    )
    statuses = comm.allgather(local_status)

    if comm.rank == 0:
        artifact_dir = ROOT / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        result = {
            "mpi_ranks": comm.size,
            "rank_exit_codes": statuses,
            "tests": [str(path.relative_to(ROOT)) for path in TESTS],
        }
        (artifact_dir / f"fvm-mpi-{comm.size}.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )

    return max(statuses)


if __name__ == "__main__":
    sys.exit(main())
