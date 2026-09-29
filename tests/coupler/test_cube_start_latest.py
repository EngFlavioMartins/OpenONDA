"""Exercise tutorial solid masks during backup, advance and automatic restore."""

from importlib.util import find_spec
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("cores", [1, 2])
def test_cube_tutorial_recovers_initial_failure_and_continues(tmp_path, cores):
    if cores > 1 and (
        find_spec("mpi4py") is None
        or find_spec("petsc4py") is None
        or not (Path(sys.executable).with_name("mpiexec").is_file() or shutil.which("mpiexec"))
    ):
        pytest.skip("MPI and PETSc are required")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    environment["TI_OFFLINE_CACHE"] = "0"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_cube_start_latest.py")),
            str(tmp_path),
            str(cores),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-8000:]
