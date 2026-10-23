"""Two-rank coupled runs: plain Python launches MPI, restarts and closes every rank."""

from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.integration
@pytest.mark.parametrize("selection", ["explicit", "latest"])
def test_partitioned_coupled_restart_keeps_boundary_collectives_in_order(tmp_path, selection):
    if find_spec("mpi4py") is None or find_spec("petsc4py") is None:
        pytest.skip("MPI and PETSc are required")
    if not (Path(sys.executable).with_name("mpiexec").is_file() or shutil.which("mpiexec")):
        pytest.skip("mpiexec is required")
    script = tmp_path / "restart.py"
    script.write_text(
        f"selection = {selection!r}\n"
        + """
from pathlib import Path
import sys
import json
import numpy as np
from openonda import coupler
from tests.support.coupler import uniform_stream_cases

directory = Path(sys.argv[1])


def solver(case_dir):
    flow, particles = uniform_stream_cases(
        case_dir, fvm_time_step_size=.01, exchange_time_step_size=.01, end_time=.02,
        cores=2, max_n_particles=1000,
    )
    return coupler.CoupledSolver(flow, particles, coupler.CouplerConfig(transfer_inset_spacings=0.0, eta_blend_width=0.0))


with solver(directory / "first") as first:
    first.start_from("latest")
    first.advance()
    first.save_backup()
resumed = solver(directory / ("first" if selection == "latest" else "resumed"))
resumed.run(start_from="latest" if selection == "latest"
            else directory / "first/solution/coupled/coupled_000001")
assert resumed.fvm_solver.time == .02
n_cells = resumed.fvm_solver.mesh_data["n_cells"]
assert np.allclose(resumed.fvm_solver.velocity[:n_cells], [1., 0., 0.], atol=1e-8)
rank = resumed.fvm_solver.parallel.rank
(directory / f"rank-{rank}.json").write_text(json.dumps({"step": resumed.step, "time": .02}))
"""
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    environment["TI_OFFLINE_CACHE_FILE_PATH"] = str(tmp_path / "taichi-cache")
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "outputs")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(2):
        assert json.loads((tmp_path / f"outputs/rank-{rank}.json").read_text()) == {
            "step": 2,
            "time": 0.02,
        }


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["run", "construction_failure", "state_check_failure"])
def test_coupled_solver_launches_and_closes_two_ranks(tmp_path, mode):
    if find_spec("mpi4py") is None or find_spec("petsc4py") is None:
        pytest.skip("MPI and PETSc are required")
    if not (Path(sys.executable).with_name("mpiexec").is_file() or shutil.which("mpiexec")):
        pytest.skip("mpiexec is required")
    script = tmp_path / "case.py"
    script.write_text("""
from pathlib import Path
import json
import sys
from mpi4py import MPI
import numpy as np
from openonda import coupler
from openonda.runtime import worker_thread_count
import openonda.vpm as vpm_api
from tests.support.coupler import uniform_stream_cases

directory = Path(sys.argv[1])
mode = sys.argv[2]
flow, particles = uniform_stream_cases(
    directory, fvm_time_step_size=.01, exchange_time_step_size=.01, end_time=.01,
    cores=2, max_n_particles=1000,
)
if mode == "construction_failure":
    def fail(case):
        raise ValueError("injected construction failure")
    vpm_api.VPMSolver = fail
solver = None
try:
    solver = coupler.CoupledSolver(flow, particles, coupler.CouplerConfig(transfer_inset_spacings=0.0, eta_blend_width=0.0))
    with solver:
        native = solver.fvm_solver
        assert native.parallel.size == 2
        assert (solver.vpm_solver is not None) == native.parallel.is_root
        assert worker_thread_count() == 2
        if mode == "state_check_failure" and solver.vpm_solver is not None:
            def fail():
                raise vpm_api.ParticleStateError("injected accepted-state limit failure")
            solver.vpm_solver.write_step_output = fail
        solver.run()
        n_cells = native.mesh_data["n_cells"]
        assert np.allclose(native.velocity[:n_cells], [1., 0., 0.], atol=1e-8)
        assert mode == "run"
    status = "completed"
except (RuntimeError, ValueError) as error:
    expected = {"construction_failure": "injected construction failure",
                "state_check_failure": "injected accepted-state limit failure"}.get(mode, "")
    if mode == "run" or expected not in str(error):
        raise
    status = "expected_failure"
rank = MPI.COMM_WORLD.Get_rank()
if solver is not None:
    assert solver._closed
    assert not solver.fvm_solver.algorithm._partitioned_linear_workspaces
(directory / f"rank-{rank}.json").write_text(json.dumps({"rank": rank, "status": status}))
""")
    environment = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TI_CPU_MAX_NUM_THREADS",
        "_OPENONDA_MPI_CHILD",
    ):
        environment.pop(key, None)
    # Locate the code under test in CI, including an uninstalled checkout.
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "outputs"), mode],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    expected = "completed" if mode == "run" else "expected_failure"
    for rank in range(2):
        assert json.loads((tmp_path / "outputs" / f"rank-{rank}.json").read_text()) == {
            "rank": rank,
            "status": expected,
        }
