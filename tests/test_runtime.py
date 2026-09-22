"""Solver construction controls numerical thread pools after imports."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_serial_public_apis_do_not_import_optional_mpi(tmp_path):
    script = """
import sys
class RejectMPI:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mpi4py', 'petsc4py'}:
            raise AssertionError('serial import attempted optional runtime: ' + fullname)
sys.meta_path.insert(0, RejectMPI())
import openonda.fvm
import openonda.vpm
import openonda.coupler
import openonda.verify_install
from source.solvers.fvm.core.parallel import ParallelContext
assert ParallelContext.create(openonda.fvm.ComputeConfig()).comm is None
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mpi_runtime_limits_loaded_pools_and_preserves_particle_worker_budget(tmp_path):
    script = """
import numpy as np
import scipy.linalg
import numba
from threadpoolctl import threadpool_info, threadpool_limits
import openonda.runtime as runtime
np.dot(np.ones((8, 8)), np.ones((8, 8)))
scipy.linalg.solve(np.eye(3), np.ones(3))
threadpool_limits(limits=4)
assert any(p['num_threads'] > 1 for p in threadpool_info())
runtime._world_size = lambda: 4
runtime.RunConfig(cpu_cores=4, parallel_mode='mpi').ensure_runtime('unused.py')
assert all(p['num_threads'] == 1 for p in threadpool_info()), threadpool_info()
assert numba.get_num_threads() == 1
assert runtime.worker_thread_count() == 4
print('runtime pools verified')
"""
    env = os.environ.copy()
    env.pop("TI_CPU_MAX_NUM_THREADS", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "runtime pools verified" in completed.stdout


def test_mpi_relaunch_inherits_thread_limits(monkeypatch):
    import openonda.runtime as runtime

    captured = {}
    monkeypatch.setattr(runtime, "_world_size", lambda: 1)
    monkeypatch.setattr(runtime, "_mpi_executable", lambda: "/mpi/mpiexec")
    monkeypatch.delenv(runtime._MPI_CHILD, raising=False)

    def limits(self, count):
        for name in runtime._THREAD_VARIABLES:
            monkeypatch.setenv(name, str(count))

    def launch(executable, command, environment):
        captured.update(command=command, env=environment)

    monkeypatch.setattr(runtime.RunConfig, "_set_thread_count", limits)
    monkeypatch.setattr(runtime.os, "execvpe", launch)
    monkeypatch.setattr(sys, "argv", ["case.py", "--name", "fine"])
    runtime.RunConfig(cpu_cores=4, parallel_mode="mpi").ensure_runtime("case.py")
    assert captured["command"][1:3] == ["-n", "4"]
    assert captured["command"][-2:] == ["--name", "fine"]
    assert all(captured["env"][key] == "1" for key in runtime._THREAD_VARIABLES)


def test_existing_wrong_mpi_world_is_rejected(monkeypatch):
    import openonda.runtime as runtime

    monkeypatch.setattr(runtime, "_world_size", lambda: 2)
    with pytest.raises(RuntimeError, match="launched 2 ranks.*cores=4"):
        runtime.RunConfig(cpu_cores=4, parallel_mode="mpi").ensure_runtime("case.py")


@pytest.mark.parametrize(("rank", "suppressed"), [(0, False), (1, True)])
def test_mpi_worker_suppresses_only_uncaught_exception_rendering(monkeypatch, rank, suppressed):
    import openonda.runtime as runtime

    def owner_hook(_type, _value, _traceback):
        return None

    monkeypatch.setattr(sys, "excepthook", owner_hook)
    monkeypatch.setitem(
        sys.modules,
        "mpi4py",
        SimpleNamespace(MPI=SimpleNamespace(COMM_WORLD=SimpleNamespace(Get_rank=lambda: rank))),
    )

    runtime._configure_mpi_exception_reporting()

    assert (sys.excepthook is not owner_hook) is suppressed
    assert sys.excepthook(RuntimeError, RuntimeError("failure"), None) is None


def test_resource_allocation_does_not_impersonate_an_mpi_launch(monkeypatch):
    import openonda.runtime as runtime
    from source.solvers.fvm.config import ComputeConfig
    from source.solvers.fvm.core.parallel import ParallelContext, detected_world_size

    for key in runtime._MPI_SIZE_VARIABLES:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_NTASKS", "64")
    assert runtime._world_size() == detected_world_size() == 1
    assert ParallelContext.create(ComputeConfig()).size == 1
    # The launched communicator may use only part of the allocation.
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    assert runtime._world_size() == detected_world_size() == 4
