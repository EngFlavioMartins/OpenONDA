"""Process-launch contract for simple Python case entry points."""

from __future__ import annotations

import os

import pytest

import openonda.runtime as runtime


def test_run_config_validates_cpu_cores():
    assert not runtime.RunConfig(cpu_cores=1).is_parallel
    assert runtime.RunConfig(cpu_cores=4).is_parallel
    with pytest.raises(ValueError, match="at least one"):
        runtime.RunConfig(cpu_cores=0)
    with pytest.raises(TypeError, match="integer"):
        runtime.RunConfig(cpu_cores=True)
    with pytest.raises(ValueError, match="parallel_mode"):
        runtime.RunConfig(parallel_mode="processes")


def test_serial_launch_does_not_exec(monkeypatch, tmp_path):
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
    monkeypatch.setattr(
        runtime.os,
        "execvpe",
        lambda *_args, **_kwargs: pytest.fail("serial execution must not relaunch"),
    )

    runtime.RunConfig(cpu_cores=1).ensure_mpi(tmp_path / "run_setup.py")


def test_parallel_launch_uses_environment_mpi_and_caps_threads(monkeypatch, tmp_path):
    captured = {}

    def capture(executable, command, environment):
        captured.update(
            executable=executable,
            command=command,
            environment=environment,
        )
        raise RuntimeError("exec captured")

    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
    monkeypatch.setattr(runtime, "_mpi_executable", lambda: "/env/bin/mpiexec")
    monkeypatch.setattr(runtime.sys, "executable", "/env/bin/python")
    monkeypatch.setattr(runtime.sys, "argv", ["run_setup.py"])
    monkeypatch.setattr(runtime.os, "execvpe", capture)

    with pytest.raises(RuntimeError, match="exec captured"):
        runtime.RunConfig(cpu_cores=4, parallel_mode="mpi").ensure_mpi(tmp_path / "run_setup.py")

    assert captured["command"][:4] == ["/env/bin/mpiexec", "-n", "4", "/env/bin/python"]
    assert captured["environment"]["_OPENONDA_MPI_CHILD"] == "1"
    assert captured["environment"]["PMIX_MCA_pif_base_retain_loopback"] == "1"
    for name in runtime._THREAD_VARIABLES:
        assert captured["environment"][name] == "1"


def test_mpi_launcher_matches_mpi4py_vendor(monkeypatch):
    monkeypatch.delenv("OPENONDA_MPIEXEC", raising=False)
    monkeypatch.setattr(runtime.sys, "executable", "/env/bin/python")
    monkeypatch.setattr(runtime, "_mpi_vendor", lambda: "Open MPI")
    monkeypatch.setattr(
        runtime.shutil,
        "which",
        lambda name: {
            "mpiexec.openmpi": "/usr/bin/mpiexec.openmpi",
            "mpiexec": "/paraview/bin/mpiexec",
        }.get(name),
    )

    assert runtime._mpi_executable() == "/usr/bin/mpiexec.openmpi"


def test_explicit_mpi_launcher_takes_precedence(monkeypatch):
    monkeypatch.setenv("OPENONDA_MPIEXEC", "custom-mpiexec")
    monkeypatch.setattr(runtime.shutil, "which", lambda name: f"/tools/{name}")

    assert runtime._mpi_executable() == "/tools/custom-mpiexec"


def test_mpi_child_must_match_configured_core_count(monkeypatch, tmp_path):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    with pytest.raises(RuntimeError, match="FVMSetup requests cores=4"):
        runtime.RunConfig(cpu_cores=4, parallel_mode="mpi").ensure_mpi(tmp_path / "run_setup.py")


def test_threaded_runtime_configures_all_worker_pools(monkeypatch, tmp_path):
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)

    runtime.RunConfig(cpu_cores=6).ensure_runtime(tmp_path / "run_setup.py")

    for name in (*runtime._THREAD_VARIABLES, "OPENONDA_CPU_THREADS"):
        assert os.environ[name] == "6"
