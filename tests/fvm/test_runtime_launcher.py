"""Process-launch contract for simple Python case entry points."""

from __future__ import annotations

import os

import pytest

import openonda_bootstrap as bootstrap


def test_run_config_validates_cpu_cores():
    assert not bootstrap.RunConfig(cpu_cores=1).is_parallel
    assert bootstrap.RunConfig(cpu_cores=4).is_parallel
    with pytest.raises(ValueError, match="at least one"):
        bootstrap.RunConfig(cpu_cores=0)
    with pytest.raises(TypeError, match="integer"):
        bootstrap.RunConfig(cpu_cores=True)
    with pytest.raises(ValueError, match="parallel_mode"):
        bootstrap.RunConfig(parallel_mode="processes")


def test_serial_launch_does_not_exec(monkeypatch, tmp_path):
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)
    monkeypatch.setattr(
        bootstrap.os,
        "execvpe",
        lambda *_args, **_kwargs: pytest.fail("serial execution must not relaunch"),
    )

    bootstrap.RunConfig(cpu_cores=1).ensure_mpi(tmp_path / "run_setup.py")


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
    monkeypatch.setattr(bootstrap, "_mpi_executable", lambda: "/env/bin/mpiexec")
    monkeypatch.setattr(bootstrap.sys, "executable", "/env/bin/python")
    monkeypatch.setattr(bootstrap.sys, "argv", ["run_setup.py"])
    monkeypatch.setattr(bootstrap.os, "execvpe", capture)

    with pytest.raises(RuntimeError, match="exec captured"):
        bootstrap.RunConfig(cpu_cores=4, parallel_mode="mpi").ensure_mpi(tmp_path / "run_setup.py")

    assert captured["command"][:4] == ["/env/bin/mpiexec", "-n", "4", "/env/bin/python"]
    assert captured["environment"]["_OPENONDA_MPI_CHILD"] == "1"
    assert captured["environment"]["PMIX_MCA_pif_base_retain_loopback"] == "1"
    for name in bootstrap._THREAD_VARIABLES:
        assert captured["environment"][name] == "1"


def test_mpi_child_must_match_configured_core_count(monkeypatch, tmp_path):
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    with pytest.raises(RuntimeError, match="FVMSetup requests cores=4"):
        bootstrap.RunConfig(cpu_cores=4, parallel_mode="mpi").ensure_mpi(tmp_path / "run_setup.py")


def test_threaded_runtime_configures_all_worker_pools(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap, "_is_canonical_environment", lambda: True)
    monkeypatch.setattr(bootstrap.sys, "version_info", (3, 11))
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE", raising=False)

    bootstrap.RunConfig(cpu_cores=6).ensure_runtime(tmp_path / "run_setup.py")

    for name in (*bootstrap._THREAD_VARIABLES, "OPENONDA_CPU_THREADS"):
        assert os.environ[name] == "6"


def test_wrong_python_relaunches_canonical_conda_environment(monkeypatch, tmp_path):
    captured = {}

    def capture(executable, command, environment):
        captured.update(
            executable=executable,
            command=command,
            environment=environment,
        )
        raise RuntimeError("exec captured")

    monkeypatch.setattr(bootstrap.sys, "version_info", (3, 10))
    monkeypatch.setattr(bootstrap.sys, "argv", ["run_setup.py"])
    monkeypatch.setattr(bootstrap, "_conda_executable", lambda: "/conda/bin/conda")
    monkeypatch.setattr(bootstrap.os, "execvpe", capture)
    monkeypatch.delenv("_OPENONDA_CONDA_CHILD", raising=False)

    with pytest.raises(RuntimeError, match="exec captured"):
        bootstrap.RunConfig().ensure_python(tmp_path / "run_setup.py")

    assert captured["command"][:7] == [
        "/conda/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "OpenONDA",
        "python",
        str((tmp_path / "run_setup.py").resolve()),
    ]
    assert captured["environment"]["_OPENONDA_CONDA_CHILD"] == "1"
    assert os.environ.get("_OPENONDA_CONDA_CHILD") is None


def test_matching_python_outside_canonical_environment_still_relaunches(monkeypatch, tmp_path):
    def capture(*_args, **_kwargs):
        raise RuntimeError("exec captured")

    monkeypatch.setattr(bootstrap.sys, "version_info", (3, 11))
    monkeypatch.setattr(bootstrap, "_is_canonical_environment", lambda: False)
    monkeypatch.setattr(bootstrap.sys, "argv", ["run_setup.py"])
    monkeypatch.setattr(bootstrap, "_conda_executable", lambda: "/conda/bin/conda")
    monkeypatch.setattr(bootstrap.os, "execvpe", capture)
    monkeypatch.delenv("_OPENONDA_CONDA_CHILD", raising=False)

    with pytest.raises(RuntimeError, match="exec captured"):
        bootstrap.RunConfig().ensure_python(tmp_path / "run_setup.py")
