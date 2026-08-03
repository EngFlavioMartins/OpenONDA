"""Process runtime used by installed OpenONDA applications."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
from typing import Literal

_MPI_CHILD = "_OPENONDA_MPI_CHILD"
_MPIEXEC_VARIABLE = "OPENONDA_MPIEXEC"
_PMIX_RETAIN_LOOPBACK = "PMIX_MCA_pif_base_retain_loopback"
_MPI_SIZE_VARIABLES = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "SLURM_NTASKS",
)
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def _world_size() -> int:
    sizes = []
    for name in _MPI_SIZE_VARIABLES:
        value = os.environ.get(name)
        if value is not None:
            sizes.append(int(value))
    return max(sizes, default=1)


def _mpi_vendor() -> str:
    """MPI implementation used by mpi4py, or an empty string if unavailable."""
    try:
        from mpi4py import MPI

        return str(MPI.get_vendor()[0])
    except (ImportError, RuntimeError):
        return ""


def _mpi_executable() -> str:
    explicit = os.environ.get(_MPIEXEC_VARIABLE, "").strip()
    if explicit:
        discovered = shutil.which(explicit)
        if discovered:
            return discovered
        raise RuntimeError(f"{_MPIEXEC_VARIABLE}={explicit!r} is not an executable launcher")

    environment_launcher = Path(sys.executable).with_name("mpiexec")
    if environment_launcher.is_file():
        return str(environment_launcher)

    vendor_launchers = {
        "Open MPI": ("mpiexec.openmpi",),
        "MPICH": ("mpiexec.mpich", "mpiexec.hydra"),
        "Intel MPI": ("mpiexec.hydra",),
    }
    for launcher in vendor_launchers.get(_mpi_vendor(), ()):
        discovered = shutil.which(launcher)
        if discovered:
            return discovered

    discovered = shutil.which("mpiexec")
    if discovered:
        return discovered
    raise RuntimeError("Parallel execution requires mpiexec in the active environment")


@dataclass(frozen=True)
class RunConfig:
    cpu_cores: int = 1
    parallel_mode: Literal["threads", "mpi"] = "threads"

    def __post_init__(self) -> None:
        if isinstance(self.cpu_cores, bool) or not isinstance(self.cpu_cores, int):
            raise TypeError("cpu_cores must be an integer")
        if self.cpu_cores < 1:
            raise ValueError("cpu_cores must be at least one")
        if self.parallel_mode not in {"threads", "mpi"}:
            raise ValueError("parallel_mode must be 'threads' or 'mpi'")

    @property
    def is_parallel(self) -> bool:
        return self.cpu_cores > 1

    def _set_thread_count(self, count: int) -> None:
        for name in (*_THREAD_VARIABLES, "OPENONDA_CPU_THREADS"):
            os.environ[name] = str(count)

    def ensure_mpi(self, script: str | Path) -> None:
        size = _world_size()
        if size > 1:
            if size != self.cpu_cores:
                raise RuntimeError(
                    f"MPI launched {size} ranks, but FVMSetup requests cores={self.cpu_cores}"
                )
            self._set_thread_count(1)
            return
        self._set_thread_count(1)
        if not self.is_parallel:
            return
        if os.environ.get(_MPI_CHILD) == "1":
            raise RuntimeError("mpiexec did not create the requested MPI communicator")

        environment = os.environ.copy()
        environment[_MPI_CHILD] = "1"
        environment.setdefault(_PMIX_RETAIN_LOOPBACK, "1")
        command = [
            _mpi_executable(),
            "-n",
            str(self.cpu_cores),
            sys.executable,
            str(Path(script).resolve()),
            *sys.argv[1:],
        ]
        os.execvpe(command[0], command, environment)

    def ensure_runtime(self, script: str | Path) -> None:
        if self.parallel_mode == "mpi":
            self.ensure_mpi(script)
            return
        if _world_size() > 1:
            raise RuntimeError("A threaded case must not be launched with mpiexec")
        self._set_thread_count(self.cpu_cores)


__all__ = ["RunConfig"]
