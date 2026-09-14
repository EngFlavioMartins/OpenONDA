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
)
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_WORKER_THREADS: int | None = None


def worker_thread_count() -> int:
    """CPU budget for owner-only particle work between collective FVM solves.

    MPI ranks each use one BLAS/Numba thread. While they wait for the particle
    owner, that owner can use the case's CPU budget for Taichi/host kernels.
    Standalone VPM defaults to the available CPUs. An explicit Taichi override
    is still honoured for advanced callers.
    """
    requested = os.environ.get("TI_CPU_MAX_NUM_THREADS")
    if requested is not None:
        count = int(requested)
        if count < 1:
            raise ValueError("TI_CPU_MAX_NUM_THREADS must be positive")
        return count
    if _WORKER_THREADS is not None:
        return _WORKER_THREADS
    affinity = getattr(os, "sched_getaffinity", None)
    return max(1, len(affinity(0)) if affinity is not None else (os.cpu_count() or 1))


def detected_world_size() -> int:
    """Read MPI launcher hints without initializing an MPI runtime.

    An allocation such as SLURM_NTASKS reserves resources; it does not mean
    this Python process already belongs to a multi-rank communicator. Only
    launcher-specific variables are evidence that MPI has started.
    """
    sizes = []
    for name in _MPI_SIZE_VARIABLES:
        value = os.environ.get(name)
        if value is not None:
            try:
                size = int(value)
                if size < 1:
                    raise ValueError("MPI world size must be positive")
            except ValueError as error:
                raise RuntimeError(f"Invalid MPI launcher variable {name}={value!r}") from error
            sizes.append(size)
    return max(sizes, default=1)


def _world_size() -> int:
    return detected_world_size()


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
    """Declare how one OpenONDA process uses CPU threads or MPI ranks.

    Parameters
    ----------
    cpu_cores : int, default=1
        Positive number of worker threads when ``parallel_mode="threads"`` or
        MPI ranks when ``parallel_mode="mpi"``.
    parallel_mode : {'threads', 'mpi'}, default='threads'
        Execution model. Threaded execution stays in the current process;
        MPI execution may replace it with ``mpiexec -n cpu_cores``.

    Notes
    -----
    :meth:`ensure_runtime` changes process-wide numerical-library thread
    limits. In MPI mode it may call :func:`os.execvpe`, so code following that
    call runs only after the process has been relaunched under MPI.
    """

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
        """Whether the configuration requests more than one worker or rank."""
        return self.cpu_cores > 1

    def _set_thread_count(self, count: int) -> None:
        global _WORKER_THREADS
        _WORKER_THREADS = self.cpu_cores
        # Numba fixes its pool capacity at import. Mutating NUMBA_NUM_THREADS
        # later breaks subsequent JIT compilation, including VPM diffusion
        # after an FVM solve in the same process. Mask active threads through
        # its runtime API, leaving the process pool capacity unchanged.
        for name in _THREAD_VARIABLES:
            os.environ[name] = str(count)
        import numba
        from threadpoolctl import threadpool_limits

        numba.set_num_threads(count)
        # NumPy/SciPy are usually imported before solver construction. Merely
        # setting environment variables here does not resize their live pools.
        threadpool_limits(limits=count)

    def ensure_mpi(self, script: str | Path) -> None:
        """Ensure that ``script`` is running with the requested MPI world size.

        Parameters
        ----------
        script : str or pathlib.Path
            Python entry-point path passed to the relaunched interpreter.

        Raises
        ------
        RuntimeError
            If an existing MPI world has the wrong size, no launcher can be
            found, or a requested child launch did not create an MPI world.

        Notes
        -----
        With ``cpu_cores > 1`` outside an MPI world, this method replaces the
        current process with ``mpiexec``. It also restricts each rank to one
        numerical-library thread to avoid oversubscription.
        """
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
        """Apply this execution policy before allocating solver resources.

        Parameters
        ----------
        script : str or pathlib.Path
            Entry point used only if MPI relaunch is necessary.

        Raises
        ------
        RuntimeError
            If threaded execution is requested inside an MPI world, or if
            MPI setup fails as described by :meth:`ensure_mpi`.

        Notes
        -----
        Thread mode updates Numba and BLAS-related process environment limits;
        MPI mode delegates to :meth:`ensure_mpi`.
        """
        if self.parallel_mode == "mpi":
            self.ensure_mpi(script)
            return
        if _world_size() > 1:
            raise RuntimeError("A threaded case must not be launched with mpiexec")
        self._set_thread_count(self.cpu_cores)


__all__ = ["RunConfig"]
