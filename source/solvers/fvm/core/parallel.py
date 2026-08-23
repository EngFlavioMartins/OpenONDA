"""MPI execution context for replicated and partitioned FVM execution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

_MPI_SIZE_ENV = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
    "MV2_COMM_WORLD_SIZE",
    "SLURM_NTASKS",
)


def detected_world_size() -> int:
    """Best-effort launcher size without importing MPI.

    Detecting this before importing ``mpi4py`` lets a rank launched under MPI
    fail with an actionable dependency error instead of silently running an
    independent serial simulation on every process.
    """
    values = []
    for name in _MPI_SIZE_ENV:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            values.append(int(raw))
        except ValueError as error:
            raise RuntimeError(f"Invalid MPI launcher variable {name}={raw!r}") from error
    return max(values, default=1)


@dataclass(frozen=True)
class ParallelContext:
    """MPI rank and communicator state for the FVM solver.

    Carries the parallel mode (``"serial"``, ``"petsc_replicated"``, or
    ``"petsc_partitioned"``), the ``mpi4py`` communicator, rank and size,
    and an optional local :class:`~source.solvers.fvm.mesh.partition.CellPartition`.

    Use the :meth:`create` factory to validate an :class:`ExecutionConfig`
    and build the appropriate context.  Properties like :attr:`is_root`,
    :attr:`is_parallel`, and :attr:`is_partitioned` provide safe guards
    around operations that may differ between serial and parallel modes.

    Examples
    --------
    >>> ctx = ParallelContext.create(execution_config)
    >>> if ctx.is_root:
    ...     log.info("Running on %d rank(s)", ctx.size)
    """

    mode: str = "serial"
    comm: Any | None = None
    mpi: Any | None = None
    rank: int = 0
    size: int = 1
    partition: Any | None = None

    @classmethod
    def create(cls, execution, *, comm=None, mpi=None) -> ParallelContext:
        """Validate an :class:`ExecutionConfig` and create its context."""
        operator = str(execution.operator_backend).lower()
        linear = str(execution.linear_backend).lower()
        mode = str(execution.parallel_mode).lower()
        output_mode = str(execution.output_mode).lower()

        unsupported = []
        if operator not in {"numpy", "numba", "taichi"}:
            unsupported.append(f"operator_backend={operator!r}")
        if linear not in {"scipy", "petsc"}:
            unsupported.append(f"linear_backend={linear!r}")
        if mode not in {"serial", "petsc_replicated", "petsc_partitioned"}:
            unsupported.append(f"parallel_mode={mode!r}")
        if output_mode not in {"synchronous", "threaded"}:
            unsupported.append(f"output_mode={output_mode!r}")
        if mode == "petsc_partitioned" and output_mode == "threaded":
            unsupported.append("output_mode='threaded' with petsc_partitioned")
        if unsupported:
            raise ValueError("Unsupported FVM execution configuration: " + ", ".join(unsupported))

        launcher_size = detected_world_size()
        if mode == "serial":
            if linear != "scipy":
                raise ValueError("parallel_mode='serial' currently requires linear_backend='scipy'")
            if launcher_size > 1:
                raise RuntimeError(
                    f"FVM serial mode was launched with {launcher_size} MPI ranks. "
                    "Use ExecutionConfig.petsc_replicated() with the fvm-parallel "
                    "dependencies, or launch one process."
                )
            return cls()

        if linear != "petsc":
            raise ValueError(f"parallel_mode={mode!r} requires linear_backend='petsc'")

        if comm is None or mpi is None:
            try:
                from mpi4py import MPI
            except ImportError as error:
                raise RuntimeError(
                    "petsc_replicated mode requires mpi4py. Install the "
                    "'fvm-parallel' optional dependency in an MPI/PETSc environment."
                ) from error
            mpi = MPI
            comm = MPI.COMM_WORLD

        size = int(comm.Get_size())
        rank = int(comm.Get_rank())
        if launcher_size > 1 and size != launcher_size:
            raise RuntimeError(
                f"mpi4py communicator size {size} disagrees with launcher size {launcher_size}"
            )
        try:
            from petsc4py import PETSc  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "petsc_replicated mode requires petsc4py linked to PETSc. "
                "Install the 'fvm-parallel' optional dependency using the same MPI."
            ) from error
        return cls(mode=mode, comm=comm, mpi=mpi, rank=rank, size=size)

    def with_partition(self, partition) -> ParallelContext:
        """Return this communicator context bound to its local mesh partition."""
        if self.mode != "petsc_partitioned":
            raise RuntimeError("Cell partitions are valid only in petsc_partitioned mode")
        if partition.rank != self.rank or partition.size != self.size:
            raise ValueError("Partition rank/size does not match the communicator")
        return ParallelContext(
            mode=self.mode,
            comm=self.comm,
            mpi=self.mpi,
            rank=self.rank,
            size=self.size,
            partition=partition,
        )

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    @property
    def is_parallel(self) -> bool:
        return self.size > 1

    @property
    def owns_replicated_output(self) -> bool:
        """Only rank zero exposes replicated fields to external consumers."""
        return self.is_root

    @property
    def is_partitioned(self) -> bool:
        return self.mode == "petsc_partitioned"

    @property
    def n_owned(self) -> int | None:
        if self.partition is None:
            return None
        return len(self.partition.owned_global_ids)

    def exchange_halo(self, values) -> None:
        """Update a local owned-plus-halo cell field in place."""
        if self.is_partitioned:
            if self.partition is None:
                raise RuntimeError("Partitioned context has no cell partition")
            self.partition.exchange_halo(values, self.comm)

    def barrier(self) -> None:
        if self.is_parallel:
            self.comm.Barrier()

    def bcast(self, value, root: int = 0):
        if not self.is_parallel:
            return value
        return self.comm.bcast(value, root=root)

    def global_sum(self, value):
        if not self.is_parallel:
            return value
        return self.comm.allreduce(value, op=self.mpi.SUM)

    def global_max(self, value):
        if not self.is_parallel:
            return value
        return self.comm.allreduce(value, op=self.mpi.MAX)

    def global_min(self, value):
        if not self.is_parallel:
            return value
        return self.comm.allreduce(value, op=self.mpi.MIN)

    def global_all(self, value: bool) -> bool:
        if not self.is_parallel:
            return bool(value)
        return bool(self.comm.allreduce(bool(value), op=self.mpi.LAND))

    def root_view(self, values, *, trailing_shape=(), dtype=np.float64):
        """Return replicated values on root and a typed empty array elsewhere."""
        if self.owns_replicated_output:
            return np.ascontiguousarray(values, dtype=dtype)
        return np.empty((0, *trailing_shape), dtype=dtype)
