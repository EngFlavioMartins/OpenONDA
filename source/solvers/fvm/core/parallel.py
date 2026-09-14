"""MPI execution context for replicated and partitioned FVM execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from openonda.runtime import detected_world_size


class _MPICommunicator(Protocol):
    """Subset of the mpi4py communicator used by the FVM runtime."""

    def Get_size(self) -> int:
        """Return the communicator size."""

        ...

    def Get_rank(self) -> int:
        """Return the caller's rank."""

        ...

    def Barrier(self) -> None:
        """Block until every rank enters the barrier."""

        ...

    def bcast(self, value: object, root: int = 0) -> object:
        """Broadcast a pickle-compatible value from ``root``."""

        ...

    def allreduce(self, value: object, op: object = None) -> object:
        """Reduce a value across ranks with the supplied MPI operation."""

        ...

    def gather(self, value: object, root: int = 0) -> object:
        """Gather one value from each rank on ``root``."""

        ...

    def send(self, value: object, dest: int, tag: int = 0) -> None:
        """Send one pickle-compatible value to ``dest``."""

        ...

    def recv(self, source: int = -1, tag: int = -1) -> object:
        """Receive one value from a matching source/tag."""

        ...

    def Irecv(self, buf: np.ndarray, source: int, tag: int) -> object:
        """Post a nonblocking numeric receive into ``buf``."""

        ...

    def Isend(self, buf: np.ndarray, dest: int, tag: int) -> object:
        """Post a nonblocking numeric send from ``buf``."""

        ...


class _MPINamespace(Protocol):
    """Reduction operators imported from ``mpi4py.MPI``."""

    SUM: object
    MAX: object
    MIN: object
    LAND: object


class _CellPartition(Protocol):
    """Partition attributes and methods required by :class:`ParallelContext`."""

    owned_global_ids: np.ndarray

    def exchange_halo(self, local_field: np.ndarray, comm: _MPICommunicator) -> None:
        """Update ghost rows in ``local_field`` through the halo schedule."""

        ...


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
    comm: _MPICommunicator | None = None
    mpi: _MPINamespace | None = None
    rank: int = 0
    size: int = 1
    partition: _CellPartition | None = None

    @classmethod
    def create(
        cls,
        execution,
        *,
        comm: _MPICommunicator | None = None,
        mpi: _MPINamespace | None = None,
    ) -> ParallelContext:
        """Validate execution settings and create the MPI context.

        Parameters
        ----------
        execution : ComputeConfig
            FVM execution policy. Supported modes are serial, PETSc replicated,
            and PETSc partitioned.
        comm, mpi : object or None, optional
            Injected communicator/MPI module for tests or an existing runtime.
            When omitted in a PETSc mode, ``mpi4py`` and ``petsc4py`` are
            imported and ``MPI.COMM_WORLD`` is used.

        Returns
        -------
        ParallelContext
            Validated rank/size context. Serial mode returns rank zero/size
            one and performs no MPI import.

        Raises
        ------
        RuntimeError
            If launcher size disagrees with the communicator or required MPI/
            PETSc dependencies are unavailable.
        ValueError
            If backend, mode, output, or linear-solver combinations are
            unsupported.
        """
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
                    "Create the solver with create_fvm_solver and set FVMSetup.cores "
                    "to the requested process count; the factory owns MPI setup."
                )
            return cls()

        if linear != "petsc":
            raise ValueError(f"parallel_mode={mode!r} requires linear_backend='petsc'")

        if comm is None or mpi is None:
            try:
                from mpi4py import MPI
            except ImportError as error:
                raise RuntimeError(
                    "Parallel FVM requires mpi4py. Install OpenONDA's 'parallel' "
                    "optional dependencies in an MPI/PETSc environment."
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
                "Parallel FVM requires petsc4py linked to PETSc. Install OpenONDA's "
                "'parallel' optional dependencies using the same MPI implementation."
            ) from error
        return cls(mode=mode, comm=comm, mpi=mpi, rank=rank, size=size)

    def with_partition(self, partition: _CellPartition) -> ParallelContext:
        """Return a copy bound to one validated local mesh partition.

        Parameters
        ----------
        partition : CellPartition
            Partition whose rank/size match this context. It stores owned and
            halo global cell IDs and performs halo exchange.

        Raises
        ------
        RuntimeError
            If this context is not partitioned.
        ValueError
            If partition rank or size disagrees with the communicator.
        """
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
        """Whether this rank is rank zero, the replicated-output owner."""
        return self.rank == 0

    @property
    def is_parallel(self) -> bool:
        """Whether the communicator contains more than one rank."""
        return self.size > 1

    @property
    def owns_replicated_output(self) -> bool:
        """Only rank zero exposes replicated fields to external consumers."""
        return self.is_root

    @property
    def is_partitioned(self) -> bool:
        """Whether fields use local owned-plus-halo PETSc partitions."""
        return self.mode == "petsc_partitioned"

    @property
    def n_owned(self) -> int | None:
        """Return local owned-cell count, or ``None`` for unpartitioned modes."""
        if self.partition is None:
            return None
        return len(self.partition.owned_global_ids)

    def exchange_halo(self, values) -> None:
        """Exchange owned values into local halo rows in place.

        Parameters
        ----------
        values : numpy.ndarray
            Local field whose leading dimension contains owned cells followed
            by halo cells. The trailing shape is preserved.

        Notes
        -----
        This is an MPI collective in partitioned mode and a no-op otherwise.
        """
        if self.is_partitioned:
            if self.partition is None:
                raise RuntimeError("Partitioned context has no cell partition")
            self.partition.exchange_halo(values, self.comm)

    def barrier(self) -> None:
        """Synchronize all ranks; a no-op in serial mode.

        Raises
        ------
        RuntimeError
            If a parallel context has no communicator.
        """
        if self.is_parallel:
            if self.comm is None:
                raise RuntimeError("Parallel context has no MPI communicator")
            self.comm.Barrier()

    def bcast(self, value, root: int = 0):
        """Broadcast ``value`` from ``root`` and return it on every rank.

        In serial mode the input is returned unchanged. In parallel mode this
        is an MPI collective and every rank must call it in the same order.

        Parameters
        ----------
        value : object
            Pickle-compatible value supplied by ``root``.
        root : int, default=0
            Source rank.

        Returns
        -------
        object
            Broadcast value on every rank.
        """
        if not self.is_parallel:
            return value
        if self.comm is None:
            raise RuntimeError("Parallel context has no MPI communicator")
        return self.comm.bcast(value, root=root)

    def global_sum(self, value):
        """Return an MPI sum reduction, or ``value`` in serial mode.

        Scalars and NumPy-compatible reduction values are accepted. The return
        type follows the communicator's `allreduce` implementation.

        Parameters
        ----------
        value : object
            Scalar or array-compatible local contribution.

        Returns
        -------
        object
            Global sum, or ``value`` unchanged in serial mode.
        """
        if not self.is_parallel:
            return value
        if self.comm is None or self.mpi is None:
            raise RuntimeError("Parallel context has no MPI reduction objects")
        return self.comm.allreduce(value, op=self.mpi.SUM)

    def global_max(self, value):
        """Return a global maximum reduction over scalar-compatible values.

        Parameters
        ----------
        value : object
            Local scalar contribution.

        Returns
        -------
        object
            Global maximum, or the input in serial mode.
        """
        if not self.is_parallel:
            return value
        if self.comm is None or self.mpi is None:
            raise RuntimeError("Parallel context has no MPI reduction objects")
        return self.comm.allreduce(value, op=self.mpi.MAX)

    def global_min(self, value):
        """Return a global minimum reduction over scalar-compatible values.

        Parameters
        ----------
        value : object
            Local scalar contribution.

        Returns
        -------
        object
            Global minimum, or the input in serial mode.
        """
        if not self.is_parallel:
            return value
        if self.comm is None or self.mpi is None:
            raise RuntimeError("Parallel context has no MPI reduction objects")
        return self.comm.allreduce(value, op=self.mpi.MIN)

    def global_all(self, value: bool) -> bool:
        """Return whether a boolean condition holds on every rank.

        Parameters
        ----------
        value : bool
            Local condition.

        Returns
        -------
        bool
            Logical AND over ranks.
        """
        if not self.is_parallel:
            return bool(value)
        if self.comm is None or self.mpi is None:
            raise RuntimeError("Parallel context has no MPI reduction objects")
        return bool(self.comm.allreduce(bool(value), op=self.mpi.LAND))

    def root_view(self, values, *, trailing_shape=(), dtype=np.float64):
        """Return values on root and a typed empty view elsewhere.

        Parameters
        ----------
        values : array-like
            Values to expose on the root rank.
        trailing_shape : tuple[int, ...], default=()
            Shape after the leading empty dimension on non-root ranks.
        dtype : numpy.dtype, default=numpy.float64
            Output dtype.

        Returns
        -------
        numpy.ndarray
            Contiguous root copy, or an empty array on non-root ranks.
        """
        if self.owns_replicated_output:
            return np.ascontiguousarray(values, dtype=dtype)
        return np.empty((0, *trailing_shape), dtype=dtype)
