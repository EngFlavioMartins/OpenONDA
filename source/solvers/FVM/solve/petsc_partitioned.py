"""Owned-row PETSc assembly for partitioned sparse systems."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from ..mesh.partition import ownership_ranges
from .linear_interface import LinearSolveError, LinearSolveResult


@dataclass(frozen=True)
class OwnedRowsCSR:
    """Local CSR rows with global column indices and a local right-hand side."""

    global_size: int
    row_start: int
    row_end: int
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    rhs: np.ndarray

    @classmethod
    def from_global(cls, matrix, rhs, rank: int, size: int) -> OwnedRowsCSR:
        """Create a test/reference partition without retaining the global matrix."""
        matrix = matrix.tocsr()
        offsets = ownership_ranges(matrix.shape[0], size)
        start, end = int(offsets[rank]), int(offsets[rank + 1])
        local = matrix[start:end].tocsr()
        return cls(
            global_size=matrix.shape[0],
            row_start=start,
            row_end=end,
            indptr=np.asarray(local.indptr, dtype=np.int64),
            indices=np.asarray(local.indices, dtype=np.int64),
            data=np.asarray(local.data, dtype=np.float64),
            rhs=np.asarray(rhs[start:end], dtype=np.float64),
        )

    @classmethod
    def from_local(cls, matrix, rhs, partition) -> OwnedRowsCSR:
        """Convert owned local rows and local columns to global PETSc indices."""
        matrix = matrix.tocsr()
        n_local = len(partition.local_global_ids)
        n_owned = len(partition.owned_global_ids)
        if matrix.shape != (n_local, n_local) or np.asarray(rhs).shape != (n_local,):
            raise ValueError("Local matrix/RHS does not match the partition field layout")
        local = matrix[:n_owned].tocsr()
        owned = partition.owned_global_ids
        if n_owned and not np.array_equal(owned, np.arange(owned[0], owned[0] + n_owned)):
            raise ValueError("PETSc owned rows must be contiguous global cell IDs")
        start = int(owned[0]) if n_owned else int(partition.global_n_cells)
        end = start + n_owned
        return cls(
            global_size=partition.global_n_cells,
            row_start=start,
            row_end=end,
            indptr=np.asarray(local.indptr, dtype=np.int64),
            indices=np.asarray(partition.local_global_ids[local.indices], dtype=np.int64),
            data=np.asarray(local.data, dtype=np.float64),
            rhs=np.asarray(rhs[:n_owned], dtype=np.float64),
        )

    def validate(self) -> None:
        local_rows = self.row_end - self.row_start
        if not 0 <= self.row_start <= self.row_end <= self.global_size:
            raise ValueError("Invalid owned row range")
        if self.indptr.shape != (local_rows + 1,) or self.rhs.shape != (local_rows,):
            raise ValueError("Owned CSR row pointers or RHS have an invalid shape")
        if self.indptr[0] != 0 or self.indptr[-1] != len(self.indices):
            raise ValueError("Owned CSR row pointers do not cover all entries")
        if len(self.indices) != len(self.data):
            raise ValueError("Owned CSR indices and data lengths differ")
        if np.any(self.indices < 0) or np.any(self.indices >= self.global_size):
            raise ValueError("Owned CSR contains an out-of-range global column")


class PartitionedLinearWorkspace:
    """Persistent PETSc objects for one partitioned equation family.

    Coefficients and RHS values are replaced for every solve; ownership,
    topology, KSP method, and null-space treatment define the allocation
    signature.  Retaining the allocation removes collective object churn
    without allowing a stale preconditioner to change convergence behaviour.
    """

    def __init__(self, context) -> None:
        self.context = context
        self.matrix = None
        self.rhs = None
        self.solution = None
        self.residual = None
        self.ksp = None
        self.nullspace = None
        self._signature = None

    def _destroy_objects(self) -> None:
        for name in ("ksp", "residual", "solution", "rhs", "matrix", "nullspace"):
            value = getattr(self, name)
            if value is not None:
                value.destroy()
                setattr(self, name, None)
        self._signature = None

    def close(self) -> None:
        """Collectively destroy PETSc objects owned by this workspace."""
        self._destroy_objects()

    def _build(self, system, method: str, constant_nullspace: bool, PETSc) -> None:
        signature = (
            system.global_size,
            system.row_start,
            system.row_end,
            tuple(np.diff(system.indptr).tolist()),
            method,
            constant_nullspace,
        )
        if self._signature == signature:
            return
        self._destroy_objects()
        matrix = PETSc.Mat().createAIJ(
            size=(system.global_size, system.global_size),
            nnz=max(int(np.max(np.diff(system.indptr), initial=0)), 1),
            comm=PETSc.COMM_WORLD,
        )
        matrix.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        actual_start, actual_end = matrix.getOwnershipRange()
        if (actual_start, actual_end) != (system.row_start, system.row_end):
            matrix.destroy()
            raise RuntimeError(
                f"PETSc owns rows {(actual_start, actual_end)}, supplied "
                f"{(system.row_start, system.row_end)}"
            )
        self.matrix = matrix
        self.rhs = PETSc.Vec().createMPI(system.global_size, comm=PETSc.COMM_WORLD)
        self.solution = self.rhs.duplicate()
        self.residual = self.rhs.duplicate()
        if constant_nullspace:
            self.nullspace = PETSc.NullSpace().create(constant=True, comm=PETSc.COMM_WORLD)
            matrix.setNullSpace(self.nullspace)
            matrix.setNearNullSpace(self.nullspace)
        ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
        ksp.setOperators(matrix)
        methods = {
            "cg": PETSc.KSP.Type.CG,
            "gmres": PETSc.KSP.Type.GMRES,
            "bicgstab": PETSc.KSP.Type.BCGS,
        }
        if method == "amg":
            ksp.setType(PETSc.KSP.Type.CG)
            ksp.getPC().setType(PETSc.PC.Type.GAMG)
        else:
            try:
                ksp.setType(methods[method])
            except KeyError as error:
                self._destroy_objects()
                raise ValueError(f"Unsupported partitioned PETSc method {method!r}") from error
            ksp.getPC().setType(
                PETSc.PC.Type.JACOBI if constant_nullspace else PETSc.PC.Type.BJACOBI
            )
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        ksp.setOptionsPrefix("fvm_partitioned_")
        ksp.setFromOptions()
        self.ksp = ksp
        self._signature = signature

    def solve(
        self,
        system: OwnedRowsCSR,
        *,
        method: str,
        tolerance: float,
        max_iterations: int,
        constant_nullspace: bool,
        initial_guess: np.ndarray | None,
    ):
        """Update numeric values and solve using persistent PETSc objects."""
        try:
            from petsc4py import PETSc
        except ImportError as error:
            raise RuntimeError("Owned-row solves require petsc4py") from error
        system.validate()
        if self.context.size != PETSc.COMM_WORLD.getSize():
            raise RuntimeError("mpi4py and PETSc communicator sizes differ")
        setup_start = time.perf_counter()
        self._build(system, method, constant_nullspace, PETSc)
        assert self.matrix is not None
        assert self.rhs is not None
        assert self.solution is not None
        assert self.residual is not None
        assert self.ksp is not None

        self.matrix.zeroEntries()
        for local_row, global_row in enumerate(range(system.row_start, system.row_end)):
            start, end = system.indptr[local_row : local_row + 2]
            if end > start:
                self.matrix.setValues(
                    global_row,
                    np.asarray(system.indices[start:end], dtype=PETSc.IntType),
                    system.data[start:end],
                )
        self.matrix.assemblyBegin()
        self.matrix.assemblyEnd()

        self.rhs.set(0.0)
        rows = np.arange(system.row_start, system.row_end, dtype=PETSc.IntType)
        if system.row_end > system.row_start:
            self.rhs.setValues(rows, system.rhs)
        self.rhs.assemblyBegin()
        self.rhs.assemblyEnd()
        if self.nullspace is not None:
            self.nullspace.remove(self.rhs)

        self.solution.set(0.0)
        if initial_guess is not None:
            guess = np.asarray(initial_guess, dtype=np.float64)
            if guess.shape != system.rhs.shape:
                raise ValueError("Partitioned initial guess must contain one value per owned row")
            if system.row_end > system.row_start:
                self.solution.setValues(rows, guess)
        self.solution.assemblyBegin()
        self.solution.assemblyEnd()
        self.ksp.setOperators(self.matrix)
        self.ksp.setTolerances(rtol=tolerance, max_it=max_iterations)
        self.ksp.setInitialGuessNonzero(initial_guess is not None)
        # Coefficients are dynamic.  PETSc's default KSP policy rebuilds the
        # PC after a numeric matrix update; do not call version-specific API
        # here merely to request the default.
        setup_seconds = time.perf_counter() - setup_start

        solve_start = time.perf_counter()
        self.ksp.solve(self.rhs, self.solution)
        solve_seconds = time.perf_counter() - solve_start
        reason_code = int(self.ksp.getConvergedReason())
        rhs_norm = float(self.rhs.norm())
        self.matrix.mult(self.solution, self.residual)
        self.residual.axpy(-1.0, self.rhs)
        residual_norm = float(self.residual.norm())
        relative_residual = residual_norm / max(rhs_norm, 1e-30)
        result = LinearSolveResult(
            backend="petsc-partitioned",
            method=method,
            preconditioner=str(self.ksp.getPC().getType()),
            nullspace="constant" if constant_nullspace else None,
            converged=(
                reason_code > 0
                and np.isfinite(relative_residual)
                and relative_residual <= max(10.0 * tolerance, 1e-12)
            ),
            reason=str(self.ksp.getConvergedReason()),
            iterations=int(self.ksp.getIterationNumber()),
            initial_residual=1.0 if rhs_norm else 0.0,
            final_residual=relative_residual,
            setup_seconds=setup_seconds,
            solve_seconds=solve_seconds,
            preconditioner_rebuilt=True,
        )
        if not result.converged:
            raise LinearSolveError(
                f"Partitioned PETSc {method} failed after {result.iterations} iterations: "
                f"{result.reason}"
            )
        return self.solution.getArray(readonly=True).copy(), result


def solve_owned_rows(
    system: OwnedRowsCSR,
    context,
    *,
    method: str = "cg",
    tolerance: float = 1e-10,
    max_iterations: int = 500,
    constant_nullspace: bool = False,
    initial_guess: np.ndarray | None = None,
):
    """One-shot compatibility wrapper around a temporary workspace."""
    workspace = PartitionedLinearWorkspace(context)
    try:
        return workspace.solve(
            system,
            method=method,
            tolerance=tolerance,
            max_iterations=max_iterations,
            constant_nullspace=constant_nullspace,
            initial_guess=initial_guess,
        )
    finally:
        workspace.close()


def solve_local_partitioned_system(
    matrix,
    rhs,
    context,
    *,
    method: str,
    tolerance: float,
    max_iterations: int,
    constant_nullspace: bool,
    initial_guess=None,
    workspace: PartitionedLinearWorkspace | None = None,
):
    """Solve owned rows and return a refreshed owned-plus-halo local vector."""
    if context.partition is None:
        raise RuntimeError("Partitioned PETSc solve requires partition metadata")
    system = OwnedRowsCSR.from_local(matrix, rhs, context.partition)
    n_owned = len(context.partition.owned_global_ids)
    guess = None if initial_guess is None else np.asarray(initial_guess)[:n_owned]
    if workspace is None:
        owned, result = solve_owned_rows(
            system,
            context,
            method=method,
            tolerance=tolerance,
            max_iterations=max_iterations,
            constant_nullspace=constant_nullspace,
            initial_guess=guess,
        )
    else:
        owned, result = workspace.solve(
            system,
            method=method,
            tolerance=tolerance,
            max_iterations=max_iterations,
            constant_nullspace=constant_nullspace,
            initial_guess=guess,
        )
    local = np.empty(len(context.partition.local_global_ids), dtype=np.float64)
    local[:n_owned] = owned
    context.exchange_halo(local)
    if constant_nullspace:
        local -= context.global_sum(float(np.sum(local[:n_owned]))) / system.global_size
    return local, result
