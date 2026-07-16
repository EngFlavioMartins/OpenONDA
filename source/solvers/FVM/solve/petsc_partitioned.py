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
    """Solve a distributed system without replicating matrix rows or the RHS."""
    try:
        from petsc4py import PETSc
    except ImportError as error:
        raise RuntimeError("Owned-row solves require petsc4py") from error

    system.validate()
    if context.size != PETSc.COMM_WORLD.getSize():
        raise RuntimeError("mpi4py and PETSc communicator sizes differ")

    setup_start = time.perf_counter()
    matrix = PETSc.Mat().createAIJ(
        size=(system.global_size, system.global_size), comm=PETSc.COMM_WORLD
    )
    actual_start, actual_end = matrix.getOwnershipRange()
    if (actual_start, actual_end) != (system.row_start, system.row_end):
        raise RuntimeError(
            f"PETSc owns rows {(actual_start, actual_end)}, supplied "
            f"{(system.row_start, system.row_end)}"
        )
    for local_row, global_row in enumerate(range(system.row_start, system.row_end)):
        start, end = system.indptr[local_row : local_row + 2]
        if end > start:
            columns = np.asarray(system.indices[start:end], dtype=PETSc.IntType)
            matrix.setValues(global_row, columns, system.data[start:end])
    matrix.assemblyBegin()
    matrix.assemblyEnd()

    nullspace = None
    if constant_nullspace:
        nullspace = PETSc.NullSpace().create(constant=True, comm=PETSc.COMM_WORLD)
        matrix.setNullSpace(nullspace)

    rhs = PETSc.Vec().createMPI(system.global_size, comm=PETSc.COMM_WORLD)
    if system.row_end > system.row_start:
        rows = np.arange(system.row_start, system.row_end, dtype=PETSc.IntType)
        rhs.setValues(rows, system.rhs)
    rhs.assemblyBegin()
    rhs.assemblyEnd()
    if nullspace is not None:
        nullspace.remove(rhs)

    solution = rhs.duplicate()
    if initial_guess is not None:
        guess = np.asarray(initial_guess, dtype=np.float64)
        if guess.shape != system.rhs.shape:
            raise ValueError("Partitioned initial guess must contain one value per owned row")
        if system.row_end > system.row_start:
            rows = np.arange(system.row_start, system.row_end, dtype=PETSc.IntType)
            solution.setValues(rows, guess)
        solution.assemblyBegin()
        solution.assemblyEnd()
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
            raise ValueError(f"Unsupported partitioned PETSc method {method!r}") from error
        ksp.getPC().setType(PETSc.PC.Type.JACOBI if constant_nullspace else PETSc.PC.Type.BJACOBI)
    ksp.setTolerances(rtol=tolerance, max_it=max_iterations)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setInitialGuessNonzero(initial_guess is not None)
    ksp.setOptionsPrefix("fvm_partitioned_")
    ksp.setFromOptions()
    setup_seconds = time.perf_counter() - setup_start

    solve_start = time.perf_counter()
    ksp.solve(rhs, solution)
    solve_seconds = time.perf_counter() - solve_start
    reason_code = int(ksp.getConvergedReason())
    rhs_norm = float(rhs.norm())
    residual = rhs.duplicate()
    matrix.mult(solution, residual)
    residual.axpy(-1.0, rhs)
    residual_norm = float(residual.norm())
    relative_residual = residual_norm / max(rhs_norm, 1e-30)
    local_solution = solution.getArray(readonly=True).copy()
    result = LinearSolveResult(
        backend="petsc-partitioned",
        method=method,
        preconditioner=str(ksp.getPC().getType()),
        nullspace="constant" if constant_nullspace else None,
        converged=(
            reason_code > 0
            and np.isfinite(relative_residual)
            and relative_residual <= max(10.0 * tolerance, 1e-12)
        ),
        reason=str(ksp.getConvergedReason()),
        iterations=int(ksp.getIterationNumber()),
        initial_residual=1.0 if rhs_norm else 0.0,
        final_residual=relative_residual,
        setup_seconds=setup_seconds,
        solve_seconds=solve_seconds,
        preconditioner_rebuilt=True,
    )

    ksp.destroy()
    residual.destroy()
    solution.destroy()
    rhs.destroy()
    matrix.destroy()
    if nullspace is not None:
        nullspace.destroy()
    if not result.converged:
        raise LinearSolveError(
            f"Partitioned PETSc {method} failed after {result.iterations} iterations: "
            f"{result.reason}"
        )
    return local_solution, result


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
):
    """Solve owned rows and return a refreshed owned-plus-halo local vector."""
    if context.partition is None:
        raise RuntimeError("Partitioned PETSc solve requires partition metadata")
    system = OwnedRowsCSR.from_local(matrix, rhs, context.partition)
    n_owned = len(context.partition.owned_global_ids)
    guess = None if initial_guess is None else np.asarray(initial_guess)[:n_owned]
    owned, result = solve_owned_rows(
        system,
        context,
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
