#!/usr/bin/env python3
"""
Linear solver interface for the FVM solver.

This module centralizes linear solver selection, ILU preconditioning and caching,
and provides a single `solve_linear_system` function that other modules call.

It is extracted from `matrix_assembly.py` to decouple matrix construction from solvers
and to make backend swapping (PETSc, GPU solvers) simpler in the future.
"""

from dataclasses import dataclass
import logging
import time

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, bicgstab, cg, gmres, spilu, spsolve

logger = logging.getLogger(__name__)
logger.propagate = False  # Don't send warnings to root logger (stderr)
_ILU_CACHE = {}
_AMG_CACHE = {}
_PYAMG_WARNING_SHOWN = False  # Track if pyamg warning already displayed
_FALLBACK_WARN_COUNT = 0  # Track iterative solver fallback warnings


class LinearSolveError(RuntimeError):
    """Raised when a configured linear backend does not converge."""


@dataclass(frozen=True)
class LinearSolveInfo:
    """Backend-neutral convergence record for a sparse solve."""

    backend: str
    method: str
    converged: bool
    reason: str
    iterations: int
    initial_residual: float
    final_residual: float
    setup_seconds: float
    solve_seconds: float


def normalized_residual(A, x, b):
    """Return a scale-aware algebraic residual for ``A x = b``.

    Unlike an update norm, this directly measures whether the discrete linear
    equation was solved.  The symmetric denominator remains meaningful for a
    near-zero right-hand side and does not hide a large ``A x``.
    """
    ax = A @ x
    residual = np.asarray(b) - np.asarray(ax)
    scale = max(float(np.linalg.norm(b)), float(np.linalg.norm(ax)), 1e-30)
    return float(np.linalg.norm(residual) / scale)


def _solve_petsc(A, b, method, equation_type, tol, maxiter, x0, parallel_context):
    """Collectively solve a replicated SciPy system with distributed PETSc.

    Every rank supplies the same global CSR matrix, but inserts only the rows
    owned by its PETSc matrix.  PETSc distributes vectors/Krylov work; the
    converged solution is then gathered to every rank because the current
    NumPy finite-volume operators still use replicated fields.
    """
    try:
        from petsc4py import PETSc
    except ImportError as error:
        raise RuntimeError(
            "linear_backend='petsc' requires petsc4py linked against PETSc"
        ) from error

    if parallel_context is None:
        raise ValueError("PETSc solves require the solver ParallelContext")
    petsc_size = int(PETSc.COMM_WORLD.getSize())
    if petsc_size != int(parallel_context.size):
        raise RuntimeError(
            f"PETSc communicator size {petsc_size} does not match MPI context "
            f"size {parallel_context.size}; petsc4py and mpi4py must use the same MPI"
        )
    if np.asarray(b).ndim != 1:
        raise ValueError("The PETSc backend currently accepts one RHS vector per solve")

    A_csr = A.tocsr()
    b_array = np.asarray(b, dtype=np.float64)
    n = A_csr.shape[0]
    if A_csr.shape != (n, n) or b_array.shape != (n,):
        raise ValueError("PETSc solve requires a square matrix and matching RHS")

    setup_start = time.perf_counter()
    max_row_nnz = max(int(np.max(np.diff(A_csr.indptr))), 1)
    mat = PETSc.Mat().createAIJ(
        size=A_csr.shape,
        nnz=max_row_nnz,
        comm=PETSc.COMM_WORLD,
    )
    mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    row_start, row_end = mat.getOwnershipRange()
    for row in range(row_start, row_end):
        start, end = A_csr.indptr[row], A_csr.indptr[row + 1]
        if end > start:
            mat.setValues(row, A_csr.indices[start:end], A_csr.data[start:end])
    mat.assemblyBegin()
    mat.assemblyEnd()

    rhs = PETSc.Vec().createMPI(n, comm=PETSc.COMM_WORLD)
    rhs_start, rhs_end = rhs.getOwnershipRange()
    if rhs_end > rhs_start:
        rows = np.arange(rhs_start, rhs_end, dtype=PETSc.IntType)
        rhs.setValues(rows, b_array[rhs_start:rhs_end])
    rhs.assemblyBegin()
    rhs.assemblyEnd()
    solution = rhs.duplicate()

    initial = np.zeros(n, dtype=np.float64) if x0 is None else np.asarray(x0, dtype=np.float64)
    if initial.shape != (n,):
        raise ValueError(f"PETSc initial guess must have shape ({n},)")
    sol_start, sol_end = solution.getOwnershipRange()
    if sol_end > sol_start:
        rows = np.arange(sol_start, sol_end, dtype=PETSc.IntType)
        solution.setValues(rows, initial[sol_start:sol_end])
    solution.assemblyBegin()
    solution.assemblyEnd()

    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp.setOperators(mat)
    pc = ksp.getPC()
    requested = str(method).lower()
    if equation_type == "pressure":
        ksp.setType(PETSc.KSP.Type.CG)
        pc.setType(PETSc.PC.Type.GAMG)
        method_name = "cg+gamg"
    else:
        ksp_types = {
            "bicgstab": PETSc.KSP.Type.BCGS,
            "gmres": PETSc.KSP.Type.GMRES,
            "cg": PETSc.KSP.Type.CG,
            "amg": PETSc.KSP.Type.GMRES,
        }
        if requested not in ksp_types:
            raise ValueError(
                f"PETSc momentum/scalar solver must be bicgstab, gmres, cg, or amg; got {method!r}"
            )
        ksp.setType(ksp_types[requested])
        pc.setType(PETSc.PC.Type.BJACOBI)
        method_name = f"{requested}+bjacobi"
    ksp.setTolerances(rtol=float(tol), max_it=int(maxiter))
    ksp.setInitialGuessNonzero(x0 is not None)
    # Allow command-line PETSc options such as ``-fvm_pressure_pc_type hypre``.
    prefix = "fvm_pressure_" if equation_type == "pressure" else "fvm_momentum_"
    ksp.setOptionsPrefix(prefix)
    ksp.setFromOptions()
    setup_seconds = time.perf_counter() - setup_start

    solve_start = time.perf_counter()
    ksp.solve(rhs, solution)
    solve_seconds = time.perf_counter() - solve_start
    reason_code = int(ksp.getConvergedReason())
    iterations = int(ksp.getIterationNumber())

    scatter, solution_all = PETSc.Scatter.toAll(solution)
    scatter.begin(
        solution,
        solution_all,
        addv=PETSc.InsertMode.INSERT_VALUES,
        mode=PETSc.ScatterMode.FORWARD,
    )
    scatter.end(
        solution,
        solution_all,
        addv=PETSc.InsertMode.INSERT_VALUES,
        mode=PETSc.ScatterMode.FORWARD,
    )
    x = solution_all.getArray(readonly=True).copy()
    initial_residual = normalized_residual(A_csr, initial, b_array)
    final_residual = normalized_residual(A_csr, x, b_array)
    reason = str(ksp.getConvergedReason())
    info = LinearSolveInfo(
        backend="petsc",
        method=method_name,
        converged=reason_code > 0,
        reason=reason,
        iterations=iterations,
        initial_residual=initial_residual,
        final_residual=final_residual,
        setup_seconds=setup_seconds,
        solve_seconds=solve_seconds,
    )

    scatter.destroy()
    solution_all.destroy()
    ksp.destroy()
    solution.destroy()
    rhs.destroy()
    mat.destroy()

    if not info.converged:
        raise LinearSolveError(
            f"PETSc {method_name} failed after {iterations} iterations: {reason}"
        )
    if not np.isfinite(final_residual):
        raise LinearSolveError("PETSc returned a non-finite algebraic residual")
    logger.info(
        "PETSc %s converged in %d iterations: residual %.3e",
        method_name,
        iterations,
        final_residual,
    )
    return x, info


def _cache_key_from_matrix(A_csc, ilu_key=None):
    """Generate a hashable cache key for an ILU preconditioner.

    If *ilu_key* is provided, uses it directly (user-defined key).
    Otherwise builds a structural key from the matrix shape, indptr,
    and indices arrays.

    Args:
        A_csc:   Matrix in CSC format.
        ilu_key: Optional user-defined key.

    Returns:
        A tuple usable as a dict key.
    """
    if ilu_key is not None:
        return ("key", ilu_key)
    return ("pattern", A_csc.shape, A_csc.indptr.tobytes(), A_csc.indices.tobytes())


def _amg_cache_key(A):
    """Return a topology-only key for an AMG hierarchy."""
    csr = A.tocsr()
    return (csr.shape, csr.indptr.tobytes(), csr.indices.tobytes())


def _get_or_build_amg(A, pyamg, reuse_tol=0.05, force_rebuild=False):
    """Build or reuse an AMG hierarchy as a preconditioner.

    Reusing the hierarchy directly as a solver would solve its stale level-0
    matrix.  Instead callers apply it as a preconditioner to CG operating on
    the current matrix, preserving the exact current linear system.
    """
    key = _amg_cache_key(A)
    cached = _AMG_CACHE.get(key)
    diagonal = A.diagonal()
    rebuild = force_rebuild or cached is None
    if cached is not None and not rebuild:
        _, old_diagonal = cached
        relative_change = np.linalg.norm(diagonal - old_diagonal) / (
            np.linalg.norm(old_diagonal) + 1e-30
        )
        rebuild = relative_change > reuse_tol
    if rebuild:
        hierarchy = pyamg.smoothed_aggregation_solver(A)
        _AMG_CACHE[key] = (hierarchy, diagonal.copy())
        return hierarchy
    return cached[0]


def _solve_pressure(A, b, amg_tol, amg_maxiter, tol, maxiter, x0, amg_reuse_tol, failure_policy):
    """Solve the pressure Poisson equation.

    Attempts an algebraic multigrid (AMG) solve via ``pyamg``.
    If ``pyamg`` is unavailable, falls back to
    :func:`_cg_pressure_fallback` (CG with diagonal preconditioner).

    Args:
        A:           Sparse matrix.
        b:           Right-hand side vector.
        amg_tol:     AMG solver tolerance.
        amg_maxiter: AMG max iterations.
        tol:         CG fallback tolerance.
        maxiter:     CG fallback max iterations.
        x0:          Initial guess (optional).

    Returns:
        Solution vector.
    """
    try:
        import pyamg
    except ImportError:
        global _PYAMG_WARNING_SHOWN
        if not _PYAMG_WARNING_SHOWN:
            print("[INFO] pyamg not available, using CG with diagonal preconditioner for pressure")
            _PYAMG_WARNING_SHOWN = True
        return _cg_pressure_fallback(A, b, tol, maxiter, x0, failure_policy)

    try:
        t0 = time.perf_counter()
        ml = _get_or_build_amg(A, pyamg, reuse_tol=amg_reuse_tol)
        M = ml.aspreconditioner(cycle="V")
        x, info = cg(A, b, M=M, rtol=amg_tol, maxiter=amg_maxiter, x0=x0)
        if info != 0:
            # One rebuild handles coefficient drift that made a cached
            # hierarchy ineffective without silently accepting a poor solve.
            ml = _get_or_build_amg(A, pyamg, reuse_tol=amg_reuse_tol, force_rebuild=True)
            M = ml.aspreconditioner(cycle="V")
            x, info = cg(A, b, M=M, rtol=amg_tol, maxiter=amg_maxiter, x0=x0)
        if info != 0:
            raise RuntimeError(f"AMG-preconditioned pressure CG did not converge (info={info})")
        logger.info(f"pyamg pressure solve time={time.perf_counter() - t0:.3f}s")
        return x
    except Exception as error:
        logger.warning("AMG pressure solve failed; using Jacobi-CG fallback: %s", error)
        return _cg_pressure_fallback(A, b, tol, maxiter, x0, failure_policy)


def _cg_pressure_fallback(A, b, tol, maxiter, x0, failure_policy):
    """Conjugate-gradient solve with a diagonal (Jacobi) preconditioner.

    Used as the fallback for the pressure equation when AMG is not
    available.  Falls back to ``spsolve`` if CG does not converge.

    Args:
        A:       Sparse matrix.
        b:       Right-hand side vector.
        tol:     Relative tolerance.
        maxiter: Max iterations.
        x0:      Initial guess (optional).

    Returns:
        Solution vector.
    """
    try:
        t0 = time.perf_counter()
        M_inv = diags(1.0 / (A.diagonal() + 1e-16))
        x, info = cg(A, b, M=M_inv, rtol=tol, maxiter=maxiter, x0=x0)
        logger.info(f"CG pressure fallback time={time.perf_counter() - t0:.3f}s")
        if info != 0:
            logger.warning(f"CG (pressure fallback) did not converge, info={info}")
            if failure_policy == "raise":
                raise LinearSolveError(
                    f"Pressure CG did not converge after {maxiter} iterations (info={info})"
                )
            return spsolve(A, b)
        return x
    except Exception as e2:
        if failure_policy == "raise":
            if isinstance(e2, LinearSolveError):
                raise
            raise LinearSolveError("Pressure iterative solve failed") from e2
        logger.error("Pressure solver fallback failed", exc_info=e2)
        return spsolve(A, b)


def _get_or_build_ilu(A_csc, reuse_ilu, ilu_key, ilu_drop_tol, ilu_fill_factor, ilu_reuse_tol, A):
    """Return an ILU preconditioner, optionally from cache.

    When ``reuse_ilu`` is ``True``, looks up the cache by key and only
    recomputes if the matrix diagonal has changed beyond
    ``ilu_reuse_tol``.  When ``reuse_ilu`` is ``False``, computes a
    transient (non-cached) ILU.

    Args:
        A_csc:          Matrix in CSC format.
        reuse_ilu:      Whether to cache and reuse the ILU factorisation.
        ilu_key:        User-defined cache key (optional).
        ilu_drop_tol:   ILU drop tolerance.
        ilu_fill_factor: ILU fill factor.
        ilu_reuse_tol:  Diagonal change threshold for rebuild.
        A:              Original matrix (for diagonal check).

    Returns:
        An ``spilu`` factorisation object.
    """
    if not reuse_ilu:
        ilu = spilu(A_csc, drop_tol=ilu_drop_tol, fill_factor=ilu_fill_factor)
        logger.info("Computed transient ILU preconditioner (not cached)")
        return ilu

    key = _cache_key_from_matrix(A_csc, ilu_key)
    cached = _ILU_CACHE.get(key)
    if cached is None:
        ilu = spilu(A_csc, drop_tol=ilu_drop_tol, fill_factor=ilu_fill_factor)
        _ILU_CACHE[key] = (ilu, A.diagonal().copy())
        logger.info("Computed and cached new ILU preconditioner")
        return ilu

    ilu_cached, diag_snapshot = cached
    if ilu_reuse_tol is not None and diag_snapshot is not None:
        cur_diag = A.diagonal()
        rel_change = np.linalg.norm(cur_diag - diag_snapshot) / (
            np.linalg.norm(diag_snapshot) + 1e-16
        )
        if rel_change > ilu_reuse_tol:
            logger.info(f"ILU cached but matrix changed (rel_change={rel_change:.3e}), rebuilding")
            ilu = spilu(A_csc, drop_tol=ilu_drop_tol, fill_factor=ilu_fill_factor)
            _ILU_CACHE[key] = (ilu, A.diagonal().copy())
            return ilu

    logger.info("Reusing ILU preconditioner (pattern key)")
    return ilu_cached


def _iterative_solve_with_M(A, b, method, M, tol, maxiter, x0, failure_policy):
    """Run an iterative solver (BiCGSTAB or GMRES) with a preconditioner.

    Falls back to ``spsolve`` if the solver does not converge.

    Args:
        A:       Sparse matrix.
        b:       Right-hand side vector.
        method:  ``"bicgstab"`` or ``"gmres"``.
        M:       Preconditioner (a ``LinearOperator``).
        tol:     Relative tolerance.
        maxiter: Max iterations.
        x0:      Initial guess (optional).

    Returns:
        Solution vector.
    """
    if method == "gmres":
        x, info = gmres(A, b, M=M, rtol=tol, maxiter=maxiter, x0=x0)
    else:
        x, info = bicgstab(A, b, M=M, rtol=tol, maxiter=maxiter, x0=x0)
    if info != 0:
        global _FALLBACK_WARN_COUNT
        _FALLBACK_WARN_COUNT += 1
        msg = f"iterative solver did not converge (info={info}), falling back to direct spsolve"
        logger.warning(msg)
        if _FALLBACK_WARN_COUNT <= 3 or _FALLBACK_WARN_COUNT % 50 == 0:
            print(f"  [WARNING] {msg} (occurrence #{_FALLBACK_WARN_COUNT})")
        if failure_policy == "raise":
            raise LinearSolveError(
                f"{method} did not converge after {maxiter} iterations (info={info})"
            )
        return spsolve(A, b)
    return x


def _solve_with_ilu(
    A,
    b,
    method,
    tol,
    maxiter,
    x0,
    reuse_ilu,
    ilu_key,
    ilu_drop_tol,
    ilu_fill_factor,
    ilu_reuse_tol,
    failure_policy,
):
    """Solve a linear system with ILU-preconditioned iterative solver.

    Builds or retrieves an ILU preconditioner, then runs the selected
    iterative method.  Falls back to plain iterative or ``spsolve``
    on failure.

    Args:
        A:               Sparse matrix.
        b:               Right-hand side.
        method:          ``"bicgstab"`` or ``"gmres"``.
        tol:             Relative tolerance.
        maxiter:         Max iterations.
        x0:              Initial guess.
        reuse_ilu:       Whether to cache ILU.
        ilu_key:         ILU cache key.
        ilu_drop_tol:    ILU drop tolerance.
        ilu_fill_factor: ILU fill factor.
        ilu_reuse_tol:   Diagonal change threshold.

    Returns:
        Solution vector.
    """
    try:
        t0 = time.perf_counter()
        A_csc = A.tocsc()
        ilu = _get_or_build_ilu(
            A_csc, reuse_ilu, ilu_key, ilu_drop_tol, ilu_fill_factor, ilu_reuse_tol, A
        )
        logger.info(f"ILU setup time={time.perf_counter() - t0:.3f}s")

        M = LinearOperator(A.shape, matvec=ilu.solve)  # type: ignore[call-arg]
        t0 = time.perf_counter()
        x = _iterative_solve_with_M(A, b, method, M, tol, maxiter, x0, failure_policy)
        logger.info(f"Iterative solver ({method}) time={time.perf_counter() - t0:.3f}s")
        return x
    except Exception as e:
        logger.warning(
            f"ILU preconditioner or iterative solver failed: {e}, trying plain iterative"
        )
        if isinstance(e, LinearSolveError):
            raise
        return _iterative_solve_plain(A, b, method, tol, maxiter, x0, failure_policy)


def _iterative_solve_plain(A, b, method, tol, maxiter, x0, failure_policy):
    """Solve with a plain iterative method (no preconditioner).

    Falls back to ``spsolve`` on convergence failure or exception.

    Args:
        A:       Sparse matrix.
        b:       Right-hand side.
        method:  ``"bicgstab"`` or ``"gmres"``.
        tol:     Relative tolerance.
        maxiter: Max iterations.
        x0:      Initial guess.

    Returns:
        Solution vector.
    """
    try:
        if method == "gmres":
            x, info = gmres(A, b, rtol=tol, maxiter=maxiter, x0=x0)
        else:
            x, info = bicgstab(A, b, rtol=tol, maxiter=maxiter, x0=x0)
        if info != 0:
            logger.warning(f"Plain iterative solver did not converge info={info}, falling back")
            if failure_policy == "raise":
                raise LinearSolveError(
                    f"Plain {method} did not converge after {maxiter} iterations (info={info})"
                )
            return spsolve(A, b)
        return x
    except Exception as e2:
        if failure_policy == "raise":
            if isinstance(e2, LinearSolveError):
                raise
            raise LinearSolveError(f"Plain {method} solve failed") from e2
        logger.error("Plain iterative solver failed, falling back to spsolve", exc_info=e2)
        return spsolve(A, b)


def solve_linear_system(
    A,
    b,
    method="spsolve",
    equation_type=None,
    tol=1e-6,
    maxiter=1000,
    x0=None,
    reuse_ilu=False,
    ilu_key=None,
    ilu_drop_tol=1e-4,
    ilu_fill_factor=10,
    ilu_reuse_tol=None,
    backend="scipy",
    parallel_context=None,
    return_info=False,
    failure_policy="direct_fallback",
    **kwargs,
):
    """Solve the linear system ``A·x = b``.

    Dispatches to the appropriate solver based on *method* and
    *equation_type*:

    - ``"spsolve"``: direct sparse solve (scipy).
    - ``equation_type="pressure"``: AMG (pyamg) with CG fallback.
    - ``equation_type="momentum"`` or ``"scalar"``: ILU-preconditioned
      iterative solver (BiCGSTAB / GMRES).
    - ``"cg"``, ``"gmres"``, ``"bicgstab"``: plain iterative solver.

    Supports initial guess ``x0``, ILU caching with diagonal-change
    rebuild heuristic, and AMG tuning via ``**kwargs``.

    Args:
        A:               Sparse matrix ``(n, n)``.
        b:               Right-hand side ``(n,)`` or ``(n, 1)``.
        method:          Solver method (default ``"spsolve"``).
        equation_type:   Optional hint for solver selection
                         (``"pressure"``, ``"momentum"``, ``"scalar"``).
        tol:             Relative residual tolerance for iterative solvers.
        maxiter:         Maximum number of iterations.
        x0:              Initial guess (optional, iterative solvers only).
        reuse_ilu:       Whether to cache and reuse the ILU factorisation.
        ilu_key:         User-defined ILU cache key.
        ilu_drop_tol:    ILU drop tolerance.
        ilu_fill_factor: ILU fill factor.
        ilu_reuse_tol:   Diagonal change threshold for ILU rebuild.
        **kwargs:        Additional arguments (e.g. ``amg_tol``, ``amg_maxiter``).

    Returns:
        Solution vector ``x``.

    Raises:
        ValueError: If *method* is not recognised.
    """
    failure_policy = str(failure_policy).lower()
    if failure_policy not in {"raise", "direct_fallback"}:
        raise ValueError(f"Unknown linear failure policy {failure_policy!r}")

    if str(backend).lower() == "petsc":
        solution, info = _solve_petsc(
            A,
            b,
            method,
            equation_type,
            tol,
            maxiter,
            x0,
            parallel_context,
        )
        return (solution, info) if return_info else solution
    if str(backend).lower() != "scipy":
        raise ValueError(f"Unknown linear backend {backend!r}")

    if return_info:
        raise ValueError("return_info is currently implemented for the PETSc backend only")

    if method == "spsolve":
        return spsolve(A, b)

    if equation_type == "pressure":
        amg_tol = kwargs.get("amg_tol", 4e-4)
        amg_maxiter = kwargs.get("amg_maxiter", maxiter)
        amg_reuse_tol = kwargs.get("amg_reuse_tol", 0.05)
        return _solve_pressure(
            A,
            b,
            amg_tol,
            amg_maxiter,
            tol,
            maxiter,
            x0,
            amg_reuse_tol,
            failure_policy,
        )

    if equation_type in ("momentum", "scalar") or method in ("bicgstab", "gmres"):
        return _solve_with_ilu(
            A,
            b,
            method,
            tol,
            maxiter,
            x0,
            reuse_ilu,
            ilu_key,
            ilu_drop_tol,
            ilu_fill_factor,
            ilu_reuse_tol,
            failure_policy,
        )

    # Generic method selector (no ILU)
    if method == "cg":
        x, info = cg(A, b, rtol=tol, maxiter=maxiter, x0=x0)
        if info != 0:
            logger.warning(f"CG did not converge, info={info}")
    elif method == "gmres":
        x, info = gmres(A, b, rtol=tol, maxiter=maxiter, x0=x0)
        if info != 0:
            logger.warning(f"GMRES did not converge, info={info}")
    elif method == "bicgstab":
        x, info = bicgstab(A, b, rtol=tol, maxiter=maxiter, x0=x0)
        if info != 0:
            logger.warning(f"BiCGSTAB did not converge, info={info}")
    else:
        raise ValueError(f"Unknown solver method: {method}")

    if info != 0:
        if failure_policy == "raise":
            raise LinearSolveError(
                f"{method} did not converge after {maxiter} iterations (info={info})"
            )
        return spsolve(A, b)

    return x
