#!/usr/bin/env python3
"""
Linear solver interface for the FVM solver.

This module centralizes linear solver selection, ILU preconditioning and caching,
and provides a single `solve_linear_system` function that other modules call.

It is extracted from `matrix_assembly.py` to decouple matrix construction from solvers
and to make backend swapping (PETSc, GPU solvers) simpler in the future.
"""

import logging
import time

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, bicgstab, cg, gmres, spilu, spsolve

logger = logging.getLogger(__name__)
logger.propagate = False  # Don't send warnings to root logger (stderr)
_ILU_CACHE = {}
_PYAMG_WARNING_SHOWN = False  # Track if pyamg warning already displayed
_FALLBACK_WARN_COUNT = 0  # Track iterative solver fallback warnings


def _cache_key_from_matrix(A_csc, ilu_key=None):
    if ilu_key is not None:
        return ("key", ilu_key)
    # Basic structural key: shape + indptr + indices
    return ("pattern", A_csc.shape, A_csc.indptr.tobytes(), A_csc.indices.tobytes())


def _solve_pressure(A, b, amg_tol, amg_maxiter, tol, maxiter, x0):
    """Solve pressure Poisson equation, preferring AMG then CG fallback."""
    try:
        import pyamg

        t0 = time.perf_counter()
        ml = pyamg.smoothed_aggregation_solver(A)
        try:
            res = ml.solve(b, tol=amg_tol, maxiter=amg_maxiter, return_residuals=True)
        except TypeError:
            res = ml.solve(b, tol=amg_tol, maxiter=amg_maxiter)
        x = res[0] if isinstance(res, tuple) else res
        logger.info(f"pyamg pressure solve time={time.perf_counter() - t0:.3f}s")
        return x
    except Exception:
        global _PYAMG_WARNING_SHOWN
        if not _PYAMG_WARNING_SHOWN:
            print("[INFO] pyamg not available, using CG with diagonal preconditioner for pressure")
            _PYAMG_WARNING_SHOWN = True
    return _cg_pressure_fallback(A, b, tol, maxiter, x0)


def _cg_pressure_fallback(A, b, tol, maxiter, x0):
    """CG solve with diagonal preconditioner as pressure fallback."""
    try:
        t0 = time.perf_counter()
        M_inv = diags(1.0 / (A.diagonal() + 1e-16))
        x, info = cg(A, b, M=M_inv, rtol=tol, maxiter=maxiter, x0=x0)
        logger.info(f"CG pressure fallback time={time.perf_counter() - t0:.3f}s")
        if info != 0:
            logger.warning(f"CG (pressure fallback) did not converge, info={info}")
        return x
    except Exception as e2:
        logger.error("Pressure solver fallback failed", exc_info=e2)
        return spsolve(A, b)


def _get_or_build_ilu(A_csc, reuse_ilu, ilu_key, ilu_drop_tol, ilu_fill_factor, ilu_reuse_tol, A):
    """Return an ILU preconditioner, using cache when reuse_ilu is set."""
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


def _iterative_solve_with_M(A, b, method, M, tol, maxiter, x0):
    """Run bicgstab or gmres with preconditioner M, returning solution vector."""
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
        return spsolve(A, b)
    return x


def _solve_with_ilu(
    A, b, method, tol, maxiter, x0, reuse_ilu, ilu_key, ilu_drop_tol, ilu_fill_factor, ilu_reuse_tol
):
    """Solve using ILU preconditioner with iterative method; fall back to spsolve on failure."""
    try:
        t0 = time.perf_counter()
        A_csc = A.tocsc()
        ilu = _get_or_build_ilu(
            A_csc, reuse_ilu, ilu_key, ilu_drop_tol, ilu_fill_factor, ilu_reuse_tol, A
        )
        logger.info(f"ILU setup time={time.perf_counter() - t0:.3f}s")

        M = LinearOperator(A.shape, matvec=ilu.solve)  # type: ignore[call-arg]
        t0 = time.perf_counter()
        x = _iterative_solve_with_M(A, b, method, M, tol, maxiter, x0)
        logger.info(f"Iterative solver ({method}) time={time.perf_counter() - t0:.3f}s")
        return x
    except Exception as e:
        logger.warning(
            f"ILU preconditioner or iterative solver failed: {e}, trying plain iterative"
        )
        return _iterative_solve_plain(A, b, method, tol, maxiter, x0)


def _iterative_solve_plain(A, b, method, tol, maxiter, x0):
    """Plain iterative solve without preconditioner; fall back to spsolve."""
    try:
        if method == "gmres":
            x, info = gmres(A, b, rtol=tol, maxiter=maxiter, x0=x0)
        else:
            x, info = bicgstab(A, b, rtol=tol, maxiter=maxiter, x0=x0)
        if info != 0:
            logger.warning(f"Plain iterative solver did not converge info={info}, falling back")
            return spsolve(A, b)
        return x
    except Exception as e2:
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
    **kwargs,
):
    """
    Solve linear system A * x = b using SciPy solvers or PyAMG for pressure.

    New features:
    - Accepts `x0` as initial guess for iterative solvers.
    - ILU cache with optional rebuild heuristic based on diagonal change (ilu_reuse_tol).

    Args: (keeps same semantics as previous implementation, with new args)
    """
    if method == "spsolve":
        return spsolve(A, b)

    if equation_type == "pressure":
        amg_tol = kwargs.get("amg_tol", 4e-4)
        amg_maxiter = kwargs.get("amg_maxiter", maxiter)
        return _solve_pressure(A, b, amg_tol, amg_maxiter, tol, maxiter, x0)

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

    return x
