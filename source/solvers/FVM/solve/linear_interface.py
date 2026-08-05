"""Sparse linear solvers and convergence telemetry."""

from dataclasses import dataclass, replace
import logging
import time
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, cg, gmres, spilu, spsolve

logger = logging.getLogger(__name__)
logger.propagate = False
_ILU_CACHE = {}
_AMG_CACHE = {}
_MAX_TRANSIENT_CACHE_ENTRIES = 16
_AMG_BUILD_SEED = 0
_FALLBACK_WARN_COUNT = 0


def _emit_warning(log_sink, message, *args) -> None:
    """Route a user-facing warning through the active FVM logger."""
    text = message % args if args else message
    if log_sink is None:
        logger.warning(text)
    else:
        log_sink.warning(text)


class LinearSolveError(RuntimeError):
    """Raised when a configured linear backend does not converge.

    Both the SciPy and PETSc backends raise this exception when the Krylov
    method fails to reach the requested tolerance, unless the
    ``direct_fallback`` failure policy has been set.
    """


@dataclass(frozen=True)
class LinearSolveResult:
    """Backend-neutral convergence record for a single sparse solve.

    Produced by :func:`solve_linear_system` and consumed by the solver's
    diagnostic pipeline and log output.  The frozen tuple fields make
    instances safe to store across time steps without accidental mutation.

    Attributes
    ----------
    backend : str
        Linear-algebra backend used (``"scipy"`` or ``"petsc"``).
    method : str
        Solver method (e.g. ``"bicgstab"``, ``"cg+gamg"``).
    preconditioner : str or None
        Preconditioner name, if one was applied.
    nullspace : str or None
        Null-space strategy (e.g. ``"constant"`` for pressure).
    converged : bool
        Whether the tolerance was met.
    reason : str
        Human-readable convergence or failure explanation.
    iterations : int
        Number of iterations performed (0 for direct solves).
    initial_residual : float
        Residual before the solve began.
    final_residual : float
        Residual after the solve completed.
    setup_seconds : float
        Wall time for preconditioner setup.
    solve_seconds : float
        Wall time for the iterative solve.
    used_fallback : bool
        Whether a direct fallback was invoked after iterative failure.
    preconditioner_rebuilt : bool or None
        Whether the preconditioner was rebuilt this call (``None`` for direct).
    """

    backend: str
    method: str
    preconditioner: str | None
    nullspace: str | None
    converged: bool
    reason: str
    iterations: int
    initial_residual: float
    final_residual: float
    setup_seconds: float
    solve_seconds: float
    used_fallback: bool = False
    preconditioner_rebuilt: bool | None = None
    equation: str | None = None


# Compatibility for callers that imported the experimental PETSc-only name.
LinearSolveInfo = LinearSolveResult


@runtime_checkable
class LinearSolver(Protocol):
    """Protocol for backend-neutral sparse linear solves.

    Any object that implements a ``solve`` method with the signature below
    satisfies the protocol and can be used as a drop-in replacement for the
    default SciPy or PETSc backends.

    Examples
    --------
    >>> class MySolver:
    ...     def solve(self, matrix, rhs, **options):
    ...         result = LinearSolveResult(...)
    ...         return solution, result
    """

    def solve(self, matrix, rhs, **_options) -> tuple[np.ndarray, LinearSolveResult]: ...


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


def deviation_norm_factor(A, b, x0):
    """OpenFOAM-style residual normalization: the deviation from the mean state.

    ``||b||`` carries the transport of whatever mean the solution rides on.
    For x-momentum in a unit free stream that bulk term is orders of magnitude
    larger than the near-wall dynamics, so any ``tol * ||b||`` convergence test
    is satisfied while the boundary layer is entirely unsolved -- with a warm
    initial guess the Krylov solve then exits at iteration zero and the flow
    freezes into an unphysical steady state (no separation, no shedding).

    This is the L2 analog of OpenFOAM's ``normFactor``: with
    ``xRef = mean(x0)``,

        normFactor = ||A x0 - A xRef|| + ||b - A xRef||

    which reduces exactly to ``||b||`` when ``x0`` is absent or zero-mean, and
    otherwise measures the part of the equation the solve is actually expected
    to resolve.
    """
    b = np.asarray(b, dtype=np.float64)
    if x0 is None:
        return max(float(np.linalg.norm(b)), 1e-30)
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.ndim == 1:
        mean_state = np.full(A.shape[0], float(x0.mean()))
    elif x0.ndim == 2:
        mean_state = np.broadcast_to(np.mean(x0, axis=0), x0.shape)
    else:
        raise ValueError("Linear initial guess must be one- or two-dimensional")
    reference = A @ mean_state
    factor = float(np.linalg.norm(A @ x0 - reference)) + float(np.linalg.norm(b - reference))
    return max(factor, 1e-30)


def openfoam_residual_target(A, b, x0, absolute_tolerance, relative_tolerance=0.0):
    """Return OpenFOAM-style initial residual, target, and norm factor.

    OpenFOAM stops a linear solve when either its absolute normalized
    tolerance is met or the residual has fallen by ``relTol`` from the value
    at entry to that solve.  ``relTol`` is therefore *not* itself an absolute
    residual target.  Keeping this conversion next to
    :func:`deviation_norm_factor` gives every serial/replicated backend the
    same semantics; the partitioned PETSc path performs the equivalent global
    calculation with distributed vectors.
    """
    b_array = np.asarray(b, dtype=np.float64)
    initial = np.zeros_like(b_array) if x0 is None else np.asarray(x0, dtype=np.float64)
    norm_factor = deviation_norm_factor(A, b_array, initial)
    initial_residual = float(np.linalg.norm(b_array - A @ initial) / norm_factor)
    target = max(float(absolute_tolerance), float(relative_tolerance) * initial_residual)
    return initial_residual, target, norm_factor


def _trivial_solution(A, b, x0, tol):
    """Residual-verified early exit: zero RHS or an already-converged ``x0``.

    Returns the verified solution, or ``None`` when an iterative solve is
    actually needed.  SciPy's Krylov methods report breakdown (``info=-10``)
    instead of convergence when started at (or extremely near) the solution,
    so skipping them here is a correctness fix as much as a speed one.
    """
    if float(np.linalg.norm(b)) == 0.0:
        return np.zeros(A.shape[0], dtype=np.float64)
    if x0 is not None:
        # Deviation-normalized test: ``tol * ||b||`` would accept a warm guess
        # whose entire error is the (mean-flow-dominated) small-scale physics.
        residual = float(np.linalg.norm(np.asarray(b) - A @ np.asarray(x0)))
        if residual <= tol * deviation_norm_factor(A, b, x0):
            return np.asarray(x0, dtype=np.float64)
    return None


def _breakdown_converged(A, b, x, tol, method, info):
    """True when a nonzero ``info`` iterate nevertheless meets the tolerance.

    SciPy signals breakdown (``info < 0``) when the recurrence scalars
    underflow, which typically happens *because* the residual already dropped
    below what the recurrences can resolve.  The algebraic residual is the
    ground truth, so accept the iterate when it verifies.
    """
    res = normalized_residual(A, x, b)
    if res <= tol:
        logger.info(
            "%s reported info=%s at converged residual %.3e <= %.1e; accepted",
            method,
            info,
            res,
            tol,
        )
        return True
    return False


def _run_krylov(A, b, method, M, tol, maxiter, x0):
    """Run a SciPy Krylov method in residual-correction form.

    Solves ``A e = r0 / ||r0||`` with ``r0 = b - A x0`` and returns
    ``x0 + ||r0|| e``.  The unit-normalized RHS keeps the recurrence scalars
    (rho, omega) away from underflow on tiny-scale systems — SciPy otherwise
    reports breakdown (``info=-10``) on RHS vectors that are pure assembly
    roundoff (~1e-19).  ``tol`` targets ``||b - A x|| <= tol * normFactor``
    with the deviation-based ``normFactor`` (see
    :func:`deviation_norm_factor`), so a mean-flow-inflated ``||b||`` cannot
    mask unresolved small-scale physics.
    """
    b_norm = float(np.linalg.norm(b))
    if b_norm == 0.0:
        return np.zeros(A.shape[0], dtype=np.float64), 0, 0
    if x0 is None:
        x0 = np.zeros(A.shape[0], dtype=np.float64)
        r0 = np.asarray(b, dtype=np.float64)
    else:
        x0 = np.asarray(x0, dtype=np.float64)
        r0 = np.asarray(b, dtype=np.float64) - A @ x0
    r0_norm = float(np.linalg.norm(r0))
    if r0_norm == 0.0:
        return x0.copy(), 0, 0
    rtol_eff = float(np.clip(tol * deviation_norm_factor(A, b, x0) / r0_norm, 1e-14, 0.99))
    iterations = 0

    def count_iteration(_value):
        nonlocal iterations
        iterations += 1

    kwargs = {"rtol": rtol_eff, "maxiter": maxiter, "callback": count_iteration}
    if M is not None:
        kwargs["M"] = M
    # Resolved at call time so tests can monkeypatch the module attributes.
    if method == "gmres":
        solve_fn = gmres
        kwargs["callback_type"] = "pr_norm"
    elif method == "cg":
        solve_fn = cg
    else:
        solve_fn = bicgstab
    e, info = solve_fn(A, r0 / r0_norm, **kwargs)
    return x0 + r0_norm * e, info, iterations


def _solve_petsc(
    A,
    b,
    method,
    equation_type,
    tol,
    rel_tol,
    maxiter,
    x0,
    parallel_context,
    nullspace,
):
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

    petsc_nullspace = None
    if nullspace is not None:
        if nullspace != "constant" or equation_type != "pressure":
            raise ValueError("PETSc currently supports only a constant pressure null space")
        petsc_nullspace = PETSc.NullSpace().create(constant=True, comm=PETSc.COMM_WORLD)
        mat.setNullSpace(petsc_nullspace)

    rhs = PETSc.Vec().createMPI(n, comm=PETSc.COMM_WORLD)
    rhs_start, rhs_end = rhs.getOwnershipRange()
    if rhs_end > rhs_start:
        rows = np.arange(rhs_start, rhs_end, dtype=PETSc.IntType)
        rhs.setValues(rows, b_array[rhs_start:rhs_end])
    rhs.assemblyBegin()
    rhs.assemblyEnd()
    if petsc_nullspace is not None:
        petsc_nullspace.remove(rhs)
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
    ksp_types = {
        "bicgstab": PETSc.KSP.Type.BCGS,
        "gmres": PETSc.KSP.Type.GMRES,
        "cg": PETSc.KSP.Type.CG,
    }
    if requested == "amg":
        if equation_type != "pressure":
            raise ValueError("PETSc AMG is supported only for pressure equations")
        ksp.setType(PETSc.KSP.Type.CG)
        pc.setType(PETSc.PC.Type.GAMG)
        method_name = "cg+gamg"
    elif requested in ksp_types:
        ksp.setType(ksp_types[requested])
        if nullspace == "constant":
            # A one-rank block-Jacobi block is the complete singular
            # Neumann matrix, so its local factorization fails. Point
            # Jacobi is null-space safe and behaves consistently across
            # communicator sizes.
            pc.setType(PETSc.PC.Type.JACOBI)
            method_name = f"{requested}+jacobi"
        else:
            pc.setType(PETSc.PC.Type.BJACOBI)
            method_name = f"{requested}+bjacobi"
    else:
        raise ValueError(f"Unknown PETSc iterative solver {method!r}")
    initial_residual, residual_target, norm_factor = openfoam_residual_target(
        A_csr, b_array, x0, tol, rel_tol
    )
    # PETSc's default test is relative to ||b||; rescale so the OpenFOAM
    # absolute-or-relative target is measured against the deviation norm.
    # deviation-based normFactor and a warm guess cannot satisfy the tolerance
    # on the strength of the mean flow alone (see deviation_norm_factor).
    b_norm = max(float(np.linalg.norm(b_array)), 1e-30)
    rtol_eff = float(np.clip(residual_target * norm_factor / b_norm, 1e-14, 0.99))
    ksp.setTolerances(rtol=rtol_eff, max_it=int(maxiter))
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
    if nullspace == "constant":
        x -= np.mean(x)
    final_residual = float(np.linalg.norm(b_array - A_csr @ x) / norm_factor)
    reason = str(ksp.getConvergedReason())
    converged = (
        reason_code > 0
        and np.isfinite(final_residual)
        and final_residual <= max(10.0 * residual_target, 1e-12)
    )
    info = LinearSolveResult(
        backend="petsc",
        method=method_name,
        preconditioner=str(ksp.getPC().getType()),
        nullspace=nullspace,
        converged=converged,
        reason=reason,
        iterations=iterations,
        initial_residual=initial_residual,
        final_residual=final_residual,
        setup_seconds=setup_seconds,
        solve_seconds=solve_seconds,
        used_fallback=False,
        preconditioner_rebuilt=True,
    )

    scatter.destroy()
    solution_all.destroy()
    ksp.destroy()
    solution.destroy()
    rhs.destroy()
    mat.destroy()
    if petsc_nullspace is not None:
        petsc_nullspace.destroy()

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

    A short *ilu_key* (e.g. the momentum component ``"x"``) namespaces the
    entry without hashing the pattern bytes every solve, but the matrix
    ``shape`` is always part of the key so a cache built for one mesh cannot
    be handed to a differently-sized system — the failure mode when two
    solvers of different resolution share a process (mesh-refinement study,
    test suite).  Without a key, the full structural pattern is hashed.

    Args:
        A_csc:   Matrix in CSC format.
        ilu_key: Optional user-defined key.

    Returns:
        A tuple usable as a dict key.
    """
    if ilu_key is not None:
        return ("key", ilu_key, A_csc.shape)
    return ("pattern", A_csc.shape, A_csc.indptr.tobytes(), A_csc.indices.tobytes())


def _amg_cache_key(A, amg_key=None):
    """Return a topology-only key for an AMG hierarchy."""
    if amg_key is not None:
        return ("key", amg_key, A.shape)
    csr = A.tocsr()
    return (csr.shape, csr.indptr.tobytes(), csr.indices.tobytes())


def clear_linear_solver_caches(cache_namespace=None) -> None:
    """Clear transient global preconditioner entries, optionally by namespace.

    Production solvers pass a short workspace namespace and call this from
    their lifecycle.  The bounded global fallback remains for standalone
    compatibility calls that have no owner.
    """
    if cache_namespace is None:
        _ILU_CACHE.clear()
        _AMG_CACHE.clear()
        return
    for cache in (_ILU_CACHE, _AMG_CACHE):
        stale = [key for key in cache if len(key) > 1 and key[1] == cache_namespace]
        for key in stale:
            cache.pop(key, None)


def _get_or_build_amg(A, pyamg, reuse_tol=0.05, force_rebuild=False, amg_key=None):
    """Build or reuse an AMG hierarchy as a preconditioner.

    Reusing the hierarchy directly as a solver would solve its stale level-0
    matrix.  Instead callers apply it as a preconditioner to CG operating on
    the current matrix, preserving the exact current linear system.
    """
    key = _amg_cache_key(A, amg_key)
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
        _rng_state = np.random.get_state()
        np.random.seed(_AMG_BUILD_SEED)
        try:
            hierarchy = pyamg.smoothed_aggregation_solver(A)
        finally:
            np.random.set_state(_rng_state)
        if len(_AMG_CACHE) >= _MAX_TRANSIENT_CACHE_ENTRIES and key not in _AMG_CACHE:
            _AMG_CACHE.pop(next(iter(_AMG_CACHE)))
        _AMG_CACHE[key] = (hierarchy, diagonal.copy())
        return hierarchy, True
    return cached[0], False


def _solve_pressure(
    A,
    b,
    amg_tol,
    amg_maxiter,
    tol,
    maxiter,
    x0,
    amg_reuse_tol,
    failure_policy,
    log_sink,
    amg_key,
):
    """Solve the pressure Poisson equation.

    Runs algebraic-multigrid-preconditioned CG via ``pyamg``. A failure
    raises unless ``failure_policy='direct_fallback'`` was requested.

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
    setup_start = time.perf_counter()
    x_early = _trivial_solution(A, b, x0, amg_tol)
    if x_early is not None:
        return x_early, {
            "preconditioner": "pyamg",
            "iterations": 0,
            "reason": "initial residual satisfied tolerance",
            "setup_seconds": time.perf_counter() - setup_start,
            "solve_seconds": 0.0,
            "used_fallback": False,
        }
    try:
        import pyamg
    except ImportError as error:
        if failure_policy == "raise":
            raise LinearSolveError("AMG pressure solve requires pyamg") from error
        _emit_warning(
            log_sink,
            "pyamg unavailable; using configured direct fallback",
        )
        setup_seconds = time.perf_counter() - setup_start
        solve_start = time.perf_counter()
        x = spsolve(A, b)
        return x, {
            "preconditioner": "pyamg",
            "iterations": 0,
            "reason": "pyamg unavailable; configured direct fallback converged",
            "setup_seconds": setup_seconds,
            "solve_seconds": time.perf_counter() - solve_start,
            "used_fallback": True,
        }

    try:
        ml, rebuilt = _get_or_build_amg(A, pyamg, reuse_tol=amg_reuse_tol, amg_key=amg_key)
        M = ml.aspreconditioner(cycle="V")
        setup_seconds = time.perf_counter() - setup_start
        solve_start = time.perf_counter()
        iterations = 0

        def count_iteration(_value):
            nonlocal iterations
            iterations += 1

        x, info = cg(A, b, M=M, rtol=amg_tol, maxiter=amg_maxiter, x0=x0, callback=count_iteration)
        if info != 0 and _breakdown_converged(A, b, x, amg_tol, "pressure CG", info):
            info = 0
        if info != 0:
            # One rebuild handles coefficient drift that made a cached
            # hierarchy ineffective without silently accepting a poor solve.
            ml, rebuilt = _get_or_build_amg(
                A, pyamg, reuse_tol=amg_reuse_tol, force_rebuild=True, amg_key=amg_key
            )
            M = ml.aspreconditioner(cycle="V")
            setup_seconds = time.perf_counter() - setup_start
            x, info = cg(
                A,
                b,
                M=M,
                rtol=amg_tol,
                maxiter=amg_maxiter,
                x0=x0,
                callback=count_iteration,
            )
            if info != 0 and _breakdown_converged(A, b, x, amg_tol, "pressure CG", info):
                info = 0
        if info != 0:
            raise RuntimeError(f"AMG-preconditioned pressure CG did not converge (info={info})")
        solve_seconds = time.perf_counter() - solve_start
        logger.info("pyamg pressure solve time=%.3fs", solve_seconds)
        return x, {
            "preconditioner": "pyamg",
            "iterations": iterations,
            "reason": "AMG-preconditioned CG converged",
            "setup_seconds": setup_seconds,
            "solve_seconds": solve_seconds,
            "used_fallback": False,
            "preconditioner_rebuilt": rebuilt,
        }
    except Exception as error:
        if failure_policy == "raise":
            raise LinearSolveError("AMG pressure solve failed") from error
        _emit_warning(
            log_sink,
            "AMG pressure solve failed; using configured direct fallback: %s",
            error,
        )
        fallback_start = time.perf_counter()
        failed_solve_seconds = fallback_start - locals().get("solve_start", fallback_start)
        x = spsolve(A, b)
        return x, {
            "preconditioner": "pyamg",
            "iterations": locals().get("iterations", 0),
            "reason": f"AMG failed; configured direct fallback converged: {error}",
            "setup_seconds": locals().get("setup_seconds", time.perf_counter() - setup_start),
            "solve_seconds": failed_solve_seconds + time.perf_counter() - fallback_start,
            "used_fallback": True,
        }


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
        return ilu, True

    key = _cache_key_from_matrix(A_csc, ilu_key)
    cached = _ILU_CACHE.get(key)
    if cached is None:
        ilu = spilu(A_csc, drop_tol=ilu_drop_tol, fill_factor=ilu_fill_factor)
        if len(_ILU_CACHE) >= _MAX_TRANSIENT_CACHE_ENTRIES:
            _ILU_CACHE.pop(next(iter(_ILU_CACHE)))
        _ILU_CACHE[key] = (ilu, A.diagonal().copy())
        logger.info("Computed and cached new ILU preconditioner")
        return ilu, True

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
            return ilu, True

    logger.info("Reusing ILU preconditioner (pattern key)")
    return ilu_cached, False


def _iterative_solve_with_M(
    A,
    b,
    method,
    M,
    tol,
    maxiter,
    x0,
    failure_policy,
    log_sink,
):
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
    x, info, iterations = _run_krylov(A, b, method, M, tol, maxiter, x0)
    if info != 0 and _breakdown_converged(A, b, x, tol, method, info):
        info = 0
    if info != 0:
        global _FALLBACK_WARN_COUNT
        _FALLBACK_WARN_COUNT += 1
        msg = f"{method} did not converge (info={info})"
        if _FALLBACK_WARN_COUNT <= 3 or _FALLBACK_WARN_COUNT % 50 == 0:
            _emit_warning(
                log_sink,
                "%s (occurrence #%d)",
                msg,
                _FALLBACK_WARN_COUNT,
            )
        if failure_policy == "raise":
            raise LinearSolveError(
                f"{method} did not converge after {maxiter} iterations (info={info})"
            )
        _emit_warning(
            log_sink,
            "Using configured direct fallback after %s failure",
            method,
        )
        return spsolve(A, b), max(iterations, int(info) if info > 0 else 0), True, msg
    return x, iterations, False, "Krylov solver converged"


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
    log_sink,
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
    setup_start = time.perf_counter()
    x_early = _trivial_solution(A, b, x0, tol)
    if x_early is not None:
        return x_early, {
            "preconditioner": "ilu",
            "iterations": 0,
            "reason": "initial residual satisfied tolerance",
            "setup_seconds": time.perf_counter() - setup_start,
            "solve_seconds": 0.0,
            "used_fallback": False,
        }
    try:
        A_csc = A.tocsc()
        ilu, rebuilt = _get_or_build_ilu(
            A_csc, reuse_ilu, ilu_key, ilu_drop_tol, ilu_fill_factor, ilu_reuse_tol, A
        )
        setup_seconds = time.perf_counter() - setup_start
        logger.info("ILU setup time=%.3fs", setup_seconds)

        M = LinearOperator(A.shape, matvec=ilu.solve)  # type: ignore[call-arg]
        solve_start = time.perf_counter()
        x, iterations, used_fallback, reason = _iterative_solve_with_M(
            A,
            b,
            method,
            M,
            tol,
            maxiter,
            x0,
            failure_policy,
            log_sink,
        )
        solve_seconds = time.perf_counter() - solve_start
        logger.info("Iterative solver (%s) time=%.3fs", method, solve_seconds)
        return x, {
            "preconditioner": "ilu",
            "iterations": iterations,
            "reason": reason,
            "setup_seconds": setup_seconds,
            "solve_seconds": solve_seconds,
            "used_fallback": used_fallback,
            "preconditioner_rebuilt": rebuilt,
        }
    except Exception as e:
        if isinstance(e, LinearSolveError):
            raise
        if failure_policy == "raise":
            raise LinearSolveError(f"{method} ILU setup or solve failed") from e
        _emit_warning(
            log_sink,
            "ILU setup failed; using configured direct fallback: %s",
            e,
        )
        setup_seconds = time.perf_counter() - setup_start
        solve_start = time.perf_counter()
        x = spsolve(A, b)
        return x, {
            "preconditioner": "ilu",
            "iterations": 0,
            "reason": f"ILU failed; configured direct fallback converged: {e}",
            "setup_seconds": setup_seconds,
            "solve_seconds": time.perf_counter() - solve_start,
            "used_fallback": True,
        }


def solve_linear_system(
    A,
    b,
    method="spsolve",
    equation_type=None,
    tol=1e-6,
    rel_tol=0.0,
    maxiter=1000,
    x0=None,
    reuse_ilu=False,
    ilu_key=None,
    ilu_drop_tol=1e-4,
    ilu_fill_factor=10,
    ilu_reuse_tol=None,
    backend="scipy",
    parallel_context=None,
    nullspace=None,
    partitioned_workspace=None,
    amg_key=None,
    return_info=False,
    failure_policy="raise",
    log_sink=None,
    matrix_values_unchanged=False,
    **kwargs,
):
    """Solve ``A·x = b`` using the explicitly selected serial or PETSc path.

    Cached ILU/AMG setup is controlled by the reuse arguments. A direct solve
    after iterative failure is permitted only with
    ``failure_policy="direct_fallback"`` and is reported in returned telemetry.
    """
    method = str(method).lower()
    failure_policy = str(failure_policy).lower()
    if failure_policy not in {"raise", "direct_fallback"}:
        raise ValueError(f"Unknown linear failure policy {failure_policy!r}")

    if str(backend).lower() == "petsc":
        if parallel_context is not None and parallel_context.is_partitioned:
            from .petsc_partitioned import solve_local_partitioned_system

            solution, info = solve_local_partitioned_system(
                A,
                b,
                parallel_context,
                method=method,
                tolerance=tol,
                relative_tolerance=rel_tol,
                max_iterations=maxiter,
                constant_nullspace=nullspace == "constant",
                initial_guess=x0,
                workspace=partitioned_workspace,
                matrix_values_unchanged=matrix_values_unchanged,
            )
        else:
            solution, info = _solve_petsc(
                A,
                b,
                method,
                equation_type,
                tol,
                rel_tol,
                maxiter,
                x0,
                parallel_context,
                nullspace,
            )
        if equation_type is not None and info.equation is None:
            info = replace(info, equation=str(equation_type))
        return (solution, info) if return_info else solution
    if str(backend).lower() != "scipy":
        raise ValueError(f"Unknown linear backend {backend!r}")
    if nullspace is not None:
        raise ValueError("Explicit null-space solves currently require the PETSc backend")

    initial_residual, residual_target, norm_factor = openfoam_residual_target(
        A, b, x0, tol, rel_tol
    )
    tol = residual_target

    def finish(solution, metadata):
        final_residual = float(np.linalg.norm(np.asarray(b) - A @ solution) / norm_factor)
        residual_limit = max(10.0 * tol, 1e-12)
        result = LinearSolveResult(
            backend="scipy",
            method=method,
            preconditioner=metadata.get("preconditioner"),
            nullspace=None,
            converged=bool(np.isfinite(final_residual) and final_residual <= residual_limit),
            reason=str(metadata["reason"]),
            iterations=int(metadata["iterations"]),
            initial_residual=initial_residual,
            final_residual=final_residual,
            setup_seconds=float(metadata["setup_seconds"]),
            solve_seconds=float(metadata["solve_seconds"]),
            used_fallback=bool(metadata.get("used_fallback", False)),
            preconditioner_rebuilt=metadata.get("preconditioner_rebuilt"),
            equation=None if equation_type is None else str(equation_type),
        )
        if not result.converged:
            raise LinearSolveError(
                f"SciPy {method} returned residual {final_residual:.3e}, "
                f"above the verified limit {residual_limit:.3e}"
            )
        return (solution, result) if return_info else solution

    if method == "spsolve":
        solve_start = time.perf_counter()
        solution = spsolve(A, b)
        return finish(
            solution,
            {
                "preconditioner": None,
                "iterations": 1,
                "reason": "sparse direct solve completed",
                "setup_seconds": 0.0,
                "solve_seconds": time.perf_counter() - solve_start,
                "used_fallback": False,
            },
        )

    if method == "amg":
        if equation_type != "pressure":
            raise ValueError("AMG is supported only for pressure equations")
        configured_amg_tol = kwargs.get("amg_tol", 4e-4)
        amg_tol = tol if configured_amg_tol is None else float(configured_amg_tol)
        amg_maxiter = kwargs.get("amg_maxiter", maxiter)
        amg_reuse_tol = kwargs.get("amg_reuse_tol", 0.05)
        solution, metadata = _solve_pressure(
            A,
            b,
            amg_tol,
            amg_maxiter,
            tol,
            maxiter,
            x0,
            amg_reuse_tol,
            failure_policy,
            log_sink,
            amg_key,
        )
        return finish(solution, metadata)

    if equation_type in ("momentum", "scalar") and method in ("bicgstab", "gmres", "cg"):
        solution, metadata = _solve_with_ilu(
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
            log_sink,
        )
        return finish(solution, metadata)

    if method in ("bicgstab", "gmres"):
        solution, metadata = _solve_with_ilu(
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
            log_sink,
        )
        return finish(solution, metadata)

    # Generic method selector (no ILU)
    if method not in {"cg", "gmres", "bicgstab"}:
        raise ValueError(f"Unknown solver method: {method}")
    solve_start = time.perf_counter()
    x, info, iterations = _run_krylov(A, b, method, None, tol, maxiter, x0)
    solve_seconds = time.perf_counter() - solve_start
    if info != 0:
        _emit_warning(log_sink, "%s did not converge, info=%s", method, info)

    if info != 0:
        if failure_policy == "raise":
            raise LinearSolveError(
                f"{method} did not converge after {maxiter} iterations (info={info})"
            )
        fallback_start = time.perf_counter()
        x = spsolve(A, b)
        return finish(
            x,
            {
                "preconditioner": None,
                "iterations": max(iterations, int(info) if info > 0 else 0),
                "reason": f"Krylov info={info}; configured direct fallback converged",
                "setup_seconds": 0.0,
                "solve_seconds": solve_seconds + time.perf_counter() - fallback_start,
                "used_fallback": True,
            },
        )

    return finish(
        x,
        {
            "preconditioner": None,
            "iterations": iterations,
            "reason": "Krylov solver converged",
            "setup_seconds": 0.0,
            "solve_seconds": solve_seconds,
            "used_fallback": False,
        },
    )
