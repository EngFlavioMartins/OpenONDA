"""
Linear solver strategy pattern for the panel method.
==================
GPU and CPU linear solvers matching the VLMLinearSolver pattern.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import abc
from dataclasses import dataclass
import logging

import numpy as np
import taichi as ti

from ....config.constants import PANEL_EPSILON

logger = logging.getLogger("vpm")

# Relative-residual acceptance limit for a panel solve. A direct dense solve
# of a well-conditioned influence matrix lands many orders below this; the
# limit exists to reject rank-deficient or inconsistent systems, which a
# direct solver returns without raising.
DEFAULT_PANEL_RESIDUAL_TOLERANCE = 1.0e-8

# Below this magnitude a BiCGSTAB scalar is treated as a breakdown rather than
# divided by.
_BREAKDOWN_EPSILON = 1.0e-300


def default_residual_tolerance(float_dtype: str, *, constrained: bool = False) -> float:
    """Acceptance limit for a panel solve at the achievable working precision.

    A solve cannot drive the relative residual below the noise floor of the
    least precise arithmetic in the chain, so a fixed ``1e-8`` limit would
    reject every healthy single-precision solve. Two things set that floor:
    the panel field dtype.  The panel kernels seed accumulators from their
    typed panel operands, so VPM's global ``default_fp`` does not reduce an
    f64 panel solve.

    The returned limits still sit far below the order-one relative residual
    of a rank-deficient or inconsistent system, so fail-fast behaviour is
    preserved at either precision.
    """
    single_precision = float_dtype == "f32"
    # For a constrained least-squares source solve this threshold applies to
    # the dimensionless projected KKT optimality and relative flux errors,
    # *not* to ||A sigma - b|| / ||b||.  The latter can be non-zero at the
    # legitimate constrained optimum when finite-resolution collocation and
    # exact per-body flux compatibility are not perfectly compatible.
    if constrained:
        return 1.0e-4 if single_precision else 1.0e-10
    return 1.0e-5 if single_precision else DEFAULT_PANEL_RESIDUAL_TOLERANCE


def relative_residual(
    influence_matrix: np.ndarray, solution: np.ndarray, right_hand_side: np.ndarray
) -> float:
    """Return ``||A x - b|| / ||b||`` (absolute residual when ``b`` vanishes).

    Non-finite solutions return infinity so a caller's tolerance test
    rejects them without a separate finiteness branch.
    """
    if solution.size == 0:
        return 0.0
    if not np.all(np.isfinite(solution)):
        return float("inf")
    absolute = float(np.linalg.norm(influence_matrix @ solution - right_hand_side))
    scale = float(np.linalg.norm(right_hand_side))
    return absolute / scale if scale > 0.0 else absolute


def constrained_least_squares_metrics(
    influence_matrix: np.ndarray,
    right_hand_side: np.ndarray,
    constraints: np.ndarray,
    solution: np.ndarray,
) -> dict[str, float]:
    """Return equation, flux, and projected-KKT metrics for ``min ||Ax-b||``.

    The ordinary equation residual is a discretisation-quality diagnostic for
    equality-constrained least squares, not by itself a convergence criterion.
    At a valid optimum the flux constraint must hold and the residual gradient
    must have no component in the feasible/null-space directions:
    ``P A.T (A x - b) = 0``.

    All reductions are performed in f64 host arithmetic so these diagnostics
    faithfully measure an f32 panel solve as well as an f64 one.
    """
    matrix = np.asarray(influence_matrix, dtype=np.float64)
    rhs = np.asarray(right_hand_side, dtype=np.float64)
    constraint_matrix = np.asarray(constraints, dtype=np.float64)
    values = np.asarray(solution, dtype=np.float64)
    if not all(np.all(np.isfinite(value)) for value in (matrix, rhs, constraint_matrix, values)):
        return {
            "discrete_equation_residual": float("inf"),
            "discrete_equation_residual_absolute": float("inf"),
            "right_hand_side_norm": float("inf"),
            "constraint_residual": float("inf"),
            "relative_constraint_residual": float("inf"),
            "projected_optimality_residual": float("inf"),
            "projected_optimality_residual_absolute": float("inf"),
        }

    equation_error = matrix @ values - rhs
    equation_absolute = float(np.linalg.norm(equation_error))
    rhs_norm = float(np.linalg.norm(rhs))
    equation_relative = equation_absolute / rhs_norm if rhs_norm > 0.0 else equation_absolute

    if constraint_matrix.size:
        constraint_error = constraint_matrix @ values
        constraint_absolute = float(np.linalg.norm(constraint_error, ord=np.inf))
        constraint_scale = float(
            np.linalg.norm(constraint_matrix, ord=np.inf) * np.linalg.norm(values, ord=np.inf)
        )
        constraint_relative = (
            constraint_absolute / constraint_scale
            if constraint_scale > 0.0
            else constraint_absolute
        )
    else:
        constraint_absolute = 0.0
        constraint_relative = 0.0

    gradient = matrix.T @ equation_error
    projected_gradient = gradient
    if constraint_matrix.size:
        # P g = g - C.T (C C.T)^-1 C g.  The closed-body constraints occupy
        # disjoint panel ranges, so this is a tiny well-conditioned system.
        gram = constraint_matrix @ constraint_matrix.T
        try:
            multipliers = np.linalg.solve(gram, constraint_matrix @ gradient)
        except np.linalg.LinAlgError:
            multipliers, _, _, _ = np.linalg.lstsq(gram, constraint_matrix @ gradient, rcond=None)
        projected_gradient = gradient - constraint_matrix.T @ multipliers
    optimality_absolute = float(np.linalg.norm(projected_gradient))
    # Scaling by ||Ax-b|| is ill-conditioned when an otherwise compatible
    # system reaches machine residual: roundoff then makes a tiny projected
    # gradient look order-one.  Scale stationarity by the natural gradient
    # magnitude of the full problem instead, ||A|| (||A|| ||x|| + ||b||).
    matrix_norm = float(np.linalg.norm(matrix, ord="fro"))
    optimality_scale = matrix_norm * (matrix_norm * float(np.linalg.norm(values)) + rhs_norm)
    optimality_relative = (
        optimality_absolute / optimality_scale if optimality_scale > 0.0 else optimality_absolute
    )

    return {
        "discrete_equation_residual": equation_relative,
        "discrete_equation_residual_absolute": equation_absolute,
        "right_hand_side_norm": rhs_norm,
        "constraint_residual": constraint_absolute,
        "relative_constraint_residual": constraint_relative,
        "projected_optimality_residual": optimality_relative,
        "projected_optimality_residual_absolute": optimality_absolute,
    }


@dataclass
class EqualityConstrainedLeastSquaresFactorization:
    """Reusable null-space QR factorization for ``min ||A x-b||, Cx=0``.

    Geometry fixes both ``A`` and ``C`` while VPM coupling changes only the
    right-hand side.  Caching the orthonormal null space and pivoted QR of
    ``A @ Z`` therefore turns subsequent solves into two matrix-vector
    products and one triangular solve without changing the mathematics.
    """

    basis: np.ndarray
    q: np.ndarray
    r: np.ndarray
    pivots: np.ndarray
    rank: int
    reduced_operator: np.ndarray | None = None

    @classmethod
    def factorize(
        cls,
        influence_matrix: np.ndarray,
        constraints: np.ndarray,
    ) -> "EqualityConstrainedLeastSquaresFactorization":
        """Build the reusable null-space and reduced QR factors."""
        import scipy.linalg as la

        matrix = np.asarray(influence_matrix)
        constraint_matrix = np.asarray(constraints)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("influence_matrix must be square")
        if constraint_matrix.ndim != 2 or constraint_matrix.shape[1] != matrix.shape[1]:
            raise ValueError("constraints must have shape (n_constraints, n_unknowns)")

        basis = la.null_space(constraint_matrix)
        reduced_operator = matrix @ basis
        q, r, pivots = la.qr(
            reduced_operator,
            mode="economic",
            pivoting=True,
            overwrite_a=False,
            check_finite=False,
        )
        diagonal = np.abs(np.diag(r))
        leading = float(diagonal.max(initial=0.0))
        tolerance = (
            max(reduced_operator.shape) * np.finfo(r.dtype).eps * leading if leading > 0.0 else 0.0
        )
        rank = int(np.count_nonzero(diagonal > tolerance))
        return cls(
            basis=basis,
            q=q,
            r=r,
            pivots=np.asarray(pivots, dtype=np.int64),
            rank=rank,
            reduced_operator=reduced_operator if rank < reduced_operator.shape[1] else None,
        )

    def solve(self, right_hand_side: np.ndarray) -> np.ndarray:
        """Solve one right-hand side using the cached factorization."""
        import scipy.linalg as la

        rhs = np.asarray(right_hand_side)
        if rhs.ndim != 1 or rhs.shape[0] != self.q.shape[0]:
            raise ValueError("right_hand_side has incompatible shape")
        n_coordinates = self.basis.shape[1]
        if n_coordinates == 0:
            return np.zeros(self.basis.shape[0], dtype=rhs.dtype)
        if self.rank < n_coordinates:
            coordinates, _, _, _ = la.lstsq(
                self.reduced_operator,
                rhs,
                lapack_driver="gelsy",
                check_finite=False,
            )
        else:
            permuted = la.solve_triangular(
                self.r,
                self.q.T @ rhs,
                lower=False,
                check_finite=False,
            )
            coordinates = np.empty_like(permuted)
            coordinates[self.pivots] = permuted
        return self.basis @ coordinates

    @property
    def memory_bytes(self) -> int:
        """Host bytes retained by the reusable factorization."""
        arrays = (self.basis, self.q, self.r, self.pivots, self.reduced_operator)
        return int(sum(array.nbytes for array in arrays if array is not None))


def solve_equality_constrained_least_squares(
    influence_matrix: np.ndarray,
    right_hand_side: np.ndarray,
    constraints: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Solve ``min ||A x - b||`` subject to ``C x = 0``.

    The source-panel operator is not assumed symmetric, so a post-solve
    subtraction is not a valid constraint treatment.  We instead construct
    an orthonormal basis ``Z`` for the feasible null space and solve
    ``min ||A @ Z @ y - b||``.  This avoids the catastrophic cancellation
    that a rank-deficient projected operator can introduce.
    """
    if constraints.size == 0:
        try:
            import scipy.linalg as la

            solution = la.solve(influence_matrix, right_hand_side)
        except Exception:
            from scipy.linalg import lstsq

            solution, _, _, _ = lstsq(influence_matrix, right_hand_side)
        return solution, relative_residual(influence_matrix, solution, right_hand_side), 0.0

    factorization = EqualityConstrainedLeastSquaresFactorization.factorize(
        influence_matrix, constraints
    )
    solution = factorization.solve(right_hand_side)

    residual = relative_residual(influence_matrix, solution, right_hand_side)
    constraint_residual = float(np.linalg.norm(constraints @ solution, ord=np.inf))
    return solution, residual, constraint_residual


class PanelLinearSolver(abc.ABC):
    """Abstract base class for panel method linear solvers."""

    @abc.abstractmethod
    def solve(
        self,
        aerodynamic_influence_coefficient: ti.template(),
        right_hand_side: ti.template(),
        x: ti.template(),
        n: int,
    ) -> bool:
        pass


class PanelScipySolver(PanelLinearSolver):
    """CPU-side solver using scipy for dense direct solve / least-squares."""

    def __init__(self, residual_tolerance: float = DEFAULT_PANEL_RESIDUAL_TOLERANCE) -> None:
        self.residual_tolerance = residual_tolerance
        self.last_residual: float | None = None
        self.last_iterations: int | None = None

    def solve(
        self,
        aerodynamic_influence_coefficient: ti.template(),
        right_hand_side: ti.template(),
        x: ti.template(),
        n: int,
    ) -> bool:
        logger.debug(f"Solving {n}x{n} system on CPU using Scipy.")
        # GPU -> CPU
        A_np = aerodynamic_influence_coefficient.to_numpy()[:n, :n]
        b_np = right_hand_side.to_numpy()[:n]

        try:
            import scipy.linalg as la

            # Try direct solve first
            sol = la.solve(A_np, b_np)
        except Exception as e:
            logger.warning(
                f"Scipy solve failed: {e}. Retrying with regularization + least-squares fallback."
            )
            from scipy.linalg import lstsq

            try:
                A_reg = A_np + PANEL_EPSILON * np.eye(n, dtype=A_np.dtype)
                sol = la.solve(A_reg, b_np)
            except Exception:
                sol, _, _, _ = lstsq(A_np, b_np)

        # CPU -> GPU
        x_full = np.zeros(x.shape[0], dtype=A_np.dtype)
        x_full[:n] = sol
        x.from_numpy(x_full)
        # A direct dense solve returns a vector for a rank-deficient or
        # inconsistent system without raising, so success is decided by the
        # residual it actually achieved, never by reaching this line.
        self.last_residual = relative_residual(A_np, sol, b_np)
        self.last_iterations = None
        return self.last_residual <= self.residual_tolerance


@ti.data_oriented
class PanelBiCGSTABSolver(PanelLinearSolver):
    """GPU-resident iterative solvers using Taichi.

    ``solve`` retains the unconstrained BiCGSTAB implementation used by the
    standalone Dirichlet formulation.  ``solve_constrained_least_squares``
    uses projected CGLS for the Neumann source problem.  The latter solves
    ``min ||A x-b||`` while projecting every gradient and search direction
    into ``ker(C)``; it is therefore mathematically equivalent to the CPU
    null-space least-squares reference without forming a dense null-space
    basis.
    """

    def __init__(
        self,
        max_n_panels: int,
        dtype: ti.template(),
        residual_tolerance: float = DEFAULT_PANEL_RESIDUAL_TOLERANCE,
    ):
        # Guard: Ensure Taichi is initialized
        if ti.lang.impl.get_runtime().prog is None:
            raise RuntimeError("PanelBiCGSTABSolver must be created after ti.init()")
        self.residual_tolerance = residual_tolerance
        self.r = ti.field(dtype, shape=max_n_panels)
        self.r_hat = ti.field(dtype, shape=max_n_panels)
        self.p = ti.field(dtype, shape=max_n_panels)
        self.v = ti.field(dtype, shape=max_n_panels)
        self.s = ti.field(dtype, shape=max_n_panels)
        self.t = ti.field(dtype, shape=max_n_panels)
        self.Ap = ti.field(dtype, shape=max_n_panels)
        self.As = ti.field(dtype, shape=max_n_panels)
        self.constraint_value = ti.field(dtype, shape=max_n_panels)
        self.constraint_body = ti.field(ti.i32, shape=max_n_panels)
        self.constraint_sum = ti.field(ti.f64, shape=max_n_panels)
        self.constraint_gram = ti.field(ti.f64, shape=max_n_panels)
        self.last_residual: float | None = None
        self.last_iterations: int | None = None

    @ti.kernel
    def dot_kernel(self, a: ti.template(), b: ti.template(), n: int) -> ti.f64:
        # Reductions accumulate in f64 whatever the field dtype: a length-n
        # sum in f32 loses precision fast, and every BiCGSTAB scalar
        # (rho, alpha, omega) is derived from these dot products.
        res = ti.cast(0.0, ti.f64)
        for i in range(n):
            res += ti.cast(a[i], ti.f64) * ti.cast(b[i], ti.f64)
        return res

    @ti.kernel
    def matmul_kernel(self, A: ti.template(), x: ti.template(), b: ti.template(), n: int):
        for i in range(n):
            acc = A[i, 0] * 0.0
            for j in range(n):
                acc += A[i, j] * x[j]
            b[i] = acc

    @ti.kernel
    def matmul_transpose_kernel(self, A: ti.template(), x: ti.template(), b: ti.template(), n: int):
        for i in range(n):
            acc = A[0, i] * 0.0
            for j in range(n):
                acc += A[j, i] * x[j]
            b[i] = acc

    @ti.kernel
    def _matrix_norm_squared(self, A: ti.template(), n: int) -> ti.f64:
        total = ti.cast(0.0, ti.f64)
        for i, j in ti.ndrange(n, n):
            value = ti.cast(A[i, j], ti.f64)
            total += value * value
        return total

    @ti.kernel
    def _clear_constraint_sums(self, n_constraints: int):
        for body in range(n_constraints):
            self.constraint_sum[body] = 0.0

    @ti.kernel
    def _accumulate_constraint_sums(self, values: ti.template(), n: int):
        for i in range(n):
            body = self.constraint_body[i]
            if body >= 0:
                ti.atomic_add(
                    self.constraint_sum[body],
                    ti.cast(self.constraint_value[i] * values[i], ti.f64),
                )

    @ti.kernel
    def _apply_constraint_projection(self, values: ti.template(), n: int):
        for i in range(n):
            body = self.constraint_body[i]
            if body >= 0:
                correction = (
                    ti.cast(self.constraint_value[i], ti.f64)
                    * self.constraint_sum[body]
                    / self.constraint_gram[body]
                )
                values[i] -= correction

    def _configure_constraints(self, constraints: np.ndarray, n: int) -> int:
        """Upload disjoint per-body constraint rows for the O(N) projector."""
        matrix = np.asarray(constraints)
        if matrix.ndim != 2 or matrix.shape[1] != n:
            raise ValueError("constraints must have shape (n_constraints, n_unknowns)")
        n_constraints = matrix.shape[0]
        nonzero = np.abs(matrix) > 0.0
        if np.any(np.sum(nonzero, axis=0) > 1):
            raise ValueError("projected GPU solve requires disjoint constraint rows")

        body = np.full(n, -1, dtype=np.int32)
        value = np.zeros(n, dtype=matrix.dtype)
        columns = np.flatnonzero(np.any(nonzero, axis=0))
        if columns.size:
            body[columns] = np.argmax(nonzero[:, columns], axis=0).astype(np.int32)
            value[columns] = matrix[body[columns], columns]
        gram = np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64)
        if np.any(gram <= 0.0):
            raise ValueError("constraint rows must have non-zero norm")

        full_value = np.zeros(self.constraint_value.shape[0], dtype=matrix.dtype)
        full_body = np.full(self.constraint_body.shape[0], -1, dtype=np.int32)
        full_gram = np.ones(self.constraint_gram.shape[0], dtype=np.float64)
        full_value[:n] = value
        full_body[:n] = body
        full_gram[:n_constraints] = gram
        self.constraint_value.from_numpy(full_value)
        self.constraint_body.from_numpy(full_body)
        self.constraint_gram.from_numpy(full_gram)
        return n_constraints

    def _project(self, values: ti.template(), n: int, n_constraints: int) -> None:
        self._clear_constraint_sums(n_constraints)
        self._accumulate_constraint_sums(values, n)
        self._apply_constraint_projection(values, n)

    @ti.kernel
    def _initialize_cgls(self, right_hand_side: ti.template(), x: ti.template(), n: int):
        for i in range(n):
            x[i] = 0.0
            self.r[i] = right_hand_side[i]

    @ti.kernel
    def _copy_search_direction(self, n: int):
        for i in range(n):
            self.p[i] = self.s[i]

    @ti.kernel
    def _cgls_step(self, x: ti.template(), alpha: ti.f64, n: int):
        for i in range(n):
            x[i] += alpha * self.p[i]
            self.r[i] -= alpha * self.v[i]

    @ti.kernel
    def _update_cgls_direction(self, beta: ti.f64, n: int):
        for i in range(n):
            self.p[i] = self.s[i] + beta * self.p[i]

    @ti.kernel
    def _equation_error(
        self, matrix_times_solution: ti.template(), right_hand_side: ti.template(), n: int
    ):
        for i in range(n):
            self.r[i] = matrix_times_solution[i] - right_hand_side[i]

    def solve_constrained_least_squares(
        self,
        aerodynamic_influence_coefficient: ti.template(),
        right_hand_side: ti.template(),
        x: ti.template(),
        n: int,
        constraints: np.ndarray,
        relative_tolerance: float,
        max_iter: int | None = None,
    ) -> bool:
        """Solve ``min ||A x-b||, Cx=0`` with projected CGLS on device."""
        if n == 0:
            self.last_residual = 0.0
            self.last_iterations = 0
            return True
        n_constraints = self._configure_constraints(constraints, n)
        self._initialize_cgls(right_hand_side, x, n)
        self.matmul_transpose_kernel(aerodynamic_influence_coefficient, self.r, self.s, n)
        self._project(self.s, n, n_constraints)
        self._copy_search_direction(n)

        gamma = float(self.dot_kernel(self.s, self.s, n))
        initial_gradient_norm = float(np.sqrt(max(gamma, 0.0)))
        absolute_tolerance = relative_tolerance * max(initial_gradient_norm, 1.0)
        if initial_gradient_norm <= absolute_tolerance:
            self.last_residual = initial_gradient_norm / max(initial_gradient_norm, 1.0)
            self.last_iterations = 0
            return True

        iteration_limit = max_iter if max_iter is not None else max(50, min(4 * n, 4000))
        converged = False
        broke_down = False
        iterations = iteration_limit
        for iteration in range(iteration_limit):
            self.matmul_kernel(aerodynamic_influence_coefficient, self.p, self.v, n)
            denominator = float(self.dot_kernel(self.v, self.v, n))
            if not np.isfinite(denominator) or denominator <= _BREAKDOWN_EPSILON:
                broke_down = True
                iterations = iteration + 1
                break
            alpha = gamma / denominator
            self._cgls_step(x, alpha, n)

            self.matmul_transpose_kernel(aerodynamic_influence_coefficient, self.r, self.s, n)
            self._project(self.s, n, n_constraints)
            gamma_new = float(self.dot_kernel(self.s, self.s, n))
            gradient_norm = float(np.sqrt(max(gamma_new, 0.0)))
            if not np.isfinite(gradient_norm):
                broke_down = True
                iterations = iteration + 1
                break
            if gradient_norm <= absolute_tolerance:
                gamma = gamma_new
                converged = True
                iterations = iteration + 1
                break
            if gamma <= _BREAKDOWN_EPSILON:
                broke_down = True
                iterations = iteration + 1
                break
            beta = gamma_new / gamma
            gamma = gamma_new
            self._update_cgls_direction(beta, n)
            # Floating-point recurrences can slowly leak out of the feasible
            # subspace; this O(N) projection keeps the invariant explicit.
            self._project(self.p, n, n_constraints)

        self._project(x, n, n_constraints)
        final_gradient_norm = float(np.sqrt(max(gamma, 0.0)))
        self.last_residual = final_gradient_norm / max(initial_gradient_norm, 1.0)
        self.last_iterations = iterations
        if broke_down:
            logger.warning(
                "Projected GPU CGLS broke down after %d iterations at relative "
                "projected-gradient residual %.3e.",
                iterations,
                self.last_residual,
            )
            return False
        if not converged:
            logger.warning(
                "Projected GPU CGLS reached relative projected-gradient residual %.3e "
                "after %d iterations.",
                self.last_residual,
                iterations,
            )
        return converged

    def constrained_metrics(
        self,
        aerodynamic_influence_coefficient: ti.template(),
        right_hand_side: ti.template(),
        x: ti.template(),
        n: int,
        constraints: np.ndarray,
    ) -> dict[str, float]:
        """Compute the constrained KKT diagnostics without downloading ``A``."""
        n_constraints = self._configure_constraints(constraints, n)
        self.matmul_kernel(aerodynamic_influence_coefficient, x, self.v, n)
        self._equation_error(self.v, right_hand_side, n)
        equation_absolute = float(np.sqrt(max(self.dot_kernel(self.r, self.r, n), 0.0)))
        rhs_norm = float(np.sqrt(max(self.dot_kernel(right_hand_side, right_hand_side, n), 0.0)))
        equation_relative = equation_absolute / rhs_norm if rhs_norm > 0.0 else equation_absolute

        self.matmul_transpose_kernel(aerodynamic_influence_coefficient, self.r, self.s, n)
        self._project(self.s, n, n_constraints)
        optimality_absolute = float(np.sqrt(max(self.dot_kernel(self.s, self.s, n), 0.0)))
        matrix_norm = float(
            np.sqrt(max(self._matrix_norm_squared(aerodynamic_influence_coefficient, n), 0.0))
        )
        solution_norm = float(np.sqrt(max(self.dot_kernel(x, x, n), 0.0)))
        optimality_scale = matrix_norm * (matrix_norm * solution_norm + rhs_norm)
        optimality_relative = (
            optimality_absolute / optimality_scale
            if optimality_scale > 0.0
            else optimality_absolute
        )

        values = x.to_numpy()[:n].astype(np.float64, copy=False)
        constraint_matrix = np.asarray(constraints, dtype=np.float64)
        constraint_error = constraint_matrix @ values
        constraint_absolute = float(np.linalg.norm(constraint_error, ord=np.inf))
        constraint_scale = float(
            np.linalg.norm(constraint_matrix, ord=np.inf) * np.linalg.norm(values, ord=np.inf)
        )
        constraint_relative = (
            constraint_absolute / constraint_scale
            if constraint_scale > 0.0
            else constraint_absolute
        )
        return {
            "discrete_equation_residual": equation_relative,
            "discrete_equation_residual_absolute": equation_absolute,
            "right_hand_side_norm": rhs_norm,
            "constraint_residual": constraint_absolute,
            "relative_constraint_residual": constraint_relative,
            "projected_optimality_residual": optimality_relative,
            "projected_optimality_residual_absolute": optimality_absolute,
        }

    # The vector updates below are kernels rather than Python loops over field
    # elements. A Python-level ``for i in range(n): field[i] = ...`` issues one
    # host-device round trip per element per iteration, which dominates the
    # solve and defeats the point of a GPU-resident solver.

    @ti.kernel
    def _residual_from_solution(
        self, A: ti.template(), x: ti.template(), b: ti.template(), n: int
    ) -> ti.f64:
        """Return ``||A x - b||^2`` recomputed from the current solution."""
        total = ti.cast(0.0, ti.f64)
        for i in range(n):
            acc = ti.cast(0.0, ti.f64)
            for j in range(n):
                acc += ti.cast(A[i, j], ti.f64) * ti.cast(x[j], ti.f64)
            total += (acc - ti.cast(b[i], ti.f64)) ** 2
        return total

    @ti.kernel
    def _init_residual(self, b: ti.template(), n: int):
        """r = b - A x (with A x already staged in r); r_hat = p = r."""
        for i in range(n):
            residual = b[i] - self.r[i]
            self.r[i] = residual
            self.r_hat[i] = residual
            self.p[i] = residual

    @ti.kernel
    def _update_direction(self, beta: ti.f64, omega: ti.f64, n: int):
        """p = r + beta * (p - omega * v)"""
        for i in range(n):
            self.p[i] = self.r[i] + beta * (self.p[i] - omega * self.v[i])

    @ti.kernel
    def _update_s(self, alpha: ti.f64, n: int):
        """s = r - alpha * v"""
        for i in range(n):
            self.s[i] = self.r[i] - alpha * self.v[i]

    @ti.kernel
    def _axpy_p(self, x: ti.template(), alpha: ti.f64, n: int):
        """x += alpha * p"""
        for i in range(n):
            x[i] += alpha * self.p[i]

    @ti.kernel
    def _update_solution(self, x: ti.template(), alpha: ti.f64, omega: ti.f64, n: int):
        """x += alpha * p + omega * s"""
        for i in range(n):
            x[i] += alpha * self.p[i] + omega * self.s[i]

    @ti.kernel
    def _update_residual(self, omega: ti.f64, n: int):
        """r = s - omega * t"""
        for i in range(n):
            self.r[i] = self.s[i] - omega * self.t[i]

    def _bicgstab_iterate(
        self,
        aerodynamic_influence_coefficient: ti.template(),
        x: ti.template(),
        n: int,
        tol: float,
        rho: float,
        alpha: float,
        w: float,
    ) -> tuple[float, float, float, bool, bool]:
        """Run one BiCGSTAB iteration.

        Returns ``(rho, alpha, omega, converged, broke_down)``.
        """
        rho_new = self.dot_kernel(self.r_hat, self.r, n)
        # Classic BiCGSTAB breakdown: r_hat becomes orthogonal to r, or the
        # stabilizer collapses. Continuing would divide by zero and fill the
        # solution with NaNs, which the caller's residual test would then
        # report as a failure with no indication of the real cause.
        if abs(rho_new) < _BREAKDOWN_EPSILON or abs(rho) < _BREAKDOWN_EPSILON:
            return rho_new, alpha, w, False, True
        if abs(w) < _BREAKDOWN_EPSILON:
            return rho_new, alpha, w, False, True

        beta = (rho_new / rho) * (alpha / w)
        rho = rho_new
        self._update_direction(beta, w, n)
        self.matmul_kernel(aerodynamic_influence_coefficient, self.p, self.v, n)

        r_hat_dot_v = self.dot_kernel(self.r_hat, self.v, n)
        if abs(r_hat_dot_v) < _BREAKDOWN_EPSILON:
            return rho, alpha, w, False, True
        alpha = rho / r_hat_dot_v

        self._update_s(alpha, n)
        if np.sqrt(self.dot_kernel(self.s, self.s, n)) < tol:
            self._axpy_p(x, alpha, n)
            return rho, alpha, w, True, False

        self.matmul_kernel(aerodynamic_influence_coefficient, self.s, self.t, n)
        t_dot_t = self.dot_kernel(self.t, self.t, n)
        if abs(t_dot_t) < _BREAKDOWN_EPSILON:
            return rho, alpha, w, False, True
        w = self.dot_kernel(self.t, self.s, n) / t_dot_t

        self._update_solution(x, alpha, w, n)
        self._update_residual(w, n)
        return rho, alpha, w, np.sqrt(self.dot_kernel(self.r, self.r, n)) < tol, False

    def solve(
        self,
        aerodynamic_influence_coefficient: ti.template(),
        right_hand_side: ti.template(),
        x: ti.template(),
        n: int,
        tol: float = 1e-7,
        max_iter: int = 1000,
    ) -> bool:
        logger.debug(f"Solving {n}x{n} system on GPU using BiCGSTAB.")
        if n == 0:
            self.last_residual = 0.0
            self.last_iterations = 0
            return True

        self.matmul_kernel(aerodynamic_influence_coefficient, x, self.r, n)
        self._init_residual(right_hand_side, n)

        rho = 1.0
        alpha = 1.0
        w = 1.0
        iterations = max_iter
        broke_down = False
        for it in range(max_iter):
            rho, alpha, w, converged, broke_down = self._bicgstab_iterate(
                aerodynamic_influence_coefficient, x, n, tol, rho, alpha, w
            )
            if converged or broke_down:
                iterations = it + 1
                break

        # The recursively updated ``self.r`` drifts from the true residual over
        # many iterations, so acceptance is decided by recomputing it from the
        # final solution. The recompute stays on device: pulling the whole
        # n-by-n matrix back to the host every solve would cost more than the
        # solve itself.
        right_hand_side_norm_squared = self.dot_kernel(right_hand_side, right_hand_side, n)
        absolute = float(
            np.sqrt(
                max(
                    self._residual_from_solution(
                        aerodynamic_influence_coefficient, x, right_hand_side, n
                    ),
                    0.0,
                )
            )
        )
        scale = float(np.sqrt(max(right_hand_side_norm_squared, 0.0)))
        self.last_residual = absolute / scale if scale > 0.0 else absolute
        self.last_iterations = iterations

        if broke_down:
            logger.warning(
                f"GPU BiCGSTAB broke down after {iterations} iterations "
                f"at relative residual {self.last_residual:.3e}."
            )
            return False
        if self.last_residual <= self.residual_tolerance:
            return True
        logger.warning(
            f"GPU BiCGSTAB reached relative residual {self.last_residual:.3e} after "
            f"{iterations} iterations, above the {self.residual_tolerance:.3e} tolerance."
        )
        return False
