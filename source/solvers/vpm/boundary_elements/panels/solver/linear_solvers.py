"""
Linear solver strategy pattern for the panel method.
==================
GPU and CPU linear solvers matching the VLMLinearSolver pattern.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import abc
import contextlib
import logging

import numpy as np
import taichi as ti

from ....config.constants import PANEL_EPSILON, TI_FLOAT

logger = logging.getLogger("vpm")

# Relative-residual acceptance limit for a panel solve. A direct dense solve
# of a well-conditioned influence matrix lands many orders below this; the
# limit exists to reject rank-deficient or inconsistent systems, which a
# direct solver returns without raising.
DEFAULT_PANEL_RESIDUAL_TOLERANCE = 1.0e-8

# Below this magnitude a BiCGSTAB scalar is treated as a breakdown rather than
# divided by.
_BREAKDOWN_EPSILON = 1.0e-300


def default_residual_tolerance(float_dtype: str) -> float:
    """Acceptance limit for a panel solve at the achievable working precision.

    A solve cannot drive the relative residual below the noise floor of the
    least precise arithmetic in the chain, so a fixed ``1e-8`` limit would
    reject every healthy single-precision solve. Two things set that floor:
    the panel field dtype, and Taichi's ``default_fp`` — the influence
    kernels build their intermediates from untyped literals, so an ``f32``
    ``default_fp`` caps accuracy near ``1e-7`` even when the fields
    themselves are ``f64``. The looser of the two governs.

    The returned limits still sit far below the order-one relative residual
    of a rank-deficient or inconsistent system, so fail-fast behaviour is
    preserved at either precision.
    """
    single_precision = float_dtype == "f32"
    # Taichi may not be initialized yet; in that case the field dtype governs.
    with contextlib.suppress(Exception):
        single_precision = single_precision or ti.lang.impl.get_runtime().default_fp == ti.f32
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
    """GPU-resident BiCGSTAB solver using Taichi."""

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
            acc = 0.0
            for j in range(n):
                acc += A[i, j] * x[j]
            b[i] = acc

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
    def _update_direction(self, beta: TI_FLOAT, omega: TI_FLOAT, n: int):
        """p = r + beta * (p - omega * v)"""
        for i in range(n):
            self.p[i] = self.r[i] + beta * (self.p[i] - omega * self.v[i])

    @ti.kernel
    def _update_s(self, alpha: TI_FLOAT, n: int):
        """s = r - alpha * v"""
        for i in range(n):
            self.s[i] = self.r[i] - alpha * self.v[i]

    @ti.kernel
    def _axpy_p(self, x: ti.template(), alpha: TI_FLOAT, n: int):
        """x += alpha * p"""
        for i in range(n):
            x[i] += alpha * self.p[i]

    @ti.kernel
    def _update_solution(self, x: ti.template(), alpha: TI_FLOAT, omega: TI_FLOAT, n: int):
        """x += alpha * p + omega * s"""
        for i in range(n):
            x[i] += alpha * self.p[i] + omega * self.s[i]

    @ti.kernel
    def _update_residual(self, omega: TI_FLOAT, n: int):
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
