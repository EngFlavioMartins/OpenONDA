"""
Linear solver strategy pattern for the panel method.
==================
GPU and CPU linear solvers matching the VLMLinearSolver pattern.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import abc
import logging

import numpy as np
import taichi as ti

from ....config.constants import PANEL_EPSILON, TI_FLOAT

logger = logging.getLogger("vpm")


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
        return True


@ti.data_oriented
class PanelBiCGSTABSolver(PanelLinearSolver):
    """GPU-resident BiCGSTAB solver using Taichi."""

    def __init__(self, max_n_panels: int, dtype: ti.template()):
        # Guard: Ensure Taichi is initialized
        if ti.lang.impl.get_runtime().prog is None:
            raise RuntimeError("PanelBiCGSTABSolver must be created after ti.init()")
        self.r = ti.field(dtype, shape=max_n_panels)
        self.r_hat = ti.field(dtype, shape=max_n_panels)
        self.p = ti.field(dtype, shape=max_n_panels)
        self.v = ti.field(dtype, shape=max_n_panels)
        self.s = ti.field(dtype, shape=max_n_panels)
        self.t = ti.field(dtype, shape=max_n_panels)
        self.Ap = ti.field(dtype, shape=max_n_panels)
        self.As = ti.field(dtype, shape=max_n_panels)

    @ti.kernel
    def dot_kernel(self, a: ti.template(), b: ti.template(), n: int) -> TI_FLOAT:
        res = 0.0
        for i in range(n):
            res += a[i] * b[i]
        return res

    @ti.kernel
    def matmul_kernel(self, A: ti.template(), x: ti.template(), b: ti.template(), n: int):
        for i in range(n):
            acc = 0.0
            for j in range(n):
                acc += A[i, j] * x[j]
            b[i] = acc

    def _bicgstab_iterate(
        self,
        aerodynamic_influence_coefficient: ti.template(),
        x: ti.template(),
        n: int,
        tol: float,
        rho: float,
        alpha: float,
        w: float,
    ) -> tuple[float, float, float, bool]:
        """Run one BiCGSTAB iteration. Returns (rho, alpha, w, converged)."""
        rho_new = self.dot_kernel(self.r_hat, self.r, n)
        beta = (rho_new / rho) * (alpha / w)
        rho = rho_new
        for i in range(n):
            self.p[i] = self.r[i] + beta * (self.p[i] - w * self.v[i])
        self.matmul_kernel(aerodynamic_influence_coefficient, self.p, self.v, n)
        alpha = rho / self.dot_kernel(self.r_hat, self.v, n)
        for i in range(n):
            self.s[i] = self.r[i] - alpha * self.v[i]
        if np.sqrt(self.dot_kernel(self.s, self.s, n)) < tol:
            for i in range(n):
                x[i] += alpha * self.p[i]
            return rho, alpha, w, True
        self.matmul_kernel(aerodynamic_influence_coefficient, self.s, self.t, n)
        w = self.dot_kernel(self.t, self.s, n) / self.dot_kernel(self.t, self.t, n)
        for i in range(n):
            x[i] += alpha * self.p[i] + w * self.s[i]
        for i in range(n):
            self.r[i] = self.s[i] - w * self.t[i]
        return rho, alpha, w, np.sqrt(self.dot_kernel(self.r, self.r, n)) < tol

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
        self.matmul_kernel(aerodynamic_influence_coefficient, x, self.r, n)
        for i in range(n):
            self.r[i] = right_hand_side[i] - self.r[i]
        for i in range(n):
            self.r_hat[i] = self.r[i]
            self.p[i] = self.r[i]
        rho = 1.0
        alpha = 1.0
        w = 1.0
        for _it in range(max_iter):
            rho, alpha, w, converged = self._bicgstab_iterate(
                aerodynamic_influence_coefficient, x, n, tol, rho, alpha, w
            )
            if converged:
                return True
        logger.warning(f"GPU BiCGSTAB failed to converge in {max_iter} iterations.")
        return False
