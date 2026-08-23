"""
Linear-solver strategies for the VLM circulation system: SciPy dense, Taichi CG,
and Taichi BiCGSTAB backends.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import scipy.linalg

# Taichi imports
import taichi as ti

# Import constants from central config
from ....config.constants import (
    EPSILON,
    NP_FLOAT,
    TI_FLOAT,
)


def _lazy_import_taichi():
    """Return the Taichi module (imported at module load; kept as a single hook)."""
    return ti


# =========================================================
# Linear Solver Base Class
# =========================================================


class VLMLinearSolver(ABC):
    """Abstract base class for VLM linear solvers."""

    @abstractmethod
    def solve(
        self,
        aerodynamic_influence_coefficient,
        right_hand_side,
        circulation,
        n_panels: int,
        max_iterations: int = 1000,
        tolerance: float = EPSILON,
    ) -> int:
        """
        Solve the linear system aerodynamic_influence_coefficient @ circulation = right_hand_side.

        Args:
            aerodynamic_influence_coefficient: Aerodynamic Influence Coefficient matrix (Taichi field or numpy)
            right_hand_side: Right-hand side vector (Taichi field or numpy)
            circulation: Solution vector - MODIFIED IN PLACE (Taichi field or numpy)
            n_panels: Number of panels (active size of the system)
            max_iterations: Maximum iterations for iterative solvers
            tolerance: Convergence tolerance for iterative solvers

        Returns:
            Number of iterations (0 for direct solvers)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable solver name."""
        pass

    @property
    @abstractmethod
    def is_gpu(self) -> bool:
        """Whether this solver operates on GPU."""
        pass


# =========================================================
# Scipy Solver (CPU, Direct)
# =========================================================


class ScipySolver(VLMLinearSolver):
    """
    CPU-based direct solver using scipy.linalg.solve.

    Pros:
    - Very robust for all matrix types
    - Efficient for small systems (< 500 panels)
    - Handles near-singular matrices gracefully

    Cons:
    - Requires GPU→CPU→GPU data transfer (slow for large systems)
    - O(N³) direct solve can be expensive for large N
    """

    @property
    def name(self) -> str:
        return "SCIPY"

    @property
    def is_gpu(self) -> bool:
        return False

    def solve(
        self,
        aerodynamic_influence_coefficient,
        right_hand_side,
        circulation,
        n_panels: int,
        max_iterations: int = 1000,
        tolerance: float = EPSILON,
    ) -> int:
        # Determine numpy dtype from aerodynamic_influence_coefficient (Taichi field or numpy)
        dtype = NP_FLOAT
        if hasattr(aerodynamic_influence_coefficient, "dtype"):
            if aerodynamic_influence_coefficient.dtype == ti.f32:
                dtype = np.float32
            elif aerodynamic_influence_coefficient.dtype == ti.f64:
                dtype = np.float64
            else:
                dtype = aerodynamic_influence_coefficient.dtype

        # Extract numpy arrays from Taichi fields and cast to inferred dtype
        if hasattr(aerodynamic_influence_coefficient, "to_numpy"):
            AIC_np = aerodynamic_influence_coefficient.to_numpy()[:n_panels, :n_panels].astype(
                dtype
            )
        else:
            AIC_np = np.asarray(
                aerodynamic_influence_coefficient[:n_panels, :n_panels], dtype=dtype
            )

        if hasattr(right_hand_side, "to_numpy"):
            rhs_np = right_hand_side.to_numpy()[:n_panels].astype(dtype)
        else:
            rhs_np = np.asarray(right_hand_side[:n_panels], dtype=dtype)

        # Solve on CPU using inferred precision LAPACK.
        # A singular aerodynamic_influence_coefficient is a physics error (degenerate geometry or zero-velocity
        # freestream): raise rather than silently regularize, which would produce
        # physically meaningless γ and could mask upstream bugs.
        circulation_np = scipy.linalg.solve(AIC_np, rhs_np)

        # Write back to Taichi field or numpy array
        if hasattr(circulation, "from_numpy"):
            circulation_full = np.zeros(circulation.shape[0], dtype=dtype)
            circulation_full[:n_panels] = circulation_np
            circulation.from_numpy(circulation_full)
        else:
            circulation[:n_panels] = circulation_np

        return 0  # Direct solver, no iterations


# =========================================================
# Taichi Conjugate Gradient Solver (GPU, Iterative)
# =========================================================


class TaichiCGSolver(VLMLinearSolver):
    """
    GPU-based Conjugate Gradient solver using Taichi.

    Implements the classic CG algorithm entirely on GPU, avoiding ALL data transfers.
    The aerodynamic_influence_coefficient matrix, RHS, and solution (circulation) are all Taichi fields that stay on device.

    Features:
    - No GPU→CPU→GPU data transfer overhead
    - Highly parallel matrix-vector products
    - Memory efficient (only stores vectors, not factorized matrix)
    - Great for large systems (> 500 panels)
    - **Batched iterations**: Only checks convergence every `batch_size` iterations

    Cons:
    - Iterative, may not converge for ill-conditioned systems
    - Performance depends on number of iterations
    - Only works for symmetric positive-definite matrices (use BiCGSTAB for VLM)

    Algorithm:
        CG solves A @ x = b for symmetric positive-definite A.
        VLM aerodynamic_influence_coefficient matrices are NOT symmetric - use BiCGSTAB instead for VLM.
    """

    def __init__(self, max_n_panels: int = 10000, batch_size: int = 50):
        """
        Initialize CG solver with workspace allocation.

        Args:
            max_n_panels: Maximum number of panels (for workspace allocation)
            batch_size: Number of iterations between convergence checks (default: 50)
        """
        self.max_n_panels = max_n_panels
        self.batch_size = batch_size
        self._workspace_initialized = False

        # Workspace fields (allocated lazily)
        self.r = None  # Residual
        self.p = None  # Search direction
        self.Ap = None  # A @ p
        self.temp = None  # Temporary storage for A^T @ (A @ p)

    def _ensure_workspace(self, dtype=TI_FLOAT):
        """Lazily initialize workspace fields."""
        if self._workspace_initialized:
            # Check if existing workspace matches requested dtype
            if self.r.dtype != dtype:
                # Reallocate if dtype changed
                self._workspace_initialized = False
            else:
                return

        ti = _lazy_import_taichi()

        self.r = ti.field(dtype=dtype, shape=(self.max_n_panels,))
        self.p = ti.field(dtype=dtype, shape=(self.max_n_panels,))
        self.Ap = ti.field(dtype=dtype, shape=(self.max_n_panels,))
        self.temp = ti.field(dtype=dtype, shape=(self.max_n_panels,))

        self._workspace_initialized = True

    @property
    def name(self) -> str:
        return "CG_GPU"

    @property
    def is_gpu(self) -> bool:
        return True

    def solve(
        self,
        aerodynamic_influence_coefficient,
        right_hand_side,
        circulation,
        n_panels: int,
        max_iterations: int = 1000,
        tolerance: float = EPSILON,
    ) -> int:
        """
        Solve aerodynamic_influence_coefficient @ circulation = right_hand_side using batched Conjugate Gradient on GPU.

        Uses batched iteration strategy: convergence is only checked every
        `batch_size` iterations to reduce kernel launch overhead.

        All operations are performed on GPU using Taichi kernels.
        circulation is modified in-place.
        """
        raise RuntimeError(
            "TaichiCGSolver is only valid for symmetric positive-definite matrices. "
            "VLM aerodynamic_influence_coefficient matrices are non-symmetric — use TaichiBiCGSTABSolver instead."
        )


# =========================================================
# Taichi BiCGSTAB Solver (GPU, Iterative, Non-Symmetric)
# =========================================================


class TaichiBiCGSTABSolver(VLMLinearSolver):
    """
    GPU-based BiCGSTAB (Bi-Conjugate Gradient Stabilized) solver using Taichi.

    BiCGSTAB is designed for NON-SYMMETRIC matrices like the VLM aerodynamic_influence_coefficient matrix.
    Unlike standard CG, it converges for general square matrices.

    Features:
    - Works with non-symmetric VLM aerodynamic_influence_coefficient matrices
    - Optional Jacobi (diagonal) preconditioning for faster convergence
    - All operations on GPU (zero data transfer overhead)
    - **Batched iterations**: Only checks convergence every `batch_size` iterations
      to reduce kernel launch overhead and GPU-CPU synchronization

    Algorithm (Right-Preconditioned BiCGSTAB):
        Solves A @ x = b by transforming to A @ M^-1 @ y = b, then x = M^-1 @ y.
        This is more stable than left-preconditioning for non-symmetric matrices.

        Key insight: Instead of computing A @ p directly, we compute:
        1. p_hat = M^-1 @ p  (apply preconditioner to search direction)
        2. v = A @ p_hat     (matvec with preconditioned direction)

        This ensures the residual r = b - A @ x is computed correctly.

    References:
        van der Vorst, H. A. (1992). "Bi-CGSTAB: A Fast and Smoothly Converging
        Variant of Bi-CG for the Solution of Nonsymmetric Linear Systems"
    """

    def __init__(
        self, max_n_panels: int = 10000, use_preconditioner: bool = True, batch_size: int = 200
    ):
        """
        Initialize BiCGSTAB solver with workspace allocation.

        Args:
            max_n_panels: Maximum number of panels (for workspace allocation)
            use_preconditioner: Enable Jacobi (diagonal) preconditioning
            batch_size: Number of iterations between convergence checks (default: 50)
                        Higher values reduce overhead but may overshoot convergence.
        """
        self.max_n_panels = max_n_panels
        self.use_preconditioner = use_preconditioner
        self.batch_size = batch_size
        self._workspace_initialized = False

        # Workspace fields (allocated lazily)
        self.r = None  # Residual
        self.r0 = None  # Initial residual shadow
        self.p = None  # Search direction
        self.p_hat = None  # Preconditioned p: M^-1 @ p
        self.v = None  # A @ p_hat
        self.s = None  # Stabilization vector
        self.s_hat = None  # Preconditioned s: M^-1 @ s
        self.t = None  # A @ s_hat
        self.M_inv = None  # Preconditioner (diagonal)

    def _ensure_workspace(self, dtype=TI_FLOAT):
        """Lazily initialize workspace fields."""
        if self._workspace_initialized:
            # Check if existing workspace matches requested dtype
            if self.r.dtype != dtype:
                # Reallocate if dtype changed
                self._workspace_initialized = False
            else:
                return

        ti = _lazy_import_taichi()

        self.r = ti.field(dtype=dtype, shape=(self.max_n_panels,))
        self.r0 = ti.field(dtype=dtype, shape=(self.max_n_panels,))
        self.p = ti.field(dtype=dtype, shape=(self.max_n_panels,))
        self.p_hat = ti.field(dtype=dtype, shape=(self.max_n_panels,))
        self.v = ti.field(dtype=dtype, shape=(self.max_n_panels,))
        self.s = ti.field(dtype=dtype, shape=(self.max_n_panels,))
        self.s_hat = ti.field(dtype=dtype, shape=(self.max_n_panels,))
        self.t = ti.field(dtype=dtype, shape=(self.max_n_panels,))
        self.M_inv = ti.field(dtype=dtype, shape=(self.max_n_panels,))

        self._workspace_initialized = True

    @property
    def name(self) -> str:
        return "BICGSTAB_GPU"

    @property
    def is_gpu(self) -> bool:
        return True

    def _matvec_p_to_v(self, aerodynamic_influence_coefficient, n: int) -> None:
        """Compute v = A @ (M⁻¹ @ p) or A @ p depending on preconditioner."""
        if self.use_preconditioner:
            _apply_precond(self.p, self.p_hat, self.M_inv, n)
            _matvec(aerodynamic_influence_coefficient, self.p_hat, self.v, n)
        else:
            _matvec(aerodynamic_influence_coefficient, self.p, self.v, n)

    def _matvec_s_to_t(self, aerodynamic_influence_coefficient, n: int) -> None:
        """Compute t = A @ (M⁻¹ @ s) or A @ s depending on preconditioner."""
        if self.use_preconditioner:
            _apply_precond(self.s, self.s_hat, self.M_inv, n)
            _matvec(aerodynamic_influence_coefficient, self.s_hat, self.t, n)
        else:
            _matvec(aerodynamic_influence_coefficient, self.s, self.t, n)

    def _update_x_full(self, circulation, alpha: float, omega: float, n: int) -> None:
        """Update x += alpha*p̂ + omega*ŝ (or p/s without preconditioner)."""
        if self.use_preconditioner:
            _bicgstab_update_x(circulation, self.p_hat, self.s_hat, alpha, omega, n)
        else:
            _bicgstab_update_x(circulation, self.p, self.s, alpha, omega, n)

    def _update_x_partial(self, circulation, alpha: float, n: int) -> None:
        """Apply partial update x += alpha*p̂ (or p without preconditioner)."""
        if self.use_preconditioner:
            _axpy(circulation, self.p_hat, alpha, n)
        else:
            _axpy(circulation, self.p, alpha, n)

    def _solve_one_batch(
        self,
        aerodynamic_influence_coefficient,
        circulation,
        rho_iter: float,
        alpha: float,
        omega: float,
        iterations: int,
        batch_iters: int,
        n: int,
        dtype,
    ) -> tuple[float, float, float, int, int | None]:
        """
        Run *batch_iters* BiCGSTAB iterations.

        Returns ``(rho_iter, alpha, omega, iterations, early_ret)`` where
        *early_ret* is the value to return on algorithm breakdown,
        or ``None`` when the batch completed normally.
        """
        for _ in range(batch_iters):
            self._matvec_p_to_v(aerodynamic_influence_coefficient, n)
            r0v = _dot_product(self.r0, self.v, n, dtype=dtype)
            if abs(r0v) < EPSILON:
                return rho_iter, alpha, omega, iterations, iterations
            alpha = rho_iter / r0v
            _bicgstab_update_s(self.s, self.r, self.v, alpha, n)
            self._matvec_s_to_t(aerodynamic_influence_coefficient, n)
            ts = _dot_product(self.t, self.s, n, dtype=dtype)
            tt = _dot_product(self.t, self.t, n, dtype=dtype)
            if abs(tt) < EPSILON:
                self._update_x_partial(circulation, alpha, n)
                return rho_iter, alpha, omega, iterations, iterations
            omega = ts / tt
            self._update_x_full(circulation, alpha, omega, n)
            _bicgstab_update_r(self.r, self.s, self.t, omega, n)
            rho_next = _dot_product(self.r0, self.r, n, dtype=dtype)
            if abs(rho_next) < EPSILON:
                return rho_iter, alpha, omega, iterations, iterations
            beta = (rho_next / rho_iter) * (alpha / omega) if abs(omega) > EPSILON else 0.0
            _bicgstab_update_p(self.p, self.r, self.v, beta, omega, n)
            rho_iter = rho_next
            iterations += 1
        return rho_iter, alpha, omega, iterations, None

    def solve(
        self,
        aerodynamic_influence_coefficient,
        right_hand_side,
        circulation,
        n_panels: int,
        max_iterations: int = 1000,
        tolerance: float = EPSILON,
    ) -> int:
        """
        Solve aerodynamic_influence_coefficient @ circulation = right_hand_side using batched BiCGSTAB on GPU.

        Uses batched iteration strategy: convergence is only checked every
        `batch_size` iterations to reduce kernel launch overhead.

        All operations are performed on GPU using Taichi kernels.
        circulation is modified in-place.
        """
        _lazy_import_taichi()

        dtype = (
            aerodynamic_influence_coefficient.dtype
            if hasattr(aerodynamic_influence_coefficient, "dtype")
            else TI_FLOAT
        )
        self._ensure_workspace(dtype=dtype)
        n = n_panels
        tol_sq = tolerance * tolerance

        if self.use_preconditioner:
            _build_jacobi_precond(aerodynamic_influence_coefficient, self.M_inv, n)

        _bicgstab_init(circulation, right_hand_side, self.r, self.r0, self.p, n)
        rho_iter = _dot_product(self.r0, self.r, n, dtype=dtype)
        if abs(rho_iter) < EPSILON:
            return 0

        iterations = 0
        alpha = 0.0
        omega = 1.0
        num_outer = (max_iterations + self.batch_size - 1) // self.batch_size

        for _outer in range(num_outer):
            batch_iters = min(self.batch_size, max_iterations - iterations)
            rho_iter, alpha, omega, iterations, early_ret = self._solve_one_batch(
                aerodynamic_influence_coefficient,
                circulation,
                rho_iter,
                alpha,
                omega,
                iterations,
                batch_iters,
                n,
                dtype,
            )
            if early_ret is not None:
                return early_ret
            r_norm_sq = _dot_product(self.r, self.r, n, dtype=dtype)
            if r_norm_sq < tol_sq:
                break

        return iterations


# =========================================================
# Taichi Kernels for CG Solver
# =========================================================


@ti.kernel
def _cg_init_kernel(
    x: ti.template(), b: ti.template(), r: ti.template(), p: ti.template(), n: ti.i32
):
    for i in range(n):
        x[i] = 0.0
        r[i] = b[i]
        p[i] = b[i]


def _cg_init(x, b, r, p, n: int):
    """Initialize CG: x=0, r=b, p=b."""
    _lazy_import_taichi()
    _cg_init_kernel(x, b, r, p, n)


@ti.kernel
def _matvec_kernel(A: ti.template(), x: ti.template(), y: ti.template(), n: ti.i32):
    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc += A[i, j] * x[j]
        y[i] = acc


def _matvec(A, x, y, n: int):
    """Compute y = A @ x (matrix-vector product on GPU)."""
    _lazy_import_taichi()
    _matvec_kernel(A, x, y, n)


# Note: Taichi parallel loops require explicit reduction to avoid race conditions.
# We use ti.atomic_add for thread-safe accumulation into a scalar field.

# Persistent result field to avoid repeated allocation
_dot_result = None


@ti.kernel
def _dot_product_reset_kernel(out: ti.template()):
    """Reset the output scalar field to zero."""
    out[None] = 0.0


@ti.kernel
def _dot_product_kernel(a: ti.template(), b: ti.template(), n: ti.i32, out: ti.template()):
    """
    Compute dot product using thread-safe atomic accumulation.

    Each thread atomically adds its contribution to the output scalar.
    This is correct for any number of threads.
    """
    # Parallel accumulation with atomic adds
    for i in range(n):
        ti.atomic_add(out[None], a[i] * b[i])


def _dot_product(a, b, n: int, dtype=TI_FLOAT) -> float:
    """
    Compute dot product a^T @ b.

    For small vectors (n < 1000) the GPU atomic-reduction path suffers
    from extreme thread contention (all threads hammer the same scalar).
    The CPU fallback copies ~8 KiB of data and uses NumPy, which is
    10-50× faster for n < 1000.
    """
    if n < 1000:
        # Fast CPU path: negligible PCIe transfer, no kernel launch overhead
        a_np = a.to_numpy()[:n]
        b_np = b.to_numpy()[:n]
        return float(np.dot(a_np, b_np))

    # GPU atomic path (kept for very large systems where PCIe transfer
    # would dominate)
    global _dot_result
    ti = _lazy_import_taichi()

    if _dot_result is None or _dot_result.dtype != dtype:
        _dot_result = ti.field(dtype=dtype, shape=())

    _dot_product_reset_kernel(_dot_result)
    _dot_product_kernel(a, b, n, _dot_result)
    return _dot_result[None]


@ti.kernel
def _cg_update_xr_kernel(
    x: ti.template(),
    r: ti.template(),
    p: ti.template(),
    Ap: ti.template(),
    alpha: ti.template(),
    n: ti.i32,
):
    for i in range(n):
        x[i] += alpha * p[i]
        r[i] -= alpha * Ap[i]


def _cg_update_xr(x, r, p, Ap, alpha: float, n: int):
    """Update x and r in CG: x += alpha*p, r -= alpha*Ap."""
    _lazy_import_taichi()
    _cg_update_xr_kernel(x, r, p, Ap, alpha, n)


@ti.kernel
def _cg_update_p_kernel(p: ti.template(), r: ti.template(), beta: ti.template(), n: ti.i32):
    for i in range(n):
        p[i] = r[i] + beta * p[i]


def _cg_update_p(p, r, beta: float, n: int):
    """Update p in CG: p = r + beta*p."""
    _lazy_import_taichi()
    _cg_update_p_kernel(p, r, beta, n)


# =========================================================
# Taichi Kernels for BiCGSTAB Solver
# =========================================================


@ti.kernel
def _build_jacobi_precond_kernel(A: ti.template(), M_inv: ti.template(), n: ti.i32):
    for i in range(n):
        diag = A[i, i]
        if ti.abs(diag) > EPSILON:
            M_inv[i] = 1.0 / diag
        else:
            M_inv[i] = 1.0  # Fallback for zero diagonal


def _build_jacobi_precond(A, M_inv, n: int):
    """Build Jacobi preconditioner: M_inv[i] = 1/A[i,i]."""
    _lazy_import_taichi()
    _build_jacobi_precond_kernel(A, M_inv, n)


@ti.kernel
def _apply_precond_kernel(x: ti.template(), y: ti.template(), M_inv: ti.template(), n: ti.i32):
    """Apply Jacobi preconditioner: y[i] = M_inv[i] * x[i]."""
    for i in range(n):
        y[i] = M_inv[i] * x[i]


def _apply_precond(x, y, M_inv, n: int):
    """Apply preconditioner: y = M^-1 @ x (element-wise for Jacobi)."""
    _lazy_import_taichi()
    _apply_precond_kernel(x, y, M_inv, n)


@ti.kernel
def _matvec_precond_kernel(
    A: ti.template(), x: ti.template(), y: ti.template(), M_inv: ti.template(), n: ti.i32
):
    for i in range(n):
        acc = 0.0
        for j in range(n):
            acc += A[i, j] * x[j]
        y[i] = M_inv[i] * acc  # Apply preconditioner


def _matvec_precond(A, x, y, M_inv, n: int):
    """Compute y = M^-1 @ (A @ x) with Jacobi preconditioning."""
    _lazy_import_taichi()
    _matvec_precond_kernel(A, x, y, M_inv, n)


@ti.kernel
def _bicgstab_init_kernel(
    x: ti.template(),
    b: ti.template(),
    r: ti.template(),
    r0: ti.template(),
    p: ti.template(),
    n: ti.i32,
):
    for i in range(n):
        x[i] = 0.0
        r[i] = b[i]
        r0[i] = b[i]
        p[i] = b[i]


def _bicgstab_init(x, b, r, r0, p, n: int):
    """Initialize BiCGSTAB: x=0, r=b, r0=r, p=r."""
    _lazy_import_taichi()
    _bicgstab_init_kernel(x, b, r, r0, p, n)


@ti.kernel
def _bicgstab_update_s_kernel(
    s: ti.template(), r: ti.template(), v: ti.template(), alpha: ti.template(), n: ti.i32
):
    for i in range(n):
        s[i] = r[i] - alpha * v[i]


def _bicgstab_update_s(s, r, v, alpha: float, n: int):
    """Compute s = r - alpha * v."""
    _lazy_import_taichi()
    _bicgstab_update_s_kernel(s, r, v, alpha, n)


@ti.kernel
def _axpy_kernel(y: ti.template(), x: ti.template(), alpha: ti.template(), n: ti.i32):
    for i in range(n):
        y[i] += alpha * x[i]


def _axpy(y, x, alpha: float, n: int):
    """Compute y = y + alpha * x."""
    _lazy_import_taichi()
    _axpy_kernel(y, x, alpha, n)


@ti.kernel
def _bicgstab_update_x_kernel(
    x: ti.template(),
    p: ti.template(),
    s: ti.template(),
    alpha: ti.template(),
    omega: ti.template(),
    n: ti.i32,
):
    for i in range(n):
        x[i] += alpha * p[i] + omega * s[i]


def _bicgstab_update_x(x, p, s, alpha: float, omega: float, n: int):
    """Compute x = x + alpha * p + omega * s."""
    _lazy_import_taichi()
    _bicgstab_update_x_kernel(x, p, s, alpha, omega, n)


@ti.kernel
def _bicgstab_update_r_kernel(
    r: ti.template(), s: ti.template(), t: ti.template(), omega: ti.template(), n: ti.i32
):
    for i in range(n):
        r[i] = s[i] - omega * t[i]


def _bicgstab_update_r(r, s, t, omega: float, n: int):
    """Compute r = s - omega * t."""
    _lazy_import_taichi()
    _bicgstab_update_r_kernel(r, s, t, omega, n)


@ti.kernel
def _bicgstab_update_p_kernel(
    p: ti.template(),
    r: ti.template(),
    v: ti.template(),
    beta: ti.template(),
    omega: ti.template(),
    n: ti.i32,
):
    for i in range(n):
        p[i] = r[i] + beta * (p[i] - omega * v[i])


def _bicgstab_update_p(p, r, v, beta: float, omega: float, n: int):
    """Compute p = r + beta * (p - omega * v)."""
    _lazy_import_taichi()
    _bicgstab_update_p_kernel(p, r, v, beta, omega, n)


# =========================================================
# Solver Factory
# =========================================================

_SOLVER_REGISTRY = {
    "SCIPY": ScipySolver,
    "CG_GPU": TaichiCGSolver,
    "BICGSTAB_GPU": TaichiBiCGSTABSolver,
}


def get_linear_solver(
    solver_type: Literal["SCIPY", "CG_GPU", "BICGSTAB_GPU"] = "SCIPY",
    max_n_panels: int = 10000,
    use_preconditioner: bool = True,
    batch_size: int = 50,
) -> VLMLinearSolver:
    """
    Get a linear solver instance by type.

    Args:
        solver_type: Solver type ('SCIPY', 'CG_GPU', or 'BICGSTAB_GPU')
        max_n_panels: Maximum number of panels (for workspace allocation)
        use_preconditioner: Enable preconditioning for iterative solvers
        batch_size: Number of iterations between convergence checks for GPU solvers
                    (default: 50). Higher values reduce overhead but may overshoot.

    Returns:
        VLMLinearSolver instance

    Example:
        >>> solver = get_linear_solver('BICGSTAB_GPU')
        >>> solver.solve(aerodynamic_influence_coefficient, right_hand_side, circulation, n_panels)

    Recommended:
        - SCIPY: Small systems (<500 panels), most robust, fastest for small N
        - BICGSTAB_GPU: Large systems (>500 panels), non-symmetric VLM matrices
        - CG_GPU: Only for symmetric positive-definite matrices (not VLM)
    """
    solver_type = solver_type.upper()

    if solver_type not in _SOLVER_REGISTRY:
        available = ", ".join(_SOLVER_REGISTRY.keys())
        raise ValueError(f"Unknown solver type '{solver_type}'. Available: {available}")

    solver_class = _SOLVER_REGISTRY[solver_type]

    # GPU solvers need max_n_panels and batch_size for workspace
    if solver_type == "CG_GPU":
        return solver_class(max_n_panels=max_n_panels, batch_size=batch_size)
    elif solver_type == "BICGSTAB_GPU":
        return solver_class(
            max_n_panels=max_n_panels, use_preconditioner=use_preconditioner, batch_size=batch_size
        )
    else:
        return solver_class()


def list_available_solvers() -> list:
    """Return list of available solver types."""
    return list(_SOLVER_REGISTRY.keys())
