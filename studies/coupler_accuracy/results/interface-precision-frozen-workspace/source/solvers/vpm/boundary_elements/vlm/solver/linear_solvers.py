"""
Linear-solver strategies for the VLM circulation system: SciPy dense, Taichi CG,
and Taichi BiCGSTAB backends.

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: January 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

from abc import ABC, abstractmethod
from typing import Literal
import warnings

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

    def __init__(self):
        """Initialize an empty matrix/LU cache, reused only for identical systems."""
        self._matrix = None
        self._factorization = None

    @property
    def name(self) -> str:
        """Return the stable registry name ``"SCIPY"``."""
        return "SCIPY"

    @property
    def is_gpu(self) -> bool:
        """Return ``False`` because the solve runs on the CPU."""
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
        """Solve the active dense VLM system with SciPy on the CPU.

        Parameters
        ----------
        aerodynamic_influence_coefficient : array-like or Taichi field, shape (N, N)
            Dense influence matrix for the active panels.  Its entries map
            panel circulation to collocation-point normal velocity.
        right_hand_side : array-like or Taichi field, shape (N,)
            Boundary-condition vector for the active panels.
        circulation : ndarray or Taichi field, shape (capacity,)
            Solution storage.  The active prefix ``[:N]`` is overwritten in
            place; a Taichi field remains device-resident after the upload.
        n_panels : int
            Active system size ``N``; unused capacity is ignored.
        max_iterations : int, default=1000
            Accepted for interface compatibility and ignored by this direct
            solver.
        tolerance : float, default=EPSILON
            Accepted for interface compatibility and ignored by SciPy's direct
            factorization.

        Returns
        -------
        int
            ``0`` because no iterative steps are performed.

        Raises
        ------
        scipy.linalg.LinAlgError
            If the active influence matrix is singular or ill-conditioned
            enough for the selected LAPACK solve to fail.
        """
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
        if self._matrix is None or not np.array_equal(AIC_np, self._matrix):
            self._matrix = AIC_np.copy()
            with warnings.catch_warnings():
                warnings.simplefilter("error", scipy.linalg.LinAlgWarning)
                self._factorization = scipy.linalg.lu_factor(AIC_np)
        circulation_np = scipy.linalg.lu_solve(self._factorization, rhs_np)
        if not np.all(np.isfinite(circulation_np)):
            raise np.linalg.LinAlgError("VLM circulation solve produced non-finite values")

        # Write back to Taichi field or numpy array
        if hasattr(circulation, "from_numpy"):
            circulation_full = np.zeros(circulation.shape[0], dtype=dtype)
            circulation_full[:n_panels] = circulation_np
            circulation.from_numpy(circulation_full)
        else:
            circulation[:n_panels] = circulation_np

        return 0  # Direct solver, no iterations


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
    - Checks convergence during iteration and verifies the true final residual

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

    def __init__(self, max_n_panels: int = 10000, use_preconditioner: bool = True):
        """
        Initialize BiCGSTAB solver with workspace allocation.

        Args:
            max_n_panels: Maximum number of panels (for workspace allocation)
            use_preconditioner: Enable Jacobi (diagonal) preconditioning
        """
        self.max_n_panels = max_n_panels
        self.use_preconditioner = use_preconditioner
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
        """Return the stable registry name ``"BICGSTAB_GPU"``."""
        return "BICGSTAB_GPU"

    @property
    def is_gpu(self) -> bool:
        """Return ``True`` because the linear algebra runs in Taichi."""
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

    def solve(
        self,
        aerodynamic_influence_coefficient,
        right_hand_side,
        circulation,
        n_panels: int,
        max_iterations: int = 1000,
        tolerance: float = EPSILON,
    ) -> int:
        """Solve with right-preconditioned BiCGSTAB and verify the true residual.

        ``tolerance`` is relative to the RHS norm. Breakdown, non-finite output,
        and exhausted iterations cannot silently publish an unconverged lattice.
        """
        dtype = aerodynamic_influence_coefficient.dtype
        self._ensure_workspace(dtype=dtype)
        n = n_panels
        if self.use_preconditioner:
            _build_jacobi_precond(aerodynamic_influence_coefficient, self.M_inv, n)
        _bicgstab_init(circulation, right_hand_side, self.r, self.r0, self.p, n)
        norm_sq = _dot_product(self.r, self.r, n, dtype=dtype)
        if norm_sq == 0.0:
            return 0
        target_sq = tolerance * tolerance * norm_sq
        rho = norm_sq
        iterations = 0
        for iteration in range(max_iterations):
            self._matvec_p_to_v(aerodynamic_influence_coefficient, n)
            denominator = _dot_product(self.r0, self.v, n, dtype=dtype)
            if denominator == 0.0 or not np.isfinite(denominator):
                break
            alpha = rho / denominator
            _bicgstab_update_s(self.s, self.r, self.v, alpha, n)
            if _dot_product(self.s, self.s, n, dtype=dtype) <= target_sq:
                self._update_x_partial(circulation, alpha, n)
                iterations = iteration + 1
                break
            self._matvec_s_to_t(aerodynamic_influence_coefficient, n)
            tt = _dot_product(self.t, self.t, n, dtype=dtype)
            if tt == 0.0 or not np.isfinite(tt):
                break
            omega = _dot_product(self.t, self.s, n, dtype=dtype) / tt
            self._update_x_full(circulation, alpha, omega, n)
            _bicgstab_update_r(self.r, self.s, self.t, omega, n)
            iterations = iteration + 1
            if _dot_product(self.r, self.r, n, dtype=dtype) <= target_sq:
                break
            rho_next = _dot_product(self.r0, self.r, n, dtype=dtype)
            if rho_next == 0.0 or omega == 0.0 or not np.isfinite(rho_next):
                break
            beta = (rho_next / rho) * (alpha / omega)
            _bicgstab_update_p(self.p, self.r, self.v, beta, omega, n)
            rho = rho_next

        matrix = aerodynamic_influence_coefficient.to_numpy()[:n, :n].astype(np.float64)
        rhs = right_hand_side.to_numpy()[:n].astype(np.float64)
        solution = circulation.to_numpy()[:n].astype(np.float64)
        relative_residual = np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs)
        if not np.isfinite(relative_residual) or relative_residual > tolerance:
            raise np.linalg.LinAlgError(
                f"VLM BiCGSTAB did not converge after {iterations} iterations: "
                f"relative residual={relative_residual:.3g}, tolerance={tolerance:.3g}"
            )
        return iterations


# =========================================================
# Shared iterative linear algebra kernels
# =========================================================


@ti.kernel
def _matvec_kernel(A: ti.template(), x: ti.template(), y: ti.template(), n: ti.i32):
    """Write the active dense matrix-vector product y = A x on the device."""
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
_dot_runtime = None


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
    global _dot_result, _dot_runtime
    ti = _lazy_import_taichi()

    runtime = ti.lang.impl.get_runtime().prog
    if _dot_result is None or _dot_result.dtype != dtype or _dot_runtime is not runtime:
        _dot_result = ti.field(dtype=dtype, shape=())
        _dot_runtime = runtime

    _dot_product_reset_kernel(_dot_result)
    _dot_product_kernel(a, b, n, _dot_result)
    return _dot_result[None]


@ti.kernel
def _build_jacobi_precond_kernel(A: ti.template(), M_inv: ti.template(), n: ti.i32):
    """Write inverse diagonals; use unity where the diagonal is numerically zero."""
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
def _bicgstab_init_kernel(
    x: ti.template(),
    b: ti.template(),
    r: ti.template(),
    r0: ti.template(),
    p: ti.template(),
    n: ti.i32,
):
    """Initialize a zero solution and identical residual, shadow residual and direction."""
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
    """Write the intermediate BiCGSTAB residual s = r - alpha v."""
    for i in range(n):
        s[i] = r[i] - alpha * v[i]


def _bicgstab_update_s(s, r, v, alpha: float, n: int):
    """Compute s = r - alpha * v."""
    _lazy_import_taichi()
    _bicgstab_update_s_kernel(s, r, v, alpha, n)


@ti.kernel
def _axpy_kernel(y: ti.template(), x: ti.template(), alpha: ti.template(), n: ti.i32):
    """Accumulate alpha x into y over the active prefix."""
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
    """Accumulate both preconditioned BiCGSTAB search directions into x."""
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
    """Write the final BiCGSTAB residual r = s - omega t."""
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
    """Update the BiCGSTAB search direction with the shadow-residual coefficient."""
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
    "BICGSTAB_GPU": TaichiBiCGSTABSolver,
}


def get_linear_solver(
    solver_type: Literal["SCIPY", "BICGSTAB_GPU"] = "SCIPY",
    max_n_panels: int = 10000,
    use_preconditioner: bool = True,
) -> VLMLinearSolver:
    """Construct a VLM linear-solver strategy by registry name.

    Parameters
    ----------
    solver_type : {"SCIPY", "BICGSTAB_GPU"}, default="SCIPY"
        Case-insensitive backend name.  SciPy is a CPU direct solve;
        BiCGSTAB is the GPU backend for non-symmetric VLM matrices.
    max_n_panels : int, default=10000
        Workspace capacity for GPU backends.
    use_preconditioner : bool, default=True
        Enable Jacobi preconditioning for BiCGSTAB.

    Returns
    -------
    VLMLinearSolver
        Newly constructed solver strategy.

    Raises
    ------
    ValueError
        If ``solver_type`` is not registered.

    Examples
    --------
    >>> solver = get_linear_solver("SCIPY")
    >>> solver.name
    'SCIPY'
    """
    solver_type = solver_type.upper()

    if solver_type not in _SOLVER_REGISTRY:
        available = ", ".join(_SOLVER_REGISTRY.keys())
        raise ValueError(f"Unknown solver type '{solver_type}'. Available: {available}")

    solver_class = _SOLVER_REGISTRY[solver_type]

    if solver_type == "BICGSTAB_GPU":
        return solver_class(max_n_panels=max_n_panels, use_preconditioner=use_preconditioner)
    return solver_class()


def list_available_solvers() -> list[str]:
    """Return the registered VLM solver names in factory order."""
    return list(_SOLVER_REGISTRY.keys())
