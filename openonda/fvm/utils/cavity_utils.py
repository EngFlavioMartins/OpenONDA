#!/usr/bin/env python3
"""
Cavity Flow Utilities for OpenONDA FVM Solver

Handles special cases for closed cavity flows where pressure
needs to be fixed at a reference point.

Based on uFVM cavity handling logic.
"""

import numpy as np
from scipy import sparse


def fix_pressure_reference(A: sparse.spmatrix, b: np.ndarray) -> tuple[sparse.spmatrix, np.ndarray]:
    """
    Fix pressure in cell 0 to avoid a singular all-Neumann system.

    Returns a copy of the pressure-correction system that enforces
    ``p'[0] = 0``. The row and column are both constrained, so a symmetric
    pressure matrix remains symmetric, and the sparsity pattern is kept so
    cached preconditioners stay valid.

    Args:
        A: Pressure correction matrix; not modified.
        b: Right-hand side; not modified.

    Returns:
        tuple: (A_fixed, b_fixed) - Modified matrix and RHS
    """
    # Pure-CSR row/column zeroing: this runs on EVERY pressure solve of an
    # all-Neumann (velocity-Dirichlet) case, so it must be O(nnz) with no
    # LIL round-trip.  Values are zeroed in place of the existing pattern,
    # which leaves the sparsity structure (and any cached ILU/AMG keyed on
    # it) unchanged.  The zero reference value needs no right-hand-side
    # column correction.
    A_csr = A.tocsr(copy=True)
    b_fixed = b.copy()

    # Zero the column, then the row, then place the unit diagonal.
    A_csr.data[A_csr.indices == 0] = 0.0
    row_start, row_end = A_csr.indptr[0], A_csr.indptr[1]
    A_csr.data[row_start:row_end] = 0.0
    diagonal = np.nonzero(A_csr.indices[row_start:row_end] == 0)[0][0]
    A_csr.data[row_start + diagonal] = 1.0
    b_fixed[0] = 0.0
    return A_csr, b_fixed
