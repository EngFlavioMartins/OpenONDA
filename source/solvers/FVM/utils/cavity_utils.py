#!/usr/bin/env python3
"""
Cavity Flow Utilities for OpenONDA FVM Solver

Handles special cases for closed cavity flows where pressure
needs to be fixed at a reference point.

Based on uFVM cavity handling logic.
"""

import numpy as np


def is_closed_cavity(boundaries):
    """
    Check if the case is a closed cavity (no outlet with fixed pressure).

    A closed cavity has all walls with no pressure outlet, requiring
    pressure to be fixed at a reference point.

    Args:
        boundaries: List of boundary patch dictionaries

    Returns:
        bool: True if closed cavity, False otherwise
    """

    # Check if any boundary has fixed pressure (outlet)
    for boundary in boundaries:
        bc_type = boundary.get("type", "")

        # Common outlet types that fix pressure
        if bc_type in ["outlet", "fixedPressure", "totalPressure"]:
            return False

        # Check if it's a pressure boundary
        if "pressure" in bc_type.lower():
            return False

        # Check bc_type_p
        bc_type_p = boundary.get("bc_type_p", "")
        if bc_type_p in ["fixedValue", "totalPressure", "inletOutlet"]:
            return False

    # No pressure outlet found - this is a closed cavity
    return True


def needs_pressure_reference(boundaries, n_elements=None):
    """
    Check if the pressure system needs a reference point constraint.

    For incompressible flow, the pressure Poisson equation is singular
    (defined up to a constant) only when no pressure boundary fixes its
    absolute level.  One Dirichlet face is sufficient to remove the constant
    null space; adding an interior reference in that case over-constrains the
    pressure correction.

    Args:
        boundaries: List of boundary patch dictionaries
        n_elements: Deprecated and ignored; kept for API compatibility.

    Returns:
        bool: True if pressure reference should be applied
    """
    for boundary in boundaries:
        bc_type_p = boundary.get("bc_type_p", "")
        bc_type = boundary.get("type", "")

        if bc_type_p in ["fixedValue", "totalPressure", "inletOutlet"] or bc_type in [
            "outlet",
            "fixedPressure",
            "totalPressure",
        ]:
            return False
    return True


def fix_pressure_reference(A, b, ref_element=0, ref_value=0.0):
    """
    Fix pressure at a reference element to avoid singular matrix.

    Modifies the pressure correction matrix to enforce:
    p'[ref_element] = ref_value

    The row and column are both constrained so a symmetric pressure matrix
    remains symmetric.  This operation is rare (closed/all-Neumann domains),
    so clarity and correctness are preferred over an in-place CSR shortcut.

    Args:
        A: Pressure correction matrix (will be modified in-place)
        b: RHS vector (will be modified in-place)
        ref_element: Element index to fix (default: 0)
        ref_value: Reference pressure value (default: 0.0)

    Returns:
        tuple: (A_fixed, b_fixed) - Modified matrix and RHS
    """
    n = A.shape[0]
    if not 0 <= ref_element < n:
        raise IndexError(f"Pressure reference cell {ref_element} outside [0, {n})")

    # Pure-CSR row/column zeroing: this runs on EVERY pressure solve of an
    # all-Neumann (velocity-Dirichlet) case, so it must be O(nnz) with no
    # LIL round-trip.  Values are zeroed in place of the existing pattern,
    # which leaves the sparsity structure (and any cached ILU/AMG keyed on
    # it) unchanged.
    A_csr = A.tocsr(copy=True)
    b_fixed = b.copy()

    col_mask = A_csr.indices == ref_element
    b_fixed -= _column_times_value(A_csr, col_mask, ref_value, n)

    # Zero the column, then the row, then place the unit diagonal.
    A_csr.data[col_mask] = 0.0
    row_start, row_end = A_csr.indptr[ref_element], A_csr.indptr[ref_element + 1]
    A_csr.data[row_start:row_end] = 0.0
    diag_hits = np.nonzero(A_csr.indices[row_start:row_end] == ref_element)[0]
    if diag_hits.size:
        A_csr.data[row_start + diag_hits[0]] = 1.0
    else:  # diagonal absent from the pattern (never for FV assemblies)
        A_csr[ref_element, ref_element] = 1.0
    b_fixed[ref_element] = ref_value
    return A_csr, b_fixed


def _column_times_value(A_csr, col_mask, ref_value, n):
    """Return ``A[:, ref] * ref_value`` as a dense vector (O(nnz))."""
    if ref_value == 0.0:
        return 0.0
    rows = np.repeat(np.arange(n), np.diff(A_csr.indptr))[col_mask]
    out = np.zeros(n, dtype=A_csr.data.dtype)
    np.add.at(out, rows, A_csr.data[col_mask] * ref_value)
    return out
