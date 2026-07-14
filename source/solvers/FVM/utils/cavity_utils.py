#!/usr/bin/env python3
"""
Cavity Flow Utilities for OpenONDA FVM Solver

Handles special cases for closed cavity flows where pressure
needs to be fixed at a reference point.

Based on uFVM cavity handling logic.
"""


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

    A_csr = A.tocsr(copy=True)
    b_fixed = b.copy()
    column = A_csr.getcol(ref_element).toarray().ravel()
    b_fixed -= column * ref_value

    A_lil = A_csr.tolil()
    A_lil[ref_element, :] = 0.0
    A_lil[:, ref_element] = 0.0
    A_lil[ref_element, ref_element] = 1.0
    b_fixed[ref_element] = ref_value
    return A_lil.tocsr(), b_fixed
