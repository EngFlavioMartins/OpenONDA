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


def needs_pressure_reference(boundaries, n_elements):
    """
    Check if the pressure system needs a reference point constraint.

    For incompressible flow, the pressure Poisson equation is singular
    (defined up to a constant) unless enough Dirichlet boundary faces
    anchor the pressure level. This function detects cases where the
    Dirichlet constraint is too weak relative to the mesh size.

    OpenFOAM always applies pRefCell/pRefValue via setReference() for
    incompressible solvers. We replicate this: if the ratio of Dirichlet
    pressure faces to total cells is below a threshold, the pressure level
    is under-constrained and a reference must be applied.

    Args:
        boundaries: List of boundary patch dictionaries
        n_elements: Total number of mesh cells

    Returns:
        bool: True if pressure reference should be applied
    """
    # Count faces with Dirichlet pressure BC
    n_dirichlet_faces = 0
    for boundary in boundaries:
        bc_type_p = boundary.get("bc_type_p", "")
        bc_type = boundary.get("type", "")

        if bc_type_p in ["fixedValue", "totalPressure", "inletOutlet"] or bc_type in [
            "outlet",
            "fixedPressure",
            "totalPressure",
        ]:
            n_dirichlet_faces += boundary.get("nFaces", 0)

    # If no Dirichlet faces at all → closed cavity → always needs reference
    if n_dirichlet_faces == 0:
        return True

    # If the Dirichlet face ratio is below 5% of cells, the constraint
    # is too weak to prevent the near-null-space pressure mode from growing.
    # Example: 10 outlet faces for 2387 cells = 0.4% → needs reference.
    ratio = n_dirichlet_faces / max(n_elements, 1)
    return ratio < 0.05


def fix_pressure_reference(A, b, ref_element=0, ref_value=0.0):
    """
    Fix pressure at a reference element to avoid singular matrix.

    Modifies the pressure correction matrix to enforce:
    p'[ref_element] = ref_value

    Operates directly on CSR data to avoid expensive LIL conversion.

    Args:
        A: Pressure correction matrix (will be modified in-place)
        b: RHS vector (will be modified in-place)
        ref_element: Element index to fix (default: 0)
        ref_value: Reference pressure value (default: 0.0)

    Returns:
        tuple: (A_fixed, b_fixed) - Modified matrix and RHS
    """
    # Ensure CSR format (no-op if already CSR)
    A_csr = A.tocsr() if hasattr(A, 'tocsr') else A

    # Zero out the reference row by clearing its data entries
    row_start = A_csr.indptr[ref_element]
    row_end = A_csr.indptr[ref_element + 1]
    A_csr.data[row_start:row_end] = 0.0

    # Set diagonal to 1.0 — locate it via indptr range
    # In a CSR matrix, A[ref, ref] is at some position in the row.
    # We find it by scanning the column indices of the ref_element row.
    found = False
    for j in range(row_start, row_end):
        if A_csr.indices[j] == ref_element:
            A_csr.data[j] = 1.0
            found = True
            break
    if not found:
        # Diagonal entry missing (should not happen in FVM matrices)
        # Fall back to LIL modification
        A_lil = A_csr.tolil()
        A_lil[ref_element, :] = 0.0
        A_lil[ref_element, ref_element] = 1.0
        b[ref_element] = ref_value
        return A_lil.tocsr(), b

    b[ref_element] = ref_value
    return A_csr, b
