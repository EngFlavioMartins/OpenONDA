"""Pressure reference for a flow with no fixed-pressure boundary, such as a closed cavity.

The pressure is then defined only up to a constant; fixing it in one cell makes the
pressure-correction system non-singular. Follows uFVM.
"""

import numpy as np
from scipy import sparse


def fix_pressure_reference(
    A: sparse.csr_matrix, b: np.ndarray
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Fix pressure in cell 0 to avoid a singular all-Neumann system.

    Returns a copy of ``A`` that enforces ``p'[0] = 0`` and sets ``b[0] = 0`` in
    place. The row and column are both constrained, so a symmetric pressure
    matrix remains symmetric, and the sparsity pattern is kept so cached
    preconditioners stay valid. ``A`` itself is not modified: it is the shared
    assembly workspace matrix, which the next momentum assembly overwrites,
    while the cached AMG hierarchy keeps the matrix it was built from as its
    finest level.

    Apply it once per assembled matrix. A reused matrix is already constrained
    and needs only ``b[0] = 0``.

    Args:
        A: Pressure correction matrix; not modified.
        b: Right-hand side; ``b[0]`` is set to zero in place.

    Returns:
        tuple: (A_fixed, b) - the constrained matrix copy and the same ``b``
    """
    # Pure-CSR row/column zeroing, O(nnz) with no LIL round-trip.  Values are
    # zeroed in place of the existing pattern, which leaves the sparsity
    # structure (and any cached ILU/AMG keyed on it) unchanged.  The zero
    # reference value needs no right-hand-side column correction.
    A_csr = A.tocsr(copy=True)

    # Zero the column, then the row, then place the unit diagonal.
    A_csr.data[A_csr.indices == 0] = 0.0
    row_start, row_end = A_csr.indptr[0], A_csr.indptr[1]
    A_csr.data[row_start:row_end] = 0.0
    diagonal = np.nonzero(A_csr.indices[row_start:row_end] == 0)[0][0]
    A_csr.data[row_start + diagonal] = 1.0
    b[0] = 0.0
    return A_csr, b
