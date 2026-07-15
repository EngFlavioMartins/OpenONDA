#!/usr/bin/env python3
"""
Matrix Assembly for OpenONDA FVM Solver

Converts flux coefficients into sparse matrix format for linear system solution.

Implements: A * phi = b
where A is the coefficient matrix and b is the RHS vector.

Converted from uFVM cfdAssembleIntoGlobalMatrixFaceFluxes.m
"""

import logging

import numpy as np
from scipy.sparse import lil_matrix

# Module logger and ILU cache
logger = logging.getLogger(__name__)
_ILU_CACHE = {}

# Cache for matrix sparsity patterns (keyed by mesh topology signature)
_SPATIAL_CACHE: dict = {}


def _sparsity_key(mesh_data):
    """Hashable key from mesh topology (owners + neighbours + n_elements)."""
    cached = mesh_data.get("_fvm_sparsity_key")
    if cached is not None:
        return cached
    n = mesh_data["n_elements"]
    n_i = mesh_data["n_interior_faces"]
    own = mesh_data["owners"]
    nei = mesh_data["neighbours"]
    coupled = mesh_data.get("boundary_neighbours")
    # Use bytes of the topology arrays + sizes
    key = (
        n,
        n_i,
        mesh_data["n_faces"],
        own.tobytes() if hasattr(own, "tobytes") else str(own),
        nei.tobytes() if hasattr(nei, "tobytes") else str(nei),
        coupled.tobytes() if hasattr(coupled, "tobytes") else str(coupled),
    )
    mesh_data["_fvm_sparsity_key"] = key
    return key


def assemble_matrix_from_fluxes(flux_data, mesh_data):
    """
    Assemble sparse coefficient matrix from face flux data.

    For each face:
    - A[owner, owner] += FluxCf[face]
    - A[owner, neighbor] += FluxFf[face]
    - A[neighbor, owner] -= FluxFf[face]
    - A[neighbor, neighbor] -= FluxCf[face]

    Args:
        flux_data: Dict with flux coefficients
            - flux_cf: Owner coefficients (n_faces,)
            - flux_ff: Neighbor coefficients (n_faces,)
        mesh_data: Mesh connectivity

    Returns:
        scipy.sparse.csr_matrix: Coefficient matrix A (n_elements, n_elements)
    """

    n_elements = mesh_data["n_elements"]
    n_interior_faces = mesh_data["n_interior_faces"]

    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]

    flux_cf = flux_data["flux_cf"]
    flux_ff = flux_data["flux_ff"]

    # Use LIL format for efficient construction
    A = lil_matrix((n_elements, n_elements), dtype=np.float64)

    # Assemble interior faces
    for i_face in range(n_interior_faces):
        own = owners[i_face]
        nei = neighbours[i_face]

        cf = flux_cf[i_face]
        ff = flux_ff[i_face]

        # Owner row
        A[own, own] += cf
        A[own, nei] += ff

        # Neighbor row
        # F_nei = -F_own = -(cf * own + ff * nei) = -cf * own - ff * nei
        # A[nei, own] += -cf
        # A[nei, nei] += -ff
        A[nei, own] -= cf
        A[nei, nei] -= ff

    # Assemble boundary faces
    # For boundary faces, only owner equation is affected
    for i_face in range(n_interior_faces, len(flux_cf)):
        own = owners[i_face]
        cf = flux_cf[i_face]

        # Boundary contribution to diagonal
        A[own, own] += cf
        boundary_neighbours = mesh_data.get("boundary_neighbours")
        if boundary_neighbours is not None and boundary_neighbours[i_face] >= 0:
            A[own, boundary_neighbours[i_face]] += flux_ff[i_face]

    # Convert to CSR format for efficient arithmetic
    return A.tocsr()


def build_sparsity_pattern(mesh_data):
    """Pre-build (rows, cols) index arrays for matrix assembly.

    The sparsity pattern depends only on mesh topology (owner/neighbour
    relationships), not on coefficient values.  For static meshes this
    can be built once and reused across all iterations, avoiding repeated
    ``np.concatenate`` and ``tocsr()`` sort overhead.

    Returns
    -------
    rows, cols : ndarray
        Index arrays that can be passed as ``indices`` to
        :func:`assemble_matrix_from_fluxes_vectorized`.
    """
    n_interior_faces = mesh_data["n_interior_faces"]
    owners_i = mesh_data["owners"][:n_interior_faces]
    neighbours = mesh_data["neighbours"][:n_interior_faces]
    has_boundaries = mesh_data["n_faces"] > n_interior_faces
    if has_boundaries:
        owners_b = mesh_data["owners"][n_interior_faces:]
        boundary_neighbours = np.asarray(
            mesh_data.get("boundary_neighbours", np.full(mesh_data["n_faces"], -1))
        )[n_interior_faces:]
        coupled = boundary_neighbours >= 0
        rows = np.concatenate(
            [owners_i, owners_i, neighbours, neighbours, owners_b, owners_b[coupled]]
        )
        cols = np.concatenate(
            [owners_i, neighbours, owners_i, neighbours, owners_b, boundary_neighbours[coupled]]
        )
    else:
        rows = np.concatenate([owners_i, owners_i, neighbours, neighbours])
        cols = np.concatenate([owners_i, neighbours, owners_i, neighbours])
    return rows, cols


def assemble_matrix_from_fluxes_vectorized(flux_data, mesh_data, indices=None):
    """
    Vectorized version of matrix assembly for better performance.

    Uses COO (coordinate) format for efficient construction from arrays.
    When called repeatedly with the same mesh topology, the sparsity
    indices are cached internally so that ``np.concatenate`` for the
    row/col arrays is only done once.

    Parameters
    ----------
    flux_data : dict
        Dict with flux_cf (owner coeff), flux_ff (neighbour coeff).
    mesh_data : dict
        Mesh connectivity (owners, neighbours, n_elements, n_interior_faces).
    indices : tuple of (rows, cols) or None
        Pre-built sparsity indices from :func:`build_sparsity_pattern`.
        When *None*, indices are fetched from an internal topology cache
        or computed on first use.
    """

    n_elements = mesh_data["n_elements"]
    n_interior_faces = mesh_data["n_interior_faces"]

    # Slices for interior faces
    owners = mesh_data["owners"][:n_interior_faces]
    neighbours = mesh_data["neighbours"][:n_interior_faces]

    flux_cf = flux_data["flux_cf"][:n_interior_faces]
    flux_ff = flux_data["flux_ff"][:n_interior_faces]

    # Handle boundary faces if present
    has_boundaries = len(flux_data["flux_cf"]) > n_interior_faces
    if has_boundaries:
        owners_b = mesh_data["owners"][n_interior_faces:]
        flux_cf_b = flux_data["flux_cf"][n_interior_faces:]
        flux_ff_b = flux_data["flux_ff"][n_interior_faces:]
        boundary_neighbours = np.asarray(
            mesh_data.get("boundary_neighbours", np.full(mesh_data["n_faces"], -1))
        )[n_interior_faces:]
        coupled = boundary_neighbours >= 0

    if indices is not None:
        rows, cols = indices
    else:
        key = _sparsity_key(mesh_data)
        cached = _SPATIAL_CACHE.get(key)
        if cached is not None:
            rows, cols = cached
        else:
            if has_boundaries:
                rows = np.concatenate(
                    [owners, owners, neighbours, neighbours, owners_b, owners_b[coupled]]
                )
                cols = np.concatenate(
                    [
                        owners,
                        neighbours,
                        owners,
                        neighbours,
                        owners_b,
                        boundary_neighbours[coupled],
                    ]
                )
            else:
                rows = np.concatenate([owners, owners, neighbours, neighbours])
                cols = np.concatenate([owners, neighbours, owners, neighbours])
            _SPATIAL_CACHE[key] = (rows, cols)

    data = np.concatenate([flux_cf, flux_ff, -flux_cf, -flux_ff])
    if has_boundaries:
        data = np.concatenate([data, flux_cf_b, flux_ff_b[coupled]])

    # Create sparse matrix from COO data
    from scipy.sparse import coo_matrix

    A_coo = coo_matrix((data, (rows, cols)), shape=(n_elements, n_elements))

    # Convert to CSR and sum duplicates
    return A_coo.tocsr()


def assemble_rhs_from_fluxes(flux_data, mesh_data):
    """
    Assemble RHS vector from explicit flux corrections.

    b[owner] -= FluxVf[face]
    b[neighbor] += FluxVf[face]

    Args:
        flux_data: Dict with flux coefficients
            - flux_vf: Explicit correction (n_faces,)
        mesh_data: Mesh connectivity

    Returns:
        numpy.ndarray: RHS vector b (n_elements,)
    """

    n_elements = mesh_data["n_elements"]
    n_interior_faces = mesh_data["n_interior_faces"]

    owners = mesh_data["owners"]
    neighbours = mesh_data["neighbours"]

    flux_vf = flux_data["flux_vf"]

    # Initialize RHS
    b = np.zeros(n_elements, dtype=np.float64)

    # Interior faces
    for i_face in range(n_interior_faces):
        own = owners[i_face]
        nei = neighbours[i_face]
        vf = flux_vf[i_face]

        b[own] -= vf
        b[nei] += vf

    # Boundary faces
    for i_face in range(n_interior_faces, len(flux_vf)):
        own = owners[i_face]
        vf = flux_vf[i_face]

        b[own] -= vf

    return b


def assemble_rhs_from_fluxes_vectorized(flux_data, mesh_data):
    """
    Vectorized RHS assembly.
    """

    n_elements = mesh_data["n_elements"]
    n_interior_faces = mesh_data["n_interior_faces"]

    owners = mesh_data["owners"][:n_interior_faces]
    neighbours = mesh_data["neighbours"][:n_interior_faces]
    flux_vf = flux_data["flux_vf"][:n_interior_faces]

    # Initialize RHS
    b = np.zeros(n_elements, dtype=np.float64)

    # Interior faces - vectorized accumulation
    np.add.at(b, owners, -flux_vf)
    np.add.at(b, neighbours, flux_vf)

    # Boundary faces
    if len(flux_data["flux_vf"]) > n_interior_faces:
        owners_b = mesh_data["owners"][n_interior_faces:]
        flux_vf_b = flux_data["flux_vf"][n_interior_faces:]
        np.add.at(b, owners_b, -flux_vf_b)

    return b


# Delegated linear solver implementation lives in `..solve.linear_interface`.
# This keeps matrix assembly focused on building numerics and allows swapping solver backends.
from ..solve.linear_interface import normalized_residual, solve_linear_system  # noqa: F401


def solve_diffusion_equation(
    phi_initial, gamma, boundaries, mesh_data, geo_data, solver="spsolve", **solver_kwargs
):
    """
    Complete workflow: assemble and solve diffusion equation.

    ∇·(γ∇φ) = 0

    Args:
        phi_initial: Initial field values
        gamma: Diffusion coefficient
        boundaries: Boundary conditions
        mesh_data: Mesh connectivity
        geo_data: Geometric data
        solver: Linear solver method
        **solver_kwargs: Solver options

    Returns:
        numpy.ndarray: Solution field
    """

    from ..fields import gradients
    from . import diffusion

    # Compute gradient
    grad_phi = gradients.compute_gradient_gauss_linear_vectorized(phi_initial, mesh_data, geo_data)

    # Assemble diffusion term
    flux_data = diffusion.assemble_diffusion_term(
        phi_initial, grad_phi, gamma, mesh_data, geo_data, boundaries
    )

    # Assemble matrix and RHS
    A = assemble_matrix_from_fluxes_vectorized(flux_data, mesh_data)
    b = assemble_rhs_from_fluxes_vectorized(flux_data, mesh_data)

    # Solve
    phi_solution = solve_linear_system(A, b, method=solver, equation_type="scalar", **solver_kwargs)

    return phi_solution
