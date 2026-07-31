#!/usr/bin/env python3
"""
Matrix Assembly for OpenONDA FVM Solver

Converts flux coefficients into sparse matrix format for linear system solution.

Implements: A * phi = b
where A is the coefficient matrix and b is the RHS vector.

Converted from uFVM cfdAssembleIntoGlobalMatrixFaceFluxes.m
"""

from dataclasses import dataclass, field
from itertools import count
import logging

from numba import njit
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

# Module logger
logger = logging.getLogger(__name__)
_WORKSPACE_IDS = count()


@dataclass(frozen=True)
class _CSRPattern:
    """Structural CSR (compressed sparse row) pattern for a static mesh.

    Stores the column indices, row pointers, and contribution-slot mapping
    that remain invariant as long as the mesh topology does not change.
    Instances are created once by :meth:`MatrixAssemblyWorkspace.create` and
    reused across every time step.

    Attributes
    ----------
    indices : np.ndarray
        Column indices of the sparse matrix (CSR format).
    indptr : np.ndarray
        Row pointers into ``indices`` (CSR format), shape ``(n_rows + 1,)``.
    contribution_slots : np.ndarray
        Mapping from flux-contribution entries to their target positions in
        the flattened coefficient array.
    """

    indices: np.ndarray
    indptr: np.ndarray
    contribution_slots: np.ndarray


@dataclass
class MatrixAssemblyWorkspace:
    """Reusable CSR storage for a static mesh topology.

    The matrix object and its structural arrays remain stable across updates;
    only ``matrix.data`` is overwritten. Callers that need to retain an older
    coefficient state must copy the returned matrix.
    """

    matrix: csr_matrix
    pattern: _CSRPattern
    include_boundaries: bool
    contributions: np.ndarray
    cache_namespace: int = field(default_factory=lambda: next(_WORKSPACE_IDS))

    @classmethod
    def create(cls, mesh_data, *, include_boundaries: bool = True):
        pattern = _csr_pattern(mesh_data, include_boundaries)
        data = np.zeros(len(pattern.indices), dtype=np.float64)
        matrix = csr_matrix(
            (data, pattern.indices, pattern.indptr),
            shape=(mesh_data["n_elements"], mesh_data["n_elements"]),
            copy=False,
        )
        return cls(
            matrix=matrix,
            pattern=pattern,
            include_boundaries=include_boundaries,
            contributions=np.empty(len(pattern.contribution_slots), dtype=np.float64),
        )

    def update(self, flux_data, mesh_data, *, backend: str = "numpy") -> csr_matrix:
        """Overwrite coefficients and return the stable CSR matrix object."""
        contributions = _fill_matrix_contributions(
            self.contributions, flux_data, mesh_data, include_boundaries=self.include_boundaries
        )
        values = _reduce_contributions(
            self.pattern.contribution_slots,
            contributions,
            len(self.pattern.indices),
            backend,
        )
        self.matrix.data[:] = values
        return self.matrix


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


def _csr_pattern(mesh_data, include_boundaries: bool) -> _CSRPattern:
    """Return mesh-owned CSR structure and contribution destination slots.

    Static mesh topology is immutable during a solve.  Keeping the pattern on
    that mesh gives it the correct lifetime and avoids retaining topology-byte
    copies for every solver ever constructed in this process.
    """
    patterns = mesh_data.setdefault("_fvm_csr_patterns", {})
    cached = patterns.get(include_boundaries)
    if cached is not None:
        return cached

    if include_boundaries:
        rows, cols = build_sparsity_pattern(mesh_data)
    else:
        n_interior = mesh_data["n_interior_faces"]
        owners = mesh_data["owners"][:n_interior]
        neighbours = mesh_data["neighbours"][:n_interior]
        rows = np.concatenate([owners, owners, neighbours, neighbours])
        cols = np.concatenate([owners, neighbours, owners, neighbours])

    n_elements = mesh_data["n_elements"]
    linear_indices = rows.astype(np.int64) * n_elements + cols
    unique_indices, contribution_slots = np.unique(linear_indices, return_inverse=True)
    unique_rows = unique_indices // n_elements
    indices = np.asarray(unique_indices % n_elements, dtype=np.int32)
    row_counts = np.bincount(unique_rows, minlength=n_elements)
    indptr = np.empty(n_elements + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(row_counts, out=indptr[1:])

    indices.setflags(write=False)
    indptr.setflags(write=False)
    contribution_slots = np.asarray(contribution_slots, dtype=np.int64)
    contribution_slots.setflags(write=False)
    pattern = _CSRPattern(indices, indptr, contribution_slots)
    patterns[include_boundaries] = pattern
    return pattern


def prepare_matrix_assembly(mesh_data) -> None:
    """Precompute the static full-face CSR structure for a mesh."""
    _csr_pattern(mesh_data, include_boundaries=True)


def _matrix_contributions(flux_data, mesh_data, *, include_boundaries: bool) -> np.ndarray:
    """Return face contributions in the order used by the cached CSR pattern."""
    n_interior_faces = mesh_data["n_interior_faces"]
    flux_cf = flux_data["flux_cf"][:n_interior_faces]
    flux_ff = flux_data["flux_ff"][:n_interior_faces]
    contributions = [flux_cf, flux_ff, -flux_cf, -flux_ff]

    if include_boundaries:
        flux_cf_b = flux_data["flux_cf"][n_interior_faces:]
        flux_ff_b = flux_data["flux_ff"][n_interior_faces:]
        boundary_neighbours = np.asarray(
            mesh_data.get("boundary_neighbours", np.full(mesh_data["n_faces"], -1))
        )[n_interior_faces:]
        coupled = boundary_neighbours >= 0
        contributions.extend((flux_cf_b, flux_ff_b[coupled]))
    return np.concatenate(contributions)


def _fill_matrix_contributions(target, flux_data, mesh_data, *, include_boundaries: bool):
    """Fill a reusable contribution buffer in cached-CSR ordering.

    Matrix updates occur for every momentum/pressure correction.  Reusing the
    buffer avoids allocating and concatenating four full interior-face arrays
    on each update while preserving the exact coefficient ordering.
    """
    n_interior = mesh_data["n_interior_faces"]
    flux_cf = np.asarray(flux_data["flux_cf"])
    flux_ff = np.asarray(flux_data["flux_ff"])
    cursor = 0
    for values, sign in (
        (flux_cf[:n_interior], 1.0),
        (flux_ff[:n_interior], 1.0),
        (flux_cf[:n_interior], -1.0),
        (flux_ff[:n_interior], -1.0),
    ):
        end = cursor + len(values)
        if sign == 1.0:
            target[cursor:end] = values
        else:
            np.negative(values, out=target[cursor:end])
        cursor = end
    if include_boundaries:
        values = flux_cf[n_interior:]
        end = cursor + len(values)
        target[cursor:end] = values
        cursor = end
        boundary_neighbours = np.asarray(
            mesh_data.get("boundary_neighbours", np.full(mesh_data["n_faces"], -1))
        )[n_interior:]
        coupled = boundary_neighbours >= 0
        values = flux_ff[n_interior:][coupled]
        end = cursor + len(values)
        target[cursor:end] = values
        cursor = end
    if cursor != len(target):
        raise RuntimeError("Matrix contribution buffer does not match cached sparsity pattern")
    return target


@njit(cache=True)
def _reduce_contributions_numba(slots, contributions, size):
    values = np.zeros(size, dtype=np.float64)
    for index in range(len(contributions)):
        values[slots[index]] += contributions[index]
    return values


def _reduce_contributions(slots, contributions, size: int, backend: str) -> np.ndarray:
    if backend == "numpy":
        return np.bincount(slots, weights=contributions, minlength=size)
    if backend == "numba":
        return _reduce_contributions_numba(slots, contributions, size)
    if backend == "taichi":
        from ..core.taichi_operators import reduce_contributions

        return reduce_contributions(slots, contributions, size)
    raise ValueError(f"Unknown operator backend {backend!r}")


def assemble_matrix_from_fluxes_vectorized(
    flux_data, mesh_data, indices=None, workspace=None, *, backend: str = "numpy"
):
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
    has_boundaries = len(flux_data["flux_cf"]) > n_interior_faces

    if workspace is not None:
        if indices is not None:
            raise ValueError("indices and workspace are mutually exclusive")
        if workspace.include_boundaries != has_boundaries:
            raise ValueError("Matrix workspace boundary layout does not match the flux arrays")
        return workspace.update(flux_data, mesh_data, backend=backend)

    contributions = _matrix_contributions(flux_data, mesh_data, include_boundaries=has_boundaries)

    if indices is not None:
        rows, cols = indices
        return csr_matrix((contributions, (rows, cols)), shape=(n_elements, n_elements))

    pattern = _csr_pattern(mesh_data, has_boundaries)
    data = _reduce_contributions(
        pattern.contribution_slots,
        contributions,
        len(pattern.indices),
        backend,
    )
    return csr_matrix(
        (data, pattern.indices, pattern.indptr),
        shape=(n_elements, n_elements),
        copy=False,
    )


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


@njit(cache=True)
def _assemble_rhs_numba(flux_vf, owners, neighbours, n_elements, n_interior_faces):
    result = np.zeros(n_elements, dtype=np.float64)
    for face in range(n_interior_faces):
        value = flux_vf[face]
        result[owners[face]] -= value
        result[neighbours[face]] += value
    for face in range(n_interior_faces, len(flux_vf)):
        result[owners[face]] -= flux_vf[face]
    return result


def assemble_rhs_from_fluxes_vectorized(flux_data, mesh_data, *, backend: str = "numpy"):
    """
    Vectorized RHS assembly.
    """

    n_elements = mesh_data["n_elements"]
    n_interior_faces = mesh_data["n_interior_faces"]

    owners = mesh_data["owners"][:n_interior_faces]
    neighbours = mesh_data["neighbours"][:n_interior_faces]
    flux_vf = flux_data["flux_vf"][:n_interior_faces]

    if backend == "numba":
        return _assemble_rhs_numba(
            np.asarray(flux_data["flux_vf"], dtype=np.float64),
            np.asarray(mesh_data["owners"], dtype=np.int64),
            np.asarray(mesh_data["neighbours"], dtype=np.int64),
            n_elements,
            n_interior_faces,
        )
    if backend == "taichi":
        from ..core.taichi_operators import assemble_rhs

        return assemble_rhs(
            flux_data["flux_vf"],
            mesh_data["owners"],
            mesh_data["neighbours"],
            n_elements,
            n_interior_faces,
        )
    if backend != "numpy":
        raise ValueError(f"Unknown operator backend {backend!r}")

    b = np.bincount(owners, weights=-flux_vf, minlength=n_elements)
    b += np.bincount(neighbours, weights=flux_vf, minlength=n_elements)

    # Boundary faces
    if len(flux_data["flux_vf"]) > n_interior_faces:
        owners_b = mesh_data["owners"][n_interior_faces:]
        flux_vf_b = flux_data["flux_vf"][n_interior_faces:]
        b += np.bincount(owners_b, weights=-flux_vf_b, minlength=n_elements)

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
