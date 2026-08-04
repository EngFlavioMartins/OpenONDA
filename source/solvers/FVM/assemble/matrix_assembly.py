"""Assemble finite-volume face coefficients into ``A φ = b``."""

from dataclasses import dataclass, field
from itertools import count
import logging

from numba import njit
import numpy as np
from scipy.sparse import csr_matrix

# Module logger
logger = logging.getLogger(__name__)
_WORKSPACE_IDS = count()


@dataclass(frozen=True)
class _CSRPattern:
    """Static CSR indices and face-contribution locations for one mesh."""

    indices: np.ndarray
    indptr: np.ndarray
    diagonal_slots: np.ndarray
    offdiagonal_slots: np.ndarray


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
        )

    def update(self, flux_data, mesh_data, *, backend: str = "numpy") -> csr_matrix:
        """Overwrite coefficients and return the stable CSR matrix object."""
        del backend  # Direct CSR accumulation is backend-independent.
        _fill_matrix_values(
            self.matrix.data,
            self.pattern,
            flux_data,
            mesh_data,
            include_boundaries=self.include_boundaries,
        )
        return self.matrix


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


@njit(cache=True)
def _find_sorted_column(indices, start, stop, column):
    """Return a column's position in one sorted CSR row."""
    lower = start
    upper = stop
    while lower < upper:
        middle = (lower + upper) // 2
        value = indices[middle]
        if value < column:
            lower = middle + 1
        else:
            upper = middle
    if lower >= stop or indices[lower] != column:
        return -1
    return lower


@njit(cache=True)
def _build_csr_pattern_numba(
    owners,
    neighbours,
    boundary_neighbours,
    n_elements,
    n_interior,
    n_faces,
    include_boundaries,
):
    """Build unique CSR rows and face-to-entry slots with bounded storage."""
    # First construct each row as diagonal plus adjacent cell IDs.  This raw
    # graph is only marginally larger than the final matrix (two entries per
    # internal face), unlike concatenate+linear-index+unique, whose rows,
    # columns, 64-bit keys, sort workspace, and inverse coexist at peak.
    row_counts = np.ones(n_elements, dtype=np.int32)
    for face in range(n_interior):
        row_counts[owners[face]] += 1
        row_counts[neighbours[face]] += 1
    if include_boundaries:
        for face in range(n_interior, n_faces):
            if boundary_neighbours[face] >= 0:
                row_counts[owners[face]] += 1

    raw_indptr = np.empty(n_elements + 1, dtype=np.int32)
    raw_indptr[0] = 0
    for row in range(n_elements):
        next_offset = int(raw_indptr[row]) + int(row_counts[row])
        if next_offset > 2_147_483_647:
            raise OverflowError("FVM CSR topology exceeds 32-bit addressing")
        raw_indptr[row + 1] = next_offset
    raw_indices = np.empty(raw_indptr[-1], dtype=np.int32)
    cursor = raw_indptr[:-1].copy()
    for row in range(n_elements):
        raw_indices[cursor[row]] = row
        cursor[row] += 1
    for face in range(n_interior):
        owner = owners[face]
        neighbour = neighbours[face]
        raw_indices[cursor[owner]] = neighbour
        cursor[owner] += 1
        raw_indices[cursor[neighbour]] = owner
        cursor[neighbour] += 1
    if include_boundaries:
        for face in range(n_interior, n_faces):
            neighbour = boundary_neighbours[face]
            if neighbour >= 0:
                owner = owners[face]
                raw_indices[cursor[owner]] = neighbour
                cursor[owner] += 1

    # Cell stencils are short. Insertion-sorting each row in place avoids an
    # all-nnz argsort while producing canonical SciPy/PETSc column ordering.
    unique_counts = np.empty(n_elements, dtype=np.int32)
    for row in range(n_elements):
        start = raw_indptr[row]
        stop = raw_indptr[row + 1]
        for position in range(start + 1, stop):
            value = raw_indices[position]
            previous = position - 1
            while previous >= start and raw_indices[previous] > value:
                raw_indices[previous + 1] = raw_indices[previous]
                previous -= 1
            raw_indices[previous + 1] = value
        count_unique = 0
        previous_value = -1
        for position in range(start, stop):
            value = raw_indices[position]
            if value != previous_value:
                count_unique += 1
                previous_value = value
        unique_counts[row] = count_unique

    indptr = np.empty(n_elements + 1, dtype=np.int32)
    indptr[0] = 0
    for row in range(n_elements):
        indptr[row + 1] = indptr[row] + unique_counts[row]
    indices = np.empty(indptr[-1], dtype=np.int32)
    for row in range(n_elements):
        source_start = raw_indptr[row]
        source_stop = raw_indptr[row + 1]
        destination = indptr[row]
        previous_value = -1
        for position in range(source_start, source_stop):
            value = raw_indices[position]
            if value != previous_value:
                indices[destination] = value
                destination += 1
                previous_value = value

    n_coupled = 0
    if include_boundaries:
        for face in range(n_interior, n_faces):
            if boundary_neighbours[face] >= 0:
                n_coupled += 1
    # Diagonal destinations repeat for every incident face. Store them once
    # per cell, plus the two genuinely face-specific off-diagonal entries.
    # This is the CSR equivalent of OpenFOAM's diag/lower/upper LDU layout.
    diagonal_slots = np.empty(n_elements, dtype=np.int32)
    for row in range(n_elements):
        diagonal_slots[row] = _find_sorted_column(indices, indptr[row], indptr[row + 1], row)
    offdiagonal_slots = np.empty(2 * n_interior + n_coupled, dtype=np.int32)
    for face in range(n_interior):
        owner = owners[face]
        neighbour = neighbours[face]
        offdiagonal_slots[face] = _find_sorted_column(
            indices, indptr[owner], indptr[owner + 1], neighbour
        )
        offdiagonal_slots[n_interior + face] = _find_sorted_column(
            indices, indptr[neighbour], indptr[neighbour + 1], owner
        )

    slot_cursor = 2 * n_interior
    if include_boundaries:
        for face in range(n_interior, n_faces):
            neighbour = boundary_neighbours[face]
            if neighbour >= 0:
                owner = owners[face]
                offdiagonal_slots[slot_cursor] = _find_sorted_column(
                    indices, indptr[owner], indptr[owner + 1], neighbour
                )
                slot_cursor += 1

    for position in range(len(diagonal_slots)):
        if diagonal_slots[position] < 0:
            raise RuntimeError("CSR diagonal is absent from its matrix row")
    for position in range(len(offdiagonal_slots)):
        if offdiagonal_slots[position] < 0:
            raise RuntimeError("CSR face contribution is absent from its matrix row")
    return indices, indptr, diagonal_slots, offdiagonal_slots


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

    n_elements = int(mesh_data["n_elements"])
    n_interior = int(mesh_data["n_interior_faces"])
    n_faces = int(mesh_data["n_faces"])
    owners = np.asarray(mesh_data["owners"], dtype=np.int32)
    neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int32)
    boundary_neighbours = mesh_data.get("boundary_neighbours")
    if boundary_neighbours is None:
        boundary_neighbours = np.full(n_faces, -1, dtype=np.int32)
    else:
        boundary_neighbours = np.asarray(boundary_neighbours, dtype=np.int32)
    indices, indptr, diagonal_slots, offdiagonal_slots = _build_csr_pattern_numba(
        owners,
        neighbours,
        boundary_neighbours,
        n_elements,
        n_interior,
        n_faces,
        include_boundaries,
    )

    indices.setflags(write=False)
    indptr.setflags(write=False)
    diagonal_slots.setflags(write=False)
    offdiagonal_slots.setflags(write=False)
    pattern = _CSRPattern(indices, indptr, diagonal_slots, offdiagonal_slots)
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
def _fill_matrix_values_numba(
    target,
    diagonal_slots,
    offdiagonal_slots,
    flux_cf,
    flux_ff,
    owners,
    neighbours,
    boundary_neighbours,
    n_interior,
    include_boundaries,
):
    """Accumulate face coefficients directly into existing CSR data."""
    target[:] = 0.0
    for face in range(n_interior):
        owner = owners[face]
        neighbour = neighbours[face]
        target[diagonal_slots[owner]] += flux_cf[face]
        target[offdiagonal_slots[face]] += flux_ff[face]
        target[offdiagonal_slots[n_interior + face]] -= flux_cf[face]
        target[diagonal_slots[neighbour]] -= flux_ff[face]
    cursor = 2 * n_interior
    if include_boundaries:
        for face in range(n_interior, len(flux_cf)):
            target[diagonal_slots[owners[face]]] += flux_cf[face]
        for face in range(n_interior, len(flux_ff)):
            if boundary_neighbours[face] >= 0:
                target[offdiagonal_slots[cursor]] += flux_ff[face]
                cursor += 1
    return cursor


def _fill_matrix_values(
    target,
    pattern,
    flux_data,
    mesh_data,
    *,
    include_boundaries: bool,
):
    """Refill CSR numeric values without a face-sized concatenation buffer."""
    n_faces = int(mesh_data["n_faces"])
    boundary_neighbours = mesh_data.get("boundary_neighbours")
    if boundary_neighbours is None:
        boundary_neighbours = np.full(n_faces, -1, dtype=np.int32)
    else:
        boundary_neighbours = np.asarray(boundary_neighbours, dtype=np.int32)
    cursor = _fill_matrix_values_numba(
        target,
        pattern.diagonal_slots,
        pattern.offdiagonal_slots,
        np.asarray(flux_data["flux_cf"], dtype=np.float64),
        np.asarray(flux_data["flux_ff"], dtype=np.float64),
        np.asarray(mesh_data["owners"], dtype=np.int32),
        np.asarray(mesh_data["neighbours"], dtype=np.int32),
        boundary_neighbours,
        int(mesh_data["n_interior_faces"]),
        include_boundaries,
    )
    if cursor != len(pattern.offdiagonal_slots):
        raise RuntimeError("Matrix contribution slots do not match the mesh faces")
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

    if indices is not None:
        contributions = _matrix_contributions(
            flux_data, mesh_data, include_boundaries=has_boundaries
        )
        rows, cols = indices
        return csr_matrix((contributions, (rows, cols)), shape=(n_elements, n_elements))

    pattern = _csr_pattern(mesh_data, has_boundaries)
    data = np.zeros(len(pattern.indices), dtype=np.float64)
    _fill_matrix_values(
        data,
        pattern,
        flux_data,
        mesh_data,
        include_boundaries=has_boundaries,
    )
    return csr_matrix(
        (data, pattern.indices, pattern.indptr),
        shape=(n_elements, n_elements),
        copy=False,
    )


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
