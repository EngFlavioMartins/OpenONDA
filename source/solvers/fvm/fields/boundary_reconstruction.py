"""Cached, noncollective least-squares reconstruction at physical boundaries.

Only boundary-owner stencils are evaluated. Processor neighbours are ordinary
real/halo cells in these stencils; their current values must already be present
in the supplied field. No halo exchange is hidden in a patch setter.
"""

from dataclasses import dataclass

import numpy as np

from .gradients import _is_empty_boundary


@dataclass(frozen=True)
class _BoundaryStencil:
    signature: tuple
    cells: np.ndarray
    row: np.ndarray
    neighbour: np.ndarray
    weighted_displacement: np.ndarray
    inverse: np.ndarray
    projector: np.ndarray


@dataclass(frozen=True)
class _BoundaryStencilCache:
    signature: tuple
    entries: dict


def _boundary_stencil(mesh_data, geo_data, include_physical_ghosts, cells):
    n_cells = int(mesh_data["n_cells"])
    n_interior = int(mesh_data["n_interior_faces"])
    signature = (
        n_cells,
        n_interior,
        int(mesh_data["n_faces"]),
        id(mesh_data["owners"]),
        id(mesh_data["neighbours"]),
        id(mesh_data.get("boundary_neighbour_cell")),
        id(geo_data["cell_centre"]),
        id(geo_data["face_centre"]),
        id(geo_data["cell_connection_vector"]),
        tuple(
            (
                int(patch["start_face"]),
                int(patch["n_faces"]),
                _is_empty_boundary(patch, allow_source_type=True),
            )
            for patch in mesh_data["boundary"]
        ),
    )
    key = (
        "_boundary_owner_lsq_with_ghosts" if include_physical_ghosts else "_boundary_owner_lsq_real"
    )
    cache = geo_data.get(key)
    if cache is None or cache.signature != signature:
        cache = _BoundaryStencilCache(signature, {})
        geo_data[key] = cache
    selected_key = cells.tobytes()
    if selected_key in cache.entries:
        return cache.entries[selected_key]

    # Build only the selected owners' stencils. A thin extruded mesh may have
    # front/back faces on every cell; evaluating all physical-boundary owners
    # for one numerical-boundary patch would become a full-domain gradient.
    cell_row = np.full(n_cells, -1, dtype=np.int64)
    cell_row[cells] = np.arange(len(cells))
    if "least_squares_owner_cell" in geo_data:
        owner = np.asarray(geo_data["least_squares_owner_cell"])
        neighbour = np.asarray(geo_data["least_squares_neighbour_field_index"])
        weighted = np.asarray(geo_data["least_squares_neighbour_weighted_displacement"])
        selected = cell_row[owner] >= 0
        if not include_physical_ghosts:
            selected &= neighbour < n_cells
        owner, neighbour, weighted = owner[selected], neighbour[selected], weighted[selected]
    else:
        interior_owner = np.asarray(mesh_data["owners"][:n_interior])
        interior_neighbour = np.asarray(mesh_data["neighbours"][:n_interior])
        owner_parts, neighbour_parts, displacement_parts = [], [], []
        for own, neighbour in (
            (interior_owner, interior_neighbour),
            (interior_neighbour, interior_owner),
        ):
            selected = cell_row[own] >= 0
            owner_parts.append(own[selected])
            neighbour_parts.append(neighbour[selected])
            displacement_parts.append(
                geo_data["cell_centre"][neighbour[selected]]
                - geo_data["cell_centre"][own[selected]]
            )
        paired = np.asarray(
            mesh_data.get("boundary_neighbour_cell", np.full(mesh_data["n_faces"], -1))
        )
        for patch in mesh_data["boundary"]:
            if _is_empty_boundary(patch, allow_source_type=True):
                continue
            start = int(patch["start_face"])
            faces = np.arange(start, start + int(patch["n_faces"]))
            own = np.asarray(mesh_data["owners"])[faces]
            selected = cell_row[own] >= 0
            if not include_physical_ghosts:
                selected &= paired[faces] >= 0
            faces, own = faces[selected], own[selected]
            coupled = paired[faces] >= 0
            neighbour = np.where(coupled, paired[faces], n_cells + faces - n_interior)
            dr = geo_data["face_centre"][faces] - geo_data["cell_centre"][own]
            dr[coupled] = geo_data["cell_connection_vector"][faces[coupled]]
            owner_parts.append(own)
            neighbour_parts.append(neighbour)
            displacement_parts.append(dr)
        owner = np.concatenate(owner_parts)
        neighbour = np.concatenate(neighbour_parts)
        dr = np.concatenate(displacement_parts)
        weighted = dr / np.maximum(np.einsum("si,si->s", dr, dr), 1e-60)[:, None]
    row = cell_row[owner]

    # w*dr has norm 1/|dr| for inverse-distance-squared weighting, hence
    # w*dr*dr^T = (w*dr)(w*dr)^T / |w*dr|^2. Recomputing the boundary-only
    # normal matrix allows physical ghosts to be omitted without a biased
    # inverse from the full native stencil.
    squared = np.einsum("si,si->s", weighted, weighted)
    moment = np.zeros((len(cells), 3, 3))
    for axis in range(3):
        for other in range(axis, 3):
            values = weighted[:, axis] * weighted[:, other] / np.maximum(squared, 1e-300)
            reduced = np.bincount(row, weights=values, minlength=len(cells))
            moment[:, axis, other] = reduced
            moment[:, other, axis] = reduced
    inverse = np.linalg.pinv(moment, rcond=1e-10, hermitian=True)
    projector = inverse @ moment
    cached = _BoundaryStencil(signature, cells, row, neighbour, weighted, inverse, projector)
    cache.entries[selected_key] = cached
    return cached


def boundary_owner_gradient(
    field_values,
    mesh_data,
    geo_data,
    face_indices,
    *,
    include_physical_ghosts=False,
    displacements=None,
):
    """Return selected boundary-owner gradients as ``(faces, axis, component)``.

    Real-cell-only reconstruction is independent of boundary patch update order.
    Setting ``include_physical_ghosts`` also consumes the supplied, lagged native
    face values; it never refreshes them or exchanges halos. Rank-deficient
    stencils use their minimum-norm LSQ gradient (e.g. a span-invariant plane).
    When extrapolating, pass ``displacements`` to reject directions that the
    stencil cannot resolve rather than silently extrapolating a zero derivative.
    The mesh geometry must be immutable during an algorithm instance.
    """
    faces = np.asarray(face_indices, dtype=np.int64)
    if (
        faces.ndim != 1
        or np.any(faces < mesh_data["n_interior_faces"])
        or np.any(faces >= mesh_data["n_faces"])
    ):
        raise ValueError("face_indices must select native boundary faces")
    values = np.asarray(field_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) < mesh_data["n_cells"]:
        raise ValueError("field_values must contain real/halo cell values")
    if not len(faces):
        return np.empty((0, 3, values.shape[1]))
    cells = np.unique(np.asarray(mesh_data["owners"])[faces])
    stencil = _boundary_stencil(mesh_data, geo_data, include_physical_ghosts, cells)
    selected = np.searchsorted(stencil.cells, np.asarray(mesh_data["owners"])[faces])
    if displacements is not None:
        displacement = np.asarray(displacements, dtype=float)
        if displacement.shape != (len(faces), 3):
            raise ValueError("displacements must have shape (selected faces, 3)")
        residual = displacement - np.einsum("fij,fj->fi", stencil.projector[selected], displacement)
        if np.any(
            np.linalg.norm(residual, axis=1) > 1e-8 * np.linalg.norm(displacement, axis=1) + 1e-13
        ):
            raise ValueError(
                "Boundary reconstruction stencil cannot resolve extrapolation direction"
            )
    if len(stencil.neighbour) and np.max(stencil.neighbour) >= len(values):
        raise ValueError("physical ghost reconstruction requires native face values")
    difference = values[stencil.neighbour] - values[stencil.cells[stencil.row]]
    rhs = np.empty((len(stencil.cells), 3, values.shape[1]))
    for axis in range(3):
        for component in range(values.shape[1]):
            rhs[:, axis, component] = np.bincount(
                stencil.row,
                weights=stencil.weighted_displacement[:, axis] * difference[:, component],
                minlength=len(stencil.cells),
            )
    gradient = np.einsum("bij,bjc->bic", stencil.inverse, rhs)
    return gradient[selected]


def boundary_resolved_displacement(mesh_data, geo_data, face_indices, displacements):
    """Project face displacements onto the real-neighbour stencil's span.

    This makes missing directional information explicit. A caller may retain
    this real-cell reconstruction and separately supply a documented, lagged
    boundary closure for the unresolved component. No fields are read and no
    communication is performed.
    """
    faces = np.asarray(face_indices, dtype=np.int64)
    displacement = np.asarray(displacements, dtype=float)
    if faces.ndim != 1 or displacement.shape != (len(faces), 3):
        raise ValueError("displacements must have shape (selected faces, 3)")
    if np.any(faces < mesh_data["n_interior_faces"]) or np.any(faces >= mesh_data["n_faces"]):
        raise ValueError("face_indices must select native boundary faces")
    if not len(faces):
        return displacement.copy()
    owners = np.asarray(mesh_data["owners"])[faces]
    stencil = _boundary_stencil(mesh_data, geo_data, False, np.unique(owners))
    selected = np.searchsorted(stencil.cells, owners)
    return np.einsum("fij,fj->fi", stencil.projector[selected], displacement)
