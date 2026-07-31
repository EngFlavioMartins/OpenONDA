from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _readonly(values):
    result = np.ascontiguousarray(values, dtype=np.float64).view()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MeshGeometry:
    """Immutable geometric quantities computed from a static mesh.

    Holds the cell centroids, volumes, face centroids, area vectors,
    interpolation weights, wall distance, and optional least-squares
    condition numbers.  All arrays are read-only and should be constructed
    once through the :meth:`from_data` factory.

    Attributes
    ----------
    points : np.ndarray
        Vertex coordinates, shape ``(n_points, 3)``.
    face_centroids : np.ndarray
        Face centre coordinates, shape ``(n_faces, 3)``.
    face_area_vectors : np.ndarray
        Face area normal vectors, shape ``(n_faces, 3)``.
    face_areas : np.ndarray
        Face area magnitudes, shape ``(n_faces,)``.
    cell_centroids : np.ndarray
        Cell centre coordinates, shape ``(n_cells, 3)``.
    cell_volumes : np.ndarray
        Cell volumes, shape ``(n_cells,)``.
    interpolation_weights : np.ndarray
        Face interpolation weights (cell-to-face), shape ``(n_faces,)``.
    cell_face_vectors : np.ndarray
        Cell-centre-to-face-centre vectors, shape ``(n_faces, 3)``.
    wall_distance : np.ndarray
        Owner-cell-centre to boundary-face distance [mesh length unit], stored
        per face with shape ``(n_faces,)``. Interior-face entries are zero.
    lsq_condition : np.ndarray or None
        Least-squares gradient stencil condition numbers (optional).
    geometry_version : int
        Version counter for cache invalidation.
    """

    points: np.ndarray
    face_centroids: np.ndarray
    face_area_vectors: np.ndarray
    face_areas: np.ndarray
    cell_centroids: np.ndarray
    cell_volumes: np.ndarray
    interpolation_weights: np.ndarray
    cell_face_vectors: np.ndarray
    wall_distance: np.ndarray
    lsq_condition: np.ndarray | None = None
    geometry_version: int = 1

    @classmethod
    def from_data(cls, mesh_data, geo_data) -> MeshGeometry:
        """Create read-only array views without freezing the source arrays.

        The facade shares memory with already-contiguous ``float64`` source
        arrays, so later source updates remain visible. Only writes through the
        facade are prohibited.
        """
        condition = geo_data.get("lsq_condition")
        return cls(
            points=_readonly(mesh_data["points"]),
            face_centroids=_readonly(geo_data["face_centroids"]),
            face_area_vectors=_readonly(geo_data["face_sf"]),
            face_areas=_readonly(geo_data["face_areas"]),
            cell_centroids=_readonly(geo_data["element_centroids"]),
            cell_volumes=_readonly(geo_data["element_volumes"]),
            interpolation_weights=_readonly(geo_data["face_weights"]),
            cell_face_vectors=_readonly(geo_data["face_cf_vector"]),
            wall_distance=_readonly(geo_data["wall_dist"]),
            lsq_condition=None if condition is None else _readonly(condition),
        )


def compute_geometry(
    points,
    faces,
    owners,
    neighbours,
    n_elements,
    n_faces,
    n_interior_faces,
    element_faces,
    *,
    logger=None,
):
    """
    Compute geometric properties of the mesh.

    Args:
        points (np.ndarray): Node coordinates (N_nodes, 3).
        faces (list of np.ndarray): List of node indices for each face.
        owners (np.ndarray): Owner cell indices.
        neighbours (np.ndarray): Neighbour cell indices.
        n_elements (int): Number of elements.
        n_faces (int): Number of faces.
        n_interior_faces (int): Number of interior faces.
        element_faces (list of list): Face indices for each element.

    Returns:
        dict: Dictionary containing geometric data:
            - face_centroids
            - face_sf
            - face_areas
            - element_centroids
            - element_volumes
            - face_weights
            - face_cf
            - face_cf_vector
            - face_ff_vector
            - wall_dist
    """

    # --- Initialize Arrays ---
    face_centroids = np.zeros((n_faces, 3), dtype=np.float64)
    face_sf = np.zeros((n_faces, 3), dtype=np.float64)
    face_areas = np.zeros(n_faces, dtype=np.float64)

    element_centroids = np.zeros((n_elements, 3), dtype=np.float64)
    element_volumes = np.zeros(n_elements, dtype=np.float64)

    face_weights = np.zeros(n_faces, dtype=np.float64)
    face_cf_vector = np.zeros((n_faces, 3), dtype=np.float64)  # faceCF in uFVM (owner to neighbour)
    face_cf = np.zeros((n_faces, 3), dtype=np.float64)  # faceCf in uFVM (owner to face)
    face_ff = np.zeros((n_faces, 3), dtype=np.float64)  # faceFf in uFVM (neighbour to face)
    wall_dist = np.zeros(n_faces, dtype=np.float64)
    wall_dist_limited = np.zeros(n_faces, dtype=np.float64)

    # --- Process Basic Face Geometry ---
    from ..io import logging

    logging.Timer.start("Basic Face Geometry")

    # 1. Convert faces to a compact padded array.  Rectilinear meshes already
    # store fixed-width quads contiguously, so avoid rebuilding them through a
    # Python loop.
    face_array = faces if isinstance(faces, np.ndarray) else None
    if face_array is not None and face_array.ndim == 2:
        padded_faces = np.ascontiguousarray(face_array, dtype=np.int32)
        max_nodes = padded_faces.shape[1]
        counts = np.full(n_faces, max_nodes, dtype=np.int32)
    else:
        max_nodes = max(len(f) for f in faces)
        padded_faces = np.full((n_faces, max_nodes), -1, dtype=np.int32)
        counts = np.zeros(n_faces, dtype=np.int32)
        for i, f in enumerate(faces):
            n_f = len(f)
            padded_faces[i, :n_f] = f
            counts[i] = n_f

    # 2. Get coordinates (n_faces, max_nodes, 3)
    # Mask invalid nodes with 0 or first point (doesn't matter if masked later,
    # but for safety let's use first point to avoid index errors if -1)
    safe_faces = padded_faces.copy()
    safe_faces[padded_faces == -1] = 0

    face_coords = points[safe_faces]  # (N_faces, max_nodes, 3)

    # 3. Compute rough center (mean of valid nodes)
    # Mask: 1 where valid, 0 where invalid from padded_faces
    mask = (padded_faces != -1).astype(np.float64)[:, :, np.newaxis]  # (N, M, 1)

    # Sum coordinates and divide by count
    face_coords_masked = face_coords * mask
    local_centre = np.sum(face_coords_masked, axis=1) / counts[:, np.newaxis]  # (N, 3)

    # 4. Vectorized Fan Triangulation
    # We iterate max_nodes times, computing triangles (C, P_i, P_{i+1})

    centroid_sum = np.zeros((n_faces, 3))
    sf_sum = np.zeros((n_faces, 3))
    area_sum = np.zeros(n_faces)

    for i in range(max_nodes):
        # Point 1: P_i
        # Point 2: P_{i+1}
        P1 = face_coords[:, i, :]
        is_last: np.ndarray = np.equal(i, counts - 1)
        idx_next = i + 1
        P2_raw = face_coords[:, idx_next, :] if idx_next < max_nodes else np.zeros((n_faces, 3))
        P2 = np.where(is_last[:, np.newaxis], face_coords[:, 0, :], P2_raw)

        # Triangle validity mask: i < counts
        valid_tri = np.less(i, counts).astype(np.float64)[:, np.newaxis]

        # Compute Triangle Properties
        # p1 = C (already broadcasted effectively as C[:,0,:])
        # p2 = P1
        # p3 = P2
        # But wait, the loop was: p1=Center, p2=Node_i, p3=Node_i+1

        local_centroid = (local_centre + P1 + P2) / 3.0

        # Cross product: (P1 - C) x (P2 - C)
        vec1 = P1 - local_centre
        vec2 = P2 - local_centre

        local_sf_vec = 0.5 * np.cross(vec1, vec2)
        local_area_val = np.linalg.norm(local_sf_vec, axis=1)  # (N,)

        # Accumulate
        centroid_sum += local_centroid * local_area_val[:, np.newaxis] * valid_tri
        sf_sum += local_sf_vec * valid_tri
        area_sum += local_area_val * valid_tri[:, 0]

    # Finalize
    # Avoid div by zero for area
    safe_area = area_sum.copy()
    safe_area[safe_area == 0] = 1.0

    face_centroids = centroid_sum / safe_area[:, np.newaxis]
    face_sf = sf_sum
    face_areas = area_sum

    logging.Timer.log(
        "Basic Face Geometry",
        sink=logger,
        detailed=True,
    )

    # --- Process Element Geometry ---
    logging.Timer.start("Element Geometry")

    # 1. Convert element faces to a padded array.  ``compute_mesh_geometry``
    # passes CSR connectivity so this avoids a million-element list-of-lists
    # on large structured meshes while retaining generic legacy support.
    if isinstance(element_faces, tuple):
        cell_faces, cell_face_offsets = element_faces
        elem_counts = np.diff(cell_face_offsets).astype(np.int32, copy=False)
        max_faces = int(np.max(elem_counts, initial=0))
        padded_elem_faces = np.full((n_elements, max_faces), -1, dtype=np.int32)
        if len(cell_faces):
            rows = np.repeat(np.arange(n_elements), elem_counts)
            starts = np.repeat(cell_face_offsets[:-1], elem_counts)
            columns = np.arange(len(cell_faces)) - starts
            padded_elem_faces[rows, columns] = cell_faces
    else:
        max_faces = max(len(f) for f in element_faces)
        padded_elem_faces = np.full((n_elements, max_faces), -1, dtype=np.int32)
        elem_counts = np.zeros(n_elements, dtype=np.int32)
        for i, f in enumerate(element_faces):
            n_f = len(f)
            padded_elem_faces[i, :n_f] = f
            elem_counts[i] = n_f

    # 2. Get face centroids (N_elem, max_faces, 3)
    safe_elem_faces = padded_elem_faces.copy()
    safe_elem_faces[padded_elem_faces == -1] = 0

    elem_face_centroids = face_centroids[safe_elem_faces]

    # 3. Compute rough center (mean of face centroids)
    mask_elem = (padded_elem_faces != -1).astype(np.float64)[:, :, np.newaxis]
    elem_centre = np.sum(elem_face_centroids * mask_elem, axis=1) / elem_counts[:, np.newaxis]

    # 4. Vectorized Element Volume
    # localVolume = dot(local_Sf, Cf) / 3
    # local_Sf = faceSign * faceSf
    # Cf = faceCentroid - elementCentroid

    local_vol_centroid_sum = np.zeros((n_elements, 3))
    local_vol_sum = np.zeros(n_elements)

    # Gather Face Sf (N_elem, max_faces, 3)
    elem_face_sf = face_sf[safe_elem_faces]

    # Gather Owners (N_elem, max_faces) - to determine sign
    face_owners = owners[safe_elem_faces]

    # Element indices broadcasted (N_elem, max_faces)
    elem_indices = np.arange(n_elements)[:, np.newaxis]

    # Sign: 1 if elem == owner, -1 otherwise
    # Note: mask out invalid faces later
    face_signs = np.where(elem_indices == face_owners, 1.0, -1.0)

    # Cf vectors (Face Center - Element Center)
    # elem_centre expanded: (N, 1, 3)
    elem_cf = elem_face_centroids - elem_centre[:, np.newaxis, :]

    # Local Sf vectors (signed)
    local_sfs = elem_face_sf * face_signs[:, :, np.newaxis]

    # Local Volumes: dot(local_Sf, Cf) / 3
    # Dot product along axis 2
    local_volumes = np.sum(local_sfs * elem_cf, axis=2) / 3.0

    # Mask invalid faces
    valid_face_mask = (padded_elem_faces != -1).astype(np.float64)
    local_volumes *= valid_face_mask

    # Local Centroids of pyramid: 0.75 * FaceCentroid + 0.25 * ElementCentroid
    local_centroids = 0.75 * elem_face_centroids + 0.25 * elem_centre[:, np.newaxis, :]

    # Accumulate
    local_vol_sum = np.sum(local_volumes, axis=1)

    # Weighted centroid sum
    # (N, M, 3) * (N, M, 1) -> sum along axis 1 -> (N, 3)
    local_vol_centroid_sum = np.sum(local_centroids * local_volumes[:, :, np.newaxis], axis=1)

    # Finalize
    # Avoid div by zero
    safe_vol = local_vol_sum.copy()
    safe_vol[safe_vol == 0] = 1.0

    element_centroids = local_vol_centroid_sum / safe_vol[:, np.newaxis]
    element_volumes = local_vol_sum
    logging.Timer.log(
        "Element Geometry",
        sink=logger,
        detailed=True,
    )

    # --- Process Secondary Face Geometry ---
    cfd_small = 1e-15  # Matches uFVM cfdSMALL?

    # Interior faces.  These operations used to be two Python face loops and
    # dominate geometry setup for structured meshes; every expression below is
    # the same owner/neighbour formula evaluated in bulk.
    if n_interior_faces:
        interior = slice(0, n_interior_faces)
        own = owners[interior]
        nei = neighbours[interior]
        normals = face_sf[interior] / face_areas[interior, np.newaxis]
        c_own = element_centroids[own]
        c_nei = element_centroids[nei]
        c_face = face_centroids[interior]
        cf = c_face - c_own
        ff = c_face - c_nei
        face_cf_vector[interior] = c_nei - c_own
        face_cf[interior] = cf
        face_ff[interior] = ff
        cf_dot_n = np.sum(cf * normals, axis=1)
        ff_dot_n = np.sum(ff * normals, axis=1)
        denominator = cf_dot_n - ff_dot_n
        weights = np.full(n_interior_faces, 0.5, dtype=np.float64)
        np.divide(cf_dot_n, denominator, out=weights, where=np.abs(denominator) >= 1e-20)
        face_weights[interior] = weights

    # Boundary faces use a virtual neighbour at the face centroid.
    if n_faces > n_interior_faces:
        boundary = slice(n_interior_faces, n_faces)
        own = owners[boundary]
        vec = face_centroids[boundary] - element_centroids[own]
        normals = face_sf[boundary] / face_areas[boundary, np.newaxis]
        face_cf_vector[boundary] = vec
        face_cf[boundary] = vec
        face_weights[boundary] = 1.0
        distance = np.sum(vec * normals, axis=1)
        wall_dist[boundary] = np.maximum(distance, cfd_small)
        wall_dist_limited[boundary] = np.maximum(
            wall_dist[boundary], 0.05 * np.linalg.norm(vec, axis=1)
        )

    return {
        "face_centroids": face_centroids,
        "face_sf": face_sf,
        "face_areas": face_areas,
        "element_centroids": element_centroids,
        "element_volumes": element_volumes,
        "face_weights": face_weights,
        "face_cf_vector": face_cf_vector,
        "face_cf": face_cf,
        "face_ff": face_ff,
        "wall_dist": wall_dist,
        "wall_dist_limited": wall_dist_limited,
    }


def compute_mesh_geometry(mesh_data, gradient_scheme="gauss", *, compute_lsq=True, logger=None):
    """Compute cell and face geometry for a validated mesh dictionary.

    ``compute_lsq=False`` is an initialization ordering tool.  It lets the
    caller install cyclic neighbour topology before constructing LSQ stencils,
    avoiding a discarded pre-periodic LSQ pass.
    """
    # Build and retain compact cell-to-face CSR connectivity.  It is shared by
    # the typed topology and VTK exporter, and avoids repeatedly materialising
    # a Python list for every cell.
    from . import topology

    element_faces = mesh_data.get("cell_faces")
    cell_face_offsets = mesh_data.get("cell_face_offsets")
    if element_faces is None or cell_face_offsets is None:
        element_faces, cell_face_offsets = topology.build_cell_face_csr(
            mesh_data["owners"],
            mesh_data["neighbours"],
            mesh_data["n_elements"],
            mesh_data["n_faces"],
        )
        mesh_data["cell_faces"] = element_faces
        mesh_data["cell_face_offsets"] = cell_face_offsets

    geo_data = compute_geometry(
        mesh_data["points"],
        mesh_data["faces"],
        mesh_data["owners"],
        mesh_data["neighbours"],
        mesh_data["n_elements"],
        mesh_data["n_faces"],
        mesh_data["n_interior_faces"],
        (element_faces, cell_face_offsets),
        logger=logger,
    )

    # Pre‑compute LSQ gradient geometry if requested
    if gradient_scheme == "lsq" and compute_lsq:
        from ..fields.gradients import compute_lsq_geometry

        geo_data.update(compute_lsq_geometry(mesh_data, geo_data))
    else:
        geo_data["gradient_scheme"] = "gauss"

    return geo_data
