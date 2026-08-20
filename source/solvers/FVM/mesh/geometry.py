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
            cell_centroids=_readonly(geo_data["cell_centroids"]),
            cell_volumes=_readonly(geo_data["cell_volumes"]),
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
    n_cells,
    n_faces,
    n_interior_faces,
    cell_faces,
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
            - face_cf_vector
            - wall_dist
    """

    # --- Initialize Arrays ---
    face_centroids = np.zeros((n_faces, 3), dtype=np.float64)
    face_sf = np.zeros((n_faces, 3), dtype=np.float64)
    face_areas = np.zeros(n_faces, dtype=np.float64)

    cell_centroids = np.zeros((n_cells, 3), dtype=np.float64)
    cell_volumes = np.zeros(n_cells, dtype=np.float64)

    face_weights = np.zeros(n_faces, dtype=np.float64)
    face_cf_vector = np.zeros((n_faces, 3), dtype=np.float64)  # faceCF in uFVM (owner to neighbour)
    wall_dist = np.zeros(n_faces, dtype=np.float64)

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

    # Vectorizing every face at once creates several ``(n_faces, n_nodes, 3)``
    # temporaries.  At reference-mesh scale their simultaneous peak is larger
    # than the final geometry itself.  Work in bounded blocks while retaining
    # the same fan-triangulation arithmetic.
    points_array = np.asarray(points, dtype=np.float64)
    face_chunk = 100_000
    for start in range(0, n_faces, face_chunk):
        stop = min(start + face_chunk, n_faces)
        face_block = padded_faces[start:stop]
        count_block = counts[start:stop]
        safe_faces = face_block.copy()
        safe_faces[safe_faces < 0] = 0
        face_coords = points_array[safe_faces]
        valid_nodes = face_block >= 0
        local_centre = np.sum(face_coords * valid_nodes[:, :, np.newaxis], axis=1)
        local_centre /= count_block[:, np.newaxis]

        centroid_sum = np.zeros((stop - start, 3), dtype=np.float64)
        sf_sum = np.zeros((stop - start, 3), dtype=np.float64)
        area_sum = np.zeros(stop - start, dtype=np.float64)
        for index in range(max_nodes):
            point1 = face_coords[:, index, :]
            is_last = index == count_block - 1
            point2_raw = (
                face_coords[:, index + 1, :] if index + 1 < max_nodes else face_coords[:, 0, :]
            )
            point2 = np.where(is_last[:, np.newaxis], face_coords[:, 0, :], point2_raw)
            valid_triangle = (index < count_block)[:, np.newaxis]
            local_sf = 0.5 * np.cross(point1 - local_centre, point2 - local_centre)
            local_area = np.linalg.norm(local_sf, axis=1)
            centroid_sum += (
                ((local_centre + point1 + point2) / 3.0)
                * local_area[:, np.newaxis]
                * valid_triangle
            )
            sf_sum += local_sf * valid_triangle
            area_sum += local_area * valid_triangle[:, 0]

        safe_area = np.where(area_sum == 0.0, 1.0, area_sum)
        face_centroids[start:stop] = centroid_sum / safe_area[:, np.newaxis]
        face_sf[start:stop] = sf_sum
        face_areas[start:stop] = area_sum

    logging.Timer.log(
        "Basic Face Geometry",
        sink=logger,
    )

    # --- Process Element Geometry ---
    logging.Timer.start("Element Geometry")

    # ``compute_mesh_geometry`` normally passes CSR connectivity.  Only the
    # counts persist globally; padded connectivity and all gathered face
    # tensors are bounded to one block.
    cell_faces_are_csr = isinstance(cell_faces, tuple)
    if cell_faces_are_csr:
        cell_faces, cell_face_offsets = cell_faces
        cell_counts = np.diff(cell_face_offsets).astype(np.int32, copy=False)
        max_faces = int(np.max(cell_counts, initial=0))
    else:
        max_faces = max(len(f) for f in cell_faces)
        cell_counts = np.fromiter((len(f) for f in cell_faces), dtype=np.int32)

    cell_chunk = 50_000
    for start in range(0, n_cells, cell_chunk):
        stop = min(start + cell_chunk, n_cells)
        count_block = cell_counts[start:stop]
        padded = np.full((stop - start, max_faces), -1, dtype=np.int32)
        if cell_faces_are_csr:
            first_entry = int(cell_face_offsets[start])
            last_entry = int(cell_face_offsets[stop])
            if last_entry > first_entry:
                rows = np.repeat(np.arange(stop - start, dtype=np.int32), count_block)
                relative_offsets = cell_face_offsets[start:stop] - first_entry
                columns = np.arange(last_entry - first_entry) - np.repeat(
                    relative_offsets, count_block
                )
                padded[rows, columns] = cell_faces[first_entry:last_entry]
        else:
            for local, faces_for_cell in enumerate(cell_faces[start:stop]):
                padded[local, : len(faces_for_cell)] = faces_for_cell

        safe_faces = padded.copy()
        safe_faces[safe_faces < 0] = 0
        valid = padded >= 0
        cell_face_centroids = face_centroids[safe_faces]
        cell_centre = (
            np.sum(cell_face_centroids * valid[:, :, np.newaxis], axis=1)
            / count_block[:, np.newaxis]
        )
        cell_face_sf = face_sf[safe_faces]
        face_owners = owners[safe_faces]
        cell_ids = np.arange(start, stop)[:, np.newaxis]
        signs = np.where(cell_ids == face_owners, 1.0, -1.0)
        local_volumes = (
            np.sum(
                cell_face_sf
                * signs[:, :, np.newaxis]
                * (cell_face_centroids - cell_centre[:, np.newaxis, :]),
                axis=2,
            )
            / 3.0
        )
        local_volumes *= valid
        local_sum = np.sum(local_volumes, axis=1)
        local_centroids = 0.75 * cell_face_centroids + 0.25 * cell_centre[:, np.newaxis, :]
        weighted = np.sum(local_centroids * local_volumes[:, :, np.newaxis], axis=1)
        safe_volume = np.where(local_sum == 0.0, 1.0, local_sum)
        cell_centroids[start:stop] = weighted / safe_volume[:, np.newaxis]
        cell_volumes[start:stop] = local_sum
    logging.Timer.log(
        "Element Geometry",
        sink=logger,
    )

    # --- Process Secondary Face Geometry ---
    cfd_small = 1e-15  # Matches uFVM cfdSMALL?

    # Interior faces.  These operations used to be two Python face loops and
    # dominate geometry setup for structured meshes; every expression below is
    # the same owner/neighbour formula evaluated in bulk.
    secondary_chunk = 200_000
    for start in range(0, n_interior_faces, secondary_chunk):
        stop = min(start + secondary_chunk, n_interior_faces)
        face_slice = slice(start, stop)
        own = owners[face_slice]
        nei = neighbours[face_slice]
        normals = face_sf[face_slice] / face_areas[face_slice, np.newaxis]
        owner_centres = cell_centroids[own]
        neighbour_centres = cell_centroids[nei]
        owner_to_face = face_centroids[face_slice] - owner_centres
        neighbour_to_face = face_centroids[face_slice] - neighbour_centres
        face_cf_vector[face_slice] = neighbour_centres - owner_centres
        owner_normal = np.sum(owner_to_face * normals, axis=1)
        neighbour_normal = np.sum(neighbour_to_face * normals, axis=1)
        denominator = owner_normal - neighbour_normal
        weights = np.full(stop - start, 0.5, dtype=np.float64)
        np.divide(
            owner_normal,
            denominator,
            out=weights,
            where=np.abs(denominator) >= 1e-20,
        )
        face_weights[face_slice] = weights

    # Boundary faces use a virtual neighbour at the face centroid.
    for start in range(n_interior_faces, n_faces, secondary_chunk):
        stop = min(start + secondary_chunk, n_faces)
        face_slice = slice(start, stop)
        own = owners[face_slice]
        vector = face_centroids[face_slice] - cell_centroids[own]
        normals = face_sf[face_slice] / face_areas[face_slice, np.newaxis]
        face_cf_vector[face_slice] = vector
        face_weights[face_slice] = 1.0
        wall_dist[face_slice] = np.maximum(np.sum(vector * normals, axis=1), cfd_small)

    return {
        "face_centroids": face_centroids,
        "face_sf": face_sf,
        "face_areas": face_areas,
        "cell_centroids": cell_centroids,
        "cell_volumes": cell_volumes,
        "face_weights": face_weights,
        "face_cf_vector": face_cf_vector,
        "wall_dist": wall_dist,
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

    cell_faces = mesh_data.get("cell_faces")
    cell_face_offsets = mesh_data.get("cell_face_offsets")
    if cell_faces is None or cell_face_offsets is None:
        cell_faces, cell_face_offsets = topology.build_cell_face_csr(
            mesh_data["owners"],
            mesh_data["neighbours"],
            mesh_data["n_cells"],
            mesh_data["n_faces"],
        )
        mesh_data["cell_faces"] = cell_faces
        mesh_data["cell_face_offsets"] = cell_face_offsets

    geo_data = compute_geometry(
        mesh_data["points"],
        mesh_data["faces"],
        mesh_data["owners"],
        mesh_data["neighbours"],
        mesh_data["n_cells"],
        mesh_data["n_faces"],
        mesh_data["n_interior_faces"],
        (cell_faces, cell_face_offsets),
        logger=logger,
    )

    # Pre‑compute LSQ gradient geometry if requested
    if gradient_scheme == "lsq" and compute_lsq:
        from ..fields.gradients import compute_lsq_geometry

        geo_data.update(compute_lsq_geometry(mesh_data, geo_data))
    else:
        geo_data["gradient_scheme"] = "gauss"

    return geo_data
