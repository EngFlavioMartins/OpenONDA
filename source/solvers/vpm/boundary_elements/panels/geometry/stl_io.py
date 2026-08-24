"""STL I/O and mesh-topology helpers for the panel solver."""

from __future__ import annotations

import importlib
import logging
import os
import struct

import numpy as np

logger = logging.getLogger("vpm")


def _compute_unit_normals(vertex_position: np.ndarray) -> np.ndarray:
    edge1 = vertex_position[:, 1, :] - vertex_position[:, 0, :]
    edge2 = vertex_position[:, 2, :] - vertex_position[:, 0, :]
    normal = np.cross(edge1, edge2)
    magnitudes = np.linalg.norm(normal, axis=1, keepdims=True)
    np.divide(normal, magnitudes, out=normal, where=magnitudes > 0.0)
    return normal


def _parse_facet_normal(line: str) -> list[float]:
    """Parse normal vector from a 'facet normal ...' STL line."""
    parts = line.split()
    if len(parts) >= 5:
        return [float(parts[-3]), float(parts[-2]), float(parts[-1])]
    return [0.0, 0.0, 0.0]


def _parse_stl_ascii_lines(handle) -> tuple[list, list]:
    """Parse lines from an open ASCII STL file handle; return (vertices_list, normals_list)."""
    vertices_list: list[list[list[float]]] = []
    normals_list: list[list[float]] = []
    current_vertices: list[list[float]] = []
    current_normal: list[float] | None = None

    for raw_line in handle:
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("facet normal"):
            current_normal = _parse_facet_normal(line)
            current_vertices = []
            continue
        if lower.startswith("vertex"):
            parts = line.split()
            if len(parts) < 4:
                continue
            current_vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            continue
        if lower.startswith("endfacet") and len(current_vertices) == 3:
            vertices_list.append(current_vertices)
            normals_list.append(current_normal if current_normal is not None else [0.0, 0.0, 0.0])
            current_vertices = []
            current_normal = None

    return vertices_list, normals_list


def _try_load_stl_with_library(filepath: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Try to load STL using numpy-stl. Returns None if library is unavailable."""
    try:
        stl_mesh = importlib.import_module("stl.mesh")
        mesh = stl_mesh.Mesh.from_file(filepath)
        vertex_position = np.asarray(mesh.vectors, dtype=np.float64)
        normal = np.asarray(mesh.normals, dtype=np.float64)
        if vertex_position.ndim != 3 or vertex_position.shape[1:] != (3, 3):
            raise ValueError(f"Unexpected STL triangle array shape: {vertex_position.shape}")
        zero_normals = np.linalg.norm(normal, axis=1) <= 0.0
        if np.any(zero_normals):
            normal = normal.copy()
            normal[zero_normals] = _compute_unit_normals(vertex_position[zero_normals])
        logger.debug(f"Loaded STL '{filepath}' using numpy-stl: {vertex_position.shape[0]} panels.")
        return vertex_position, normal
    except ImportError:
        logger.debug("numpy-stl is unavailable; using manual STL parser.")
        return None


def _detect_and_load_binary_stl(filepath: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Try to detect and load a binary STL file. Returns None if the file is not binary."""
    with open(filepath, "rb") as handle:
        header = handle.read(80)
        count_bytes = handle.read(4)
    if len(count_bytes) != 4:
        return None
    tri_count = struct.unpack("<I", count_bytes)[0]
    expected_size = 84 + tri_count * 50
    actual_size = os.path.getsize(filepath)
    if actual_size != expected_size:
        return None
    starts_with_solid = header[:5].lower() == b"solid"
    if starts_with_solid:
        logger.debug("Detected binary STL with 'solid' header.")
    return _load_stl_binary(filepath)


def _load_stl_ascii(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    with open(filepath, encoding="utf-8", errors="ignore") as handle:
        vertices_list, normals_list = _parse_stl_ascii_lines(handle)

    if not vertices_list:
        raise ValueError(f"No triangular facets found in ASCII STL: {filepath}")

    vertex_position = np.asarray(vertices_list, dtype=np.float64)
    normal = np.asarray(normals_list, dtype=np.float64)

    degenerate = np.linalg.norm(normal, axis=1) <= 0.0
    if np.any(degenerate):
        normal[degenerate] = _compute_unit_normals(vertex_position[degenerate])

    logger.debug(f"Loaded ASCII STL '{filepath}': {vertex_position.shape[0]} panels.")
    return vertex_position, normal


def load_stl(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Load an STL file and return triangle vertex_position and unit normal.

    Attempts to use the optional ``numpy-stl`` library first, then falls back
    to manual binary or ASCII STL parsing.

    Parameters
    ----------
    filepath : str
        Path to the STL file (ASCII or binary format).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``vertex_position`` with shape ``(N, 3, 3)``, ``normal`` with shape ``(N, 3)``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.

    Examples
    --------
    >>> vertex_position, normal = load_stl("body.stl")
    >>> vertex_position.shape
    (N, 3, 3)
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"STL file not found: {filepath}")

    result = _try_load_stl_with_library(filepath)
    if result is not None:
        return result

    result = _detect_and_load_binary_stl(filepath)
    if result is not None:
        return result

    return _load_stl_ascii(filepath)


def _load_stl_binary(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Load binary STL file and return ``(vertex_position, normal)``."""
    with open(filepath, "rb") as handle:
        _ = handle.read(80)
        count_bytes = handle.read(4)
        if len(count_bytes) != 4:
            raise ValueError(f"Invalid binary STL header in '{filepath}'")

        tri_count = struct.unpack("<I", count_bytes)[0]
        expected_size = 84 + tri_count * 50
        actual_size = os.path.getsize(filepath)
        if actual_size < expected_size:
            raise ValueError(
                f"Binary STL appears truncated: expected at least {expected_size} bytes, got {actual_size}"
            )

        normal = np.empty((tri_count, 3), dtype=np.float64)
        vertex_position = np.empty((tri_count, 3, 3), dtype=np.float64)

        for index in range(tri_count):
            record = handle.read(50)
            if len(record) != 50:
                raise ValueError(f"Unexpected EOF while reading triangle {index} in '{filepath}'")
            values = struct.unpack("<12fH", record)
            normal[index, :] = values[0:3]
            vertex_position[index, 0, :] = values[3:6]
            vertex_position[index, 1, :] = values[6:9]
            vertex_position[index, 2, :] = values[9:12]

    degenerate = np.linalg.norm(normal, axis=1) <= 0.0
    if np.any(degenerate):
        normal[degenerate] = _compute_unit_normals(vertex_position[degenerate])

    logger.debug(f"Loaded binary STL '{filepath}': {tri_count} panels.")
    return vertex_position, normal


def save_stl(filepath: str, vertex_position: np.ndarray, normal: np.ndarray | None = None) -> None:
    """Save triangular panels to a binary STL file.

    Uses the optional ``numpy-stl`` library when available; otherwise writes
    a binary STL file directly.

    Parameters
    ----------
    filepath : str
        Target STL path.
    vertex_position : np.ndarray
        Triangle vertex_position with shape ``(N, 3, 3)``.
    normal : np.ndarray | None
        Optional unit normal with shape ``(N, 3)``. Computed automatically
        when ``None``.

    Raises
    ------
    ValueError
        If ``vertex_position`` or ``normal`` have invalid shapes.

    Examples
    --------
    >>> save_stl("output.stl", vertex_position, normal)
    """
    triangles = np.asarray(vertex_position, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("vertex_position must have shape (N, 3, 3)")

    panel_count = triangles.shape[0]
    if normal is None:
        normals_arr = _compute_unit_normals(triangles)
    else:
        normals_arr = np.asarray(normal, dtype=np.float64)
        if normals_arr.shape != (panel_count, 3):
            raise ValueError("normal must have shape (N, 3)")

        magnitudes = np.linalg.norm(normals_arr, axis=1, keepdims=True)
        np.divide(normals_arr, magnitudes, out=normals_arr, where=magnitudes > 0.0)

    try:
        stl_mesh = importlib.import_module("stl.mesh")
        mesh = stl_mesh.Mesh(np.zeros(panel_count, dtype=stl_mesh.Mesh.dtype))
        mesh.vectors = triangles.astype(np.float32)
        mesh.normals = normals_arr.astype(np.float32)
        mesh.save(filepath)
        logger.debug(f"Saved STL '{filepath}' using numpy-stl: {panel_count} panels.")
        return
    except ImportError:
        logger.debug("numpy-stl is unavailable; using manual binary STL writer.")

    _save_stl_binary(filepath, triangles, normals_arr)


def _save_stl_binary(filepath: str, vertex_position: np.ndarray, normal: np.ndarray) -> None:
    """Write binary STL file from triangle vertex_position and normal."""
    triangles = np.asarray(vertex_position, dtype=np.float64)
    normal_arr = np.asarray(normal, dtype=np.float64)

    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("vertex_position must have shape (N, 3, 3)")
    if normal_arr.shape != (triangles.shape[0], 3):
        raise ValueError("normal must have shape (N, 3)")

    panel_count = int(triangles.shape[0])
    header_text = b"OpenONDA binary STL"
    header = (header_text + b" " * 80)[:80]

    with open(filepath, "wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<I", panel_count))

        triangles32 = triangles.astype(np.float32, copy=False)
        normals32 = normal_arr.astype(np.float32, copy=False)
        for index in range(panel_count):
            packed = struct.pack(
                "<12fH",
                normals32[index, 0],
                normals32[index, 1],
                normals32[index, 2],
                triangles32[index, 0, 0],
                triangles32[index, 0, 1],
                triangles32[index, 0, 2],
                triangles32[index, 1, 0],
                triangles32[index, 1, 1],
                triangles32[index, 1, 2],
                triangles32[index, 2, 0],
                triangles32[index, 2, 1],
                triangles32[index, 2, 2],
                0,
            )
            handle.write(packed)

    logger.debug(f"Saved binary STL '{filepath}': {panel_count} panels.")


def _unique_face_vertex_ids_from_triangles(triangles: np.ndarray, decimals: int = 12) -> np.ndarray:
    rounded = np.round(triangles, decimals=decimals)
    key_to_index: dict[tuple[float, float, float], int] = {}
    face_vertex_ids = np.empty((triangles.shape[0], 3), dtype=np.int64)
    next_index = 0

    for panel in range(triangles.shape[0]):
        for local in range(3):
            key = tuple(float(value) for value in rounded[panel, local, :])
            if key not in key_to_index:
                key_to_index[key] = next_index
                next_index += 1
            face_vertex_ids[panel, local] = key_to_index[key]
    return face_vertex_ids


def _add_neighbor(neighbors: np.ndarray, panel: int, candidate: int, max_neighbors: int) -> None:
    row = neighbors[panel]
    for slot in range(max_neighbors):
        if row[slot] == candidate:
            return
        if row[slot] == -1:
            row[slot] = candidate
            return
    logger.warning(
        f"Panel {panel} exceeded max_neighbors={max_neighbors}; neighbor {candidate} was dropped."
    )


def _mark_leading_edge(te_mask: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    panel_count = triangles.shape[0]
    if panel_count == 0:
        return np.zeros(0, dtype=np.int32)

    panel_centre = triangles.mean(axis=1)
    x_centre = panel_centre[:, 0]
    x_min = float(np.min(x_centre))
    x_max = float(np.max(x_centre))
    span = x_max - x_min

    if span <= 0.0:
        le_mask = np.zeros(panel_count, dtype=np.int32)
        if np.any(~te_mask.astype(bool)):
            le_mask[int(np.argmin(x_centre))] = 1
        return le_mask

    threshold = x_min + 0.05 * span
    le_mask = (x_centre <= threshold).astype(np.int32)

    non_te = te_mask == 0
    if np.any(non_te):
        le_mask = (le_mask & non_te.astype(np.int32)).astype(np.int32)
        if np.sum(le_mask) == 0:
            idx = int(np.argmin(np.where(non_te, x_centre, np.inf)))
            if np.isfinite(x_centre[idx]):
                le_mask[idx] = 1
    return le_mask


def _validate_face_input(vertices_arr: np.ndarray, faces_arr: np.ndarray) -> None:
    """Validate vertex/face arrays when faces are provided explicitly."""
    if vertices_arr.ndim != 2 or vertices_arr.shape[1] != 3:
        raise ValueError("When faces is provided, vertex_position must have shape (V, 3)")
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
        raise ValueError("faces must have shape (N, 3)")
    if np.any(faces_arr < 0) or np.any(faces_arr >= vertices_arr.shape[0]):
        raise ValueError("faces contain out-of-bounds vertex indices")


def _build_panel_edge_map(panel_count: int, panel_vertex_ids: np.ndarray) -> dict:
    """Build edge→panel-list adjacency map for a triangular mesh."""
    edge_map: dict[tuple[int, int], list[int]] = {}
    for panel in range(panel_count):
        face = panel_vertex_ids[panel]
        edges = (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        )
        for a, b in edges:
            edge = (a, b) if a < b else (b, a)
            edge_map.setdefault(edge, []).append(panel)
    return edge_map


def _populate_neighbors_from_edges(
    edge_map: dict, neighbors: np.ndarray, is_te_panel: np.ndarray, max_neighbors: int
) -> None:
    """Fill neighbor arrays and trailing-edge mask from the edge map."""
    for panels in edge_map.values():
        panel_refs = list(dict.fromkeys(panels))
        if len(panel_refs) == 1:
            is_te_panel[panel_refs[0]] = 1
            continue
        if len(panel_refs) > 2:
            logger.warning(f"Non-manifold edge shared by {len(panel_refs)} panels.")
        for panel in panel_refs:
            for nbr in panel_refs:
                if panel != nbr:
                    _add_neighbor(neighbors, panel, nbr, max_neighbors)


def build_mesh_topology(
    vertex_position: np.ndarray,
    faces: np.ndarray | None = None,
    max_neighbors: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build panel adjacency and boundary masks for triangular panel meshes.

    Identifies panel neighbours by shared edges, marks trailing-edge (open
    boundary) panels, and marks leading-edge panels based on the x-coordinate
    of each triangle centre.

    Parameters
    ----------
    vertex_position : np.ndarray
        If ``faces is None``: triangle coordinates with shape ``(N, 3, 3)``.
        If ``faces is not None``: unique vertex coordinates with shape ``(V, 3)``.
    faces : np.ndarray | None
        Optional connectivity with shape ``(N, 3)``.
    max_neighbors : int
        Number of neighbor slots per panel (default ``8``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``neighbor_indices`` shape ``(N, max_neighbors)`` with ``-1`` padding,
        ``is_TE_panel`` shape ``(N,)`` int32,
        ``is_LE_panel`` shape ``(N,)`` int32.

    Raises
    ------
    ValueError
        If ``max_neighbors`` is not positive, or if vertex/face shapes are
        invalid.

    Examples
    --------
    >>> neighbors, te_mask, le_mask = build_mesh_topology(vertex_position)
    >>> neighbors.shape
    (N, 8)
    """
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive")

    vertices_arr = np.asarray(vertex_position, dtype=np.float64)

    if faces is not None:
        faces_arr = np.asarray(faces, dtype=np.int64)
        _validate_face_input(vertices_arr, faces_arr)
        triangles = vertices_arr[faces_arr]
        panel_vertex_ids = faces_arr
    else:
        if vertices_arr.ndim != 3 or vertices_arr.shape[1:] != (3, 3):
            raise ValueError("vertex_position must have shape (N, 3, 3) when faces is None")
        triangles = vertices_arr
        panel_vertex_ids = _unique_face_vertex_ids_from_triangles(triangles)

    panel_count = int(triangles.shape[0])
    neighbors = np.full((panel_count, max_neighbors), -1, dtype=np.int32)
    is_te_panel = np.zeros(panel_count, dtype=np.int32)

    edge_map = _build_panel_edge_map(panel_count, panel_vertex_ids)
    _populate_neighbors_from_edges(edge_map, neighbors, is_te_panel, max_neighbors)

    is_le_panel = _mark_leading_edge(is_te_panel, triangles).astype(np.int32)
    return neighbors, is_te_panel, is_le_panel
