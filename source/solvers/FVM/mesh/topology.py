"""Typed cell/face topology and connectivity helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _readonly(values, dtype=None):
    result = np.ascontiguousarray(values, dtype=dtype).view()
    result.setflags(write=False)
    return result


def build_cell_face_csr(owners, neighbours, n_elements, n_faces):
    """Return cell-to-face connectivity as compact CSR arrays.

    This replaces the former list-of-lists construction.  On the cubeFlow
    reference mesh that list alone created more than a million Python lists;
    CSR keeps the same topology in two numeric arrays and is also directly
    usable by geometry and VTK code.
    """
    owners = np.asarray(owners)
    neighbours = np.asarray(neighbours)
    n_interior = len(neighbours)
    face_ids = np.arange(n_faces, dtype=np.int32)
    cells = np.concatenate(
        (
            owners[:n_interior],
            neighbours,
            owners[n_interior:],
        )
    )
    faces = np.concatenate(
        (
            face_ids[:n_interior],
            face_ids[:n_interior],
            face_ids[n_interior:],
        )
    )
    order = np.argsort(cells, kind="stable")
    counts = np.bincount(cells, minlength=n_elements)
    offsets = np.empty(n_elements + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return np.ascontiguousarray(faces[order], dtype=np.int32), offsets


@dataclass(frozen=True)
class BoundaryPatch:
    """Stable identity of one mesh patch, independent of operator state.

    Attributes
    ----------
    name : str
        User-assigned patch name (e.g. ``"inlet"``).
    start_face : int
        Index of the first face belonging to this patch.
    n_faces : int
        Number of faces in the patch.
    source_type : str or None
        Original boundary type as read from the mesh file.
    physical_tag : int or None
        GMSH physical group tag, if the mesh came from GMSH.
    """

    name: str
    start_face: int
    n_faces: int
    source_type: str | None = None
    physical_tag: int | None = None


@dataclass(frozen=True)
class MeshTopology:
    """Immutable cell-face topology used at backend boundaries.

    Stores the CSR cell-to-face connectivity, owner/neighbour arrays, and
    the list of boundary patches.  All arrays are read-only views into
    already-contiguous storage; this object is intended to be constructed
    once and shared across all time steps.

    Attributes
    ----------
    face_nodes : np.ndarray
        Flattened node indices for each face.
    face_node_offsets : np.ndarray
        Offsets into ``face_nodes`` for each face (polygonal faces).
    owners : np.ndarray
        Owner cell for every face.
    neighbours : np.ndarray
        Neighbour cell for interior faces (``-1`` for boundary faces).
    cell_faces : np.ndarray
        Flattened face indices for each cell.
    cell_face_offsets : np.ndarray
        Offsets into ``cell_faces`` for each cell.
    patches : tuple[BoundaryPatch, ...]
        Boundary patches of the mesh.
    global_cell_ids : np.ndarray
        Globally unique cell identifiers.
    global_face_ids : np.ndarray
        Globally unique face identifiers.
    source_cell_ids : np.ndarray
        Original cell numbering from the mesh file.
    cell_type_codes : np.ndarray
        VTK cell-type codes for export.
    cell_orders : np.ndarray
        VTK cell-order arrays for higher-order elements.
    topology_version : int
        Monotonically increasing version counter for cache invalidation.
    """

    face_nodes: np.ndarray
    face_node_offsets: np.ndarray
    owners: np.ndarray
    neighbours: np.ndarray
    cell_faces: np.ndarray
    cell_face_offsets: np.ndarray
    patches: tuple[BoundaryPatch, ...]
    global_cell_ids: np.ndarray
    global_face_ids: np.ndarray
    source_cell_ids: np.ndarray
    cell_type_codes: np.ndarray
    cell_orders: np.ndarray
    topology_version: int = 1

    @classmethod
    def from_mesh_data(cls, mesh_data) -> MeshTopology:
        """Normalize legacy mesh data into read-only CSR-style array views.

        Already-contiguous arrays share memory with ``mesh_data`` without
        making the source arrays read-only.
        """
        faces = mesh_data["faces"]
        face_array = faces if isinstance(faces, np.ndarray) else None
        if face_array is not None and face_array.ndim == 2:
            face_nodes = np.ascontiguousarray(face_array.ravel(), dtype=np.int32)
            nodes_per_face = face_array.shape[1]
            face_node_offsets = np.arange(len(face_array) + 1, dtype=np.int64) * nodes_per_face
        else:
            face_node_offsets = np.zeros(len(faces) + 1, dtype=np.int64)
            for index, face in enumerate(faces):
                face_node_offsets[index + 1] = face_node_offsets[index] + len(face)
            face_nodes = np.concatenate(faces).astype(np.int32, copy=False)

        flattened_cell_faces = mesh_data.get("cell_faces")
        cell_face_offsets = mesh_data.get("cell_face_offsets")
        if flattened_cell_faces is None or cell_face_offsets is None:
            flattened_cell_faces, cell_face_offsets = build_cell_face_csr(
                mesh_data["owners"],
                mesh_data["neighbours"],
                mesh_data["n_elements"],
                mesh_data["n_faces"],
            )
        patches = tuple(
            BoundaryPatch(
                name=str(patch["name"]),
                start_face=int(patch["startFace"]),
                n_faces=int(patch["nFaces"]),
                source_type=patch.get("type"),
                physical_tag=patch.get("physical_tag"),
            )
            for patch in mesh_data["boundary"]
        )
        n_cells = int(mesh_data["n_elements"])
        n_faces = int(mesh_data["n_faces"])
        return cls(
            face_nodes=_readonly(face_nodes, np.int32),
            face_node_offsets=_readonly(face_node_offsets, np.int64),
            owners=_readonly(mesh_data["owners"], np.int32),
            neighbours=_readonly(mesh_data["neighbours"], np.int32),
            cell_faces=_readonly(flattened_cell_faces, np.int32),
            cell_face_offsets=_readonly(cell_face_offsets, np.int64),
            patches=patches,
            global_cell_ids=_readonly(
                mesh_data.get("global_cell_ids", np.arange(n_cells)), np.int64
            ),
            global_face_ids=_readonly(
                mesh_data.get("global_face_ids", np.arange(n_faces)), np.int64
            ),
            source_cell_ids=_readonly(
                mesh_data.get("source_cell_ids", np.arange(n_cells)), np.int64
            ),
            cell_type_codes=_readonly(
                mesh_data.get("cell_type_codes", np.full(n_cells, -1)), np.int32
            ),
            cell_orders=_readonly(mesh_data.get("cell_orders", np.ones(n_cells)), np.int8),
        )


def get_element_faces(owners, neighbours, n_elements, n_faces):
    """Build the face-index list for every cell.

    ``neighbours`` contains one entry per interior face; remaining faces are
    boundary faces and belong only to their owner cell.
    """
    n_interior_faces = len(neighbours)
    element_faces: list[list[int]] = [[] for _ in range(n_elements)]

    for face_index in range(n_interior_faces):
        owner = owners[face_index]
        neighbour = neighbours[face_index]
        element_faces[owner].append(face_index)
        element_faces[neighbour].append(face_index)

    for face_index in range(n_interior_faces, n_faces):
        element_faces[owners[face_index]].append(face_index)

    return element_faces
