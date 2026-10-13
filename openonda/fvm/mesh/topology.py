"""Typed cell/face topology and connectivity helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openonda.core._numba import cacheable_njit as njit


def _readonly(values: np.ndarray, dtype: type | np.dtype | None = None) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=dtype).view()
    result.setflags(write=False)
    return result


@njit(cache=True)
def _fill_cell_face_indices(
    owners: np.ndarray, neighbours: np.ndarray, offsets: np.ndarray, n_interior: int, n_faces: int
) -> np.ndarray:
    """Fill cell-face CSR in one pass without an argsort-sized workspace."""
    result = np.empty(offsets[-1], dtype=np.int32)
    cursor = offsets[:-1].copy()
    for face in range(n_faces):
        owner = owners[face]
        result[cursor[owner]] = face
        cursor[owner] += 1
        if face < n_interior:
            neighbour = neighbours[face]
            result[cursor[neighbour]] = face
            cursor[neighbour] += 1
    return result


def build_cell_face_csr(
    owners: np.ndarray, neighbours: np.ndarray, n_cells: int, n_faces: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-to-face connectivity as compact CSR arrays.

    CSR keeps the topology in two numeric arrays, instead of one Python list
    per cell, and is directly usable by geometry and VTK code.
    """
    owners = np.ascontiguousarray(owners, dtype=np.int32)
    neighbours = np.ascontiguousarray(neighbours, dtype=np.int32)
    n_interior = len(neighbours)
    counts = np.bincount(owners, minlength=n_cells)
    counts += np.bincount(neighbours, minlength=n_cells)
    if int(counts.sum()) > np.iinfo(np.int32).max:
        raise OverflowError("FVM cell-face topology exceeds 32-bit addressing")
    offsets = np.empty(n_cells + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return _fill_cell_face_indices(owners, neighbours, offsets, n_interior, n_faces), offsets


def pack_face_nodes(faces: list[np.ndarray] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return contiguous polygon connectivity without padding ragged faces."""
    if isinstance(faces, np.ndarray) and faces.ndim == 2:
        nodes = np.ascontiguousarray(faces, dtype=np.int64).reshape(-1)
        offsets = np.arange(len(faces) + 1, dtype=np.int64) * faces.shape[1]
    else:
        offsets = np.empty(len(faces) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(np.fromiter(map(len, faces), dtype=np.int64, count=len(faces)), out=offsets[1:])
        nodes = (
            np.concatenate(faces).astype(np.int64, copy=False)
            if len(faces)
            else np.empty(0, dtype=np.int64)
        )
    return nodes, offsets


@dataclass(frozen=True)
class BoundaryPatch:
    """Fixed description of one mesh patch, independent of operator state.

    Attributes
    ----------
    name : str
        User-assigned patch name (e.g. ``"inlet"``).
    start_face : int
        Index of the first face belonging to this patch.
    n_faces : int
        Number of faces in the patch.
    physical_tag : int or None
        GMSH physical group tag, if the mesh came from GMSH.
    """

    name: str
    start_face: int
    n_faces: int
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
    cell_face_indices : np.ndarray
        Flattened face indices for each cell.
    cell_face_offset : np.ndarray
        Offsets into ``cell_face_indices`` for each cell.
    patches : tuple[BoundaryPatch, ...]
        Boundary patches of the mesh.
    global_cell_id : np.ndarray
        Globally unique cell identifiers.
    global_face_id : np.ndarray
        Globally unique face identifiers.
    source_cell_id : np.ndarray
        Original cell numbering from the mesh file.
    cell_type_code : np.ndarray
        VTK cell-type codes for export.
    cell_order : np.ndarray
        VTK cell-order arrays for higher-order elements.
    """

    face_nodes: np.ndarray
    face_node_offsets: np.ndarray
    owners: np.ndarray
    neighbours: np.ndarray
    cell_face_indices: np.ndarray
    cell_face_offset: np.ndarray
    patches: tuple[BoundaryPatch, ...]
    global_cell_id: np.ndarray
    global_face_id: np.ndarray
    source_cell_id: np.ndarray
    cell_type_code: np.ndarray
    cell_order: np.ndarray

    @classmethod
    def from_mesh_data(cls, mesh_data: dict[str, object]) -> MeshTopology:
        """Convert native mesh mappings into read-only CSR-style array views.

        Already-contiguous arrays share memory with ``mesh_data`` without
        making the source arrays read-only.
        """
        faces = mesh_data["faces"]
        face_array = faces if isinstance(faces, np.ndarray) else None
        if face_array is not None and face_array.ndim == 2:
            face_nodes = np.ascontiguousarray(face_array.ravel(), dtype=np.int32)
            nodes_per_face = face_array.shape[1]
            if len(face_array) * nodes_per_face > np.iinfo(np.int32).max:
                raise OverflowError("FVM face-node topology exceeds 32-bit addressing")
            face_node_offsets = np.arange(len(face_array) + 1, dtype=np.int32) * nodes_per_face
        else:
            face_node_offsets = np.zeros(len(faces) + 1, dtype=np.int32)
            for index, face in enumerate(faces):
                next_offset = int(face_node_offsets[index]) + len(face)
                if next_offset > np.iinfo(np.int32).max:
                    raise OverflowError("FVM face-node topology exceeds 32-bit addressing")
                face_node_offsets[index + 1] = next_offset
            face_nodes = np.concatenate(faces).astype(np.int32, copy=False)

        flattened_cell_face_indices = mesh_data.get("cell_face_indices")
        cell_face_offset = mesh_data.get("cell_face_offset")
        if flattened_cell_face_indices is None or cell_face_offset is None:
            flattened_cell_face_indices, cell_face_offset = build_cell_face_csr(
                mesh_data["owners"],
                mesh_data["neighbours"],
                mesh_data["n_cells"],
                mesh_data["n_faces"],
            )
        patches = tuple(
            BoundaryPatch(
                name=str(patch["name"]),
                start_face=int(patch["start_face"]),
                n_faces=int(patch["n_faces"]),
                physical_tag=patch.get("physical_tag"),
            )
            for patch in mesh_data["boundary"]
        )
        n_cells = int(mesh_data["n_cells"])
        n_faces = int(mesh_data["n_faces"])
        return cls(
            face_nodes=_readonly(face_nodes, np.int32),
            face_node_offsets=_readonly(face_node_offsets, np.int32),
            owners=_readonly(mesh_data["owners"], np.int32),
            neighbours=_readonly(mesh_data["neighbours"], np.int32),
            cell_face_indices=_readonly(flattened_cell_face_indices, np.int32),
            cell_face_offset=_readonly(cell_face_offset, np.int32),
            patches=patches,
            global_cell_id=_readonly(mesh_data.get("global_cell_id", np.arange(n_cells)), np.int64),
            global_face_id=_readonly(mesh_data.get("global_face_id", np.arange(n_faces)), np.int64),
            source_cell_id=_readonly(mesh_data.get("source_cell_id", np.arange(n_cells)), np.int64),
            cell_type_code=_readonly(
                mesh_data.get("cell_type_code", np.full(n_cells, -1)), np.int32
            ),
            cell_order=_readonly(mesh_data.get("cell_order", np.ones(n_cells)), np.int8),
        )
