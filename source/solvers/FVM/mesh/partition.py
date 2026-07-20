"""Deterministic cell partition metadata and halo exchange."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np


def _visualization_mesh(mesh_data, cell_ids: np.ndarray) -> dict:
    """Build complete, compact cells for owned-plus-ghost visualization.

    The solver's localized mesh contains exactly the faces needed to advance
    owned cells.  That is intentionally insufficient to draw complete halo
    cells.  Visualization therefore gets a separate topology assembled from
    the global cell-to-face connectivity during the one-time localization.
    """
    from .topology import build_cell_face_csr

    n_global_cells = int(mesh_data["n_elements"])
    global_cell_vertices = mesh_data.get("cell_vertices")
    global_cell_types = np.asarray(
        mesh_data.get("cell_type_codes", np.full(n_global_cells, -1)),
        dtype=np.int32,
    )
    if (
        global_cell_vertices is not None
        and np.asarray(global_cell_vertices).ndim == 2
        and np.asarray(global_cell_vertices).shape[1] == 8
        and np.all(global_cell_types[cell_ids] == 5)
    ):
        selected_vertices = np.asarray(global_cell_vertices, dtype=np.int64)[cell_ids]
        used_points, compact_vertices = np.unique(
            selected_vertices,
            return_inverse=True,
        )
        return {
            "points": np.ascontiguousarray(np.asarray(mesh_data["points"])[used_points]),
            "faces": np.empty((0, 0), dtype=np.int32),
            "owners": np.empty(0, dtype=np.int64),
            "neighbours": np.empty(0, dtype=np.int64),
            "n_elements": len(cell_ids),
            "n_faces": 0,
            "n_interior_faces": 0,
            "boundary": [],
            "cell_faces": np.empty(0, dtype=np.int32),
            "cell_face_offsets": np.zeros(len(cell_ids) + 1, dtype=np.int64),
            "cell_vertices": np.ascontiguousarray(
                compact_vertices.reshape((-1, 8)),
                dtype=np.int32,
            ),
            "cell_type_codes": np.ascontiguousarray(global_cell_types[cell_ids]),
            "cell_orders": np.ascontiguousarray(
                np.asarray(
                    mesh_data.get("cell_orders", np.ones(n_global_cells)),
                    dtype=np.int8,
                )[cell_ids]
            ),
            "global_point_ids": np.ascontiguousarray(used_points),
            "global_face_ids": np.empty(0, dtype=np.int64),
            "global_cell_ids": np.ascontiguousarray(cell_ids),
            "source_cell_ids": np.ascontiguousarray(
                np.asarray(mesh_data.get("source_cell_ids", np.arange(n_global_cells)))[cell_ids]
            ),
        }

    cell_faces = mesh_data.get("cell_faces")
    cell_face_offsets = mesh_data.get("cell_face_offsets")
    if cell_faces is None or cell_face_offsets is None:
        cell_faces, cell_face_offsets = build_cell_face_csr(
            mesh_data["owners"],
            mesh_data["neighbours"],
            n_global_cells,
            mesh_data["n_faces"],
        )
    cell_faces = np.asarray(cell_faces, dtype=np.int64)
    cell_face_offsets = np.asarray(cell_face_offsets, dtype=np.int64)
    counts = np.diff(cell_face_offsets)[cell_ids]
    selected_offsets = np.empty(len(cell_ids) + 1, dtype=np.int64)
    selected_offsets[0] = 0
    np.cumsum(counts, out=selected_offsets[1:])

    if selected_offsets[-1]:
        starts = cell_face_offsets[cell_ids]
        entry_starts = np.repeat(starts, counts)
        local_starts = np.repeat(selected_offsets[:-1], counts)
        entries = entry_starts + np.arange(selected_offsets[-1]) - local_starts
        selected_global_faces = cell_faces[entries]
        global_face_ids, selected_cell_faces = np.unique(
            selected_global_faces,
            return_inverse=True,
        )
    else:
        global_face_ids = np.empty(0, dtype=np.int64)
        selected_cell_faces = np.empty(0, dtype=np.int64)

    source_faces = mesh_data["faces"]
    face_array = source_faces if isinstance(source_faces, np.ndarray) else None
    if face_array is not None and face_array.ndim == 2:
        selected_faces = np.asarray(face_array[global_face_ids], dtype=np.int64)
        used_points = (
            np.unique(selected_faces) if selected_faces.size else np.empty(0, dtype=np.int64)
        )
    else:
        selected_faces = [
            np.asarray(source_faces[index], dtype=np.int64) for index in global_face_ids
        ]
        used_points = (
            np.unique(np.concatenate(selected_faces))
            if selected_faces
            else np.empty(0, dtype=np.int64)
        )
    point_lookup = np.full(len(mesh_data["points"]), -1, dtype=np.int64)
    point_lookup[used_points] = np.arange(len(used_points))
    if isinstance(selected_faces, np.ndarray):
        faces = np.ascontiguousarray(point_lookup[selected_faces], dtype=np.int32)
    else:
        faces = [
            np.ascontiguousarray(point_lookup[face], dtype=np.int32) for face in selected_faces
        ]

    visualization = {
        "points": np.ascontiguousarray(np.asarray(mesh_data["points"])[used_points]),
        "faces": faces,
        # The exporter consumes the explicit cell-face CSR below.  These
        # placeholders keep the ordinary mesh schema intact.
        "owners": np.zeros(len(global_face_ids), dtype=np.int64),
        "neighbours": np.empty(0, dtype=np.int64),
        "n_elements": len(cell_ids),
        "n_faces": len(global_face_ids),
        "n_interior_faces": len(global_face_ids),
        "boundary": [],
        "cell_faces": np.ascontiguousarray(selected_cell_faces, dtype=np.int32),
        "cell_face_offsets": selected_offsets,
        "global_point_ids": np.ascontiguousarray(used_points),
        "global_face_ids": np.ascontiguousarray(global_face_ids),
        "global_cell_ids": np.ascontiguousarray(cell_ids),
        "source_cell_ids": np.ascontiguousarray(
            np.asarray(mesh_data.get("source_cell_ids", np.arange(n_global_cells)))[cell_ids]
        ),
    }
    for key, default, dtype in (
        ("cell_type_codes", -1, np.int32),
        ("cell_orders", 1, np.int8),
    ):
        values = np.asarray(mesh_data.get(key, np.full(n_global_cells, default)), dtype=dtype)
        visualization[key] = np.ascontiguousarray(values[cell_ids])
    if "cell_vertices" in mesh_data:
        visualization["cell_vertices"] = np.ascontiguousarray(
            point_lookup[np.asarray(mesh_data["cell_vertices"])[cell_ids]],
            dtype=np.int32,
        )
    return visualization


def ownership_ranges(n_cells: int, n_ranks: int) -> np.ndarray:
    """Return balanced contiguous global-cell ownership offsets."""
    if n_cells < 1 or n_ranks < 1:
        raise ValueError("n_cells and n_ranks must be positive")
    counts = np.full(n_ranks, n_cells // n_ranks, dtype=np.int64)
    counts[: n_cells % n_ranks] += 1
    offsets = np.empty(n_ranks + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets


def _rank_for_cells(cell_ids: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.searchsorted(offsets[1:], cell_ids, side="right").astype(np.int32)


@dataclass(frozen=True)
class HaloSchedule:
    """Global cell IDs sent to and received from each neighboring rank."""

    send_global_ids: tuple[np.ndarray, ...]
    receive_global_ids: tuple[np.ndarray, ...]

    @property
    def neighbor_ranks(self) -> tuple[int, ...]:
        return tuple(
            rank
            for rank, (send, receive) in enumerate(
                zip(self.send_global_ids, self.receive_global_ids, strict=True)
            )
            if len(send) or len(receive)
        )


@dataclass(frozen=True)
class CellPartition:
    """Owned/ghost cells and face fragments for one MPI rank."""

    rank: int
    size: int
    global_n_cells: int
    owned_global_ids: np.ndarray
    ghost_global_ids: np.ndarray
    local_global_ids: np.ndarray
    local_face_global_ids: np.ndarray
    processor_face_global_ids: np.ndarray
    physical_boundary_face_global_ids: np.ndarray
    halo: HaloSchedule
    _global_to_local: dict[int, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Halo exchange happens several times per PIMPLE correction.  Keep the
        # map rather than rebuilding a Python dictionary for every exchange.
        object.__setattr__(
            self,
            "_global_to_local",
            {int(global_id): index for index, global_id in enumerate(self.local_global_ids)},
        )

    @classmethod
    def from_mesh_data(cls, mesh_data, rank: int, size: int) -> CellPartition:
        if not 0 <= rank < size:
            raise ValueError(f"rank {rank} outside communicator size {size}")
        n_cells = int(mesh_data["n_elements"])
        n_interior = int(mesh_data["n_interior_faces"])
        offsets = ownership_ranges(n_cells, size)
        owned = np.arange(offsets[rank], offsets[rank + 1], dtype=np.int64)
        owners = np.asarray(mesh_data["owners"], dtype=np.int64)
        neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int64)
        owner_ranks = _rank_for_cells(owners[:n_interior], offsets)
        neighbour_ranks = _rank_for_cells(neighbours[:n_interior], offsets)

        owned_face = (owner_ranks == rank) | (neighbour_ranks == rank)
        local_interior_faces = np.flatnonzero(owned_face).astype(np.int64)
        processor_mask = owned_face & (owner_ranks != neighbour_ranks)
        processor_faces = np.flatnonzero(processor_mask).astype(np.int64)

        physical_faces = np.arange(n_interior, int(mesh_data["n_faces"]), dtype=np.int64)
        physical_owner_ranks = _rank_for_cells(owners[physical_faces], offsets)
        physical_faces = physical_faces[physical_owner_ranks == rank]
        local_faces = np.concatenate((local_interior_faces, physical_faces))

        send_sets = [set() for _ in range(size)]
        receive_sets = [set() for _ in range(size)]
        ghost_set: set[int] = set()
        for face in processor_faces:
            owner = int(owners[face])
            neighbour = int(neighbours[face])
            owner_rank = int(owner_ranks[face])
            neighbour_rank = int(neighbour_ranks[face])
            if owner_rank == rank:
                send_sets[neighbour_rank].add(owner)
                receive_sets[neighbour_rank].add(neighbour)
                ghost_set.add(neighbour)
            else:
                send_sets[owner_rank].add(neighbour)
                receive_sets[owner_rank].add(owner)
                ghost_set.add(owner)

        ghosts = np.asarray(sorted(ghost_set), dtype=np.int64)
        send = tuple(np.asarray(sorted(values), dtype=np.int64) for values in send_sets)
        receive = tuple(np.asarray(sorted(values), dtype=np.int64) for values in receive_sets)
        return cls(
            rank=rank,
            size=size,
            global_n_cells=n_cells,
            owned_global_ids=owned,
            ghost_global_ids=ghosts,
            local_global_ids=np.concatenate((owned, ghosts)),
            local_face_global_ids=local_faces,
            processor_face_global_ids=processor_faces,
            physical_boundary_face_global_ids=physical_faces,
            halo=HaloSchedule(send, receive),
        )

    def global_to_local(self, global_ids) -> np.ndarray:
        """Map global cell IDs into the owned-then-ghost local layout."""
        requested = np.asarray(global_ids, dtype=np.int64)
        positions = np.asarray(
            [self._global_to_local.get(int(value), -1) for value in requested], dtype=np.int64
        )
        if np.any(positions < 0):
            missing = requested[positions < 0]
            raise KeyError(
                f"Cells are not present in rank {self.rank} partition: {missing.tolist()}"
            )
        return positions

    def exchange_halo(self, local_field: np.ndarray, comm) -> None:
        """Collectively update ghost values using one object exchange per rank."""
        values = np.asarray(local_field)
        if values.shape[0] != len(self.local_global_ids):
            raise ValueError("local_field leading dimension must match local_global_ids")
        payload = []
        for global_ids in self.halo.send_global_ids:
            local_ids = (
                self.global_to_local(global_ids) if len(global_ids) else np.empty(0, dtype=int)
            )
            payload.append(values[local_ids].copy())
        received = comm.alltoall(payload)
        for rank, global_ids in enumerate(self.halo.receive_global_ids):
            if len(global_ids):
                incoming = np.asarray(received[rank])
                expected_shape = (len(global_ids), *values.shape[1:])
                if incoming.shape != expected_shape:
                    raise RuntimeError(
                        f"Halo payload from rank {rank} has shape {incoming.shape}; "
                        f"expected {expected_shape}"
                    )
                values[self.global_to_local(global_ids)] = incoming


def localize_mesh_and_geometry(mesh_data, geo_data, rank: int, size: int):
    """Build an owned-plus-halo mesh view with compact point connectivity."""
    gradient_scheme = geo_data.get("gradient_scheme", "gauss")
    partition = CellPartition.from_mesh_data(mesh_data, rank, size)
    n_global_cells = int(mesh_data["n_elements"])
    n_global_faces = int(mesh_data["n_faces"])
    n_global_interior = int(mesh_data["n_interior_faces"])
    face_ids = partition.local_face_global_ids
    interior_count = int(np.count_nonzero(face_ids < n_global_interior))
    if np.any(face_ids[:interior_count] >= n_global_interior):
        raise RuntimeError("Partition interior faces are not stored contiguously")

    cell_lookup = np.full(n_global_cells, -1, dtype=np.int64)
    cell_lookup[partition.local_global_ids] = np.arange(len(partition.local_global_ids))
    global_owners = np.asarray(mesh_data["owners"], dtype=np.int64)[face_ids]
    owners = cell_lookup[global_owners]
    global_neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int64)
    neighbours = cell_lookup[global_neighbours[face_ids[:interior_count]]]
    if np.any(owners < 0) or np.any(neighbours < 0):
        raise RuntimeError("Localized face references a cell outside the owned/halo view")

    global_faces = mesh_data["faces"]
    face_array = global_faces if isinstance(global_faces, np.ndarray) else None
    if face_array is not None and face_array.ndim == 2:
        selected_faces = np.asarray(face_array[face_ids], dtype=np.int64)
        used_points = np.unique(selected_faces)
    else:
        selected_faces = [np.asarray(global_faces[index], dtype=np.int64) for index in face_ids]
        used_points = np.unique(np.concatenate(selected_faces))
    point_lookup = np.full(len(mesh_data["points"]), -1, dtype=np.int64)
    point_lookup[used_points] = np.arange(len(used_points))
    if isinstance(selected_faces, np.ndarray):
        faces = np.ascontiguousarray(point_lookup[selected_faces], dtype=np.int32)
    else:
        faces = [point_lookup[face] for face in selected_faces]

    face_position = {int(global_id): local for local, global_id in enumerate(face_ids)}
    boundaries = []
    for patch in mesh_data["boundary"]:
        first = int(patch["startFace"])
        last = first + int(patch["nFaces"])
        local_faces = [
            face_position[index] for index in range(first, last) if index in face_position
        ]
        if not local_faces:
            continue
        expected = list(range(local_faces[0], local_faces[0] + len(local_faces)))
        if local_faces != expected:
            raise RuntimeError(f"Localized boundary patch {patch['name']!r} is not contiguous")
        localized = deepcopy(patch)
        localized["startFace"] = local_faces[0]
        localized["nFaces"] = len(local_faces)
        boundaries.append(localized)

    local_cell_ids = partition.local_global_ids
    local_mesh = {
        "points": np.ascontiguousarray(np.asarray(mesh_data["points"])[used_points]),
        "faces": faces,
        "owners": np.ascontiguousarray(owners, dtype=np.int64),
        "neighbours": np.ascontiguousarray(neighbours, dtype=np.int64),
        "n_elements": len(local_cell_ids),
        "n_faces": len(face_ids),
        "n_interior_faces": interior_count,
        "boundary": boundaries,
        "boundary_neighbours": np.full(len(face_ids), -1, dtype=np.int64),
        "global_cell_ids": np.ascontiguousarray(local_cell_ids),
        "global_face_ids": np.ascontiguousarray(face_ids),
        "source_cell_ids": np.ascontiguousarray(
            np.asarray(mesh_data.get("source_cell_ids", np.arange(n_global_cells)))[local_cell_ids]
        ),
        "global_boundary_names": tuple(str(patch["name"]) for patch in mesh_data["boundary"]),
        "partition": partition,
        "_n_owned": len(partition.owned_global_ids),
        "_visualization_mesh": _visualization_mesh(mesh_data, local_cell_ids),
    }
    for key, default, dtype in (
        ("cell_type_codes", -1, np.int32),
        ("cell_orders", 1, np.int8),
    ):
        values = np.asarray(mesh_data.get(key, np.full(n_global_cells, default)), dtype=dtype)
        local_mesh[key] = np.ascontiguousarray(values[local_cell_ids])

    if "cell_vertices" in mesh_data:
        local_mesh["cell_vertices"] = np.ascontiguousarray(
            point_lookup[np.asarray(mesh_data["cell_vertices"])[local_cell_ids]], dtype=np.int32
        )

    local_geo = {"gradient_scheme": gradient_scheme}
    for key, values in geo_data.items():
        if key == "gradient_scheme" or key.startswith("lsq_"):
            continue
        array = np.asarray(values)
        if array.ndim > 0 and array.shape[0] == n_global_faces:
            local_geo[key] = np.ascontiguousarray(array[face_ids])
        elif array.ndim > 0 and array.shape[0] == n_global_cells:
            local_geo[key] = np.ascontiguousarray(array[local_cell_ids])
        else:
            local_geo[key] = deepcopy(values)
    if gradient_scheme == "lsq":
        # Recompute stencils in the compact local indexing.  Every owned cell
        # contains all of its adjacent faces; ghost-cell gradients are filled
        # by halo exchange after the owned reconstruction.
        from ..fields.gradients import compute_lsq_geometry

        local_geo.update(compute_lsq_geometry(local_mesh, local_geo))
    return local_mesh, local_geo, partition
