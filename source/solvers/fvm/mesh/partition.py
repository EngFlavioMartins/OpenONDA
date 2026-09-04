"""Deterministic cell partition metadata and halo exchange."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


def _visualization_mesh(mesh_data, cell_ids: np.ndarray) -> dict[str, Any]:
    """Build complete, compact cells for owned-plus-ghost visualization.

    The solver's localized mesh contains exactly the faces needed to advance
    owned cells.  That is intentionally insufficient to draw complete halo
    cells.  Visualization therefore gets a separate topology assembled from
    the global cell-to-face connectivity during the one-time localization.
    """
    from .topology import build_cell_face_csr

    n_global_cells = int(mesh_data["n_cells"])
    global_cell_vertex_indices = mesh_data.get("cell_vertex_indices")
    global_cell_types = np.asarray(
        mesh_data.get("cell_type_code", np.full(n_global_cells, -1)),
        dtype=np.int32,
    )
    if (
        global_cell_vertex_indices is not None
        and np.asarray(global_cell_vertex_indices).ndim == 2
        and np.asarray(global_cell_vertex_indices).shape[1] == 8
        and np.all(global_cell_types[cell_ids] == 5)
    ):
        selected_vertices = np.asarray(global_cell_vertex_indices)[cell_ids]
        used_points, compact_vertices = np.unique(
            selected_vertices,
            return_inverse=True,
        )
        return {
            "vertex_position": np.ascontiguousarray(
                np.asarray(mesh_data["vertex_position"])[used_points]
            ),
            "faces": np.empty((0, 0), dtype=np.int32),
            "owners": np.empty(0, dtype=np.int64),
            "neighbours": np.empty(0, dtype=np.int64),
            "n_cells": len(cell_ids),
            "n_faces": 0,
            "n_interior_faces": 0,
            "boundary": [],
            "cell_face_indices": np.empty(0, dtype=np.int32),
            "cell_face_offset": np.zeros(len(cell_ids) + 1, dtype=np.int64),
            "cell_vertex_indices": np.ascontiguousarray(
                compact_vertices.reshape((-1, 8)),
                dtype=np.int32,
            ),
            "cell_type_code": np.ascontiguousarray(global_cell_types[cell_ids]),
            "cell_order": np.ascontiguousarray(
                np.asarray(
                    mesh_data.get("cell_order", np.ones(n_global_cells)),
                    dtype=np.int8,
                )[cell_ids]
            ),
            "global_point_ids": np.ascontiguousarray(used_points),
            "global_face_id": np.empty(0, dtype=np.int64),
            "global_cell_id": np.ascontiguousarray(cell_ids),
            "source_cell_id": np.ascontiguousarray(
                np.asarray(mesh_data.get("source_cell_id", np.arange(n_global_cells)))[cell_ids]
            ),
        }

    cell_face_indices = mesh_data.get("cell_face_indices")
    cell_face_offset = mesh_data.get("cell_face_offset")
    if cell_face_indices is None or cell_face_offset is None:
        cell_face_indices, cell_face_offset = build_cell_face_csr(
            mesh_data["owners"],
            mesh_data["neighbours"],
            n_global_cells,
            mesh_data["n_faces"],
        )
    cell_face_indices = np.asarray(cell_face_indices, dtype=np.int64)
    cell_face_offset = np.asarray(cell_face_offset, dtype=np.int64)
    counts = np.diff(cell_face_offset)[cell_ids]
    selected_offsets = np.empty(len(cell_ids) + 1, dtype=np.int64)
    selected_offsets[0] = 0
    np.cumsum(counts, out=selected_offsets[1:])

    if selected_offsets[-1]:
        starts = cell_face_offset[cell_ids]
        entry_starts = np.repeat(starts, counts)
        local_starts = np.repeat(selected_offsets[:-1], counts)
        entries = entry_starts + np.arange(selected_offsets[-1]) - local_starts
        selected_global_faces = cell_face_indices[entries]
        global_face_id, selected_cell_face_indices = np.unique(
            selected_global_faces,
            return_inverse=True,
        )
        selected_cells = np.repeat(cell_ids, counts)
        cell_face_reversed = np.zeros(len(selected_global_faces), dtype=bool)
        global_interior = int(mesh_data["n_interior_faces"])
        interior_entry = selected_global_faces < global_interior
        cell_face_reversed[interior_entry] = (
            np.asarray(mesh_data["neighbours"], dtype=np.int64)[
                selected_global_faces[interior_entry]
            ]
            == selected_cells[interior_entry]
        )
    else:
        global_face_id = np.empty(0, dtype=np.int64)
        selected_cell_face_indices = np.empty(0, dtype=np.int64)
        cell_face_reversed = np.empty(0, dtype=bool)

    source_faces = mesh_data["faces"]
    face_array = source_faces if isinstance(source_faces, np.ndarray) else None
    if face_array is not None and face_array.ndim == 2:
        selected_faces = np.asarray(face_array[global_face_id], dtype=np.int64)
        used_points = (
            np.unique(selected_faces) if selected_faces.size else np.empty(0, dtype=np.int64)
        )
    else:
        selected_faces = [
            np.asarray(source_faces[index], dtype=np.int64) for index in global_face_id
        ]
        used_points = (
            np.unique(np.concatenate(selected_faces))
            if selected_faces
            else np.empty(0, dtype=np.int64)
        )
    point_lookup = np.full(len(mesh_data["vertex_position"]), -1, dtype=np.int64)
    point_lookup[used_points] = np.arange(len(used_points))
    if isinstance(selected_faces, np.ndarray):
        faces = np.ascontiguousarray(point_lookup[selected_faces], dtype=np.int32)
    else:
        faces = [
            np.ascontiguousarray(point_lookup[face], dtype=np.int32) for face in selected_faces
        ]

    visualization = {
        "vertex_position": np.ascontiguousarray(
            np.asarray(mesh_data["vertex_position"])[used_points]
        ),
        "faces": faces,
        # The exporter consumes the explicit cell-face CSR below.  These
        # placeholders keep the ordinary mesh schema intact.
        "owners": np.zeros(len(global_face_id), dtype=np.int64),
        "neighbours": np.empty(0, dtype=np.int64),
        "n_cells": len(cell_ids),
        "n_faces": len(global_face_id),
        "n_interior_faces": len(global_face_id),
        "boundary": [],
        "cell_face_indices": np.ascontiguousarray(selected_cell_face_indices, dtype=np.int32),
        "cell_face_offset": selected_offsets,
        "cell_face_reversed": np.ascontiguousarray(cell_face_reversed),
        "global_point_ids": np.ascontiguousarray(used_points),
        "global_face_id": np.ascontiguousarray(global_face_id),
        "global_cell_id": np.ascontiguousarray(cell_ids),
        "source_cell_id": np.ascontiguousarray(
            np.asarray(mesh_data.get("source_cell_id", np.arange(n_global_cells)))[cell_ids]
        ),
    }
    for key, default, dtype in (
        ("cell_type_code", -1, np.int32),
        ("cell_order", 1, np.int8),
    ):
        values = np.asarray(mesh_data.get(key, np.full(n_global_cells, default)), dtype=dtype)
        visualization[key] = np.ascontiguousarray(values[cell_ids])
    if "cell_vertex_indices" in mesh_data:
        visualization["cell_vertex_indices"] = np.ascontiguousarray(
            point_lookup[np.asarray(mesh_data["cell_vertex_indices"])[cell_ids]],
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
    """Global and local IDs for halo-cell exchange with neighbouring ranks.

    For each rank in the communicator, stores the global cell IDs and local
    (owned-plus-halo) indices of the cells that are sent (this rank's owned
    cells that are ghosts on another rank) and received (this rank's ghost
    cells that are owned on another rank).

    Attributes
    ----------
    send_global_ids : tuple[np.ndarray, ...]
        Per-rank global cell IDs to send.
    receive_global_ids : tuple[np.ndarray, ...]
        Per-rank global cell IDs to receive.
    send_local_indices : tuple[np.ndarray, ...]
        Per-rank local (owned) indices for the cells to send.
    receive_local_indices : tuple[np.ndarray, ...]
        Per-rank local (halo) indices where received values are stored.
    """

    send_global_ids: tuple[np.ndarray, ...]
    receive_global_ids: tuple[np.ndarray, ...]
    send_local_indices: tuple[np.ndarray, ...]
    receive_local_indices: tuple[np.ndarray, ...]

    @property
    def neighbour_ranks(self) -> tuple[int, ...]:
        """Return a tuple of ranks that exchange at least one cell."""
        return tuple(
            rank
            for rank, (send, receive) in enumerate(
                zip(self.send_global_ids, self.receive_global_ids, strict=True)
            )
            if len(send) or len(receive)
        )


@dataclass(frozen=True)
class CellPartition:
    """Owned/ghost cell partitioning for one MPI rank.

    Stores the mapping from global cell IDs to local indices, the set of
    owned cells and ghost cells, and the :class:`HaloSchedule` for
    communication.  Constructed via the :meth:`from_mesh_data` factory.

    Attributes
    ----------
    rank : int
        This MPI rank.
    size : int
        Total number of MPI ranks.
    n_global_cells : int
        Total number of cells in the global (unpartitioned) mesh.
    owned_global_ids : np.ndarray
        Global cell IDs owned by this rank.
    ghost_global_ids : np.ndarray
        Global cell IDs that are ghosts on this rank.
    local_global_ids : np.ndarray
        All global IDs in local index order (owned first, then ghosts).
    local_face_global_ids : np.ndarray
        Global face IDs for all locally relevant faces.
    processor_face_global_ids : np.ndarray
        Global face IDs for processor (inter-rank) boundary faces.
    physical_boundary_face_global_ids : np.ndarray
        Global face IDs for physical boundary patches.
    halo : HaloSchedule
        Halo-exchange schedule for neighbour communication.
    """

    rank: int
    size: int
    n_global_cells: int
    owned_global_ids: np.ndarray
    ghost_global_ids: np.ndarray
    local_global_ids: np.ndarray
    local_face_global_ids: np.ndarray
    processor_face_global_ids: np.ndarray
    physical_boundary_face_global_ids: np.ndarray
    halo: HaloSchedule

    @classmethod
    def from_mesh_data(cls, mesh_data, rank: int, size: int) -> CellPartition:
        if not 0 <= rank < size:
            raise ValueError(f"rank {rank} outside communicator size {size}")
        n_cells = int(mesh_data["n_cells"])
        n_faces = int(mesh_data["n_faces"])
        int32_limit = np.iinfo(np.int32).max
        if n_cells > int32_limit or n_faces > int32_limit:
            raise OverflowError(
                "Partitioned FVM topology exceeds 32-bit global indexing: "
                f"cells={n_cells}, faces={n_faces}, limit={int32_limit}"
            )
        n_interior = int(mesh_data["n_interior_faces"])
        offsets = ownership_ranges(n_cells, size)
        owned = np.arange(offsets[rank], offsets[rank + 1], dtype=np.int32)
        owners = np.asarray(mesh_data["owners"], dtype=np.int32)
        neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int32)
        owner_ranks = _rank_for_cells(owners[:n_interior], offsets)
        neighbour_ranks = _rank_for_cells(neighbours[:n_interior], offsets)

        owned_face = (owner_ranks == rank) | (neighbour_ranks == rank)
        local_interior_faces = np.flatnonzero(owned_face).astype(np.int32)
        processor_mask = owned_face & (owner_ranks != neighbour_ranks)
        processor_faces = np.flatnonzero(processor_mask).astype(np.int32)

        physical_faces = np.arange(n_interior, n_faces, dtype=np.int32)
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

        ghosts = np.asarray(sorted(ghost_set), dtype=np.int32)
        send = tuple(np.asarray(sorted(values), dtype=np.int32) for values in send_sets)
        receive = tuple(np.asarray(sorted(values), dtype=np.int32) for values in receive_sets)
        local_global_ids = np.concatenate((owned, ghosts))
        # Owned IDs form one contiguous range and ghost IDs are sorted.  Use
        # those two compact arrays directly instead of retaining a Python
        # int->int dictionary with one entry per local cell.
        owned_start = int(offsets[rank])
        send_local = tuple(np.asarray(ids - owned_start, dtype=np.int32) for ids in send)
        receive_local = tuple(
            np.asarray(np.searchsorted(ghosts, ids) + len(owned), dtype=np.int32) for ids in receive
        )
        return cls(
            rank=rank,
            size=size,
            n_global_cells=n_cells,
            owned_global_ids=owned,
            ghost_global_ids=ghosts,
            local_global_ids=local_global_ids,
            local_face_global_ids=local_faces,
            processor_face_global_ids=processor_faces,
            physical_boundary_face_global_ids=physical_faces,
            halo=HaloSchedule(send, receive, send_local, receive_local),
        )

    def global_to_local(self, global_ids) -> np.ndarray:
        """Map global cell IDs into the owned-then-ghost local layout."""
        requested = np.asarray(global_ids, dtype=np.int32)
        requested_flat = requested.ravel()
        positions_flat = np.full(requested_flat.shape, -1, dtype=np.int32)
        if len(self.owned_global_ids):
            first = int(self.owned_global_ids[0])
            last = int(self.owned_global_ids[-1])
            owned = (requested_flat >= first) & (requested_flat <= last)
            positions_flat[owned] = requested_flat[owned] - first
        else:
            owned = np.zeros(requested_flat.shape, dtype=bool)
        remaining = ~owned
        if np.any(remaining) and len(self.ghost_global_ids):
            values = requested_flat[remaining]
            indices = np.searchsorted(self.ghost_global_ids, values)
            valid = indices < len(self.ghost_global_ids)
            if np.any(valid):
                valid_positions = np.flatnonzero(remaining)[valid]
                matched = self.ghost_global_ids[indices[valid]] == values[valid]
                positions_flat[valid_positions[matched]] = (
                    len(self.owned_global_ids) + indices[valid][matched]
                )
        if np.any(positions_flat < 0):
            missing = requested_flat[positions_flat < 0]
            raise KeyError(
                f"Cells are not present in rank {self.rank} partition: {missing.tolist()}"
            )
        return positions_flat.reshape(requested.shape)

    def exchange_halo(self, local_field: np.ndarray, comm) -> None:
        """Update ghost values with numeric nonblocking neighbour messages.

        The schedule stores local indices, so the steady path neither maps
        global IDs nor builds an all-rank Python payload list.  ``mpi4py``
        infers the explicit MPI datatype from the contiguous NumPy buffers;
        object and strided fields are rejected rather than silently pickled.
        """
        values = np.asarray(local_field)
        if values.shape[0] != len(self.local_global_ids):
            raise ValueError("local_field leading dimension must match local_global_ids")
        if values.dtype.hasobject or not (
            np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_)
        ):
            raise TypeError("Halo exchange requires a numeric NumPy dtype")
        if not values.flags.c_contiguous:
            raise ValueError("Halo exchange requires a C-contiguous field array")

        receives: list[tuple[np.ndarray, np.ndarray]] = []
        requests = []
        for rank in self.halo.neighbour_ranks:
            receive_indices = self.halo.receive_local_indices[rank]
            if len(receive_indices):
                incoming = np.empty((len(receive_indices), *values.shape[1:]), dtype=values.dtype)
                receives.append((receive_indices, incoming))
                requests.append(comm.Irecv(incoming, source=rank, tag=9107))
        # Keep gathered sends alive until Wait completes.  Advanced indexing
        # necessarily materializes the packing buffer, but it is numeric and
        # bounded by the interface rather than an all-rank object collective.
        send_buffers = []
        for rank in self.halo.neighbour_ranks:
            send_indices = self.halo.send_local_indices[rank]
            if len(send_indices):
                outgoing = np.ascontiguousarray(values[send_indices])
                send_buffers.append(outgoing)
                requests.append(comm.Isend(outgoing, dest=rank, tag=9107))
        for request in requests:
            request.Wait()
        for receive_indices, incoming in receives:
            values[receive_indices] = incoming


def localize_mesh_and_geometry(
    mesh_data,
    geo_data,
    rank: int,
    size: int,
    *,
    include_visualization_ghosts: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], CellPartition]:
    """Build an owned-plus-halo mesh view with compact point connectivity.

    ``include_visualization_ghosts`` retains complete halo-cell topology only
    when VTK output requests a ghost layer. Owned-only output can use the
    numerical local mesh directly and avoids a second mesh copy per rank.
    """
    gradient_scheme = geo_data.get("gradient_scheme", "gauss")
    partition = CellPartition.from_mesh_data(mesh_data, rank, size)
    n_global_cells = int(mesh_data["n_cells"])
    n_global_faces = int(mesh_data["n_faces"])
    n_global_interior = int(mesh_data["n_interior_faces"])
    face_ids = partition.local_face_global_ids
    interior_count = int(np.count_nonzero(face_ids < n_global_interior))
    if np.any(face_ids[:interior_count] >= n_global_interior):
        raise RuntimeError("Partition interior faces are not stored contiguously")

    cell_lookup = np.full(n_global_cells, -1, dtype=np.int32)
    cell_lookup[partition.local_global_ids] = np.arange(len(partition.local_global_ids))
    global_owners = np.asarray(mesh_data["owners"], dtype=np.int32)[face_ids]
    owners = cell_lookup[global_owners]
    global_neighbours = np.asarray(mesh_data["neighbours"], dtype=np.int32)
    neighbours = cell_lookup[global_neighbours[face_ids[:interior_count]]]
    if np.any(owners < 0) or np.any(neighbours < 0):
        raise RuntimeError("Localized face references a cell outside the owned/halo view")

    global_faces = mesh_data["faces"]
    face_array = global_faces if isinstance(global_faces, np.ndarray) else None
    if face_array is not None and face_array.ndim == 2:
        selected_faces = np.asarray(face_array[face_ids])
        used_points = np.unique(selected_faces)
    else:
        selected_faces = [np.asarray(global_faces[index], dtype=np.int64) for index in face_ids]
        used_points = np.unique(np.concatenate(selected_faces))
    if "cell_vertex_indices" in mesh_data:
        # Ghost cells are stored completely in the local view, and their far-side
        # corners need not be referenced by any local face.  Include every local
        # cell corner so the compact point set keeps hex vertex indices valid for
        # owned and ghost cells alike instead of emitting -1 entries.
        local_cell_vertex_indices = np.asarray(mesh_data["cell_vertex_indices"], dtype=np.int64)[
            partition.local_global_ids
        ]
        used_points = np.unique(np.concatenate((used_points, local_cell_vertex_indices.ravel())))
    point_lookup = np.full(len(mesh_data["vertex_position"]), -1, dtype=np.int32)
    point_lookup[used_points] = np.arange(len(used_points))
    if isinstance(selected_faces, np.ndarray):
        faces = np.ascontiguousarray(point_lookup[selected_faces], dtype=np.int32)
    else:
        faces = [point_lookup[face] for face in selected_faces]

    boundaries = []
    for patch in mesh_data["boundary"]:
        first = int(patch["start_face"])
        last = first + int(patch["n_faces"])
        local_first = int(np.searchsorted(face_ids, first, side="left"))
        local_last = int(np.searchsorted(face_ids, last, side="left"))
        if local_first == local_last:
            continue
        selected_global = face_ids[local_first:local_last]
        if np.any((selected_global < first) | (selected_global >= last)):
            raise RuntimeError(f"Localized boundary patch {patch['name']!r} is not contiguous")
        localized = deepcopy(patch)
        localized["start_face"] = local_first
        localized["n_faces"] = local_last - local_first
        boundaries.append(localized)

    local_cell_ids = partition.local_global_ids
    local_mesh: dict[str, Any] = {
        "vertex_position": np.ascontiguousarray(
            np.asarray(mesh_data["vertex_position"])[used_points]
        ),
        "faces": faces,
        "owners": np.ascontiguousarray(owners, dtype=np.int32),
        "neighbours": np.ascontiguousarray(neighbours, dtype=np.int32),
        "n_cells": len(local_cell_ids),
        "n_faces": len(face_ids),
        "n_interior_faces": interior_count,
        "boundary": boundaries,
        "boundary_neighbour_cell": np.full(len(face_ids), -1, dtype=np.int32),
        "global_cell_id": np.ascontiguousarray(local_cell_ids),
        "global_face_id": np.ascontiguousarray(face_ids),
        "source_cell_id": np.ascontiguousarray(
            np.asarray(
                mesh_data.get("source_cell_id", np.arange(n_global_cells, dtype=np.int32)),
                dtype=np.int32,
            )[local_cell_ids]
        ),
        "global_boundary_names": tuple(str(patch["name"]) for patch in mesh_data["boundary"]),
        "partition": partition,
        "_n_owned": len(partition.owned_global_ids),
    }
    # The numerical local mesh stores complete topology for owned cells but
    # only the faces needed to advance halo cells.  VTKExporter constructs its
    # grid eagerly, so even owned-only output must receive a visualization
    # mesh that excludes those intentionally incomplete halo cells.
    visualization_cell_ids = (
        local_cell_ids if include_visualization_ghosts else partition.owned_global_ids
    )
    local_mesh["_visualization_mesh"] = _visualization_mesh(
        mesh_data,
        visualization_cell_ids,
    )
    for key, default, dtype in (
        ("cell_type_code", -1, np.int32),
        ("cell_order", 1, np.int8),
    ):
        values = np.asarray(mesh_data.get(key, np.full(n_global_cells, default)), dtype=dtype)
        local_mesh[key] = np.ascontiguousarray(values[local_cell_ids])

    if "cell_vertex_indices" in mesh_data:
        local_mesh["cell_vertex_indices"] = np.ascontiguousarray(
            point_lookup[np.asarray(mesh_data["cell_vertex_indices"])[local_cell_ids]],
            dtype=np.int32,
        )

    local_geo: dict[str, Any] = {"gradient_scheme": gradient_scheme}
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
