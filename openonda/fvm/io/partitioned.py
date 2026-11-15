"""Conflict-free partitioned backup output and reconstruction."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from openonda.core.file_output import (
    npz_size_bound,
    require_free_space,
    write_json_atomic,
    write_npz_atomic,
    write_text_atomic,
)
from openonda.core.parallel import collective_phase
from openonda.core.write_precision import cast_for_write

from ..config.output import Output
from .backup import FORMAT_VERSION, decode_state, encode_state
from .vtk_exporter import VTKExporter

if TYPE_CHECKING:
    from ..core.solver import FVMSolver
    from ..mesh.partition import CellPartition


def _prune_partitioned_generations(target: Path, current: dict, previous: dict | None) -> None:
    """Keep current and previous committed generations after checkpoint-information writing.

    Only this writer's UUID-named rank archives are eligible; unrelated files,
    temporary writes and symlinks remain untouched.
    """
    keep = set(current["files"])
    if previous is not None:
        keep.update(previous["files"])
        write_text_atomic(
            target / "checkpoint_info.previous.json", json.dumps(previous, indent=2) + "\n"
        )
    for path in target.iterdir():
        if (
            re.fullmatch(r"rank-[0-9]{5}-[0-9a-f]{32}\.npz", path.name)
            and path.name not in keep
            and path.is_file()
            and not path.is_symlink()
        ):
            path.unlink()


def _resolve_backup_file(target: Path, name: str) -> Path:
    """Resolve one checkpoint-information file without allowing path traversal."""
    if not isinstance(name, str) or not name:
        raise ValueError("Partitioned backup file names must be non-empty strings")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        raise ValueError(f"Unsafe partitioned backup file path: {name!r}")
    return target / relative


def save_partitioned_solver_backup(solver: FVMSolver, directory: str | Path) -> Path:
    """Publish a complete backup without invalidating the prior generation."""
    from .backup import state_arrays

    target = Path(directory)
    comm = solver.parallel.comm
    previous_info = None
    with collective_phase(comm, "partitioned backup preparation"):
        if solver.parallel.is_root:
            target.mkdir(parents=True, exist_ok=True)
            if (target / "checkpoint_info.json").is_file():
                previous_info = json.loads((target / "checkpoint_info.json").read_text())
                # Fail before writing if an existing checkpoint information is not a usable
                # checkpoint description; guessing which old archives to retain is unsafe.
                for name in previous_info["files"]:
                    _resolve_backup_file(target, name)
    partition = solver.parallel.partition
    arrays = {
        "global_cell_id": partition.local_global_ids,
        "global_face_id": solver.mesh_data["global_face_id"],
        **state_arrays(solver),
    }

    # All ranks write their temporary archives concurrently, so reserve space
    # for the aggregate uncompressed upper bound before any old restart can be
    # affected.  The previous checkpoint information remains valid throughout this operation.
    local_archive_bytes = npz_size_bound(arrays)
    archive_bytes = int(solver.parallel.global_sum(local_archive_bytes))
    with collective_phase(comm, "partitioned backup space check"):
        if solver.parallel.is_root:
            require_free_space(target, archive_bytes)

    generation = solver.parallel.bcast(
        uuid4().hex if solver.parallel.is_root else None,
        root=0,
    )
    files = [f"rank-{rank:05d}-{generation}.npz" for rank in range(partition.size)]
    with collective_phase(comm, "partitioned backup write"):
        write_npz_atomic(
            target / files[partition.rank], encode_state(arrays), check_free_space=False
        )

    checkpoint_info = None
    with collective_phase(comm, "partitioned backup checkpoint information"):
        if solver.parallel.is_root:
            checkpoint_info = {
                "format_version": FORMAT_VERSION,
                "generation": generation,
                "configuration_sha256": solver._configuration_sha256,
                "mesh_sha256": solver._mesh_sha256,
                "n_global_cells": partition.n_global_cells,
                "n_ranks": partition.size,
                "files": files,
            }
            write_json_atomic(target / "checkpoint_info.json", checkpoint_info)
    with collective_phase(comm, "partitioned backup cleanup"):
        if solver.parallel.is_root:
            _prune_partitioned_generations(target, checkpoint_info, previous_info)
    return target


def load_partitioned_solver_backup(
    solver: FVMSolver,
    directory: str | Path,
) -> None:
    """Collectively stage and restore a complete partitioned backup."""
    from .backup import (
        restore_restart_state,
        state_arrays,
        validate_restart_state,
    )

    target = Path(directory)
    comm = solver.parallel.comm
    checkpoint_info = None
    with collective_phase(comm, "partitioned backup information"):
        if solver.parallel.is_root:
            checkpoint_info = json.loads(
                (target / "checkpoint_info.json").read_text(encoding="utf-8")
            )
    checkpoint_info = solver.parallel.bcast(checkpoint_info, root=0)

    restart_state = None
    with collective_phase(comm, "partitioned backup validation"):
        expected_info_keys = {
            "format_version",
            "generation",
            "configuration_sha256",
            "mesh_sha256",
            "n_global_cells",
            "n_ranks",
            "files",
        }
        if not isinstance(checkpoint_info, dict) or set(checkpoint_info) != expected_info_keys:
            raise ValueError("Partitioned backup information has invalid fields")
        if checkpoint_info.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported partitioned FVM backup version")
        if checkpoint_info.get("n_ranks") != solver.parallel.size:
            raise ValueError("Partitioned backup communicator size does not match")
        if checkpoint_info.get("mesh_sha256") != solver._mesh_sha256:
            raise ValueError("Partitioned backup mesh hash does not match")
        if checkpoint_info.get("configuration_sha256") != solver._configuration_sha256:
            raise ValueError("Partitioned backup configuration hash does not match")
        files = checkpoint_info.get("files")
        if not isinstance(files, list) or len(files) != solver.parallel.size:
            raise ValueError("Partitioned backup file description is incomplete")
        generation = checkpoint_info.get("generation")
        expected_files = [
            f"rank-{file_rank:05d}-{generation}.npz" for file_rank in range(solver.parallel.size)
        ]
        if not isinstance(generation, str) or not generation or files != expected_files:
            raise ValueError("Partitioned backup contains mixed or invalid generations")

        rank = solver.parallel.rank
        rank_file = _resolve_backup_file(target, files[rank])
        with np.load(rank_file, allow_pickle=False) as archive:
            state = decode_state(
                {name: np.array(archive[name], copy=True) for name in archive.files}
            )
        required_state = {"global_cell_id", "global_face_id", *state_arrays(solver)}
        state.pop("storage_layout", None)
        unexpected = sorted(set(state) - required_state)
        missing = sorted(required_state - set(state))
        if missing or unexpected:
            raise ValueError(
                f"Invalid partitioned backup fields; missing={missing}, unexpected={unexpected}"
            )

        partition = solver.parallel.partition
        if not np.array_equal(state.pop("global_cell_id"), partition.local_global_ids):
            raise ValueError("Partitioned backup cell IDs do not match")
        if not np.array_equal(state.pop("global_face_id"), solver.mesh_data["global_face_id"]):
            raise ValueError("Partitioned backup face IDs do not match")
        restart_state = validate_restart_state(solver, state)
    assert restart_state is not None
    signatures = solver.parallel.comm.allgather(
        (
            restart_state.time,
            restart_state.step,
            restart_state.time_step_size,
            restart_state.accepted_time_step_size,
            restart_state.previous_time_step_size,
        )
    )
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise RuntimeError("Partitioned restart validation found inconsistent clocks across ranks")
    restore_restart_state(solver, restart_state)
    solver.parallel.barrier()


def _vtk_xml_type(values: np.ndarray) -> str:
    """Return the VTK XML scalar type used by :class:`VTKExporter`."""
    dtype = np.asarray(values).dtype
    if np.issubdtype(dtype, np.floating):
        return f"Float{dtype.itemsize * 8}"
    if np.issubdtype(dtype, np.signedinteger):
        return f"Int{dtype.itemsize * 8}"
    if np.issubdtype(dtype, np.unsignedinteger):
        return f"UInt{dtype.itemsize * 8}"
    raise TypeError(f"VTK output does not support field dtype {dtype}")


def _field_components(values: np.ndarray) -> int:
    array = np.asarray(values)
    if array.ndim == 1:
        return 1
    if array.ndim == 2:
        return int(array.shape[1])
    raise ValueError("VTK fields must be one-dimensional scalars or two-dimensional vectors")


def write_partition_vtu(
    directory: str | Path,
    stem: str,
    mesh_data: dict[str, object],
    partition: CellPartition,
    fields: dict[str, np.ndarray],
    comm: object,
    *,
    output: Output,
    exporter: VTKExporter,
) -> Path:
    """Atomically write one rank piece and a root parallel collection.

    With one visualization ghost layer, each piece contains owned cells
    followed by face-adjacent halo cells marked with ``vtkGhostType``.  This
    lets ParaView's cell-to-point conversion remain smooth across partitions
    without changing the solver's rank assignment or numerical halo settings.
    """
    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be a non-empty filename component")
    target = Path(directory)
    n_owned = len(partition.owned_global_ids)
    local_count = len(partition.local_global_ids)
    with collective_phase(comm, "partition visualization piece"):
        target.mkdir(parents=True, exist_ok=True)
        local_fields = {}
        for name, values in fields.items():
            array = np.asarray(values)
            if array.ndim == 0 or array.shape[0] != local_count:
                raise ValueError(f"Field {name!r} does not match the local partition")
            _field_components(array)
            local_fields[name] = array

        piece_name = f"{stem}-rank-{partition.rank:05d}.vtu"
        if output.ghost_layers == 1:
            # The halo exchange fills the ghost rows in place, on copies.
            local_fields = {
                name: np.ascontiguousarray(array).copy() for name, array in local_fields.items()
            }
            if len(partition.ghost_global_ids):
                for values in local_fields.values():
                    partition.exchange_halo(values, comm)
            piece_fields = {
                name: cast_for_write(values, output.precision)
                for name, values in local_fields.items()
            }
            ghost_types = np.zeros(local_count, dtype=np.uint8)
            # vtkDataSetAttributes::DUPLICATECELL marks overlap supplied only
            # for parallel filters and prevents duplicate contributions.
            ghost_types[n_owned:] = 1
            piece_fields["vtkGhostType"] = ghost_types
            piece_fields["global_cell_id"] = np.ascontiguousarray(
                partition.local_global_ids,
                dtype=np.int64,
            )
            visualization_mesh = mesh_data["_visualization_mesh"]
            point_fields = {
                "global_point_id": np.ascontiguousarray(
                    visualization_mesh.get(
                        "global_point_ids",
                        np.arange(len(visualization_mesh["vertex_position"])),
                    ),
                    dtype=np.int64,
                )
            }
            exporter.export(
                str(target / piece_name),
                piece_fields,
                point_fields=point_fields,
            )
        else:
            # The visualization mesh holds exactly the owned cells.
            piece_fields = {name: values[:n_owned] for name, values in local_fields.items()}
            piece_fields["global_cell_id"] = np.ascontiguousarray(
                partition.owned_global_ids,
                dtype=np.int64,
            )
            exporter.export(str(target / piece_name), piece_fields)

    collection = target / f"{stem}.pvtu"
    with collective_phase(comm, "partition visualization collection"):
        if partition.rank == 0:
            lines = [
                '<?xml version="1.0"?>',
                '<VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian">',
                f'  <PUnstructuredGrid GhostLevel="{output.ghost_layers}">',
                "    <PCellData>",
            ]
            for name, values in piece_fields.items():
                components = _field_components(values)
                lines.append(
                    f'      <PDataArray type="{_vtk_xml_type(values)}" Name="{escape(name)}" '
                    f'NumberOfComponents="{components}"/>'
                )
            lines.append("    </PCellData>")
            if output.ghost_layers == 1:
                lines.extend(
                    [
                        "    <PPointData>",
                        '      <PDataArray type="Int64" Name="global_point_id" NumberOfComponents="1"/>',
                        "    </PPointData>",
                    ]
                )
            lines.extend(
                [
                    "    <PPoints>",
                    '      <PDataArray type="Float64" NumberOfComponents="3"/>',
                    "    </PPoints>",
                ]
            )
            lines.extend(
                f'    <Piece Source="{escape(f"{stem}-rank-{rank:05d}.vtu", quote=True)}"/>'
                for rank in range(partition.size)
            )
            lines.extend(["  </PUnstructuredGrid>", "</VTKFile>"])
            write_text_atomic(collection, "\n".join(lines) + "\n")
    return collection
