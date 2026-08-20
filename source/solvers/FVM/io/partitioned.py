"""Conflict-free partitioned checkpoint output and reconstruction."""

from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4

import numpy as np

from ..config.types import OutputSetup
from .storage import InsufficientStorageError, require_free_space
from .vtk_exporter import VTKExporter, atomic_write_text

PARTITIONED_CHECKPOINT_VERSION = 2


def _atomic_npz(path: Path, arrays) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _archive_upper_bound(arrays) -> int:
    """Conservative capacity estimate for a compressed NumPy archive."""
    return sum(int(np.asarray(value).nbytes) + 4096 for value in arrays.values()) + (4 << 20)


def _error_payload(error: Exception, *, rank: int) -> dict[str, object]:
    return {
        "rank": int(rank),
        "type": type(error).__name__,
        "errno": getattr(error, "errno", None),
        "message": str(error),
    }


def _raise_collective_checkpoint_error(payload: dict[str, object]) -> None:
    code = payload.get("errno")
    message = (
        f"partitioned checkpoint failed on rank {payload['rank']} "
        f"({payload['type']}): {payload['message']}"
    )
    if isinstance(code, int):
        raise OSError(code, message)
    raise RuntimeError(message)


def save_partitioned_solver_checkpoint(solver, directory) -> Path:
    """Publish a complete checkpoint without invalidating the prior generation."""
    from .checkpoint import config_hash

    target = Path(directory)
    preparation_error = None
    if solver.parallel.is_root:
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            preparation_error = _error_payload(error, rank=solver.parallel.rank)
    preparation_error = solver.parallel.bcast(preparation_error, root=0)
    if preparation_error is not None:
        _raise_collective_checkpoint_error(preparation_error)
    partition = solver.parallel.partition
    arrays = {
        "global_cell_ids": partition.local_global_ids,
        "global_face_ids": solver.mesh_data["global_face_ids"],
        "U": solver.U,
        "p": solver.p,
        "phi": solver.phi,
        "phi_old": solver.phi_old,
        "phi_old_old": solver.phi_old_old,
        "U_old": solver.U_old,
        "U_old_old": solver.U_old_old,
        "nut": np.asarray([]) if solver.nut is None else solver.nut,
        "time": np.asarray(solver.time),
        "step": np.asarray(solver.step),
        "n_committed": np.asarray(solver._n_committed),
        "dt": np.asarray(solver.time_step_size),
        "current_dt": np.asarray(solver._current_time_step_size),
        "cfl_max": np.asarray(solver.cfl_max),
        "time_since_last_write": np.asarray(solver._time_since_last_write),
        "acceptance_counts": np.asarray(
            [solver._acceptance_counts[name] for name in sorted(solver._acceptance_counts)],
            dtype=np.int64,
        ),
    }

    # All ranks write their temporary archives concurrently, so reserve space
    # for the aggregate uncompressed upper bound before any old restart can be
    # affected.  The old manifest remains valid throughout this operation.
    local_payload_bytes = _archive_upper_bound(arrays)
    payload_bytes = int(solver.parallel.global_sum(local_payload_bytes))
    capacity_error = None
    if solver.parallel.is_root:
        try:
            require_free_space(target, payload_bytes)
        except InsufficientStorageError as error:
            capacity_error = {
                "path": str(error.path),
                "required_bytes": error.required_bytes,
                "free_bytes": error.free_bytes,
            }
    capacity_error = solver.parallel.bcast(capacity_error, root=0)
    if capacity_error is not None:
        raise InsufficientStorageError(
            capacity_error["path"],
            int(capacity_error["required_bytes"]),
            int(capacity_error["free_bytes"]),
        )

    generation = solver.parallel.bcast(
        uuid4().hex if solver.parallel.is_root else None,
        root=0,
    )
    files = [f"rank-{rank:05d}-{generation}.npz" for rank in range(partition.size)]
    local_error = None
    try:
        _atomic_npz(target / files[partition.rank], arrays)
    except Exception as error:
        local_error = _error_payload(error, rank=partition.rank)
    errors = solver.parallel.comm.allgather(local_error)
    failure = next((error for error in errors if error is not None), None)
    if failure is not None:
        _raise_collective_checkpoint_error(failure)

    manifest_error = None
    if solver.parallel.is_root:
        manifest = {
            "format_version": PARTITIONED_CHECKPOINT_VERSION,
            "generation": generation,
            "config_hash": config_hash(solver.setup),
            "mesh_hash": solver.mesh_data["global_mesh_hash"],
            "global_n_cells": partition.global_n_cells,
            "ranks": partition.size,
            "files": files,
        }
        temporary = target / f".manifest-{generation}.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(json.dumps(manifest, indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target / "manifest.json")
        except Exception as error:
            if temporary.exists():
                temporary.unlink()
            manifest_error = _error_payload(error, rank=partition.rank)
    manifest_error = solver.parallel.bcast(manifest_error, root=0)
    if manifest_error is not None:
        _raise_collective_checkpoint_error(manifest_error)
    return target


def load_partitioned_solver_checkpoint(
    solver, directory, *, allow_config_change: bool = False
) -> None:
    """Restore a complete checkpoint for the same mesh and communicator size."""
    from .checkpoint import config_hash

    target = Path(directory)
    manifest = None
    if solver.parallel.is_root:
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    manifest = solver.parallel.bcast(manifest, root=0)
    version = manifest.get("format_version")
    if version not in (1, PARTITIONED_CHECKPOINT_VERSION):
        raise ValueError("Unsupported partitioned FVM checkpoint version")
    if manifest.get("ranks") != solver.parallel.size:
        raise ValueError("Partitioned checkpoint communicator size does not match")
    if manifest.get("mesh_hash") != solver.mesh_data.get("global_mesh_hash"):
        raise ValueError("Partitioned checkpoint mesh hash does not match")
    if not allow_config_change and manifest.get("config_hash") != config_hash(solver.setup):
        raise ValueError(
            "Partitioned checkpoint configuration hash does not match "
            f"(allow_config_change={allow_config_change})"
        )

    rank = solver.parallel.rank
    with np.load(target / manifest["files"][rank], allow_pickle=False) as archive:
        state = {name: np.array(archive[name], copy=True) for name in archive.files}
    if version == 1:
        state["phi_old"] = state["phi"].copy()
        state["phi_old_old"] = state["phi"].copy()
    partition = solver.parallel.partition
    if not np.array_equal(state.pop("global_cell_ids"), partition.local_global_ids):
        raise ValueError("Partitioned checkpoint cell IDs do not match")
    if not np.array_equal(state.pop("global_face_ids"), solver.mesh_data["global_face_ids"]):
        raise ValueError("Partitioned checkpoint face IDs do not match")
    for name in ("U", "p", "phi", "phi_old", "phi_old_old", "U_old", "U_old_old"):
        destination = np.asarray(getattr(solver, name))
        if state[name].shape != destination.shape or not np.all(np.isfinite(state[name])):
            raise ValueError(f"Partitioned checkpoint field {name} is incompatible")
        destination[:] = state[name]
    nut = state["nut"]
    if nut.size and (
        nut.shape != (solver.mesh_data["n_elements"],)
        or not np.all(np.isfinite(nut))
        or np.any(nut < 0.0)
    ):
        raise ValueError("Partitioned checkpoint turbulent viscosity is incompatible")
    solver.nut = None if not nut.size else nut
    solver.time = float(state["time"])
    solver.time_step_size = float(state["dt"])
    solver._current_time_step_size = float(state["current_dt"])
    solver.cfl_max = float(state["cfl_max"])
    solver._time_since_last_write = float(state["time_since_last_write"])
    solver.step = int(state["step"])
    solver._n_committed = int(state["n_committed"])
    acceptance_names = sorted(solver._acceptance_counts)
    if state["acceptance_counts"].shape != (len(acceptance_names),):
        raise ValueError("Partitioned checkpoint acceptance state is incompatible")
    solver._acceptance_counts.update(
        zip(acceptance_names, map(int, state["acceptance_counts"]), strict=True)
    )
    solver._last_residuals = None
    solver.last_diagnostics = None
    solver.parallel.barrier()


def write_partition_checkpoint(directory, partition, fields: dict[str, np.ndarray], comm) -> None:
    """Write one owned-cell archive per rank and a root manifest."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    n_owned = len(partition.owned_global_ids)
    payload = {"global_cell_ids": partition.owned_global_ids}
    for name, values in fields.items():
        array = np.asarray(values)
        if array.shape[0] != len(partition.local_global_ids):
            raise ValueError(f"Field {name!r} does not match the local partition")
        payload[name] = array[:n_owned]
    np.savez_compressed(target / f"rank-{partition.rank:05d}.npz", **payload)
    comm.Barrier()
    if partition.rank == 0:
        manifest = {
            "schema_version": 1,
            "global_n_cells": partition.global_n_cells,
            "ranks": partition.size,
            "files": [f"rank-{rank:05d}.npz" for rank in range(partition.size)],
            "fields": sorted(fields),
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    comm.Barrier()


def reconstruct_partition_checkpoint(directory) -> dict[str, np.ndarray]:
    """Reconstruct globally ordered fields for visualization or comparison."""
    target = Path(directory)
    manifest = json.loads((target / "manifest.json").read_text())
    fields: dict[str, np.ndarray] = {}
    for filename in manifest["files"]:
        with np.load(target / filename, allow_pickle=False) as archive:
            global_ids = archive["global_cell_ids"]
            for name in manifest["fields"]:
                values = archive[name]
                if name not in fields:
                    fields[name] = np.empty(
                        (manifest["global_n_cells"], *values.shape[1:]), dtype=values.dtype
                    )
                fields[name][global_ids] = values
    return fields


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
    directory,
    stem: str,
    mesh_data,
    partition,
    fields: dict[str, np.ndarray],
    comm,
    *,
    output: OutputSetup | None = None,
    exporter: VTKExporter | None = None,
) -> Path:
    """Atomically publish one rank piece and a root parallel collection.

    With one visualization ghost layer, each piece contains owned cells
    followed by face-adjacent halo cells marked with ``vtkGhostType``.  This
    lets ParaView's cell-to-point conversion remain smooth across partitions
    without changing the solver's ownership or numerical halo policy.
    """
    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be a non-empty filename component")
    output = output or OutputSetup()
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    n_owned = len(partition.owned_global_ids)
    local_count = len(partition.local_global_ids)
    local_fields = {}
    for name, values in fields.items():
        array = np.asarray(values)
        if array.shape[0] != local_count:
            raise ValueError(f"Field {name!r} does not match the local partition")
        _field_components(array)
        local_fields[name] = np.ascontiguousarray(array).copy()

    piece_name = f"{stem}-rank-{partition.rank:05d}.vtu"
    if output.ghost_layers == 1:
        if len(partition.ghost_global_ids):
            for values in local_fields.values():
                partition.exchange_halo(values, comm)
        _output_dtype = np.float32 if output.precision == "float32" else np.float64
        if _output_dtype != np.float64:
            _fmin = float(np.finfo(_output_dtype).min)
            _fmax = float(np.finfo(_output_dtype).max)
            for name in list(local_fields):
                array = local_fields[name]
                if np.issubdtype(array.dtype, np.floating):
                    local_fields[name] = np.clip(array, _fmin, _fmax).astype(_output_dtype)
        piece_fields = dict(local_fields)
        ghost_types = np.zeros(local_count, dtype=np.uint8)
        # vtkDataSetAttributes::DUPLICATECELL marks overlap supplied only for
        # parallel filters and prevents duplicate contributions.
        ghost_types[n_owned:] = 1
        piece_fields["vtkGhostType"] = ghost_types
        piece_fields["GlobalCellIds"] = np.ascontiguousarray(
            partition.local_global_ids,
            dtype=np.int64,
        )
        visualization_mesh = mesh_data.get("_visualization_mesh")
        if visualization_mesh is None:
            if mesh_data["n_elements"] != local_count:
                raise ValueError("Partitioned ghost output requires a visualization mesh")
            visualization_mesh = mesh_data
        point_fields = {
            "GlobalPointIds": np.ascontiguousarray(
                visualization_mesh.get(
                    "global_point_ids",
                    np.arange(len(visualization_mesh["points"])),
                ),
                dtype=np.int64,
            )
        }
        writer = exporter or VTKExporter(visualization_mesh, output)
        writer.export(
            str(target / piece_name),
            piece_fields,
            point_fields=point_fields,
        )
    else:
        piece_fields = {name: values[:n_owned] for name, values in local_fields.items()}
        piece_fields["GlobalCellIds"] = np.ascontiguousarray(
            partition.owned_global_ids,
            dtype=np.int64,
        )
        if mesh_data["n_elements"] == partition.global_n_cells:
            cell_ids = partition.owned_global_ids
        else:
            cell_ids = np.arange(n_owned, dtype=np.int64)
        export_mesh = mesh_data.get("_visualization_mesh", mesh_data)
        writer = exporter or VTKExporter(export_mesh, output)
        writer.export_cells(str(target / piece_name), cell_ids, piece_fields)
    comm.Barrier()

    collection = target / f"{stem}.pvtu"
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
                    '      <PDataArray type="Int64" Name="GlobalPointIds" NumberOfComponents="1"/>',
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
        atomic_write_text(collection, "\n".join(lines) + "\n")
    comm.Barrier()
    return collection
