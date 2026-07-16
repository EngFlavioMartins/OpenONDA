"""Conflict-free partitioned checkpoint output and reconstruction."""

from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .vtk_exporter import VTKExporter

PARTITIONED_CHECKPOINT_VERSION = 1


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


def save_partitioned_solver_checkpoint(solver, directory) -> Path:
    """Atomically write complete rank-local PIMPLE state and a root manifest."""
    from .checkpoint import config_hash

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    partition = solver.parallel.partition
    arrays = {
        "global_cell_ids": partition.local_global_ids,
        "global_face_ids": solver.mesh_data["global_face_ids"],
        "U": solver.U,
        "p": solver.p,
        "phi": solver.phi,
        "U_old": solver.U_old,
        "U_old_old": solver.U_old_old,
        "nut": np.asarray([]) if solver.nut is None else solver.nut,
        "flow_time": np.asarray(solver.flow_time),
        "time_step": np.asarray(solver.time_step),
        "n_committed": np.asarray(solver._n_committed),
        "dt": np.asarray(solver.dt),
        "current_dt": np.asarray(solver._current_dt),
        "cfl_max": np.asarray(solver.cfl_max),
        "time_since_last_write": np.asarray(solver._time_since_last_write),
        "force_log_counter": np.asarray(solver._force_log_counter),
        "acceptance_counts": np.asarray(
            [solver._acceptance_counts[name] for name in sorted(solver._acceptance_counts)],
            dtype=np.int64,
        ),
    }
    _atomic_npz(target / f"rank-{partition.rank:05d}.npz", arrays)
    solver.parallel.barrier()
    if solver.parallel.is_root:
        manifest = {
            "format_version": PARTITIONED_CHECKPOINT_VERSION,
            "config_hash": config_hash(solver.config),
            "mesh_hash": solver.mesh_data["global_mesh_hash"],
            "global_n_cells": partition.global_n_cells,
            "ranks": partition.size,
            "files": [f"rank-{rank:05d}.npz" for rank in range(partition.size)],
        }
        temporary = target / ".manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target / "manifest.json")
    solver.parallel.barrier()
    return target


def load_partitioned_solver_checkpoint(solver, directory) -> None:
    """Restore a complete checkpoint for the same mesh and communicator size."""
    from .checkpoint import config_hash

    target = Path(directory)
    manifest = None
    if solver.parallel.is_root:
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    manifest = solver.parallel.bcast(manifest, root=0)
    if manifest.get("format_version") != PARTITIONED_CHECKPOINT_VERSION:
        raise ValueError("Unsupported partitioned FVM checkpoint version")
    if manifest.get("ranks") != solver.parallel.size:
        raise ValueError("Partitioned checkpoint communicator size does not match")
    if manifest.get("mesh_hash") != solver.mesh_data.get("global_mesh_hash"):
        raise ValueError("Partitioned checkpoint mesh hash does not match")
    if manifest.get("config_hash") != config_hash(solver.config):
        raise ValueError("Partitioned checkpoint configuration hash does not match")

    rank = solver.parallel.rank
    with np.load(target / manifest["files"][rank], allow_pickle=False) as archive:
        state = {name: np.array(archive[name], copy=True) for name in archive.files}
    partition = solver.parallel.partition
    if not np.array_equal(state.pop("global_cell_ids"), partition.local_global_ids):
        raise ValueError("Partitioned checkpoint cell IDs do not match")
    if not np.array_equal(state.pop("global_face_ids"), solver.mesh_data["global_face_ids"]):
        raise ValueError("Partitioned checkpoint face IDs do not match")
    for name in ("U", "p", "phi", "U_old", "U_old_old"):
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
    solver.flow_time = float(state["flow_time"])
    solver.dt = float(state["dt"])
    solver._current_dt = float(state["current_dt"])
    solver.cfl_max = float(state["cfl_max"])
    solver._time_since_last_write = float(state["time_since_last_write"])
    solver.time_step = int(state["time_step"])
    solver._n_committed = int(state["n_committed"])
    solver._force_log_counter = int(state["force_log_counter"])
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


def write_partition_vtu(
    directory,
    stem: str,
    mesh_data,
    partition,
    fields: dict[str, np.ndarray],
    comm,
) -> Path:
    """Write one owned-cell VTU per rank and a root PVTU collection."""
    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be a non-empty filename component")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    n_owned = len(partition.owned_global_ids)
    owned_fields = {}
    for name, values in fields.items():
        array = np.asarray(values)
        if array.shape[0] != len(partition.local_global_ids):
            raise ValueError(f"Field {name!r} does not match the local partition")
        owned_fields[name] = array[:n_owned]

    piece_name = f"{stem}-rank-{partition.rank:05d}.vtu"
    if mesh_data["n_elements"] == partition.global_n_cells:
        cell_ids = partition.owned_global_ids
    else:
        cell_ids = np.arange(n_owned, dtype=np.int64)
    VTKExporter(mesh_data).export_cells(str(target / piece_name), cell_ids, owned_fields)
    comm.Barrier()

    collection = target / f"{stem}.pvtu"
    if partition.rank == 0:
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian">',
            '  <PUnstructuredGrid GhostLevel="0">',
            "    <PCellData>",
        ]
        for name, values in owned_fields.items():
            components = values.shape[1] if values.ndim == 2 else 1
            lines.append(
                f'      <PDataArray type="Float64" Name="{escape(name)}" '
                f'NumberOfComponents="{components}"/>'
            )
        lines.extend(
            [
                "    </PCellData>",
                "    <PPoints>",
                '      <PDataArray type="Float64" NumberOfComponents="3"/>',
                "    </PPoints>",
            ]
        )
        lines.extend(
            f'    <Piece Source="{stem}-rank-{rank:05d}.vtu"/>' for rank in range(partition.size)
        )
        lines.extend(["  </PUnstructuredGrid>", "</VTKFile>"])
        collection.write_text("\n".join(lines) + "\n", encoding="utf-8")
    comm.Barrier()
    return collection
