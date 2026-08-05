"""Versioned, atomic restart files for the native FVM solver."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .storage import require_free_space

FORMAT_VERSION = 2


def _update_digest(digest, value) -> None:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"array")
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        # ``ndarray.tobytes()`` duplicates the complete array.  Mesh hashes are
        # computed while rank zero still owns the global mesh, so that copy can
        # be hundreds of megabytes.  hashlib accepts a contiguous buffer
        # directly and consumes it without changing the digest.
        digest.update(memoryview(array).cast("B"))
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=str):
            _update_digest(digest, str(key))
            _update_digest(digest, value[key])
    elif isinstance(value, list | tuple):
        digest.update(b"sequence")
        for item in value:
            _update_digest(digest, item)
    elif isinstance(value, np.generic):
        _update_digest(digest, value.item())
    else:
        digest.update(type(value).__name__.encode())
        digest.update(repr(value).encode())


def _hash(value) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def config_hash(config) -> str:
    """Return a deterministic hash of every public FVM configuration field."""
    return _hash(asdict(config))


def mesh_hash(mesh_data) -> str:
    """Hash mesh topology, coordinates, and stable patch identity."""
    patches = [
        {
            "name": patch["name"],
            "startFace": patch["startFace"],
            "nFaces": patch["nFaces"],
            "type": patch.get("type"),
        }
        for patch in mesh_data["boundary"]
    ]
    identity = {
        "points": mesh_data["points"],
        "faces": mesh_data["faces"],
        "owners": mesh_data["owners"],
        "neighbours": mesh_data["neighbours"],
        "boundary": patches,
        "n_elements": mesh_data["n_elements"],
        "n_faces": mesh_data["n_faces"],
        "n_interior_faces": mesh_data["n_interior_faces"],
    }
    return _hash(identity)


def save_checkpoint(solver, path) -> Path:
    """Atomically save all state needed to continue BDF1/BDF2 integration."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": FORMAT_VERSION,
        "config_hash": config_hash(solver.config),
        "mesh_hash": mesh_hash(solver.mesh_data),
    }
    arrays = {
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
        "U": solver.U,
        "p": solver.p,
        "phi": solver.phi,
        "phi_old": solver.phi_old,
        "phi_old_old": solver.phi_old_old,
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
    payload_bytes = sum(int(np.asarray(value).nbytes) + 4096 for value in arrays.values())
    require_free_space(destination, payload_bytes + (4 << 20))
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return destination


def load_checkpoint(solver, path, *, allow_config_change: bool = False) -> None:
    """Validate and restore a checkpoint without accepting partial state."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        base_required = {
            "metadata",
            "U",
            "p",
            "phi",
            "U_old",
            "U_old_old",
            "nut",
            "flow_time",
            "time_step",
            "n_committed",
            "dt",
            "current_dt",
            "cfl_max",
            "time_since_last_write",
            "force_log_counter",
            "acceptance_counts",
        }
        if "metadata" not in archive.files:
            raise ValueError("Incomplete FVM checkpoint; missing: metadata")
        metadata = json.loads(str(archive["metadata"]))
        version = metadata.get("format_version")
        if version not in (1, FORMAT_VERSION):
            raise ValueError(
                f"Unsupported FVM checkpoint version {version!r}; expected 1 or {FORMAT_VERSION}"
            )
        required = set(base_required)
        if version >= 2:
            required.update(("phi_old", "phi_old_old"))
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Incomplete FVM checkpoint; missing: {', '.join(missing)}")
        expected_mesh = mesh_hash(solver.mesh_data)
        if metadata.get("mesh_hash") != expected_mesh:
            raise ValueError("FVM checkpoint mesh hash does not match the active mesh")
        expected_config = config_hash(solver.config)
        if not allow_config_change and metadata.get("config_hash") != expected_config:
            raise ValueError("FVM checkpoint configuration hash does not match the active case")

        state = {name: np.array(archive[name], copy=True) for name in required - {"metadata"}}
        if version == 1:
            # Version 1 predates transient Rhie--Chow flux history.  It cannot
            # reproduce the first resumed ddtCorr exactly, but using the
            # committed flux for both old levels is the conservative backward-
            # compatibility choice.  All version-2 restarts are exact.
            state["phi_old"] = state["phi"].copy()
            state["phi_old_old"] = state["phi"].copy()

    for name in ("U", "p", "phi", "phi_old", "phi_old_old", "U_old", "U_old_old"):
        expected_shape = np.asarray(getattr(solver, name)).shape
        if state[name].shape != expected_shape:
            raise ValueError(
                f"Checkpoint field {name} has shape {state[name].shape}; expected {expected_shape}"
            )
        if not np.all(np.isfinite(state[name])):
            raise ValueError(f"Checkpoint field {name} contains non-finite values")
    nut = state["nut"]
    if solver.turbulence is None and nut.size:
        raise ValueError("Checkpoint contains turbulence state for a laminar solver")
    if solver.turbulence is not None and nut.shape != (solver.mesh_data["n_elements"],):
        raise ValueError("Checkpoint turbulent viscosity shape is incompatible with the mesh")
    if nut.size and (not np.all(np.isfinite(nut)) or np.any(nut < 0.0)):
        raise ValueError("Checkpoint turbulent viscosity is invalid")

    solver.U[:] = state["U"]
    solver.p[:] = state["p"]
    solver.phi[:] = state["phi"]
    solver.phi_old[:] = state["phi_old"]
    solver.phi_old_old[:] = state["phi_old_old"]
    solver.U_old[:] = state["U_old"]
    solver.U_old_old[:] = state["U_old_old"]
    solver.nut = None if not nut.size else nut
    solver.flow_time = float(state["flow_time"])
    solver.time_step = int(state["time_step"])
    solver._n_committed = int(state["n_committed"])
    solver.dt = float(state["dt"])
    solver._current_dt = float(state["current_dt"])
    solver.cfl_max = float(state["cfl_max"])
    solver._time_since_last_write = float(state["time_since_last_write"])
    solver._force_log_counter = int(state["force_log_counter"])
    acceptance_names = sorted(solver._acceptance_counts)
    if state["acceptance_counts"].shape != (len(acceptance_names),):
        raise ValueError("Checkpoint acceptance-policy state is incompatible")
    solver._acceptance_counts.update(
        zip(acceptance_names, (int(value) for value in state["acceptance_counts"]), strict=True)
    )
    solver._last_residuals = None
    solver.last_diagnostics = None
