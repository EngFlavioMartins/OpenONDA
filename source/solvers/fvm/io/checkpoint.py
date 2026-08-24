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

FORMAT_VERSION = 7


def _update_digest(digest, value) -> None:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"array")
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes(order="C"))
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


def _setup_dict(setup) -> dict:
    from source.solvers.fvm.sampling.base import sampler_to_dict

    data = asdict(setup)
    if getattr(setup, "samplers", ()):
        data["samplers"] = [sampler_to_dict(sampler) for sampler in setup.samplers]
    return data


def config_hash(setup) -> str:
    """Return a deterministic hash of the canonical FVM setup."""
    return _hash(_setup_dict(setup))


def mesh_hash(mesh_data) -> str:
    """Hash canonical mesh topology, coordinates, and stable patch identity."""
    patches = [
        {
            "name": patch["name"],
            "start_face": patch["start_face"],
            "n_faces": patch["n_faces"],
            "type": patch.get("type"),
        }
        for patch in mesh_data["boundary"]
    ]
    identity = {
        "vertex_position": mesh_data["vertex_position"],
        "faces": mesh_data["faces"],
        "owners": mesh_data["owners"],
        "neighbours": mesh_data["neighbours"],
        "boundary": patches,
        "n_cells": mesh_data["n_cells"],
        "n_faces": mesh_data["n_faces"],
        "n_interior_faces": mesh_data["n_interior_faces"],
    }
    return _hash(identity)


def save_checkpoint(solver, path) -> Path:
    """Atomically save all state required for an exact restart."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format_version": FORMAT_VERSION,
        "config_hash": config_hash(solver.setup),
        "mesh_hash": mesh_hash(solver.mesh_data),
    }
    arrays = {
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
        "velocity": solver.velocity,
        "kinematic_pressure": solver.kinematic_pressure,
        "volumetric_face_flux": solver.volumetric_face_flux,
        "volumetric_face_flux_old": solver.volumetric_face_flux_old,
        "volumetric_face_flux_older": solver.volumetric_face_flux_older,
        "velocity_old": solver.velocity_old,
        "velocity_older": solver.velocity_older,
        "eddy_viscosity": (
            np.asarray([]) if solver.eddy_viscosity is None else solver.eddy_viscosity
        ),
        "time": np.asarray(solver.time),
        "step": np.asarray(solver.step),
        "n_committed_time_steps": np.asarray(solver._n_committed_time_steps),
        "time_step_size": np.asarray(solver.time_step_size),
        "accepted_time_step_size": np.asarray(solver._accepted_time_step_size),
        "max_courant_number": np.asarray(solver.max_courant_number),
        "time_since_last_write": np.asarray(solver._time_since_last_write),
        "n_consecutive_accepted_steps": np.asarray(
            [
                solver._n_consecutive_accepted_steps[name]
                for name in sorted(solver._n_consecutive_accepted_steps)
            ],
            dtype=np.int64,
        ),
    }

    payload_bytes = sum(int(np.asarray(value).nbytes) + 4096 for value in arrays.values())
    require_free_space(
        destination,
        payload_bytes + (4 << 20),
    )

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
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
    """Validate and restore one canonical FVM checkpoint."""
    source = Path(path)
    required = {
        "metadata",
        "velocity",
        "kinematic_pressure",
        "volumetric_face_flux",
        "volumetric_face_flux_old",
        "volumetric_face_flux_older",
        "velocity_old",
        "velocity_older",
        "eddy_viscosity",
        "time",
        "step",
        "n_committed_time_steps",
        "time_step_size",
        "accepted_time_step_size",
        "max_courant_number",
        "time_since_last_write",
        "n_consecutive_accepted_steps",
    }
    with np.load(source, allow_pickle=False) as archive:
        archive_keys = set(archive.files)
        if archive_keys != required:
            missing = sorted(required - archive_keys)
            unexpected = sorted(archive_keys - required)
            raise ValueError(
                f"Invalid FVM checkpoint fields; missing={missing}, unexpected={unexpected}"
            )
        metadata = json.loads(str(archive["metadata"]))
        metadata_keys = set(metadata)
        expected_metadata = {"format_version", "config_hash", "mesh_hash"}
        if metadata_keys != expected_metadata:
            raise ValueError(
                "Invalid FVM checkpoint metadata; "
                f"missing={sorted(expected_metadata - metadata_keys)}, "
                f"unexpected={sorted(metadata_keys - expected_metadata)}"
            )
        version = int(metadata.get("format_version", -1))
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported FVM checkpoint version {version!r}; expected {FORMAT_VERSION}"
            )
        if metadata.get("mesh_hash") != mesh_hash(solver.mesh_data):
            raise ValueError("FVM checkpoint mesh hash does not match the active mesh")
        if not allow_config_change and metadata.get("config_hash") != config_hash(solver.setup):
            raise ValueError("FVM checkpoint configuration hash does not match the active case")
        state = {
            name: np.array(archive[name], copy=True) for name in archive.files if name != "metadata"
        }

    missing = sorted((required - {"metadata"}) - set(state))
    if missing:
        raise ValueError("Incomplete FVM checkpoint; missing: " + ", ".join(missing))

    fields = {
        "velocity": "velocity",
        "kinematic_pressure": "kinematic_pressure",
        "volumetric_face_flux": "volumetric_face_flux",
        "volumetric_face_flux_old": "volumetric_face_flux_old",
        "volumetric_face_flux_older": "volumetric_face_flux_older",
        "velocity_old": "velocity_old",
        "velocity_older": "velocity_older",
    }
    for field_name, attribute_name in fields.items():
        active = np.asarray(getattr(solver, attribute_name))
        values = state[field_name]
        if values.shape != active.shape:
            raise ValueError(
                f"Checkpoint field {field_name} has shape {values.shape}; expected {active.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Checkpoint field {field_name} contains non-finite values")

    eddy_viscosity = state["eddy_viscosity"]
    if solver.turbulence is None and eddy_viscosity.size:
        raise ValueError("Checkpoint contains turbulence state for a laminar solver")
    if solver.turbulence is not None and eddy_viscosity.shape != (solver.mesh_data["n_cells"],):
        raise ValueError("Checkpoint eddy-viscosity shape is incompatible with the mesh")
    if eddy_viscosity.size and (
        not np.all(np.isfinite(eddy_viscosity)) or np.any(eddy_viscosity < 0.0)
    ):
        raise ValueError("Checkpoint eddy viscosity is invalid")

    for name in (
        "velocity",
        "kinematic_pressure",
        "volumetric_face_flux",
        "volumetric_face_flux_old",
        "volumetric_face_flux_older",
        "velocity_old",
        "velocity_older",
    ):
        getattr(solver, name)[:] = state[name]
    solver.eddy_viscosity = None if not eddy_viscosity.size else eddy_viscosity
    solver.time = float(state["time"])
    solver.step = int(state["step"])
    solver._n_committed_time_steps = int(state["n_committed_time_steps"])
    solver.time_step_size = float(state["time_step_size"])
    solver._accepted_time_step_size = float(state["accepted_time_step_size"])
    solver.max_courant_number = float(state["max_courant_number"])
    solver._time_since_last_write = float(state["time_since_last_write"])
    acceptance_names = sorted(solver._n_consecutive_accepted_steps)
    if state["n_consecutive_accepted_steps"].shape != (len(acceptance_names),):
        raise ValueError("Checkpoint acceptance-policy state is incompatible")
    solver._n_consecutive_accepted_steps.update(
        zip(acceptance_names, (int(v) for v in state["n_consecutive_accepted_steps"]), strict=True)
    )
    solver._last_residuals = None
    solver.last_diagnostics = None
