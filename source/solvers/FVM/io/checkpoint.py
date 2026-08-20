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

FORMAT_VERSION = 3


def _update_digest(digest, value) -> None:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"array")
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
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


def _setup_dict(setup) -> dict:
    from source.solvers.FVM.sampling.base import sampler_to_dict

    data = asdict(setup)
    if getattr(setup, "samplers", ()):
        data["samplers"] = [sampler_to_dict(sampler) for sampler in setup.samplers]
    return data


def config_hash(setup) -> str:
    """Return a deterministic hash of the canonical FVM setup."""
    return _hash(_setup_dict(setup))


def _legacy_setup_dict(setup) -> dict:
    """Reconstruct the pre-nomenclature serialized setup for old hashes."""
    data = _setup_dict(setup)

    def rename(section: dict, mapping: dict[str, str]) -> None:
        for canonical, legacy in mapping.items():
            if canonical in section:
                section[legacy] = section.pop(canonical)

    rename(
        data.get("time", {}),
        {
            "output_interval_steps": "output_interval_steps",
            "output_interval_time": "output_interval_time",
        },
    )
    rename(
        data.get("linear", {}),
        {
            "momentum_tolerance": "momentum_tolerance",
            "momentum_relative_tolerance": "momentum_relative_tolerance",
            "momentum_final_relative_tolerance": "momentum_final_relative_tolerance",
            "momentum_max_iterations": "momentum_max_iterations",
            "pressure_tolerance": "pressure_tolerance",
            "pressure_relative_tolerance": "pressure_relative_tolerance",
            "pressure_final_relative_tolerance": "pressure_final_relative_tolerance",
            "pressure_max_iterations": "pressure_max_iterations",
            "amg_tolerance": "amg_tolerance",
            "amg_max_iterations": "amg_max_iterations",
            "amg_reuse_tolerance": "amg_reuse_tolerance",
            "ilu_drop_tolerance": "ilu_drop_tolerance",
            "ilu_reuse_tolerance": "ilu_reuse_tolerance",
        },
    )
    rename(
        data.get("pimple", {}),
        {
            "velocity_relaxation": "velocity_relaxation",
            "pressure_relaxation": "pressure_relaxation",
        },
    )
    rename(
        data.get("transport", {}),
        {"kinematic_viscosity": "nu"},
    )
    rename(
        data.get("logging", {}),
        {"interval_steps": "interval"},
    )

    boundary_mapping = {
        "velocity_type": "type_velocity",
        "velocity_value": "velocity_value",
        "pressure_type": "type_p",
        "kinematic_pressure_value": "kinematic_pressure_value",
        "flux_type": "type_phi",
        "flux_value": "flux_value",
        "eddy_viscosity_type": "type_nut",
        "eddy_viscosity_value": "eddy_viscosity_value",
    }
    for boundary in data.get("boundaries", []):
        rename(boundary, boundary_mapping)

    turbulence = data.get("turbulence")
    if turbulence:
        model = str(turbulence.get("model", "None")).lower()
        if model == "wale":
            model_coefficient = turbulence.get("c_w", 0.325)
        elif model == "sigma":
            model_coefficient = turbulence.get("c_sigma", 1.35)
        else:
            model_coefficient = turbulence.get("c_s", 0.17)

        turbulence["Cs"] = model_coefficient
        turbulence["Ck"] = turbulence.pop("c_k", 0.094)
        turbulence["Ce"] = turbulence.pop("c_e", 1.048)
        turbulence.pop("c_s", None)
        turbulence.pop("c_w", None)
        turbulence.pop("c_sigma", None)

    if "initial_kinematic_pressure" in data:
        data["initial_kinematic_pressure"] = data.pop("initial_kinematic_pressure")

    return data


def legacy_config_hash(setup) -> str:
    """Hash the setup using the pre-migration serialized vocabulary."""
    return _hash(_legacy_setup_dict(setup))


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
        "points": mesh_data["points"],
        "faces": mesh_data["faces"],
        "owners": mesh_data["owners"],
        "neighbours": mesh_data["neighbours"],
        "boundary": patches,
        "n_cells": mesh_data["n_cells"],
        "n_faces": mesh_data["n_faces"],
        "n_interior_faces": mesh_data["n_interior_faces"],
    }
    return _hash(identity)


def legacy_mesh_hash(mesh_data) -> str:
    """Hash a canonical mesh as the pre-migration mesh dictionary."""
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
        "points": mesh_data["points"],
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
        "face_flux": solver.face_flux,
        "face_flux_old": solver.face_flux_old,
        "face_flux_older": solver.face_flux_older,
        "velocity_old": solver.velocity_old,
        "velocity_older": solver.velocity_older,
        "eddy_viscosity": (
            np.asarray([]) if solver.eddy_viscosity is None else solver.eddy_viscosity
        ),
        "time": np.asarray(solver.time),
        "step": np.asarray(solver.step),
        "n_committed": np.asarray(solver._n_committed),
        "time_step_size": np.asarray(solver.time_step_size),
        "current_time_step_size": np.asarray(solver._current_time_step_size),
        "cfl_max": np.asarray(solver.cfl_max),
        "time_since_last_write": np.asarray(solver._time_since_last_write),
        "acceptance_counts": np.asarray(
            [solver._acceptance_counts[name] for name in sorted(solver._acceptance_counts)],
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


_FIELD_ALIASES = {
    "velocity": ("velocity", "U"),
    "kinematic_pressure": ("kinematic_pressure", "p"),
    "face_flux": ("face_flux", "phi"),
    "face_flux_old": ("face_flux_old", "phi_old"),
    "face_flux_older": ("face_flux_older", "phi_old_old"),
    "velocity_old": ("velocity_old", "U_old"),
    "velocity_older": ("velocity_older", "U_old_old"),
    "eddy_viscosity": ("eddy_viscosity", "nut"),
    "time_step_size": ("time_step_size", "dt"),
    "current_time_step_size": (
        "current_time_step_size",
        "current_dt",
    ),
}


def _read_alias(archive, canonical: str):
    aliases = _FIELD_ALIASES.get(canonical, (canonical,))
    for name in aliases:
        if name in archive.files:
            return np.array(archive[name], copy=True)
    raise KeyError(canonical)


def _has_alias(archive, canonical: str) -> bool:
    return any(name in archive.files for name in _FIELD_ALIASES.get(canonical, (canonical,)))


def load_checkpoint(
    solver,
    path,
    *,
    allow_config_change: bool = False,
) -> None:
    """Validate and restore canonical or pre-migration FVM checkpoints."""
    source = Path(path)

    with np.load(source, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError("Incomplete FVM checkpoint; missing: metadata")

        metadata = json.loads(str(archive["metadata"]))
        version = int(metadata.get("format_version", 1))
        if version not in (1, 2, FORMAT_VERSION):
            raise ValueError(
                f"Unsupported FVM checkpoint version {version!r}; "
                f"expected 1, 2, or {FORMAT_VERSION}"
            )

        required = {
            "velocity",
            "kinematic_pressure",
            "face_flux",
            "velocity_old",
            "velocity_older",
            "eddy_viscosity",
            "time",
            "step",
            "n_committed",
            "time_step_size",
            "current_time_step_size",
            "cfl_max",
            "time_since_last_write",
            "acceptance_counts",
        }
        if version >= 2:
            required.update(("face_flux_old", "face_flux_older"))

        missing = sorted(
            name
            for name in required
            if (name not in archive.files and not _has_alias(archive, name))
        )
        if missing:
            raise ValueError("Incomplete FVM checkpoint; missing: " + ", ".join(missing))

        active_mesh_hash = mesh_hash(solver.mesh_data)
        saved_mesh_hash = metadata.get("mesh_hash")
        mesh_matches = saved_mesh_hash == active_mesh_hash
        if not mesh_matches and version < FORMAT_VERSION:
            mesh_matches = saved_mesh_hash == legacy_mesh_hash(solver.mesh_data)
        if not mesh_matches:
            raise ValueError("FVM checkpoint mesh hash does not match the active mesh")

        if not allow_config_change:
            saved_config_hash = metadata.get("config_hash")
            config_matches = saved_config_hash == config_hash(solver.setup)
            if not config_matches and version < FORMAT_VERSION:
                config_matches = saved_config_hash == legacy_config_hash(solver.setup)
            if not config_matches:
                raise ValueError("FVM checkpoint configuration hash does not match the active case")

        state = {}
        for name in required:
            if name in _FIELD_ALIASES:
                state[name] = _read_alias(archive, name)
            else:
                state[name] = np.array(
                    archive[name],
                    copy=True,
                )

        if version == 1:
            state["face_flux_old"] = state["face_flux"].copy()
            state["face_flux_older"] = state["face_flux"].copy()

    field_attributes = {
        "velocity": "velocity",
        "kinematic_pressure": "kinematic_pressure",
        "face_flux": "face_flux",
        "face_flux_old": "face_flux_old",
        "face_flux_older": "face_flux_older",
        "velocity_old": "velocity_old",
        "velocity_older": "velocity_older",
    }
    for field_name, attribute_name in field_attributes.items():
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

    solver.velocity[:] = state["velocity"]
    solver.kinematic_pressure[:] = state["kinematic_pressure"]
    solver.face_flux[:] = state["face_flux"]
    solver.face_flux_old[:] = state["face_flux_old"]
    solver.face_flux_older[:] = state["face_flux_older"]
    solver.velocity_old[:] = state["velocity_old"]
    solver.velocity_older[:] = state["velocity_older"]
    solver.eddy_viscosity = None if not eddy_viscosity.size else eddy_viscosity

    solver.time = float(state["time"])
    solver.step = int(state["step"])
    solver._n_committed = int(state["n_committed"])
    solver.time_step_size = float(state["time_step_size"])
    solver._current_time_step_size = float(state["current_time_step_size"])
    solver.cfl_max = float(state["cfl_max"])
    solver._time_since_last_write = float(state["time_since_last_write"])

    acceptance_names = sorted(solver._acceptance_counts)
    if state["acceptance_counts"].shape != (len(acceptance_names),):
        raise ValueError("Checkpoint acceptance-policy state is incompatible")
    solver._acceptance_counts.update(
        zip(
            acceptance_names,
            (int(value) for value in state["acceptance_counts"]),
            strict=True,
        )
    )

    solver._last_residuals = None
    solver.last_diagnostics = None
