"""Versioned, atomic restart files for the native FVM solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .storage import require_free_space

FORMAT_VERSION = 10


@dataclass(frozen=True)
class RestartPayload:
    """Fully validated restart data, detached from a live solver."""

    fields: dict[str, np.ndarray]
    eddy_viscosity: np.ndarray
    time: float
    step: int
    n_committed_time_steps: int
    time_step_size: float
    accepted_time_step_size: float
    previous_time_step_size: float
    kinematic_viscosity: float
    max_courant_number: float
    n_consecutive_accepted_steps: dict[str, int]


# Deflate finds almost nothing in raw float64: the mantissa bytes look like
# noise. Two lossless transforms expose the structure that is really there.
# A byte-plane shuffle groups each array's exponent bytes together, and the
# history fields are stored as a bit XOR against the level they follow, which
# leaves mostly zeros because consecutive time levels share their leading
# bytes. Both invert exactly, so a restart is still bit-for-bit.
_HISTORY_REFERENCE = {
    "velocity_old": "velocity",
    "velocity_older": "velocity",
    "volumetric_face_flux_old": "volumetric_face_flux",
    "volumetric_face_flux_older": "volumetric_face_flux",
}
_SHUFFLE_THRESHOLD_BYTES = 4096


def _shuffled(array: np.ndarray) -> np.ndarray:
    """Return the byte-plane transpose of *array*."""
    contiguous = np.ascontiguousarray(array)
    return contiguous.view(np.uint8).reshape(-1, contiguous.dtype.itemsize).T.copy()


def _unshuffled(planes: np.ndarray, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    """Invert :func:`_shuffled` back to the original dtype and shape."""
    return np.ascontiguousarray(planes.T).view(dtype).reshape(shape)


def _contiguous(values) -> np.ndarray:
    """Return a contiguous array without changing a scalar's shape."""
    array = np.asarray(values)
    return array if array.ndim == 0 else np.ascontiguousarray(array)


def encode_state(arrays: dict) -> dict:
    """Return the stored form of one backup payload."""
    encoded: dict = {}
    layout: dict[str, list] = {}
    for name, value in arrays.items():
        array = _contiguous(value)
        reference = _HISTORY_REFERENCE.get(name)
        if reference is not None:
            array = array.view(np.uint64) ^ _contiguous(arrays[reference]).view(np.uint64)
        if array.ndim and array.nbytes >= _SHUFFLE_THRESHOLD_BYTES:
            layout[name] = [str(array.dtype), list(array.shape)]
            array = _shuffled(array)
        encoded[name] = array
    encoded["storage_layout"] = np.asarray(json.dumps(layout, sort_keys=True))
    return encoded


def decode_state(stored: dict) -> dict:
    """Invert :func:`encode_state`, returning the original arrays."""
    layout = json.loads(str(stored["storage_layout"]))
    decoded = {name: value for name, value in stored.items() if name != "storage_layout"}
    for name, (dtype_name, shape) in layout.items():
        decoded[name] = _unshuffled(decoded[name], np.dtype(dtype_name), tuple(shape))
    for name, reference in _HISTORY_REFERENCE.items():
        if name in decoded:
            decoded[name] = (
                decoded[name].view(np.uint64)
                ^ np.ascontiguousarray(decoded[reference]).view(np.uint64)
            ).view(_contiguous(decoded[reference]).dtype)
    return decoded


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


def _setup_dict(setup, *, numerical_only: bool = True) -> dict:
    from source.solvers.fvm.sampling.base import sampler_to_dict

    data = asdict(setup)
    if numerical_only:
        # Output cadence, log policy, restart location and the requested
        # horizon are lifecycle controls.  They may change when extending or
        # relocating a run without changing the equations represented by a
        # numerical restart.
        data.pop("output", None)
        data.pop("logging", None)
        data.pop("backup", None)
        data.pop("samplers", None)
        time_data = dict(data.get("time") or {})
        time_data.pop("end_time", None)
        time_data.pop("output_schedule", None)
        data["time"] = time_data
    if not numerical_only and getattr(setup, "samplers", ()):
        data["samplers"] = [sampler_to_dict(sampler) for sampler in setup.samplers]
    return data


def _solver_setup(solver):
    """Return the detached setup snapshot admitted by the solver."""
    return getattr(solver, "_resolved_setup", solver.setup)


def config_hash(setup) -> str:
    """Return a deterministic hash of equation-affecting FVM controls."""
    return _hash(_setup_dict(setup, numerical_only=True))


def full_config_hash(setup) -> str:
    """Return a hash including lifecycle/output policy for provenance."""
    return _hash(_setup_dict(setup, numerical_only=False))


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


def save_backup(solver, path) -> Path:
    """Atomically save all state required for an exact restart."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format_version": FORMAT_VERSION,
        "config_hash": config_hash(_solver_setup(solver)),
        "mesh_hash": mesh_hash(solver.mesh_data),
        "kinematic_viscosity": float(
            getattr(
                solver, "_kinematic_viscosity", _solver_setup(solver).transport.kinematic_viscosity
            )
        ),
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
        "previous_time_step_size": np.asarray(solver._previous_time_step_size),
        "max_courant_number": np.asarray(solver.max_courant_number),
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

    stored = encode_state(arrays)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **stored)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise

    return destination


def _read_scalar(state: dict, name: str, *, kind: str):
    """Read one scalar with an exact archive shape and numeric kind."""
    value = np.asarray(state[name])
    if value.shape != ():
        raise ValueError(f"Backup field {name} must be a scalar; got shape {value.shape}")
    if kind == "float":
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"Backup field {name} must use a floating-point dtype")
        scalar = float(value.item())
        if not np.isfinite(scalar):
            raise ValueError(f"Backup field {name} must be finite")
        return scalar
    if kind == "int":
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError(f"Backup field {name} must use an integer dtype")
        return int(value.item())
    raise AssertionError(f"unknown scalar kind {kind!r}")


def stage_restart_payload(
    solver,
    state: dict,
    *,
    allow_config_change: bool = False,
    kinematic_viscosity: float | None = None,
) -> RestartPayload:
    """Validate a decoded backup without mutating *solver*.

    This is intentionally shared by serial and partitioned admission.  A
    caller can collect validation errors from every rank before publishing any
    fields, which makes restart rejection transactional at the solver boundary.
    """
    field_names = (
        "velocity",
        "kinematic_pressure",
        "volumetric_face_flux",
        "volumetric_face_flux_old",
        "volumetric_face_flux_older",
        "velocity_old",
        "velocity_older",
    )
    fields: dict[str, np.ndarray] = {}
    for field_name in field_names:
        active = np.asarray(getattr(solver, field_name))
        values = np.asarray(state[field_name])
        if values.shape != active.shape:
            raise ValueError(
                f"Backup field {field_name} has shape {values.shape}; expected {active.shape}"
            )
        if values.dtype != np.dtype(np.float64):
            raise ValueError(f"Backup field {field_name} must use float64; got {values.dtype}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Backup field {field_name} contains non-finite values")
        fields[field_name] = np.ascontiguousarray(values)

    eddy_viscosity = np.asarray(state["eddy_viscosity"])
    if eddy_viscosity.dtype != np.dtype(np.float64):
        raise ValueError(f"Backup eddy viscosity must use float64; got {eddy_viscosity.dtype}")
    if solver.turbulence is None:
        if eddy_viscosity.size:
            raise ValueError("Backup contains turbulence state for a laminar solver")
    elif eddy_viscosity.shape != (solver.mesh_data["n_cells"],):
        raise ValueError("Backup eddy-viscosity shape is incompatible with the mesh")
    if eddy_viscosity.size and (
        not np.all(np.isfinite(eddy_viscosity)) or np.any(eddy_viscosity < 0.0)
    ):
        raise ValueError("Backup eddy viscosity is invalid")

    time = _read_scalar(state, "time", kind="float")
    step = _read_scalar(state, "step", kind="int")
    n_committed = _read_scalar(state, "n_committed_time_steps", kind="int")
    time_step_size = _read_scalar(state, "time_step_size", kind="float")
    accepted_time_step_size = _read_scalar(state, "accepted_time_step_size", kind="float")
    previous_time_step_size = _read_scalar(state, "previous_time_step_size", kind="float")
    max_courant_number = _read_scalar(state, "max_courant_number", kind="float")
    if step < 0 or n_committed < 0:
        raise ValueError("Backup step counters must be non-negative")
    if step != n_committed:
        raise ValueError(
            "Backup step and n_committed_time_steps must describe the same accepted state"
        )
    if any(
        value <= 0.0 for value in (time_step_size, accepted_time_step_size, previous_time_step_size)
    ):
        raise ValueError("Backup time-step sizes must be finite and positive")
    if max_courant_number < 0.0:
        raise ValueError("Backup max_courant_number must be finite and non-negative")

    if kinematic_viscosity is None:
        kinematic_viscosity = float(
            getattr(
                solver, "_kinematic_viscosity", _solver_setup(solver).transport.kinematic_viscosity
            )
        )
    if not np.isfinite(kinematic_viscosity) or kinematic_viscosity <= 0.0:
        raise ValueError("Restart molecular viscosity must be finite and positive")

    start_time = float(solver._time_config.start_time)
    if time < start_time - max(1.0e-12, abs(start_time) * 1.0e-12):
        raise ValueError(f"Backup time {time} precedes the configured start time {start_time}")
    if not allow_config_change:
        end_time = float(solver._time_config.end_time)
        if time > end_time + max(1.0e-12, abs(end_time) * 1.0e-12):
            raise ValueError(f"Backup time {time} exceeds the configured end time {end_time}")

    counter_values = np.asarray(state["n_consecutive_accepted_steps"])
    names = sorted(solver._n_consecutive_accepted_steps)
    if counter_values.shape != (len(names),):
        raise ValueError("Backup acceptance-limit state is incompatible")
    if not np.issubdtype(counter_values.dtype, np.integer):
        raise ValueError("Backup acceptance counters must use an integer dtype")
    counters = {name: int(value) for name, value in zip(names, counter_values, strict=True)}
    if any(value < 0 for value in counters.values()):
        raise ValueError("Backup acceptance counters must be non-negative")

    return RestartPayload(
        fields=fields,
        eddy_viscosity=np.ascontiguousarray(eddy_viscosity),
        time=time,
        step=step,
        n_committed_time_steps=n_committed,
        time_step_size=time_step_size,
        accepted_time_step_size=accepted_time_step_size,
        previous_time_step_size=previous_time_step_size,
        kinematic_viscosity=kinematic_viscosity,
        max_courant_number=max_courant_number,
        n_consecutive_accepted_steps=counters,
    )


def publish_restart_payload(solver, payload: RestartPayload) -> None:
    """Publish a staged payload after all admission checks have succeeded."""
    for name, values in payload.fields.items():
        getattr(solver, name)[:] = values
    solver.eddy_viscosity = (
        None if not payload.eddy_viscosity.size else payload.eddy_viscosity.copy()
    )
    solver.time = payload.time
    solver.step = payload.step
    solver._n_committed_time_steps = payload.n_committed_time_steps
    solver.time_step_size = payload.time_step_size
    solver._accepted_time_step_size = payload.accepted_time_step_size
    solver._previous_time_step_size = payload.previous_time_step_size
    solver._kinematic_viscosity = payload.kinematic_viscosity
    solver.max_courant_number = payload.max_courant_number
    solver._n_consecutive_accepted_steps.update(payload.n_consecutive_accepted_steps)
    solver._last_residuals = None
    solver.last_diagnostics = None
    solver._invalidate_derived_fields()
    if hasattr(solver, "_publish_state"):
        solver._publish_state()
    if hasattr(solver, "_pending_step_size"):
        solver._pending_step_size = None
        solver._pending_acceptance_counters = None
        solver._step_phase = "accepted"
        solver._evolution_failure = None


def capture_restart_payload(solver) -> RestartPayload:
    """Copy the canonical accepted state into memory for coupling sweeps.

    This has the same numerical content as a disk restart, without compression,
    mesh hashing or filesystem traffic. Each MPI rank captures its local state.
    """
    if solver._step_phase != "accepted":
        raise RuntimeError("A coupling snapshot requires an accepted FVM state")
    names = (
        "velocity",
        "kinematic_pressure",
        "volumetric_face_flux",
        "volumetric_face_flux_old",
        "volumetric_face_flux_older",
        "velocity_old",
        "velocity_older",
    )
    return RestartPayload(
        fields={name: getattr(solver, name).copy() for name in names},
        eddy_viscosity=(
            np.empty(0) if solver.eddy_viscosity is None else solver.eddy_viscosity.copy()
        ),
        time=solver.time,
        step=solver.step,
        n_committed_time_steps=solver._n_committed_time_steps,
        time_step_size=solver.time_step_size,
        accepted_time_step_size=solver._accepted_time_step_size,
        previous_time_step_size=solver._previous_time_step_size,
        kinematic_viscosity=solver._kinematic_viscosity,
        max_courant_number=solver.max_courant_number,
        n_consecutive_accepted_steps=solver._n_consecutive_accepted_steps.copy(),
    )


def _load_backup_local(solver, path, *, allow_config_change: bool = False) -> RestartPayload:
    """Validate and restore one canonical FVM backup."""
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
        "previous_time_step_size",
        "max_courant_number",
        "n_consecutive_accepted_steps",
        "storage_layout",
    }
    with np.load(source, allow_pickle=False) as archive:
        archive_keys = set(archive.files)
        if archive_keys != required:
            missing = sorted(required - archive_keys)
            unexpected = sorted(archive_keys - required)
            raise ValueError(
                f"Invalid FVM backup fields; missing={missing}, unexpected={unexpected}"
            )
        metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
        metadata_keys = set(metadata)
        expected_metadata = {
            "format_version",
            "config_hash",
            "mesh_hash",
            "kinematic_viscosity",
        }
        if metadata_keys != expected_metadata:
            raise ValueError(
                "Invalid FVM backup metadata; "
                f"missing={sorted(expected_metadata - metadata_keys)}, "
                f"unexpected={sorted(metadata_keys - expected_metadata)}"
            )
        version = int(metadata.get("format_version", -1))
        if version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported FVM backup version {version!r}; expected {FORMAT_VERSION}"
            )
        if metadata.get("mesh_hash") != mesh_hash(solver.mesh_data):
            raise ValueError("FVM backup mesh hash does not match the active mesh")
        if not allow_config_change and metadata.get("config_hash") != config_hash(
            _solver_setup(solver)
        ):
            raise ValueError("FVM backup configuration hash does not match the active case")
        archived_viscosity = metadata.get("kinematic_viscosity")
        if (
            isinstance(archived_viscosity, bool)
            or not isinstance(archived_viscosity, int | float)
            or not np.isfinite(float(archived_viscosity))
            or float(archived_viscosity) <= 0.0
        ):
            raise ValueError("FVM backup molecular viscosity identity is invalid")
        if not allow_config_change and not np.isclose(
            float(archived_viscosity),
            float(
                getattr(
                    solver,
                    "_kinematic_viscosity",
                    _solver_setup(solver).transport.kinematic_viscosity,
                )
            ),
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError("FVM backup molecular viscosity does not match the active case")
        state = decode_state({name: np.array(archive[name], copy=True) for name in archive.files})
        state.pop("metadata", None)

    missing = sorted((required - {"metadata", "storage_layout"}) - set(state))
    if missing:
        raise ValueError("Incomplete FVM backup; missing: " + ", ".join(missing))
    payload = stage_restart_payload(
        solver,
        state,
        allow_config_change=allow_config_change,
        kinematic_viscosity=float(metadata["kinematic_viscosity"]),
    )

    return payload


def load_backup(solver, path, *, allow_config_change: bool = False) -> None:
    """Collectively admit and publish one canonical FVM backup.

    Replicated ranks stage independently, exchange only validation status, and
    publish the payload only after every rank has accepted it.  This keeps a
    corrupt or malformed archive from leaving peers at different clocks.
    """
    parallel = getattr(solver, "parallel", None)
    payload: RestartPayload | None = None
    local_error = None
    try:
        payload = _load_backup_local(
            solver,
            path,
            allow_config_change=allow_config_change,
        )
    except BaseException as error:
        local_error = {
            "rank": int(getattr(parallel, "rank", 0)),
            "type": type(error).__name__,
            "message": str(error),
        }

    if parallel is not None and parallel.is_parallel and not parallel.is_partitioned:
        errors = parallel.comm.allgather(local_error)
        failure = next((item for item in errors if item is not None), None)
        if failure is not None:
            raise RuntimeError(
                "FVM restart admission failed on rank "
                f"{failure['rank']} ({failure['type']}): {failure['message']}"
            )
        assert payload is not None
        signatures = parallel.comm.allgather(
            (
                payload.time,
                payload.step,
                payload.time_step_size,
                payload.accepted_time_step_size,
                payload.previous_time_step_size,
            )
        )
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise RuntimeError("FVM restart admission found inconsistent clocks across ranks")
    elif local_error is not None:
        raise RuntimeError(
            f"FVM restart admission failed ({local_error['type']}): {local_error['message']}"
        )

    if payload is None:
        raise RuntimeError("FVM restart admission produced no payload")
    publish_restart_payload(solver, payload)
