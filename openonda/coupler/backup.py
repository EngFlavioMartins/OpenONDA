"""Numbered restart backups of the coupled FVM--VPM state.

A coupled backup ``solution/coupled/coupled_{step:06d}/`` holds the FVM
backup, the VPM backup and the VPM interface trace of one exchange, described
by ``checkpoint_info.json``. It is written to a hidden staging directory and
renamed into place once complete.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import UTC, datetime
import hashlib
import json
import logging
from pathlib import Path
import shutil

import numpy as np

from openonda.core.file_output import rewind_jsonl, write_json_atomic
from openonda.core.simulation.parallel import collective_phase
from openonda.fvm.io.backup import decode_state, encode_state
from openonda.vpm.config.restart_changes import validate_exact_configuration_changes

from .reporting import format_coupler_log

BACKUP_FORMAT_VERSION = 14
BOUNDARY_SCHEMA_VERSION = 4
_BOUNDARY_KEYS = (
    "has_velocity",
    "velocity",
    "has_normal_velocity",
    "normal_velocity",
    "has_tangential_gradient",
    "tangential_gradient",
)
_INFO_KEYS = frozenset(
    (
        "format_version",
        "kind",
        "created_utc",
        "configuration_sha256",
        "configuration",
        "step",
        "time",
        "fvm_step",
        "vpm_step",
        "n_fvm_substeps",
        "files",
        "file_sha256",
    )
)

logger = logging.getLogger("openonda.coupler")


def coupled_backup_path(solution_directory: str | Path, step: int) -> Path:
    """Return the numbered coupled backup ``solution/coupled/coupled_{step:06d}``."""
    return Path(solution_directory) / "coupled" / f"coupled_{step:06d}"


def configuration_digest(configuration: dict) -> str:
    """Return the SHA-256 digest of a JSON-serializable configuration mapping."""
    text = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def file_digest(path: Path) -> str:
    """Return the SHA-256 digest of a file, or of a directory with its relative names."""
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    for item in files:
        if path.is_dir():
            digest.update(item.relative_to(path).as_posix().encode())
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def restart_configuration(coupler) -> dict:
    """Return the coupling settings a continued run must share with its backup.

    The FVM and VPM backups validate their own configurations.
    """
    configuration = dict(coupler.config.to_dict())
    transfer = coupler.vorticity_transfer
    revisions = [body.revision for body in transfer._solid_bodies]
    if any(revision is None for revision in revisions):
        raise RuntimeError("Solid bodies need stable geometry revisions for coupled restart")
    if revisions:
        configuration["solid_geometry"] = {"wall_revisions": revisions}
    if transfer._lattice_anchor is not None:
        configuration["transfer_lattice"] = {
            "anchor": np.asarray(transfer._lattice_anchor).tolist()
        }
    return configuration


def save_coupled_backup(coupler, path: Path) -> Path:
    """Write the FVM, VPM and interface-trace backup of the current exchange.

    Collective under MPI. Returns ``path`` on every rank.
    """
    comm = coupler._comm
    fvm, vpm = coupler.fvm_solver, coupler.vpm_solver
    step = coupler.fvm_solver.step // coupler.n_fvm_substeps
    with collective_phase(comm, "coupled backup clock validation"):
        if fvm.step != step * coupler.n_fvm_substeps:
            raise ValueError("The FVM state is not at a completed exchange")
        if vpm is not None and (vpm.step != step or vpm.time != fvm.time):
            raise ValueError("The FVM and VPM steps or times differ")
    staging = path.with_name(f".{path.name}.tmp")
    with collective_phase(comm, "coupled backup staging"):
        if coupler._is_root:
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
    fvm_file = f"fvm_{step:06d}" if fvm.parallel.is_partitioned else f"fvm_{step:06d}.npz"
    fvm.save_backup(staging / fvm_file)
    with collective_phase(comm, "coupled backup commit"):
        if coupler._is_root:
            files = {
                "fvm": fvm_file,
                "vpm": vpm.save_backup(staging / f"vpm_{step:06d}.h5").name,
                "boundary_condition": f"boundary_condition_{step:06d}.npz",
            }
            trace = {
                "velocity": (coupler._velocity_boundary_condition_old, (0, 3)),
                "normal_velocity": (coupler._normal_velocity_boundary_condition_old, (0,)),
                "tangential_gradient": (
                    coupler._tangential_gradient_boundary_condition_old,
                    (0, 3),
                ),
            }
            state = {"boundary_schema_version": np.asarray(BOUNDARY_SCHEMA_VERSION)}
            for name, (values, empty_shape) in trace.items():
                state[f"has_{name}"] = np.asarray(values is not None)
                state[name] = np.empty(empty_shape) if values is None else values
            with (staging / files["boundary_condition"]).open("wb") as stream:
                np.savez_compressed(stream, **encode_state(state))
            configuration = restart_configuration(coupler)
            write_json_atomic(
                staging / "checkpoint_info.json",
                {
                    "format_version": BACKUP_FORMAT_VERSION,
                    "kind": "openonda.coupled_backup",
                    "created_utc": datetime.now(UTC).isoformat(),
                    "configuration_sha256": configuration_digest(configuration),
                    "configuration": configuration,
                    "step": step,
                    "time": float(fvm.time),
                    "fvm_step": int(fvm.step),
                    "vpm_step": int(vpm.step),
                    "n_fvm_substeps": int(coupler.n_fvm_substeps),
                    "files": files,
                    "file_sha256": {
                        name: file_digest(staging / file) for name, file in files.items()
                    },
                },
            )
            if path.exists():
                shutil.rmtree(path)
            staging.rename(path)
            logger.info(
                format_coupler_log(
                    "coupled backup",
                    ("exchange", step),
                    ("time", f"{fvm.time:.6g}", "s"),
                    ("directory", str(path)),
                )
            )
    return path


def _read_checkpoint_info(path: Path) -> dict:
    """Read and check the description and file digests of a coupled backup."""
    info = json.loads((path / "checkpoint_info.json").read_text(encoding="utf-8"))
    if not isinstance(info, dict) or set(info) != _INFO_KEYS:
        raise ValueError(f"{path} is not a coupled backup")
    if info["format_version"] != BACKUP_FORMAT_VERSION or info["kind"] != "openonda.coupled_backup":
        raise ValueError(f"Unsupported coupled backup format in {path}")
    if info["configuration_sha256"] != configuration_digest(info["configuration"]):
        raise ValueError(f"The configuration digest of {path} does not match")
    files = info["files"]
    if set(files) != {"fvm", "vpm", "boundary_condition"} or set(info["file_sha256"]) != set(files):
        raise ValueError(f"Incomplete coupled backup {path}")
    for name, file in files.items():
        relative = Path(file)
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError(f"Unsafe coupled backup file name {file!r}")
        if (
            not (path / relative).exists()
            or file_digest(path / relative) != info["file_sha256"][name]
        ):
            raise ValueError(f"Coupled backup file {path / relative} is missing or changed")
    for name in ("step", "fvm_step", "vpm_step", "n_fvm_substeps"):
        if type(info[name]) is not int or info[name] < 0:
            raise ValueError(f"Invalid coupled backup clock {name!r}")
    if (
        info["vpm_step"] != info["step"]
        or info["fvm_step"] != info["step"] * info["n_fvm_substeps"]
    ):
        raise ValueError("The coupled backup steps do not match")
    return info


def _load_boundary_trace(path: Path) -> tuple:
    """Return the stored VPM interface trace (velocity, normal velocity, tangential gradient)."""
    with np.load(path, allow_pickle=False) as stored:
        state = decode_state({name: np.array(stored[name], copy=True) for name in stored.files})
    if int(state.get("boundary_schema_version", -1)) != BOUNDARY_SCHEMA_VERSION or any(
        name not in state for name in _BOUNDARY_KEYS
    ):
        raise ValueError(f"Unsupported coupled boundary backup {path}")
    return tuple(
        state[name].copy() if bool(state[f"has_{name}"]) else None
        for name in ("velocity", "normal_velocity", "tangential_gradient")
    )


def load_coupled_backup(
    coupler,
    path: Path,
    *,
    allowed_config_differences: Collection[str] = (),
    expected_config_differences: Mapping[str, tuple[object, object]] | None = None,
) -> int:
    """Restore both solvers and the interface trace; return the exchange count.

    Configuration differences are refused unless listed. Paths starting with
    ``vpm.`` are passed to the VPM backup reader without the prefix; the
    others name coupling settings. The VPM is restored and validated before
    the FVM. Collective under MPI.
    """
    comm = coupler._comm
    if isinstance(allowed_config_differences, str | bytes):
        raise TypeError("allowed_config_differences must be a collection of exact paths")
    allowed = tuple(allowed_config_differences)
    expected = dict(expected_config_differences or {})
    if set(expected) - set(allowed):
        raise ValueError("Configuration expectations require their exact allowed paths")
    info = None
    with collective_phase(comm, "coupled backup validation and VPM restore"):
        if coupler._is_root:
            info = _read_checkpoint_info(path)
            changes = validate_exact_configuration_changes(
                restart_configuration(coupler),
                info["configuration"],
                allowed_config_differences=tuple(p for p in allowed if not p.startswith("vpm.")),
                expected_config_differences={
                    p: v for p, v in expected.items() if not p.startswith("vpm.")
                },
            )
            if info["n_fvm_substeps"] != coupler.n_fvm_substeps:
                raise ValueError("The coupled backup has a different number of FVM substeps")
            coupler.vpm_solver.load_backup(
                path / info["files"]["vpm"],
                allowed_config_differences=tuple(p[4:] for p in allowed if p.startswith("vpm.")),
                expected_config_differences={
                    p[4:]: v for p, v in expected.items() if p.startswith("vpm.")
                },
            )
            if coupler.vpm_solver.step != info["step"]:
                raise ValueError("The VPM backup step differs from the coupled backup")
            if changes:
                logger.warning(
                    "Loading a coupled backup with allowed configuration differences: "
                    + ", ".join(sorted(change["path"] for change in changes))
                )
    if comm is not None and comm.Get_size() > 1:
        info = comm.bcast(info, root=0)
    coupler.fvm_solver.load_backup(path / info["files"]["fvm"])
    step = int(info["step"])
    exchange_time = coupler.exchange_time(step)
    with collective_phase(comm, "coupled backup clock and trace restore"):
        if coupler.fvm_solver.step != info["fvm_step"]:
            raise ValueError("The FVM backup step differs from the coupled backup")
        tolerance = 1.0e-12 + 4.0 * info["fvm_step"] * np.finfo(np.float64).eps * exchange_time
        if abs(coupler.fvm_solver.time - exchange_time) > tolerance:
            raise ValueError(
                f"Coupled backup time {coupler.fvm_solver.time!r} s is not the time "
                f"{exchange_time!r} s of exchange {step}"
            )
        if coupler._is_root:
            (
                coupler._velocity_boundary_condition_old,
                coupler._normal_velocity_boundary_condition_old,
                coupler._tangential_gradient_boundary_condition_old,
            ) = _load_boundary_trace(path / info["files"]["boundary_condition"])
            coupler._normal_velocity_boundary_condition = None
            coupler._tangential_gradient_boundary_condition = None
            rewind_jsonl(coupler.solution_dir / "coupler_diagnostics.jsonl", exchange_time)
    # Continue on the exchange clock so both solvers write identical times.
    coupler.fvm_solver.time = exchange_time
    if coupler._is_root:
        coupler.vpm_solver.time = exchange_time
    return step


__all__ = [
    "BACKUP_FORMAT_VERSION",
    "configuration_digest",
    "coupled_backup_path",
    "load_coupled_backup",
    "save_coupled_backup",
]
