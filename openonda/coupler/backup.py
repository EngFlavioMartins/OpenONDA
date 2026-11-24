"""Numbered restart backups of the coupled FVM--VPM state.

A coupled backup ``solution/coupled/coupled_{step:06d}/`` holds the FVM
backup and the boundary data of every FVM domain and the VPM backup of one
coupling step, described by ``checkpoint_info.json``. It is written to a hidden staging directory and
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

from openonda.core.digest import configuration_digest
from openonda.core.file_output import rewind_jsonl, write_json_atomic
from openonda.core.parallel import collective_phase
from openonda.fvm.io.backup import decode_state, encode_state
from openonda.vpm.config.restart_changes import validate_exact_configuration_changes

from .reporting import format_coupler_log

BACKUP_FORMAT_VERSION = 20
BOUNDARY_SCHEMA_VERSION = 6
_BOUNDARY_FIELDS = ("velocity", "normal_velocity", "normal_gradient")
_CORRECTIONS = ("boundary_data_correction", "previous_boundary_data_correction")
_BOUNDARY_KEYS = (
    *(key for name in _BOUNDARY_FIELDS for key in (f"has_{name}", name)),
    *(
        key
        for correction in _CORRECTIONS
        for key in (
            f"has_{correction}",
            f"{correction}_interval",
            *(f"{correction}_{name}" for name in _BOUNDARY_FIELDS),
        )
    ),
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
        "vpm_time_step_size",
        "vpm_step",
        "domains",
        "files",
        "file_sha256",
    )
)
_DOMAIN_KEYS = frozenset(("fvm_time_step_size", "fvm_step", "fvm", "boundary_condition"))

logger = logging.getLogger("openonda.coupler")


def coupled_backup_path(solution_directory: str | Path, step: int) -> Path:
    """Return the numbered coupled backup ``solution/coupled/coupled_{step:06d}``."""
    return Path(solution_directory) / "coupled" / f"coupled_{step:06d}"


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

    The settings, walls and transfer-lattice anchor of every domain, by name.
    The FVM and VPM backups validate their own configurations.
    """
    configuration: dict[str, object] = {"domains": {}}
    for domain in coupler.domains:
        settings = dict(domain.config.to_dict())
        transfer = domain.vorticity_transfer
        revisions = [body.revision for body in transfer._solid_bodies]
        if any(revision is None for revision in revisions):
            raise RuntimeError("Solid bodies need stable geometry revisions for coupled restart")
        if revisions:
            settings["solid_geometry"] = {"wall_revisions": revisions}
        if transfer._lattice_anchor is not None:
            settings["transfer_lattice"] = {"anchor": np.asarray(transfer._lattice_anchor).tolist()}
        configuration["domains"][domain.name] = settings
    return configuration


def _boundary_state(domain) -> dict:
    """The boundary data of one domain and its two boundary-data corrections, for ``np.savez``."""
    boundary_data = {
        "velocity": (domain._velocity_boundary_condition_old, (0, 3)),
        "normal_velocity": (domain._normal_velocity_boundary_condition_old, (0,)),
        "normal_gradient": (domain._normal_gradient_boundary_condition_old, (0, 3)),
    }
    state = {"boundary_schema_version": np.asarray(BOUNDARY_SCHEMA_VERSION)}
    for name, (values, empty_shape) in boundary_data.items():
        state[f"has_{name}"] = np.asarray(values is not None)
        state[name] = np.empty(empty_shape) if values is None else values
    # The first estimate of the next interval extrapolates these corrections.
    for correction in _CORRECTIONS:
        values = getattr(domain, correction)
        state[f"has_{correction}"] = np.asarray(values is not None)
        state[f"{correction}_interval"] = np.asarray(
            float(getattr(domain, f"{correction}_interval"))
        )
        for name, (_, empty_shape) in boundary_data.items():
            stored = None if values is None else values[_BOUNDARY_FIELDS.index(name)]
            state[f"{correction}_{name}"] = np.empty(empty_shape) if stored is None else stored
    return state


def save_coupled_backup(coupler, path: Path) -> Path:
    """Write the FVM, VPM and boundary-data backup of the current coupling step.

    Collective under MPI. Returns ``path`` on every rank.
    """
    comm = coupler._comm
    vpm = coupler.vpm_solver
    step = coupler.step
    with collective_phase(comm, "coupled backup clock validation"):
        for domain in coupler.domains:
            fvm = domain.fvm_solver
            if vpm is not None and (
                vpm.step != step
                or vpm.time != fvm.time
                or not np.isclose(
                    vpm.time_step_size, coupler.vpm_time_step_size, rtol=0.0, atol=1.0e-14
                )
                or not np.isclose(
                    fvm._accepted_time_step_size,
                    domain.fvm_time_step_size,
                    rtol=0.0,
                    atol=1.0e-14,
                )
            ):
                raise ValueError(
                    f"The FVM and VPM backup clocks of {domain.name!r} differ: "
                    f"steps={fvm.step}/{vpm.step}/{step}, "
                    f"times={fvm.time!r}/{vpm.time!r}, "
                    f"time steps={fvm._accepted_time_step_size!r}/{vpm.time_step_size!r} "
                    f"(expected {domain.fvm_time_step_size!r}/{coupler.vpm_time_step_size!r})"
                )
            if vpm is None and not np.isclose(coupler.time, fvm.time, rtol=0.0, atol=1.0e-12):
                raise ValueError(f"The coupled clock differs from the FVM time of {domain.name!r}")
    staging = path.with_name(f".{path.name}.tmp")
    with collective_phase(comm, "coupled backup staging"):
        if coupler._is_root:
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
    fvm_files = {}
    for domain in coupler.domains:
        fvm = domain.fvm_solver
        stem = f"fvm_{domain.name}_{step:06d}"
        fvm_files[domain.name] = stem if fvm.parallel.is_partitioned else f"{stem}.npz"
        fvm.save_backup(staging / fvm_files[domain.name])
    with collective_phase(comm, "coupled backup commit"):
        if coupler._is_root:
            files = {"vpm": vpm.save_backup(staging / f"vpm_{step:06d}.h5").name}
            domains = {}
            for domain in coupler.domains:
                fvm = domain.fvm_solver
                boundary_file = f"boundary_condition_{domain.name}_{step:06d}.npz"
                with (staging / boundary_file).open("wb") as stream:
                    np.savez_compressed(stream, **encode_state(_boundary_state(domain)))
                files[f"fvm.{domain.name}"] = fvm_files[domain.name]
                files[f"boundary_condition.{domain.name}"] = boundary_file
                domains[domain.name] = {
                    "fvm_time_step_size": float(fvm._accepted_time_step_size),
                    "fvm_step": int(fvm.step),
                    "fvm": f"fvm.{domain.name}",
                    "boundary_condition": f"boundary_condition.{domain.name}",
                }
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
                    "time": float(coupler.time),
                    "vpm_time_step_size": float(vpm.time_step_size),
                    "vpm_step": int(vpm.step),
                    "domains": domains,
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
                    ("coupling step", step),
                    ("time", f"{coupler.time:.6g}", "s"),
                    ("directory", str(path)),
                )
            )
    return path


def _read_checkpoint_info(path: Path) -> dict:
    """Read and check the description and file digests of a coupled backup."""
    info = json.loads((path / "checkpoint_info.json").read_text(encoding="utf-8"))
    if not isinstance(info, dict):
        raise ValueError(f"{path} is not a coupled backup")
    if info.get("format_version") != BACKUP_FORMAT_VERSION or set(info) != _INFO_KEYS:
        raise ValueError(f"{path} is not a supported coupled backup")
    if info["kind"] != "openonda.coupled_backup":
        raise ValueError(f"Unsupported coupled backup format in {path}")
    if info["configuration_sha256"] != configuration_digest(info["configuration"]):
        raise ValueError(f"The configuration digest of {path} does not match")
    files = info["files"]
    domains = info["domains"]
    if not isinstance(domains, dict) or not domains:
        raise ValueError(f"Coupled backup {path} names no FVM domain")
    expected_files = {"vpm"}
    for name, entry in domains.items():
        if not isinstance(entry, dict) or set(entry) != _DOMAIN_KEYS:
            raise ValueError(f"Incomplete coupled backup entry of {name!r} in {path}")
        expected_files.update((entry["fvm"], entry["boundary_condition"]))
    if set(files) != expected_files or set(info["file_sha256"]) != set(files):
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
    clocks = [("step", info["step"]), ("vpm_step", info["vpm_step"])]
    clocks.extend((f"{name}.fvm_step", entry["fvm_step"]) for name, entry in domains.items())
    for name, value in clocks:
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid coupled backup clock {name!r}")
    if info["vpm_step"] != info["step"]:
        raise ValueError("The coupled backup steps do not match")
    sizes = [("vpm_time_step_size", info["vpm_time_step_size"])]
    sizes.extend(
        (f"{name}.fvm_time_step_size", entry["fvm_time_step_size"])
        for name, entry in domains.items()
    )
    for name, value in sizes:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not np.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"Invalid coupled backup time-step size {name!r}")
    if not np.isfinite(float(info["time"])) or float(info["time"]) < 0.0:
        raise ValueError("Invalid coupled backup physical time")
    return info


def _load_boundary_data(path: Path) -> tuple:
    """Return the stored boundary data and the two boundary-data corrections.

    The boundary data are the velocity, normal velocity and normal gradient of u_t;
    each correction is ``(values or None, interval length in s)``.
    """
    with np.load(path, allow_pickle=False) as stored:
        state = decode_state({name: np.array(stored[name], copy=True) for name in stored.files})
    if int(state.get("boundary_schema_version", -1)) != BOUNDARY_SCHEMA_VERSION or any(
        name not in state for name in _BOUNDARY_KEYS
    ):
        raise ValueError(f"Unsupported coupled boundary backup {path}")
    data = tuple(
        state[name].copy() if bool(state[f"has_{name}"]) else None for name in _BOUNDARY_FIELDS
    )
    corrections = tuple(
        (
            tuple(state[f"{correction}_{name}"].copy() for name in _BOUNDARY_FIELDS)
            if bool(state[f"has_{correction}"])
            else None,
            float(state[f"{correction}_interval"]),
        )
        for correction in _CORRECTIONS
    )
    return data, corrections


def load_coupled_backup(
    coupler,
    path: Path,
    *,
    allowed_config_differences: Collection[str] = (),
    expected_config_differences: Mapping[str, tuple[object, object]] | None = None,
) -> int:
    """Restore both solvers and the boundary data; return the coupling step.

    Configuration differences are refused unless
    listed. Paths starting with ``vpm.`` are passed to the VPM backup reader
    without the prefix; the others name coupling settings. The FVM domains are
    restored first: the walls and strain-excluded regions of the meshes that
    move in the particle frame then take their restored poses before the
    particles are restored and checked against them. Collective under MPI.
    """
    comm = coupler._comm
    if isinstance(allowed_config_differences, str | bytes):
        raise TypeError("allowed_config_differences must be a collection of exact paths")
    allowed = tuple(allowed_config_differences)
    expected = dict(expected_config_differences or {})
    if set(expected) - set(allowed):
        raise ValueError("Configuration expectations require their exact allowed paths")
    info = None
    with collective_phase(comm, "coupled backup validation"):
        if coupler._is_root:
            info = _read_checkpoint_info(path)
            names = sorted(domain.name for domain in coupler.domains)
            if sorted(info["domains"]) != names:
                raise ValueError(
                    f"The coupled backup {path} holds the FVM domains {sorted(info['domains'])}, "
                    f"the run {names}"
                )
            changes = validate_exact_configuration_changes(
                restart_configuration(coupler),
                info["configuration"],
                allowed_config_differences=tuple(p for p in allowed if not p.startswith("vpm.")),
                expected_config_differences={
                    p: v for p, v in expected.items() if not p.startswith("vpm.")
                },
            )
            if changes:
                logger.warning(
                    "Loading a coupled backup with allowed configuration differences: "
                    + ", ".join(sorted(change["path"] for change in changes))
                )
    if comm is not None and comm.Get_size() > 1:
        info = comm.bcast(info, root=0)
    for domain in coupler.domains:
        entry = info["domains"][domain.name]
        domain.fvm_solver.load_backup(path / info["files"][entry["fvm"]])
    step = int(info["step"])
    coupling_time = float(info["time"])
    with collective_phase(comm, "VPM restore"):
        if coupler._is_root:
            coupler._place_moving_geometry(None, coupling_time)
            coupler.vpm_solver.load_backup(
                path / info["files"]["vpm"],
                allowed_config_differences=tuple(p[4:] for p in allowed if p.startswith("vpm.")),
                expected_config_differences={
                    p[4:]: v for p, v in expected.items() if p.startswith("vpm.")
                },
            )
            if coupler.vpm_solver.step != step:
                raise ValueError("The VPM backup step differs from the coupled backup")
    with collective_phase(comm, "coupled backup clock and boundary-data restore"):
        tolerance = max(1.0e-12, 4.0 * abs(coupling_time) * np.finfo(np.float64).eps)
        for domain in coupler.domains:
            entry = info["domains"][domain.name]
            fvm = domain.fvm_solver
            if fvm.step != entry["fvm_step"]:
                raise ValueError(
                    f"The FVM backup step of {domain.name!r} differs from the coupled backup"
                )
            if abs(fvm.time - coupling_time) > tolerance:
                raise ValueError(
                    f"Coupled backup time {coupling_time!r} s differs from the FVM time "
                    f"{fvm.time!r} s of {domain.name!r}"
                )
            if not np.isclose(
                fvm.time_step_size, float(entry["fvm_time_step_size"]), rtol=0.0, atol=1.0e-14
            ):
                raise ValueError(
                    f"The FVM time-step size of {domain.name!r} differs from the coupled backup"
                )
            domain.fvm_time_step_size = fvm.time_step_size
        if coupler._is_root and not np.isclose(
            coupler.vpm_solver.time, coupling_time, rtol=0.0, atol=tolerance
        ):
            raise ValueError("The VPM time differs from the coupled backup time")
        if coupler._is_root and not np.isclose(
            coupler.vpm_solver.time_step_size,
            float(info["vpm_time_step_size"]),
            rtol=0.0,
            atol=1.0e-14,
        ):
            raise ValueError("The VPM time-step size differs from the coupled backup")
        coupler._selected_interval_end = None
        coupler.vpm_time_step_size = float(info["vpm_time_step_size"])
        if coupler._is_root:
            coupler.vpm_solver.time_step_size = coupler.vpm_time_step_size
        coupler._coupling_step = step
        coupler._accepted_time = coupling_time
        if coupler._is_root:
            for domain in coupler.domains:
                entry = info["domains"][domain.name]
                data, corrections = _load_boundary_data(
                    path / info["files"][entry["boundary_condition"]]
                )
                (
                    domain._velocity_boundary_condition_old,
                    domain._normal_velocity_boundary_condition_old,
                    domain._normal_gradient_boundary_condition_old,
                ) = data
                for correction, (values, interval) in zip(_CORRECTIONS, corrections, strict=True):
                    setattr(domain, correction, values)
                    setattr(domain, f"{correction}_interval", interval)
                domain._normal_velocity_boundary_condition = None
                domain._normal_gradient_boundary_condition = None
            rewind_jsonl(coupler.solution_dir / "coupler_diagnostics.jsonl", coupling_time)
    # All solvers retain the same accepted physical time after restoration.
    for domain in coupler.domains:
        domain.fvm_solver.time = coupling_time
    if coupler._is_root:
        coupler.vpm_solver.time = coupling_time
    return step


__all__ = [
    "BACKUP_FORMAT_VERSION",
    "coupled_backup_path",
    "load_coupled_backup",
    "save_coupled_backup",
]
