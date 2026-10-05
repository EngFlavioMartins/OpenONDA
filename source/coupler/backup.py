"""Atomic restart backups for coupled FVM--VPM state."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import UTC, datetime
import hashlib
import json
import logging
from numbers import Integral
import os
from pathlib import Path
import shutil
import tempfile
from typing import Protocol

import h5py
import numpy as np

from source import log_style
from source.simulation.parallel import collective_phase
from source.solution_layout import component_directory
from source.solvers.fvm.io.backup import decode_state, encode_state
from source.solvers.vpm.config.case import Numerics
from source.solvers.vpm.config.configuration_values import numerical_configuration
from source.solvers.vpm.config.restart import (
    _configuration_mismatches,
    restart_configuration_values,
)
from source.solvers.vpm.config.restart_changes import (
    MISSING_CONFIGURATION_VALUE,
    validate_configuration_changes,
    validate_exact_configuration_changes,
)
from source.solvers.vpm.io.backup import _BackupIO
from source.solvers.vpm.io.vlm_backup import export_vlm_backup

BACKUP_DIRECTORY = "backups"
BACKUP_FORMAT_VERSION = 13


def config_mapping_digest(config: dict) -> str:
    """Hash an already-serialized configuration mapping."""
    configuration_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(configuration_json.encode()).hexdigest()


class _MappingConfig(Protocol):
    def to_dict(self) -> dict:
        """Return the object's serialized configuration mapping."""


def config_digest(config: _MappingConfig) -> str:
    """Hash an internal configuration object that provides a mapping interface."""
    return config_mapping_digest(config.to_dict())


def _vpm_numerical_config(vpm_setup: Numerics) -> dict:
    """Return only VPM settings that can affect the continued solution."""
    return numerical_configuration(vpm_setup)


def _backup_config(coupler) -> dict:
    """Build the strict restart configuration for all coupled numerical components."""
    if coupler.vpm_solver is None:
        raise RuntimeError("Initialize the coupler before backuping configuration")
    config = dict(coupler.setup.to_dict())
    config["vpm"] = _vpm_numerical_config(coupler.vpm_solver.setup)
    transfer = getattr(coupler, "vorticity_transfer", None)
    if transfer is not None:
        bodies = getattr(transfer, "_solid_bodies", ())
        if bodies:
            revisions = [getattr(body, "revision", None) for body in bodies]
            if any(revision is None for revision in revisions):
                raise RuntimeError(
                    "Solid bodies need stable geometry revisions for coupled restart"
                )
            config["solid_geometry"] = {"wall_revisions": revisions}
        anchor = getattr(transfer, "_lattice_anchor", None)
        if anchor is not None:
            config["transfer_lattice"] = {"anchor": np.asarray(anchor).tolist()}
    return config


def checkpoint_path_hash(path: Path) -> str:
    """Hash one backup checkpoint_file, including relative names for directories."""
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode())
            with child.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
    else:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _resolve_checkpoint_path(target: Path, checkpoint_file: str) -> Path:
    """Resolve a checkpoint file without allowing backup path escape."""
    if not isinstance(checkpoint_file, str) or not checkpoint_file:
        raise ValueError("Coupled backup checkpoint_file names must be non-empty strings")
    relative = Path(checkpoint_file)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe coupled backup checkpoint_file path: {checkpoint_file!r}")
    target_resolved = target.resolve()
    resolved = (target / relative).resolve()
    if resolved != target_resolved and target_resolved not in resolved.parents:
        raise ValueError(
            f"Coupled backup checkpoint_file escapes its directory: {checkpoint_file!r}"
        )
    return resolved


def _config_differences(
    stored: object,
    current: object,
    *,
    prefix: str = "",
) -> list[tuple[str, object, object]]:
    """Return recursive leaf differences with stable dotted paths."""
    if isinstance(stored, dict) and isinstance(current, dict):
        incompatible_paths = None
        if prefix == "vpm":
            stored = restart_configuration_values(stored)
            current = restart_configuration_values(current)
            incompatible_paths = set(_configuration_mismatches(current, stored))
        differences: list[tuple[str, object, object]] = []
        for key in sorted(set(stored) | set(current)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in stored:
                differences.append((path, "<missing>", current[key]))
            elif key not in current:
                differences.append((path, stored[key], "<missing>"))
            else:
                differences.extend(_config_differences(stored[key], current[key], prefix=path))
        if incompatible_paths is not None:
            differences = [
                difference
                for difference in differences
                if any(
                    path == difference[0].removeprefix("vpm.")
                    or path.startswith(difference[0].removeprefix("vpm.") + "[")
                    for path in incompatible_paths
                )
            ]
        return differences
    if stored != current:
        return [(prefix, stored, current)]
    return []


def config_diff(stored: dict | None, current: dict) -> list[str]:
    """Return recursive ``path: old -> new`` configuration differences."""
    if stored is None:
        return []
    return [
        f"{path}: {old!r} -> {new!r}" for path, old, new in _config_differences(stored, current)
    ]


def config_difference_paths(stored: dict | None, current: dict) -> set[str]:
    """Return the exact recursive configuration paths whose values differ."""
    if stored is None:
        return set()
    return {path for path, _old, _new in _config_differences(stored, current)}


def _validate_coupled_configuration(stored, current, allowed, expectations):
    """Apply VPM compatibility only inside VPM; all other namespaces are exact."""
    if isinstance(allowed, str | bytes) or not isinstance(allowed, Collection):
        raise TypeError("allowed_config_differences must be a collection of exact paths")
    paths = tuple(allowed)
    if any(type(path) is not str for path in paths):
        raise ValueError("configuration permissions require exact dotted paths")
    if expectations is None:
        expectations = {}
    elif not isinstance(expectations, Mapping):
        raise TypeError("expected_config_differences must map exact paths to (stored, current)")
    else:
        expectations = dict(expectations)
    if any(type(path) is not str for path in expectations) or set(expectations) - set(paths):
        raise ValueError("configuration expectations require their exact allowed paths")
    old_vpm, new_vpm = stored.get("vpm"), current.get("vpm")
    if not isinstance(old_vpm, dict) or not isinstance(new_vpm, dict):
        raise ValueError("coupled restart requires stored and current VPM configuration mappings")
    try:
        vpm_changes = validate_configuration_changes(
            new_vpm,
            old_vpm,
            allowed_config_differences=tuple(path[4:] for path in paths if path.startswith("vpm.")),
            expected_config_differences={
                path[4:]: value for path, value in expectations.items() if path.startswith("vpm.")
            },
        )
    except (TypeError, ValueError) as exc:
        changed = _configuration_mismatches(
            restart_configuration_values(new_vpm), restart_configuration_values(old_vpm)
        )
        detail = ", ".join("vpm." + path for path in changed)
        raise type(exc)(f"{exc}; coupled VPM difference paths: {detail}") from exc
    new_other = {key: value for key, value in current.items() if key != "vpm"}
    old_other = {key: value for key, value in stored.items() if key != "vpm"}
    other_changes = validate_exact_configuration_changes(
        new_other,
        old_other,
        allowed_config_differences=tuple(path for path in paths if not path.startswith("vpm.")),
        expected_config_differences={
            path: value for path, value in expectations.items() if not path.startswith("vpm.")
        },
    )
    # The detached validated values, rather than the caller's mutable map,
    # become the exact permission snapshot forwarded to the real VPM reader.
    vpm_paths = tuple(change["path"] for change in vpm_changes)
    vpm_expectations = {
        change["path"]: tuple(
            change[side]["value"] if change[side]["present"] else MISSING_CONFIGURATION_VALUE
            for side in ("stored", "current")
        )
        for change in vpm_changes
    }
    changes = (
        *other_changes,
        *({**change, "path": "vpm." + change["path"]} for change in vpm_changes),
    )
    return changes, vpm_paths, vpm_expectations


def _inspect_coupled_vpm_checkpoint(coupler, path, checkpoint_info, allowed, expectations):
    """Read real native VPM validation before any coupled state/history load."""
    solver = coupler.vpm_solver
    step, time, dt, _count = _BackupIO.inspect(
        solver,
        path,
        allowed_config_differences=allowed,
        expected_config_differences=expectations,
    )
    with h5py.File(path, "r") as archive:
        stored_config = json.loads(archive["solver"].attrs["numerical_configuration"])
    checkpoint_vpm_config = checkpoint_info["config"]["vpm"]
    if config_mapping_digest(stored_config) != config_mapping_digest(checkpoint_vpm_config):
        raise ValueError(
            "hash_verified VPM HDF5 configuration differs from coupled checkpoint VPM configuration"
        )
    for name in ("coupling_step", "vpm_step", "fvm_step", "n_fvm_substeps"):
        value = checkpoint_info[name]
        if type(value) is not int or value < (1 if name == "n_fvm_substeps" else 0):
            raise ValueError(f"invalid coupled backup clock {name}")
    if (
        checkpoint_info["vpm_step"] != step
        or checkpoint_info["coupling_step"] != step
        or checkpoint_info["n_fvm_substeps"] != coupler.n_fvm_substeps
        or checkpoint_info["fvm_step"] != step * checkpoint_info["n_fvm_substeps"]
    ):
        raise ValueError("coupled backup step/subcycle clocks do not match native VPM checkpoint")
    clock = checkpoint_info["time"]
    if (
        isinstance(clock, bool)
        or not isinstance(clock, int | float)
        or not np.isfinite(clock)
        or clock < 0
        or not np.isclose(clock, time, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("coupled backup time does not match native VPM checkpoint")
    if dt != checkpoint_vpm_config.get("time_step_size"):
        raise ValueError(
            "native VPM checkpoint time-step attribute differs from hash-verified configuration"
        )


def _rewind_coupler_diagnostics(path: Path, time: float, history: list | None = None) -> None:
    """Rewind coupled diagnostics and retain the superseded branch.

    Coupled diagnostics are written by the coupler rather than either solver,
    so the solver-owned restart I/O cannot restore this stream.  The archive
    suffix deliberately does not match ``coupler_diagnostics.jsonl``: recursive
    cost/report discovery must see only the active history.
    """
    if not path.is_file():
        if history is not None:
            history[:] = [
                record
                for record in history
                if isinstance(record, dict) and float(record.get("time", -np.inf)) <= time + 1.0e-12
            ]
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept: list[str] = []
    needs_rewrite = False
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            if index == len(lines) - 1 and not line.endswith(("\n", "\r")):
                needs_rewrite = True
                break
            raise ValueError(f"Coupler diagnostics has an invalid record: {path}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Coupler diagnostics has an invalid record: {path}")
        try:
            row_time = float(record["time"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Coupler diagnostics has an invalid record: {path}") from error
        if not np.isfinite(row_time):
            raise ValueError(f"Coupler diagnostics has a non-finite time: {path}")
        if row_time <= time + 1.0e-12:
            if not line.endswith(("\n", "\r")):
                line += "\n"
                needs_rewrite = True
            kept.append(line)
        else:
            needs_rewrite = True
    if needs_rewrite:
        branch_root = path.parent / "restart_history"
        branch_root.mkdir(parents=True, exist_ok=True)
        branch = Path(tempfile.mkdtemp(prefix="before-", dir=branch_root))
        shutil.copy2(path, branch / f"{path.name}.superseded")
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_text("".join(kept), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    if history is not None:
        history[:] = [
            record
            for record in history
            if isinstance(record, dict) and float(record.get("time", -np.inf)) <= time + 1.0e-12
        ]


def _same_coupled_time(left: float, right: float, fvm_step: int) -> bool:
    """Compare physical clocks allowing only accumulated float64 roundoff."""
    if not np.isfinite(left) or not np.isfinite(right):
        return False
    # FVM adds each accepted substep; VPM rounds its macro-step clock.
    rounding = fvm_step * np.finfo(np.float64).eps
    tolerance = 1e-12 + rounding / (1.0 - rounding) * max(abs(left), abs(right))
    return abs(left - right) <= tolerance


def save_coupled_backup(coupler, directory, *, coupling_step: int | None = None) -> Path:
    """Save one synchronized state without overwriting the committed generation."""
    comm = getattr(coupler, "_comm", None)
    with collective_phase(comm, "coupled backup clock validation"):
        if coupler.fvm_solver is None or (coupler._is_master and coupler.vpm_solver is None):
            raise RuntimeError("Initialize the coupler before saving a backup")
        step = (
            coupling_step
            if coupling_step is not None
            else coupler.fvm_solver.step // coupler.n_fvm_substeps
        )
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
            raise ValueError("Backup coupling_step must be a non-negative integer")
        step = int(step)
        if coupler.fvm_solver.step != step * coupler.n_fvm_substeps:
            raise ValueError("FVM backup step is not a completed coupling exchange")
        clock = float(coupler.fvm_solver.time)
        if not np.isfinite(clock) or clock < 0:
            raise ValueError("FVM backup time must be finite and non-negative")
        if coupler._is_master and (
            coupler.vpm_solver.step != step
            or not _same_coupled_time(clock, coupler.vpm_solver.time, coupler.fvm_solver.step)
        ):
            raise ValueError("FVM and VPM backup steps/times are not synchronized")
    if comm is not None and comm.Get_size() > 1:
        clocks = comm.allgather((step, clock))
        root_step, root_clock = clocks[0]
        if any(
            s != root_step
            or not _same_coupled_time(t, root_clock, root_step * coupler.n_fvm_substeps)
            for s, t in clocks
        ):
            raise ValueError("FVM backup steps/times differ across MPI ranks")
    target = Path(directory)
    generation: str | None = None
    with collective_phase(comm, "coupled backup directory creation"):
        if coupler._is_master:
            target.mkdir(parents=True, exist_ok=True)
            generation = Path(tempfile.mkdtemp(prefix="checkpoint-", dir=target)).name
    if comm is not None and comm.Get_size() > 1:
        generation = comm.bcast(generation, root=0)
    if generation is None:
        raise RuntimeError("Coupled backup directory was not created")
    staging = target / generation
    suffix = f"{step:06d}"
    partitioned = coupler.fvm_solver.parallel.is_partitioned
    fvm_file = f"fvm_{suffix}" if partitioned else f"fvm_{suffix}.npz"

    # Partitioned FVM backups are collective; every rank must enter first.
    coupler.fvm_solver.save_state(staging / fvm_file)
    if not coupler._is_master:
        return target
    if coupler.vpm_solver is None:
        raise RuntimeError("Initialize the coupler before saving a backup")

    coupler.vpm_solver._save_backup_to(str(staging / f"vpm_{suffix}"))
    boundary_file = f"vpm_boundary_condition_{suffix}.npz"
    boundary_temporary = staging / f".{boundary_file}.tmp"
    boundary_state = {
        "boundary_schema_version": np.asarray(4, dtype=np.int64),
        "has_velocity": np.asarray(coupler._velocity_boundary_condition_old is not None),
        "velocity": (
            np.empty((0, 3))
            if coupler._velocity_boundary_condition_old is None
            else coupler._velocity_boundary_condition_old
        ),
        "has_normal_velocity": np.asarray(
            coupler._normal_velocity_boundary_condition_old is not None
        ),
        "normal_velocity": (
            np.empty(0)
            if coupler._normal_velocity_boundary_condition_old is None
            else coupler._normal_velocity_boundary_condition_old
        ),
        "has_tangential_gradient": np.asarray(
            coupler._tangential_gradient_boundary_condition_old is not None
        ),
        "tangential_gradient": (
            np.empty((0, 3))
            if coupler._tangential_gradient_boundary_condition_old is None
            else coupler._tangential_gradient_boundary_condition_old
        ),
    }
    try:
        with open(boundary_temporary, "wb") as stream:
            np.savez_compressed(stream, **encode_state(boundary_state))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(boundary_temporary, staging / boundary_file)
    finally:
        boundary_temporary.unlink(missing_ok=True)

    backup_config = _backup_config(coupler)
    checkpoint_info = {
        "format_version": BACKUP_FORMAT_VERSION,
        "kind": "openonda.coupled_backup",
        "created_utc": datetime.now(UTC).isoformat(),
        "backend": "fvm",
        "config_sha256": config_mapping_digest(backup_config),
        "config": backup_config,
        "coupling_step": step,
        "time": float(coupler.vpm_solver.time),
        "fvm_step": int(coupler.fvm_solver.step),
        "vpm_step": int(coupler.vpm_solver.step),
        "n_fvm_substeps": int(coupler.n_fvm_substeps),
        "checkpoint_files": {
            "fvm": f"{generation}/{fvm_file}",
            "vpm": f"{generation}/vpm_{suffix}.h5",
            "vpm_vtu": f"{generation}/vpm_{suffix}.vtu",
            "vpm_boundary_condition": f"{generation}/{boundary_file}",
        },
    }
    checkpoint_info["file_sha256"] = {
        name: checkpoint_path_hash(target / checkpoint_file)
        for name, checkpoint_file in checkpoint_info["checkpoint_files"].items()
    }
    checkpoint_info_temporary = target / "checkpoint_info.json.tmp"
    with checkpoint_info_temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(checkpoint_info, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(checkpoint_info_temporary, target / "checkpoint_info.json")

    keep = {"checkpoint_info.json", generation}
    stale = {
        *target.glob("checkpoint-*"),
        *target.glob("fvm_*"),
        *target.glob("vpm_*"),
        *target.glob("vpm_boundary_condition_*"),
    }
    for checkpoint_file in stale:
        if checkpoint_file.name in keep or not checkpoint_file.exists():
            continue
        if checkpoint_file.is_dir():
            shutil.rmtree(checkpoint_file)
        else:
            checkpoint_file.unlink()

    logging.getLogger("coupler").info(
        log_style.Event(
            "coupled backup",
            (
                ("checkpoint_info", str(target / "checkpoint_info.json")),
                ("VPM checkpoint", str(target / checkpoint_info["checkpoint_files"]["vpm"])),
            ),
        ),
    )
    return target


def publish_vpm_snapshot(backup_directory, output_directory) -> tuple[Path, Path]:
    """Publish the post-renewal VPM state as a user-facing time-series frame.

    The atomic coupled backup remains a rolling restart checkpoint_file. This
    function copies its already-written VPM HDF5/VTK pair into
    ``solution/vpm/`` and updates the root-level ``vpm.pvd`` collection. If
    the saved state contains a VLM surface, it also publishes the corresponding
    ``solution/vlm/`` frame and ``vlm.pvd`` collection.
    """
    backup = Path(backup_directory)
    output = Path(output_directory)
    checkpoint_info = json.loads((backup / "checkpoint_info.json").read_text(encoding="utf-8"))
    checkpoint_files = checkpoint_info.get("checkpoint_files", {})
    source_h5 = _resolve_checkpoint_path(backup, checkpoint_files.get("vpm", ""))
    source_vtu = _resolve_checkpoint_path(backup, checkpoint_files.get("vpm_vtu", ""))
    if not source_h5.is_file() or not source_vtu.is_file():
        raise FileNotFoundError("Coupled backup does not contain a complete VPM snapshot")

    output.mkdir(parents=True, exist_ok=True)
    frame_directory = component_directory(output, "vpm")
    frame_directory.mkdir(parents=True, exist_ok=True)
    destinations = (frame_directory / source_h5.name, frame_directory / source_vtu.name)
    for source, destination in zip((source_h5, source_vtu), destinations, strict=True):
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    if h5py.is_hdf5(destinations[0]):
        export_vlm_backup(destinations[0])
    _BackupIO.write_pvd(output)
    logging.getLogger("coupler").info(
        log_style.Event(
            "VPM snapshot", (("HDF5", str(destinations[0])), ("VTK", str(destinations[1])))
        ),
    )
    return destinations


def load_coupled_backup(
    coupler,
    directory,
    *,
    comm=None,
    allowed_config_differences: Collection[str] = (),
    expected_config_differences: Mapping[str, tuple[object, object]] | None = None,
) -> int:
    """Restore both solvers and the VPM boundary-history state.

    Configuration matching remains strict unless a caller explicitly names
    the exact paths allowed to differ for a controlled restart experiment.
    Checkpoint file integrity and every unlisted configuration field remain strict.
    Structural changes additionally require exact stored/current expectations.
    Native VPM validation completes collectively before FVM state/history load;
    this is not a state update rollback for later FVM or output writing errors.
    """
    if coupler.fvm_solver is None or (coupler._is_master and coupler.vpm_solver is None):
        raise RuntimeError("Initialize the coupler before loading a backup")

    target = Path(directory)
    error: str | None = None
    checkpoint_info: dict | None = None
    checkpoint_files: dict[str, str] = {}
    checkpoint_paths: dict[str, Path] = {}
    vpm_allowed: tuple[str, ...] = ()
    vpm_expectations: dict[str, tuple[object, object]] = {}
    if coupler._is_master:
        try:
            checkpoint_info = json.loads(
                (target / "checkpoint_info.json").read_text(encoding="utf-8")
            )
        except OSError as exc:
            error = f"Cannot read coupled backup information at {target}: {exc}"
        except json.JSONDecodeError as exc:
            error = f"Invalid coupled backup information at {target}: {exc}"
        if error is None and not isinstance(checkpoint_info, dict):
            error = "Coupled backup information must be a JSON object"
        if error is None:
            assert isinstance(checkpoint_info, dict)
            expected_info_keys = {
                "format_version",
                "kind",
                "created_utc",
                "backend",
                "config_sha256",
                "config",
                "coupling_step",
                "time",
                "fvm_step",
                "vpm_step",
                "n_fvm_substeps",
                "checkpoint_files",
                "file_sha256",
            }
            version = checkpoint_info.get("format_version")
            if (
                set(checkpoint_info) != expected_info_keys
                or version != BACKUP_FORMAT_VERSION
                or checkpoint_info.get("kind") != "openonda.coupled_backup"
                or checkpoint_info.get("backend") != "fvm"
            ):
                error = "Unsupported coupled backup format or backend"
            else:
                stored_checkpoint_files = checkpoint_info.get("checkpoint_files", {})
                if not isinstance(stored_checkpoint_files, dict):
                    error = "Coupled backup checkpoint_files must be a mapping"
                else:
                    checkpoint_files = dict(stored_checkpoint_files)
            if error is None:
                checkpoint_info["checkpoint_files"] = checkpoint_files
                try:
                    checkpoint_paths = {
                        name: _resolve_checkpoint_path(target, checkpoint_file)
                        for name, checkpoint_file in checkpoint_files.items()
                    }
                except ValueError as exc:
                    error = str(exc)
                    checkpoint_paths = {}
                file_hashes = checkpoint_info.get("file_sha256", {})
                if file_hashes and not isinstance(file_hashes, dict):
                    error = "Coupled backup file_sha256 must be a mapping"
                if error is None and (
                    not isinstance(file_hashes, dict) or set(file_hashes) != set(checkpoint_files)
                ):
                    error = (
                        f"Coupled backup format {BACKUP_FORMAT_VERSION} requires one "
                        "SHA-256 digest "
                        "for every declared checkpoint_file"
                    )
                if error is None and file_hashes:
                    for name, expected_hash in file_hashes.items():
                        checkpoint_file = checkpoint_files.get(name)
                        if not checkpoint_file or name not in checkpoint_paths:
                            error = f"Coupled backup information hashes unknown checkpoint_file {name!r}"
                            break
                        checkpoint_path = checkpoint_paths[name]
                        if (
                            not isinstance(expected_hash, str)
                            or len(expected_hash) != 64
                            or not checkpoint_path.exists()
                            or checkpoint_path_hash(checkpoint_path) != expected_hash
                        ):
                            error = (
                                f"Coupled backup checkpoint_file hash mismatch: {checkpoint_file}"
                            )
                            break
            required_checkpoint_files = [
                "fvm",
                "vpm",
                "vpm_vtu",
                "vpm_boundary_condition",
            ]
            missing = [
                name
                for name in required_checkpoint_files
                if not checkpoint_files.get(name)
                or name not in checkpoint_paths
                or not checkpoint_paths[name].exists()
            ]
            if error is None and missing:
                error = f"Incomplete coupled backup; missing: {', '.join(missing)}"
            elif error is None:
                stored_config = checkpoint_info.get("config")
                if not isinstance(stored_config, dict):
                    error = "Coupled backup configuration must be a mapping"
                elif checkpoint_info.get("config_sha256") != config_mapping_digest(stored_config):
                    error = "Coupled backup stored configuration hash mismatch"
                else:
                    try:
                        current_config = _backup_config(coupler)
                        changes, vpm_allowed, vpm_expectations = _validate_coupled_configuration(
                            stored_config,
                            current_config,
                            allowed_config_differences,
                            expected_config_differences,
                        )
                        _inspect_coupled_vpm_checkpoint(
                            coupler,
                            checkpoint_paths["vpm"],
                            checkpoint_info,
                            vpm_allowed,
                            vpm_expectations,
                        )
                        if changes:
                            logging.getLogger("coupler").warning(
                                "Loading a coupled backup with explicitly allowed "
                                "configuration differences: "
                                + ", ".join(sorted(change["path"] for change in changes))
                            )
                    except BaseException as exc:
                        error = f"Coupled restart validation failed: {type(exc).__name__}: {exc}"
    if comm is not None and comm.Get_size() > 1:
        error, checkpoint_info = comm.bcast(
            (error, checkpoint_info) if coupler._is_master else None,
            root=0,
        )
    if error is not None:
        raise ValueError(error)
    assert checkpoint_info is not None
    checkpoint_files = checkpoint_info["checkpoint_files"]

    coupler.fvm_solver.load_state(target / checkpoint_files["fvm"])
    expected_fvm_step = int(checkpoint_info["vpm_step"]) * coupler.n_fvm_substeps
    if coupler.fvm_solver.step != expected_fvm_step:
        raise ValueError(
            f"Coupled backup time-step mismatch: FVM={coupler.fvm_solver.step}, "
            f"expected {expected_fvm_step} from VPM={checkpoint_info['vpm_step']}"
        )

    if coupler._is_master:
        try:
            assert coupler.vpm_solver is not None
            coupler.vpm_solver._load_backup_from(
                str(target / checkpoint_files["vpm"]),
                allowed_config_differences=vpm_allowed,
                expected_config_differences=vpm_expectations,
            )
            with np.load(
                target / checkpoint_files["vpm_boundary_condition"], allow_pickle=False
            ) as boundary:
                expected_boundary_keys = {
                    "boundary_schema_version",
                    "has_velocity",
                    "velocity",
                    "has_normal_velocity",
                    "normal_velocity",
                    "has_tangential_gradient",
                    "tangential_gradient",
                    "storage_layout",
                }
                if set(boundary.files) != expected_boundary_keys:
                    raise ValueError("Coupled boundary backup has invalid fields")
                boundary_state = decode_state(
                    {name: np.array(boundary[name], copy=True) for name in boundary.files}
                )
                if (
                    "boundary_schema_version" not in boundary_state
                    or int(boundary_state["boundary_schema_version"]) != 4
                ):
                    raise ValueError("Unsupported coupled boundary backup schema")
                coupler._velocity_boundary_condition_old = (
                    boundary_state["velocity"].copy()
                    if bool(boundary_state["has_velocity"])
                    else None
                )
                coupler._normal_velocity_boundary_condition_old = (
                    boundary_state["normal_velocity"].copy()
                    if bool(boundary_state["has_normal_velocity"])
                    else None
                )
                coupler._tangential_gradient_boundary_condition_old = (
                    boundary_state["tangential_gradient"].copy()
                    if bool(boundary_state["has_tangential_gradient"])
                    else None
                )
                coupler._normal_velocity_boundary_condition = None
                coupler._tangential_gradient_boundary_condition = None
            if not _same_coupled_time(
                coupler.fvm_solver.time, coupler.vpm_solver.time, coupler.fvm_solver.step
            ):
                error = f"Coupled backup time mismatch: FVM={coupler.fvm_solver.time}, VPM={coupler.vpm_solver.time}"
            if error is None and getattr(coupler, "solution_dir", None) is not None:
                try:
                    _rewind_coupler_diagnostics(
                        Path(coupler.solution_dir) / "coupler_diagnostics.jsonl",
                        float(checkpoint_info["time"]),
                        getattr(coupler, "coupling_diagnostics", None),
                    )
                except BaseException as exc:
                    error = f"Coupled diagnostics restart time alignment failed: {type(exc).__name__}: {exc}"
        except Exception as exc:
            error = f"Coupled VPM restart failed: {type(exc).__name__}: {exc}"
    if comm is not None and comm.Get_size() > 1:
        error = comm.bcast(error if coupler._is_master else None, root=0)
    if error is not None:
        raise ValueError(error)
    coupling_step = int(checkpoint_info["coupling_step"])
    if coupler.vorticity_transfer is None:
        raise RuntimeError("Coupled backup load requires an initialized vorticity transfer")
    # A fresh run performs one initial synchronization plus one transfer after
    # every completed coupling interval.  Preserve that cadence so resumed
    # diagnostics and their cost remain identical to an uninterrupted run.
    coupler.vorticity_transfer.step = coupling_step + 1
    return coupling_step


__all__ = [
    "BACKUP_DIRECTORY",
    "BACKUP_FORMAT_VERSION",
    "config_diff",
    "config_difference_paths",
    "config_digest",
    "config_mapping_digest",
    "load_coupled_backup",
    "publish_vpm_snapshot",
    "save_coupled_backup",
]
