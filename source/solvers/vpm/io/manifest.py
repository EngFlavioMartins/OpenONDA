"""Reproducibility and lifecycle manifest for VPM runs.

The numerical configuration fingerprint is deliberately separate from the
run plan and artifact paths.  A user can therefore extend a run or change its
logging destination without weakening the identity of the equations that a
restart must match.
"""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any

import numpy as np

from source.version import __version__

from ..config import constants as constants_module
from ..config.fingerprint import numerical_configuration


def _git_identity(repository: Path) -> tuple[str | None, bool | None]:
    """Return checkout identity when optional Git metadata is available."""
    try:
        import pygit2
    except ImportError:
        return None, None
    try:
        discovered = pygit2.discover_repository(str(repository))
        if discovered is None:
            return None, None
        git_repository = pygit2.Repository(discovered)
        return str(git_repository.head.target), bool(git_repository.status())
    except (KeyError, pygit2.GitError, OSError):
        return None, None


def _package_versions() -> dict[str, str | None]:
    """Capture versions that can affect a reproducibility review."""
    packages: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "numba", "pyamg", "mpi4py", "petsc4py", "taichi"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return packages


def _schedule_identity(schedule: object | None) -> dict[str, Any] | None:
    """Return only serializable, user-visible schedule controls."""
    if schedule is None:
        return None
    result: dict[str, Any] = {
        "type": f"{type(schedule).__module__}.{type(schedule).__qualname__}",
    }
    for name in ("interval", "initial", "is_final_only", "at_end"):
        if hasattr(schedule, name):
            value = getattr(schedule, name)
            if isinstance(value, np.generic):
                value = value.item()
            result[name] = value
    return result


def _sampler_identity(sampler: object) -> dict[str, Any]:
    """Describe a sampler without serializing its live callable state."""
    result: dict[str, Any] = {
        "type": f"{type(sampler).__module__}.{type(sampler).__qualname__}",
        "file_name": getattr(sampler, "file_name", None),
        "schedule": _schedule_identity(getattr(sampler, "schedule", None)),
        "initial": bool(getattr(sampler, "initial", False)),
    }
    return result


def _case_configuration(solver: Any, numerical: dict[str, Any]) -> dict[str, Any]:
    """Build the non-numerical case identity used by the full manifest hash."""
    case = solver.case
    run = case.run
    backup = case.backup
    samplers = case.samplers
    return {
        "numerical": numerical,
        "run": {
            "steps": int(run.steps),
            "initial_samples": bool(run.initial_samples),
            "final_backup": bool(run.final_backup),
            "health_limit_action": str(run.health_limit_action),
            "wall_time_limit_seconds": run.wall_time_limit_seconds,
        },
        "backup": {
            "interval_steps": int(backup.interval_steps),
            "directory": str(backup.directory),
            "log_directory": str(backup.log_directory),
        },
        "samplers": {
            "directory": samplers.directory,
            "items": [_sampler_identity(sampler) for sampler in samplers.samples],
        },
        "initial_conditions": [
            f"{type(initial_condition).__module__}.{type(initial_condition).__qualname__}"
            for initial_condition in case.initial_conditions
        ],
        "initial_weak_particle_percent": float(case.initial_weak_particle_percent),
    }


def _sha256(value: object) -> str:
    """Hash a deterministic JSON representation."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_default(value: object) -> object:
    """Serialize NumPy scalars while rejecting hidden live runtime objects."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} in a VPM manifest")


def build_manifest(solver: Any, *, status: str | None = None, failure=None) -> dict[str, Any]:
    """Collect source, environment, execution, component, and lifecycle identity."""
    numerical = numerical_configuration(solver.setup)
    configuration = _case_configuration(solver, numerical)
    repository = Path(__file__).resolve().parents[4]
    revision, dirty = _git_identity(repository)
    viscous = solver.setup.viscous
    induction = solver.induction
    stabilization_manager = getattr(solver, "stabilization", None)
    active_stabilization = (
        stabilization_manager.active_mechanisms() if stabilization_manager is not None else ()
    )
    environment_overrides = {
        name: os.environ[name]
        for name in ("TI_OFFLINE_CACHE_FILE_PATH", "NUMBA_NUM_THREADS")
        if name in os.environ
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "solver": "VPM",
        "distribution_version": __version__,
        "git_revision": revision,
        "git_dirty": dirty,
        "config_hash": _sha256(numerical),
        "full_config_hash": _sha256(configuration),
        "configuration": configuration,
        "active_components": {
            "induction": f"{type(induction).__module__}.{type(induction).__qualname__}",
            "viscous": str(getattr(viscous, "scheme", "unknown")),
            "turbulence": str(getattr(solver.setup.turbulence, "flow_model", "unknown")),
            "stabilization": [str(name) for name in active_stabilization],
            "vlm": getattr(solver, "vlm_solver", None) is not None,
            "initial_conditions": len(solver.case.initial_conditions),
        },
        "execution": {
            "compute_device": str(getattr(solver, "compute_device", "UNKNOWN")),
            "precision": str(getattr(solver, "precision", solver.setup.precision)),
            "random_seed": int(solver.setup.random_seed),
            "device_memory_fraction": float(solver.setup.device_memory_fraction),
            "mpi_size": 1,
        },
        "runtime_overrides": {
            "kinematic_viscosity": getattr(
                solver,
                "_kinematic_viscosity",
                getattr(viscous, "kinematic_viscosity", None),
            ),
        },
        "environment_overrides": environment_overrides,
        "python": sys.version,
        "packages": _package_versions(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "taichi_backend": str(
                getattr(
                    solver,
                    "_backend_name",
                    getattr(constants_module, "TAICHI_BACKEND", "UNKNOWN"),
                )
            ),
        },
        "state": {
            "step": int(solver.step),
            "time": float(solver.time),
        },
    }
    if status is not None:
        manifest["lifecycle"] = {
            "status": str(status),
            "error": None if failure is None else f"{type(failure).__name__}: {failure}",
        }
    return manifest


def write_manifest(solver: Any, path: str | Path, *, status: str, failure=None) -> Path:
    """Atomically write a machine-readable terminal VPM manifest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        build_manifest(solver, status=status, failure=failure),
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=_json_default,
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return destination
