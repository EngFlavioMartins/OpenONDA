"""Reproducibility manifest for FVM verification and benchmark runs."""

from __future__ import annotations

from dataclasses import asdict
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any

import numpy as np

from .checkpoint import config_hash, mesh_hash


def _git_identity(repository: Path) -> tuple[str | None, bool | None]:
    """Return checkout identity when developer Git metadata is available."""
    try:
        import pygit2
    except ImportError:
        return None, None
    discovered = pygit2.discover_repository(str(repository))
    if discovered is None:
        return None, None
    try:
        git_repository = pygit2.Repository(discovered)
        return str(git_repository.head.target), bool(git_repository.status())
    except (KeyError, pygit2.GitError):
        return None, None


def build_manifest(solver) -> dict[str, Any]:
    """Collect source, environment, execution, mesh, and configuration identity."""
    packages: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "numba", "pyamg", "mpi4py", "petsc4py", "taichi"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    repository = Path(__file__).resolve().parents[4]
    revision, dirty = _git_identity(repository)
    return {
        "schema_version": 1,
        "distribution_version": metadata.version("OpenONDA"),
        "git_revision": revision,
        "git_dirty": dirty,
        "config_hash": config_hash(solver.setup),
        "mesh_hash": mesh_hash(solver.mesh_data),
        "configuration": asdict(solver.setup),
        "execution": asdict(solver.setup.execution),
        "mesh_quality": solver.mesh_quality,
        "mesh": {
            "cells": int(solver.mesh_data["n_cells"]),
            "faces": int(solver.mesh_data["n_faces"]),
            "points": int(solver.mesh_data.get("n_points", len(solver.mesh_data["points"]))),
            "provenance": solver.mesh_data.get("provenance"),
        },
        "python": sys.version,
        "packages": packages,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
    }


def write_manifest(solver, path) -> Path:
    """Atomically write a machine-readable run manifest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__} in an FVM manifest")

    payload = json.dumps(
        build_manifest(solver),
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=json_default,
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
