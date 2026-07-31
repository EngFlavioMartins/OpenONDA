"""Physics-first construction helpers for the native FVM solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

from openonda.runtime import RunConfig

from .config.types import FVMSetup

MeshSource = str | Path | dict[str, Any] | Callable[[], dict[str, Any] | tuple[dict[str, Any], Any]]


def _load_mesh_file(path: str | Path) -> dict[str, Any]:
    """Load a mesh file into FVM ``mesh_data``.

    Currently supports Gmsh ``.msh`` files via :class:`GmshImporter`, so a case
    can pass ``mesh="path/to/mesh.msh"`` straight to :func:`setup_fvm_solver`.
    """
    path = Path(path)
    if path.suffix.lower() == ".msh":
        from .mesh.gmsh_importer import GmshImporter

        importer = GmshImporter()
        try:
            importer.load_mesh(str(path))
            return importer.get_mesh_data()
        finally:
            importer.finalize()
    raise ValueError(f"Unsupported mesh file {path.name!r}; expected a Gmsh '.msh' file")


def _runtime_setup(setup: FVMSetup) -> FVMSetup:
    """Return the internal execution form of a user-facing setup."""
    if setup.cores == 1:
        return setup
    execution = replace(
        setup.execution,
        linear_backend="petsc",
        parallel_mode="petsc_partitioned",
        output_mode="synchronous",
    )
    output = replace(setup.output, asynchronous=False)
    return replace(setup, execution=execution, output=output)


def _materialize_mesh(mesh: MeshSource | None, *, is_root: bool) -> dict[str, Any] | None:
    if mesh is None or not is_root:
        return None
    if isinstance(mesh, str | Path):
        return _load_mesh_file(mesh)
    generated = mesh() if callable(mesh) else mesh
    if isinstance(generated, tuple):
        generated = generated[0]
    if not isinstance(generated, dict):
        raise TypeError("mesh must be a path, a mesh dictionary, or a callable returning one")
    return generated


def setup_fvm_solver(
    setup: FVMSetup,
    *,
    case_dir: str | Path | None = None,
    mesh: MeshSource | None = None,
):
    """Construct an FVM solver from a physics setup.

    ``setup.cores`` is the complete parallel interface. When it is greater
    than one this function relaunches the current case under MPI, selects the
    partitioned PETSc backend, builds the global mesh on rank zero, and
    distributes owned-plus-halo partitions internally.
    """
    RunConfig(cpu_cores=setup.cores, parallel_mode="mpi").ensure_runtime(sys.argv[0])

    runtime_setup = _runtime_setup(setup)
    is_root = True
    if setup.cores > 1:
        from mpi4py import MPI

        is_root = MPI.COMM_WORLD.Get_rank() == 0
    mesh_data = _materialize_mesh(mesh, is_root=is_root)

    from .core.solver import Solver

    return Solver(
        runtime_setup,
        case_dir=str(Path(case_dir).resolve()) if case_dir is not None else None,
        mesh_data=mesh_data,
    )


__all__ = ["setup_fvm_solver"]
