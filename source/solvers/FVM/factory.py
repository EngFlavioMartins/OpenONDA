"""Physics-first construction helpers for the native FVM solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

from openonda.runtime import RunConfig

from .config import FVMSetup

MeshSource = (
    str
    | Path
    | dict[str, Any]
    | Callable[
        [],
        dict[str, Any] | tuple[dict[str, Any], Any],
    ]
)


def _load_mesh_file(
    path: str | Path,
) -> dict[str, Any]:
    """Load a supported mesh file into solver-native mesh data."""
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


def _runtime_setup(
    setup: FVMSetup,
) -> FVMSetup:
    """Return the execution form of a user-facing setup."""
    if setup.cores == 1:
        return setup

    parallel_mode = (
        "petsc_replicated"
        if setup.execution.parallel_mode == "petsc_replicated"
        else "petsc_partitioned"
    )
    execution = replace(
        setup.execution,
        linear_backend="petsc",
        parallel_mode=parallel_mode,
        output_mode="synchronous",
    )
    output = replace(
        setup.output,
        asynchronous=False,
    )
    return replace(
        setup,
        execution=execution,
        output=output,
    )


def _materialize_mesh(
    mesh: MeshSource | None,
    *,
    is_root: bool,
) -> dict[str, Any] | None:
    if mesh is None or not is_root:
        return None
    if isinstance(mesh, str | Path):
        return _load_mesh_file(mesh)

    generated = mesh() if callable(mesh) else mesh
    if isinstance(generated, tuple):
        generated = generated[0]
    if not isinstance(generated, dict):
        raise TypeError("mesh must be a path, mesh dictionary, or callable returning one")
    return generated


def create_fvm_solver(
    setup: FVMSetup,
    *,
    case_dir: str | Path | None = None,
    mesh: MeshSource | None = None,
):
    """Construct an FVM solver from an ``FVMSetup``."""
    RunConfig(
        cpu_cores=setup.cores,
        parallel_mode="mpi",
    ).ensure_runtime(sys.argv[0])

    runtime_setup = _runtime_setup(setup)
    materialize_mesh_here = True

    if setup.cores > 1:
        from mpi4py import MPI

        materialize_mesh_here = (
            runtime_setup.execution.parallel_mode == "petsc_replicated"
            or MPI.COMM_WORLD.Get_rank() == 0
        )

    from .core.solver import FVMSolver

    return FVMSolver(
        runtime_setup,
        case_dir=(str(Path(case_dir).resolve()) if case_dir is not None else None),
        mesh_data=_materialize_mesh(
            mesh,
            is_root=materialize_mesh_here,
        ),
    )


__all__ = ["create_fvm_solver"]
