"""Physics-first construction helpers for the native FVM solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from openonda.runtime import RunConfig

from .config import FVMSetup

if TYPE_CHECKING:
    from .core.solver import FVMSolver


@runtime_checkable
class BuildableMesh(Protocol):
    """Declarative mesh object materialized by the solver factory."""

    def build(self) -> dict[str, Any] | tuple[dict[str, Any], Any]: ...


MeshSource = (
    str
    | Path
    | dict[str, Any]
    | BuildableMesh
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

    if path.suffix.lower() == ".npz":
        from .io.mesh_storage import load_native_mesh

        return load_native_mesh(path)

    raise ValueError(
        f"Unsupported mesh file {path.name!r}; expected a native '.npz' or Gmsh '.msh' file"
    )


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

    if callable(mesh):
        generated = mesh()
    elif isinstance(mesh, BuildableMesh):
        generated = mesh.build()
    else:
        generated = mesh
    if isinstance(generated, tuple):
        generated = generated[0]
    if not isinstance(generated, dict):
        raise TypeError("mesh must be a path, mesh dictionary, or callable returning one")
    return generated


def _save_generated_mesh(mesh_data: dict[str, Any], solution_dir: Path, output: Any) -> None:
    """Back up every input mesh before solver admission; preserve earlier copies."""
    import tempfile

    from .io.mesh_storage import save_native_mesh
    from .io.vtk_exporter import VTKExporter

    solution_dir.mkdir(parents=True, exist_ok=True)
    fields: dict[str, Any] = {}
    for source_name, output_name in (
        ("cell_sizes", "cell_size"),
        ("cell_levels", "refinement_level"),
        ("boundary_layer_index", "boundary_layer_index"),
    ):
        values = mesh_data.get(source_name)
        if values is not None:
            fields[output_name] = values
    # Export before geometric/LSQ admission so a rejected mesh remains inspectable.
    # Finish both new files before moving any previous successful backup.
    with tempfile.TemporaryDirectory(prefix=".mesh-export-", dir=solution_dir) as temporary:
        staging = Path(temporary)
        save_native_mesh(mesh_data, staging / "mesh.npz")
        VTKExporter(mesh_data, output).export(str(staging / "mesh.vtu"), fields)
        existing = [solution_dir / name for name in ("mesh.npz", "mesh.vtu")]
        if any(path.exists() for path in existing):
            previous = Path(tempfile.mkdtemp(prefix="mesh-backup-", dir=solution_dir))
            for path in existing:
                if path.exists():
                    path.rename(previous / path.name)
        for name in ("mesh.npz", "mesh.vtu"):
            (staging / name).replace(solution_dir / name)
    # Enrich a valid backup without making its availability depend on geometry.
    from .mesh.geometry import compute_mesh_geometry

    geometry = compute_mesh_geometry(mesh_data, compute_lsq=False)
    fields["cell_volume"] = geometry["cell_volume"]
    VTKExporter(mesh_data, output).export(str(solution_dir / "mesh.vtu"), fields)


def create_fvm_solver(
    setup: FVMSetup,
    *,
    case_dir: str | Path | None = None,
    solution_dir: str | Path | None = None,
    samples_dir: str | Path | None = None,
    mesh: MeshSource | None = None,
) -> FVMSolver:
    """Construct an FVM solver from an ``FVMSetup``."""
    RunConfig(
        cpu_cores=setup.cores,
        parallel_mode="mpi",
    ).ensure_runtime(sys.argv[0])

    runtime_setup = _runtime_setup(setup)
    resolved_case_dir = Path(case_dir).resolve() if case_dir is not None else Path.cwd().resolve()
    resolved_solution_dir = (
        Path(solution_dir).resolve() if solution_dir is not None else resolved_case_dir / "solution"
    )
    resolved_samples_dir = (
        Path(samples_dir).resolve() if samples_dir is not None else resolved_case_dir / "samples"
    )
    materialize_mesh_here = True
    is_root = True

    if setup.cores > 1:
        from mpi4py import MPI

        is_root = MPI.COMM_WORLD.Get_rank() == 0
        materialize_mesh_here = (
            runtime_setup.execution.parallel_mode == "petsc_replicated" or is_root
        )

    mesh_data = _materialize_mesh(mesh, is_root=materialize_mesh_here)
    if is_root and mesh_data is not None:
        _save_generated_mesh(mesh_data, resolved_solution_dir, runtime_setup.output)

    from .core.solver import FVMSolver

    return FVMSolver(
        runtime_setup,
        case_dir=str(resolved_case_dir),
        solution_dir=str(resolved_solution_dir),
        samples_dir=str(resolved_samples_dir),
        mesh_data=mesh_data,
    )


__all__ = ["create_fvm_solver"]
