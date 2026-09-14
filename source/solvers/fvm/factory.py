"""Physics-first construction helpers for the native FVM solver."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from openonda.runtime import RunConfig
from source.simulation.paths import CasePaths

from .config import FVMSetup
from .config.types import validate_fvm_setup
from .mesh.progress import mesh_event, mesh_stage, mesher_log_session

if TYPE_CHECKING:
    from .core.solver import FVMSolver


@runtime_checkable
class BuildableMesh(Protocol):
    """Structural interface for a mesh object materialized by the FVM factory.

    Implementations are normally mesher configuration objects. ``build`` must
    return either a solver-native mesh mapping or ``(mesh_mapping, report)``;
    the report is retained by the producer and ignored during solver creation.
    """

    def build(self) -> dict[str, Any] | tuple[dict[str, Any], Any]:
        """Materialize mesh connectivity and geometry input without mutating a solver."""
        ...


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
        mesh_event("mesh source", kind="file", path=Path(mesh).resolve())
        return _load_mesh_file(mesh)

    mesh_event("mesh source", kind=type(mesh).__name__)
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
    from .io.vtk_exporter import VTKExporter, mesh_cell_fields

    solution_dir.mkdir(parents=True, exist_ok=True)
    fields = mesh_cell_fields(mesh_data)
    # Export before geometric/LSQ admission so a rejected mesh remains inspectable.
    # Finish both new files before moving any previous successful backup.
    with mesh_stage("mesh backup export") as stage:
        with tempfile.TemporaryDirectory(prefix=".mesh-export-", dir=solution_dir) as temporary:
            staging = Path(temporary)
            save_native_mesh(mesh_data, staging / "mesh.npz")
            exporter = VTKExporter(mesh_data, output)
            exporter.export(str(staging / "mesh.vtu"), fields)
            existing = [solution_dir / name for name in ("mesh.npz", "mesh.vtu")]
            if any(path.exists() for path in existing):
                previous = Path(tempfile.mkdtemp(prefix="mesh-backup-", dir=solution_dir))
                for path in existing:
                    if path.exists():
                        path.rename(previous / path.name)
            for name in ("mesh.npz", "mesh.vtu"):
                (staging / name).replace(solution_dir / name)
        stage.details(
            native=solution_dir / "mesh.npz",
            visualisation=solution_dir / "mesh.vtu",
        )
    # Enrich a valid backup without making its availability depend on geometry.
    from .mesh.geometry import compute_mesh_geometry

    with mesh_stage("mesh geometry and final visualisation") as stage:
        geometry = compute_mesh_geometry(mesh_data, compute_lsq=False)
        fields = mesh_cell_fields(mesh_data, geometry["cell_volume"])
        exporter.export(str(solution_dir / "mesh.vtu"), fields)
        stage.details(cells=mesh_data.get("n_cells"), faces=mesh_data.get("n_faces"))


def _prepare_output_directories(
    solution_dir: Path,
    samples_dir: Path,
    *,
    create_samples: bool,
) -> None:
    """Create the case-owned output directories before potentially slow setup work."""
    solution_dir.mkdir(parents=True, exist_ok=True)
    if create_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)


def create_fvm_solver(
    setup: FVMSetup,
    *,
    case_dir: str | Path | None = None,
    solution_dir: str | Path | None = None,
    samples_dir: str | Path | None = None,
    mesh: MeshSource | None = None,
) -> FVMSolver:
    """Validate configuration, materialize a mesh, and construct an FVM solver.

    Parameters
    ----------
    setup : FVMSetup
        Low-level solver configuration. New applications normally construct an
        :class:`~source.solvers.fvm.config.case.FVMCase` and call its factory
        path instead.
    case_dir : str or pathlib.Path or None, optional
        Case root. ``None`` uses the current working directory.
    solution_dir, samples_dir : str or pathlib.Path or None, optional
        Artifact destinations. Relative paths are resolved below ``case_dir``;
        omitted paths use the legacy ``solution/`` and canonical ``samples/``
        locations.
    mesh : path-like, mapping, BuildableMesh, callable, or None, optional
        Mesh source. ``.npz`` and Gmsh ``.msh`` files are supported. A callable
        or buildable object may return a mesh mapping or ``(mapping, report)``.

    Returns
    -------
    FVMSolver
        Initialized solver owning mutable fields, output paths, and parallel
        resources.

    Raises
    ------
    TypeError, ValueError
        If configuration or the materialized mesh violates its contract.
    RuntimeError
        If the requested threaded/MPI runtime cannot be established.

    Notes
    -----
    This function creates artifact directories and writes a native/VTK backup
    of a generated mesh before solver admission. In multi-rank runs the mesh is
    built on the rank required by the configured parallel layout.
    """
    validate_fvm_setup(setup)
    runtime_setup = _runtime_setup(setup)
    validate_fvm_setup(runtime_setup)
    resolved_case_dir = Path(case_dir).resolve() if case_dir is not None else Path.cwd().resolve()
    paths = CasePaths.resolve(
        resolved_case_dir,
        solution_dir=solution_dir,
        samples_dir=samples_dir,
        # Keep every FVM construction path on the same portable default.
        solution_default="solution",
    )
    resolved_solution_dir = paths.solution_dir
    resolved_samples_dir = paths.samples_dir
    mesher_log_path = resolved_solution_dir / "mesher.log"
    samples_requested = bool(runtime_setup.samplers) or samples_dir is not None

    RunConfig(
        cpu_cores=setup.cores,
        parallel_mode="mpi",
    ).ensure_runtime(sys.argv[0])

    materialize_mesh_here = True
    is_root = True
    comm = None

    if setup.cores > 1:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        is_root = MPI.COMM_WORLD.Get_rank() == 0
        materialize_mesh_here = (
            runtime_setup.execution.parallel_mode == "petsc_replicated" or is_root
        )

    startup_logger = None

    def _error_payload(error: BaseException | None, stage: str):
        if error is None:
            return None
        return {
            "stage": stage,
            "rank": int(comm.Get_rank()) if comm is not None else 0,
            "type": type(error).__name__,
            "message": str(error),
        }

    def _raise_collective_failure(error: BaseException | None, stage: str) -> None:
        """Make a root/local setup failure visible to every MPI rank."""
        if comm is None or comm.Get_size() == 1:
            if error is not None:
                raise error
            return
        failures = comm.allgather(_error_payload(error, stage))
        failure = next((item for item in failures if item is not None), None)
        if failure is not None:
            raise RuntimeError(
                f"FVM collective construction failed during {failure['stage']} on rank "
                f"{failure['rank']} ({failure['type']}): {failure['message']}"
            )

    path_error = None
    try:
        _prepare_output_directories(
            resolved_solution_dir,
            resolved_samples_dir,
            create_samples=samples_requested,
        )
    except BaseException as error:
        path_error = error
    _raise_collective_failure(path_error, "output directory preparation")

    logger_error = None
    if is_root:
        try:
            from .io.logging import Logging

            startup_logger = Logging(
                resolved_case_dir,
                solution_dir=resolved_solution_dir,
                config=runtime_setup.logging,
            )
            startup_logger.section(
                "FVM STARTUP",
                [
                    ("case", runtime_setup.case_name),
                    ("solution directory", str(resolved_solution_dir)),
                    ("samples directory", str(resolved_samples_dir)),
                    ("mesher log", str(mesher_log_path)),
                    ("next", "materializing mesh"),
                ],
                flush=True,
            )
        except BaseException as error:
            logger_error = error
    if logger_error is not None and startup_logger is not None:
        with suppress(BaseException):
            startup_logger.close(status="failed", failure=logger_error)
    _raise_collective_failure(logger_error, "startup logging")

    try:
        with mesher_log_session(
            mesher_log_path if is_root else None,
            reporter=(
                (lambda message: startup_logger.message(message, flush=True))
                if startup_logger is not None
                else None
            ),
        ):
            with mesh_stage("mesh materialization") as materialization:
                mesh_data = _materialize_mesh(mesh, is_root=materialize_mesh_here)
                if mesh_data is not None:
                    materialization.details(
                        cells=mesh_data.get("n_cells"),
                        faces=mesh_data.get("n_faces"),
                        points=mesh_data.get("n_points"),
                    )
            if is_root and mesh_data is not None:
                _save_generated_mesh(mesh_data, resolved_solution_dir, runtime_setup.output)
        if startup_logger is not None:
            startup_logger.info("component=fvm_startup status=mesh_materialized", flush=True)
    except BaseException as error:
        _raise_collective_failure(error, "mesh materialization/export")
        raise

    try:
        from .core.solver import FVMSolver

        solver = FVMSolver(
            runtime_setup,
            case_dir=str(resolved_case_dir),
            solution_dir=str(resolved_solution_dir),
            samples_dir=(str(resolved_samples_dir) if samples_requested else None),
            mesh_data=mesh_data,
            logger=startup_logger,
        )
        return solver
    except BaseException as error:
        if startup_logger is not None:
            try:
                startup_logger.warning("component=fvm_startup status=failed", flush=True)
                startup_logger.close(status="failed", failure=error)
            except BaseException:
                pass
        raise


__all__ = ["create_fvm_solver"]
