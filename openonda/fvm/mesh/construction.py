"""Build, load and store the mesh of an FVM case."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .progress import mesh_event, mesh_stage


def load_mesh_file(path: str | Path) -> dict[str, Any]:
    """Load a native ``.npz`` or Gmsh ``.msh`` mesh file."""
    path = Path(path)
    if path.suffix.lower() == ".msh":
        from .gmsh_importer import GmshImporter

        importer = GmshImporter()
        try:
            importer.load_mesh(str(path))
            return importer.get_mesh_data()
        finally:
            importer.finalize()
    if path.suffix.lower() == ".npz":
        from ..io.mesh_storage import load_native_mesh

        return load_native_mesh(path)
    raise ValueError(
        f"Unsupported mesh file {path.name!r}; expected a native '.npz' or Gmsh '.msh' file"
    )


def build_or_load_mesh(source: object, *, cache_path: Path) -> dict[str, Any]:
    """Return the native mesh mapping of a case mesh source.

    Parameters
    ----------
    source : path, mapping, mesher object or callable
        Mesh file, native mesh mapping, an object with ``build()`` or a
        callable returning the mapping or ``(mapping, report)``.
    cache_path : pathlib.Path
        Stored native mesh reused by a Cartesian mesher with identical input.
    """
    if isinstance(source, str | Path):
        mesh_event("mesh source", kind="file", path=Path(source).resolve())
        return load_mesh_file(source)

    mesh_event("mesh source", kind=type(source).__name__)
    from .cache import build_or_load_cached_mesh, can_reuse_mesh

    if can_reuse_mesh(source):
        generated = build_or_load_cached_mesh(source, cache_path)
    elif callable(source):
        generated = source()
    elif callable(getattr(source, "build", None)):
        generated = source.build()
    else:
        generated = source
    if callable(getattr(generated, "build", None)):
        generated = generated.build()
    if isinstance(generated, tuple):
        generated = generated[0]
    if not isinstance(generated, dict):
        raise TypeError(
            "FVMCase.mesh must be a path, mesh mapping, mesher or callable returning one"
        )
    return generated


def save_mesh(mesh_data: dict[str, Any], path: Path, output: Any) -> None:
    """Write the native mesh and its ParaView view ``<path>.vtu``.

    The view is written before geometric validation, so a rejected mesh
    remains inspectable.
    """
    from ..io.mesh_storage import save_native_mesh
    from ..io.vtk_exporter import VTKExporter, mesh_cell_fields
    from .geometry import compute_mesh_geometry

    path.parent.mkdir(parents=True, exist_ok=True)
    view = path.with_suffix(".vtu")
    exporter = VTKExporter(mesh_data, output)
    with mesh_stage("mesh export") as stage:
        save_native_mesh(mesh_data, path)
        exporter.export(str(view), mesh_cell_fields(mesh_data))
        stage.details(native=path, visualisation=view)
    with mesh_stage("mesh geometry and final visualisation") as stage:
        geometry = compute_mesh_geometry(mesh_data, compute_lsq=False)
        exporter.export(str(view), mesh_cell_fields(mesh_data, geometry["cell_volume"]))
        stage.details(cells=mesh_data.get("n_cells"), faces=mesh_data.get("n_faces"))
