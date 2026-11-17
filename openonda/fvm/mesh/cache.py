"""Specification-based reuse at the native FVM mesh construction boundary.

A stored mesh is reused when its key matches: the digest of the mesher's
specification (sizes, domain, refinements and the hash of every surface file),
of the source files that build the mesh, and of the versions of the libraries
whose arithmetic enters it. The source files are the Cartesian mesher
(``mesh/cartesian``), the geometry, topology, validation and surface modules it
calls, the VTK exporter that builds the grid its section and intersection checks
read, and the Numba wrapper. Files that only store, partition or log a mesh
(``partition.py``, ``coupled.py``, ``progress.py``, ``rectilinear.py``) are not
part of the key.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import TypeGuard
from zipfile import BadZipFile

import numba
import numpy as np
import scipy
import vtk

from ..io.mesh_storage import load_native_mesh
from .cartesian.config import (
    BoxDomain,
    BoxPatches,
    BoxRefinement,
    PatchRefinement,
    STLSurface,
)
from .cartesian.extrusion import ExtrudedCartesianMesher
from .cartesian.mesher import CartesianMesher
from .data import MeshData
from .progress import mesh_event

_OPENONDA_ROOT = Path(__file__).resolve().parents[2]
_MESH_SOURCES = (
    "fvm/mesh/data.py",
    "fvm/mesh/geometry.py",
    "fvm/mesh/surface_classification.py",
    "fvm/mesh/topology.py",
    "fvm/mesh/triangulated_surface.py",
    "fvm/mesh/validation.py",
    "fvm/io/mesh_storage.py",
    "fvm/io/vtk_exporter.py",
    "core/_numba.py",
)


def meshing_source_files() -> tuple[Path, ...]:
    """Return the source files whose bytes enter the mesh key, in hashing order."""
    cartesian = sorted((_OPENONDA_ROOT / "fvm/mesh/cartesian").glob("*.py"))
    return (*cartesian, *(_OPENONDA_ROOT / name for name in _MESH_SOURCES))


def can_reuse_mesh(mesher: object) -> TypeGuard[CartesianMesher | ExtrudedCartesianMesher]:
    """Reuse only native implementations whose complete inputs are known."""
    if type(mesher) is ExtrudedCartesianMesher:
        return (
            can_reuse_mesh(mesher.source)
            and type(mesher.domain) is BoxDomain
            and type(mesher.domain.patches) is BoxPatches
        )
    if type(mesher) is not CartesianMesher:
        return False
    return (
        all(
            type(domain) is BoxDomain and type(domain.patches) is BoxPatches
            for domain in (mesher.domain, mesher.requested_domain)
        )
        and all(type(surface) is STLSurface for surface in mesher.surfaces)
        and all(type(value) is BoxRefinement for value in mesher.refinements)
        and all(type(value) is PatchRefinement for value in mesher.patch_refinements)
    )


def _specification(mesher: CartesianMesher | ExtrudedCartesianMesher) -> dict[str, object]:
    if isinstance(mesher, ExtrudedCartesianMesher):
        return {
            "type": "extruded",
            "source": _specification(mesher.source),
            "domain": asdict(mesher.domain),
            "levels": tuple(float(value) for value in mesher.levels),
        }
    data = {
        name: getattr(mesher, name)
        for name in (
            "max_cell_size",
            "boundary_cell_size",
            "min_cell_size",
            "cell_size_anchor",
            "surface_may_cross_domain_boundary",
        )
    }
    data["type"] = "cartesian"
    data["domain"] = asdict(mesher.domain)
    data["requested_domain"] = asdict(mesher.requested_domain)
    data["surfaces"] = [
        {"sha256": surface.sha256, "patch": surface.patch} for surface in mesher.surfaces
    ]
    for name in ("refinements", "patch_refinements"):
        data[name] = [asdict(value) for value in getattr(mesher, name)]
    return data


def meshing_input_hash(mesher: CartesianMesher | ExtrudedCartesianMesher) -> str:
    """Hash meshing inputs and implementation without machine or output paths."""
    if not can_reuse_mesh(mesher):
        raise TypeError("Mesh input_hash requires native meshing implementations and inputs")
    digest = hashlib.sha256(
        json.dumps(_specification(mesher), sort_keys=True, allow_nan=False).encode()
    )
    for path in meshing_source_files():
        digest.update(path.relative_to(_OPENONDA_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    for library in (np, numba, scipy, vtk):
        digest.update(f"{library.__name__}=={library.__version__}".encode())
    return digest.hexdigest()


def build_or_load_cached_mesh(
    mesher: CartesianMesher | ExtrudedCartesianMesher, path: Path
) -> tuple[MeshData, bool]:
    """Reuse matching native input; let the factory save generated meshes.

    Returns the mesh and whether it was loaded from ``path``, where it is already stored.
    """
    input_hash = meshing_input_hash(mesher)
    if path.is_file():
        try:
            mesh = load_native_mesh(path)
        except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
            mesh = None
        if mesh is not None and mesh.get("meshing_input_hash") == input_hash:
            mesh_event("mesh cache hit", path=path.resolve(), cells=mesh["n_cells"])
            return mesh, True
    mesh = mesher.build()
    mesh["meshing_input_hash"] = input_hash
    return mesh, False
