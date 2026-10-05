"""Specification-based reuse at the native FVM mesh construction boundary."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile

from ..io.mesh_storage import load_native_mesh
from .cartesian.config import (
    BoundaryLayers,
    BoxDomain,
    BoxPatches,
    BoxRefinement,
    ConeRefinement,
    FeatureRefinement,
    LineRefinement,
    PatchRefinement,
    SphereRefinement,
    STLSurface,
)
from .cartesian.extrusion import ExtrudedCartesianMesher
from .cartesian.mesher import CartesianMesher
from .progress import mesh_event


def can_reuse_mesh(mesher) -> bool:
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
        and all(
            type(value) in (BoxRefinement, SphereRefinement, ConeRefinement, LineRefinement)
            for value in mesher.refinements
        )
        and all(type(value) is PatchRefinement for value in mesher.patch_refinements)
        and all(type(value) is BoundaryLayers for value in mesher.boundary_layers)
        and (mesher.features is None or type(mesher.features) is FeatureRefinement)
    )


def _specification(mesher):
    if type(mesher) is ExtrudedCartesianMesher:
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
        {"sha256": surface.sha256, "patch": surface.patch, "allow_open": surface.allow_open}
        for surface in mesher.surfaces
    ]
    for name in ("refinements", "patch_refinements", "boundary_layers"):
        data[name] = [asdict(value) for value in getattr(mesher, name)]
    data["features"] = None if mesher.features is None else asdict(mesher.features)
    return data


def meshing_input_hash(mesher) -> str:
    """Hash meshing inputs and implementation without machine or output paths."""
    if not can_reuse_mesh(mesher):
        raise TypeError("Mesh input_hash requires native meshing implementations and inputs")
    digest = hashlib.sha256(
        json.dumps(_specification(mesher), sort_keys=True, allow_nan=False).encode()
    )
    source = Path(__file__).parent
    for path in sorted(source.rglob("*.py")):
        if path != Path(__file__):
            digest.update(path.relative_to(source).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build_or_load_cached_mesh(mesher, path: Path):
    """Reuse matching native input; let the factory save generated meshes."""
    input_hash = meshing_input_hash(mesher)
    if path.is_file():
        try:
            mesh = load_native_mesh(path)
        except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
            mesh = None
        if mesh is not None and mesh.get("meshing_input_hash") == input_hash:
            mesh_event("mesh cache hit", path=path.resolve(), cells=mesh["n_cells"])
            return mesh
    mesh = mesher.build()
    mesh["meshing_input_hash"] = input_hash
    return mesh
