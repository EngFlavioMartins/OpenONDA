"""Reuse a native Cartesian mesh only when its complete specification matches."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from ..io.mesh_storage import load_native_mesh, save_native_mesh
from .cartesian.mesher import CartesianMesher
from .progress import mesh_event


class CachedMesh:
    """Cache one Cartesian mesher's result as a portable native archive.

    Geometry content, all meshing controls, and mesher source code form the
    identity. A changed specification rebuilds automatically. Solver controls
    and output paths do not change the mesh. Delete the archive to force a build.
    """

    def __init__(self, mesher: CartesianMesher, path: str | Path):
        if not isinstance(mesher, CartesianMesher):
            raise TypeError("CachedMesh currently supports CartesianMesher")
        self.mesher, self.path = mesher, Path(path)

    def identity(self) -> str:
        m = self.mesher
        data = {
            name: getattr(m, name)
            for name in (
                "max_cell_size",
                "boundary_cell_size",
                "min_cell_size",
                "surface_may_cross_domain_boundary",
            )
        }
        data["domain"] = asdict(m.domain)
        data["surfaces"] = [
            {"sha256": s.sha256, "patch": s.patch, "allow_open": s.allow_open} for s in m.surfaces
        ]
        for name in ("refinements", "patch_refinements", "boundary_layers"):
            data[name] = [asdict(value) for value in getattr(m, name)]
        data["features"] = None if m.features is None else asdict(m.features)
        source = Path(__file__).parent
        digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode())
        for path in sorted(source.rglob("*.py")):
            if path != Path(__file__):
                digest.update(str(path.relative_to(source)).encode())
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def build(self):
        identity = self.identity()
        if self.path.is_file():
            mesh = load_native_mesh(self.path)
            if mesh.get("cartesian_cache_identity") == identity:
                mesh_event("mesh cache hit", path=self.path.resolve(), cells=mesh["n_cells"])
                return mesh
        mesh = self.mesher.build()
        mesh["cartesian_cache_identity"] = identity
        save_native_mesh(mesh, self.path)
        return mesh
