"""Admit a mesh cache against the local case's explicit mesher inputs."""

import hashlib
import json
from pathlib import Path
import numpy as np
import openonda.fvm.mesher as msh


def mesh_cache_identity(mesh: msh.ExtrudedCartesianMesher) -> str:
    """Include surface contents, mesher controls/code and extrusion geometry."""
    from source.solvers.fvm.mesh.cache import CachedMesh

    specification = {
        "source": CachedMesh(mesh.source, Path("unused.npz")).identity(),
        "bounds": mesh.domain.bounds,
        "levels": mesh.levels,
    }
    return hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()


def cached_mesh_matches_case(
    path: Path, mesh: msh.ExtrudedCartesianMesher, *, require_identity: bool = False
) -> bool:
    """Admit a cached mesh only for the current resolved box and resolution."""
    try:
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
        identity = metadata.get("cylinder_mesh_cache_identity")
        if (require_identity or identity is not None) and identity != mesh_cache_identity(mesh):
            return False
        generation = metadata.get("mesh_generation", {})
        bounds = generation.get("domain", ())
        levels = generation.get("extrusion_levels", ())
        return bool(
            len(bounds) == 6
            and np.allclose(bounds, mesh.domain.bounds, rtol=0, atol=1e-12)
            and len(levels) == len(mesh.levels)
            and np.allclose(levels, mesh.levels, rtol=0, atol=1e-12)
            and generation.get("resolved_background_cell_size") == mesh.max_cell_size
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def resolve_mesh(builder, directory, *, reuse):
    cache = Path(directory) / "solution/fvm/mesh.npz"
    if not reuse:
        return builder
    if cache.is_file():
        if not cached_mesh_matches_case(cache, builder):
            raise ValueError(
                "Cached cylinder mesh differs from this setup. Run ./allrun.sh --fresh "
                "to archive the previous results and start from zero."
            )
        return cache

    def generate():
        result = builder.build()
        result["cylinder_mesh_cache_identity"] = mesh_cache_identity(builder)
        return result

    return generate
