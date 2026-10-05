"""Small native mesh fixtures for solver consistency checks."""

import numpy as np


def structured_box(
    nx: int,
    ny: int,
    nz: int,
    lx: float = 1.0,
    ly: float = 1.0,
    lz: float = 1.0,
) -> dict:
    """Build a small native rectilinear box for FVM consistency checks."""
    if min(nx, ny, nz) < 1 or min(lx, ly, lz) <= 0.0:
        raise ValueError("Cell counts and box lengths must be positive")
    from source.solvers.fvm.mesh.rectilinear import box_mesh_3d

    mesh = box_mesh_3d(
        np.linspace(0.0, lx, nx + 1),
        np.linspace(0.0, ly, ny + 1),
        np.linspace(0.0, lz, nz + 1),
    )
    names = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    for patch, name in zip(mesh["boundary"], names, strict=True):
        patch["name"] = name
    return mesh
