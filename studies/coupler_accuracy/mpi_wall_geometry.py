#!/usr/bin/env python3
"""Qualify collective wall extraction: mpiexec -n 2 python this_file.py."""

from types import SimpleNamespace

from mpi4py import MPI
import numpy as np

from source.solvers.fvm.core.parallel import ParallelContext
from source.solvers.fvm.coupling.coupler_interface import CouplerInterfaceMixin
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.mesh.partition import localize_mesh_and_geometry
from source.solvers.fvm.mesh.rectilinear import box_mesh_3d

comm = MPI.COMM_WORLD
if comm.size < 2:
    raise RuntimeError("Use at least two MPI ranks")
axis = np.linspace(-1, 1, 9)
mesh = box_mesh_3d(axis, axis, axis, hole_box=(-0.5, 0.5) * 3, wall_patch_name="body")
geometry = compute_mesh_geometry(mesh, gradient_scheme="lsq", compute_lsq=False)
local_mesh, local_geometry, partition = localize_mesh_and_geometry(
    mesh, geometry, comm.rank, comm.size
)
view = CouplerInterfaceMixin()
view.mesh_data = local_mesh
view.geo_data = local_geometry
view.boundaries = local_mesh["boundary"]
view.setup = SimpleNamespace(boundaries=[SimpleNamespace(name="body", mesh_type="wall")])
view.parallel = ParallelContext(
    mode="petsc_partitioned",
    comm=comm,
    mpi=MPI,
    rank=comm.rank,
    size=comm.size,
    partition=partition,
)
triangles = view.get_wall_surface_triangles()
if comm.rank == 0:
    area = 0.5 * np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    np.testing.assert_allclose(np.linalg.norm(area, axis=1).sum(), 6, rtol=0, atol=1e-14)
    assert np.all(np.einsum("ij,ij->i", area, triangles.mean(axis=1)) < 0)
    print(f"PASS: {comm.size} ranks, {len(triangles)} triangles, wall area = 6", flush=True)
else:
    assert triangles.shape == (0, 3, 3)
