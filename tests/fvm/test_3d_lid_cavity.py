"""3D lid‑driven cavity flow — regression test (structured hex mesh).

Top wall (y = H) moves at constant speed; all other walls no‑slip.
Verifies clockwise primary vortex and physically reasonable centre‑plane
velocity magnitudes.
"""

import gmsh
import numpy as np
import pytest

from source.solvers.FVM import (
    BoundaryConfig,
    FVMConfig,
    Solver,
    SolverParams,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.mesh.gmsh_importer import GmshImporter

L = 1.0
U_LID = 1.0
NU = 0.01
NX, NY, NZ = (20, 20, 3)


@pytest.mark.slow
class Test3DLidDrivenCavity:
    def test_lid_driven_cavity(self, tmp_path):
        gmsh.initialize()
        try:
            gmsh.model.add("cavity")
            gmsh.model.occ.addBox(0, 0, 0, L, L, 0.1)
            gmsh.model.occ.synchronize()

            tol = 1e-6
            for dim, tag in gmsh.model.getEntities(1):
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
                dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
                if dx > tol and dy < tol and dz < tol:
                    gmsh.model.mesh.setTransfiniteCurve(tag, NX)
                elif dy > tol and dx < tol and dz < tol:
                    gmsh.model.mesh.setTransfiniteCurve(tag, NY)
                elif dz > tol and dx < tol and dy < tol:
                    gmsh.model.mesh.setTransfiniteCurve(tag, NZ)

            for surf in gmsh.model.getEntities(2):
                gmsh.model.mesh.setTransfiniteSurface(surf[1])
                gmsh.model.mesh.setRecombine(surf[0], surf[1])
            for vol in gmsh.model.getEntities(3):
                gmsh.model.mesh.setTransfiniteVolume(vol[1])

            gmsh.model.mesh.generate(3)

            tol = 1e-4
            surfaces = gmsh.model.getEntities(2)
            lid, walls = [], []
            x_min, x_max = 0.0, L
            y_min, y_max = 0.0, L
            for dim, tag in surfaces:
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
                if abs(ymin - y_max) < tol and abs(ymax - y_max) < tol:
                    lid.append(tag)
                else:
                    walls.append(tag)

            gmsh.model.addPhysicalGroup(2, lid, 1, "lid")
            gmsh.model.addPhysicalGroup(2, walls, 2, "walls")

            imp = GmshImporter()
            mesh = imp.get_mesh_data()
        finally:
            gmsh.finalize()

        config = FVMConfig(
            case_name="cavity",
            time=TimeConfig.transient(dt=0.01, duration=2.0, write_interval=200),
            solver=SolverParams.pimple(
                n_correctors=2,
                n_outer=1,
                linear_solver="spsolve",
                convection_scheme="upwind",
            ),
            transport=TransportConfig(density=1.0, nu=NU),
            boundaries=[
                BoundaryConfig("lid",
                                type_U="fixedValue", value_U=[U_LID, 0.0, 0.0],
                                type_p="zeroGradient"),
                BoundaryConfig.wall("walls"),
            ],
            initial_U=[0.0, 0.0, 0.0],
            initial_p=0.0,
        )

        solver = Solver(config, str(tmp_path / "case"), mesh_data=mesh)
        n_steps = int(2.0 / 0.01)
        for _ in range(n_steps):
            solver.evolve()

        U = solver.U[: mesh["n_elements"]]
        from source.solvers.FVM.mesh.geometry import compute_mesh_geometry

        geo = compute_mesh_geometry(mesh)
        cents = geo["element_centroids"]

        top = cents[:, 1] > 0.8 * L
        bot = cents[:, 1] < 0.2 * L
        if top.sum() > 0 and bot.sum() > 0:
            Ux_top = np.mean(U[top, 0])
            Ux_bot = np.mean(U[bot, 0])
            assert Ux_top > 0, f"Near‑lid flow should be rightward: {Ux_top:.4f}"
            assert Ux_bot < 0, f"Near‑bottom return flow should be leftward: {Ux_bot:.4f}"

        centre_mask = (
            (np.abs(cents[:, 0] - 0.5 * L) < 0.15)
            & (np.abs(cents[:, 1] - 0.5 * L) < 0.15)
            & (np.abs(cents[:, 2] - 0.05) < 0.03)
        )
        if centre_mask.sum() > 0:
            mag = np.linalg.norm(U[centre_mask], axis=1).mean()
            assert 0 < mag < U_LID, (
                f"Centre speed {mag:.4f} should be positive and below lid speed"
            )
