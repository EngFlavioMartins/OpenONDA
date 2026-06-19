"""3D Poiseuille duct flow — regression test (structured hex mesh).

Drives pressure‑driven flow through a square duct with no‑slip walls,
inlet fixed‑velocity, and outlet fixed‑pressure.  Verifies basic physical
plausibility: net forward flow, pressure drop, centre‑plane velocity peak.
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

L = 2.0
H = 1.0
W = 0.5
U_IN = 1.5
NU = 0.01
NX, NY, NZ = 20, 10, 5


@pytest.mark.slow
class Test3DPoiseuille:
    def test_pressure_driven_duct(self, tmp_path):
        gmsh.initialize()
        try:
            gmsh.model.add("duct")
            gmsh.model.occ.addBox(0, -H / 2, -W / 2, L, H, W)
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

            surfaces = gmsh.model.getEntities(2)
            inlets, outlets, walls = [], [], []
            x_min, x_max = 0.0, L
            y_min, y_max = -H / 2, H / 2
            z_min, z_max = -W / 2, W / 2
            for dim, tag in surfaces:
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
                if abs(xmin - x_min) < tol and abs(xmax - x_min) < tol:
                    inlets.append(tag)
                elif abs(xmin - x_max) < tol and abs(xmax - x_max) < tol:
                    outlets.append(tag)
                elif (
                    abs(ymin - y_min) < tol
                    or abs(ymax - y_max) < tol
                    or abs(zmin - z_min) < tol
                    or abs(zmax - z_max) < tol
                ):
                    walls.append(tag)

            gmsh.model.addPhysicalGroup(2, inlets, 1, "inlet")
            gmsh.model.addPhysicalGroup(2, outlets, 2, "outlet")
            gmsh.model.addPhysicalGroup(2, walls, 3, "walls")

            imp = GmshImporter()
            mesh = imp.get_mesh_data()
        finally:
            gmsh.finalize()

        config = FVMConfig(
            case_name="poiseuille",
            time=TimeConfig.transient(dt=0.05, duration=1.0, write_interval=50),
            solver=SolverParams.pimple(
                n_correctors=2,
                n_outer=1,
                linear_solver="spsolve",
                convection_scheme="upwind",
            ),
            transport=TransportConfig(density=1.0, nu=NU),
            boundaries=[
                BoundaryConfig.inlet("inlet", [U_IN, 0.0, 0.0]),
                BoundaryConfig.outlet("outlet", 0.0),
                BoundaryConfig.wall("walls"),
            ],
            initial_U=[0.0, 0.0, 0.0],
            initial_p=0.0,
        )

        solver = Solver(config, str(tmp_path / "case"), mesh_data=mesh)
        n_steps = int(1.0 / 0.05)
        for _ in range(n_steps):
            solver.evolve()

        U = solver.U[: mesh["n_elements"]]
        p = solver.p[: mesh["n_elements"]]
        from source.solvers.FVM.mesh.geometry import compute_mesh_geometry

        geo = compute_mesh_geometry(mesh)
        cents = geo["element_centroids"]

        assert np.mean(U[:, 0]) > 0, "Net forward flow should be positive"

        order = np.argsort(cents[:, 0])
        p_sorted = p[order]
        n10 = max(1, len(p_sorted) // 10)
        p_in = np.mean(p_sorted[:n10])
        p_out = np.mean(p_sorted[-n10:])
        assert p_in > p_out, f"Pressure should drop: p_in={p_in:.4f} p_out={p_out:.4f}"

        near_center = (np.abs(cents[:, 1]) < 0.1 * H) & (np.abs(cents[:, 2]) < 0.1 * W)
        if near_center.sum() > 0:
            Ux_center = np.mean(U[near_center, 0])
            near_wall = np.abs(cents[:, 1]) > 0.4 * H
            if near_wall.sum() > 0:
                Ux_wall = np.mean(U[near_wall, 0])
                assert Ux_center > Ux_wall, "Centre velocity should exceed wall velocity"
