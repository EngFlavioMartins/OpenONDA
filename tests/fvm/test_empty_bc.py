"""Test the `empty` boundary condition for quasi-3D (2D) extruded meshes.

The `empty` BC is used on the front/back faces of a 2D extruded mesh.
It enforces zero normal flux (no flow through the face) and zero normal
gradient of scalar quantities.

This test verifies:
  1. PIMPLE solver runs without divergence on a quasi-3D mesh
  2. Uz is negligible everywhere (2D behaviour enforced by empty BC)
  3. Basic flow physics hold
"""

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh", reason="Gmsh FVM test dependency is not installed")

from source.solvers.FVM import (
    BoundaryConfig,
    FVMSetup,
    LinearSolverConfig,
    PimpleControl,
    SchemesConfig,
    Solver,
    TimeConfig,
    TransportConfig,
)
from source.solvers.FVM.mesh.gmsh_importer import GmshImporter

L = 2.0
H = 1.0
U_IN = 1.0
NU = 0.01
NX, NY, NZ = 10, 6, 1  # single cell in z — quasi-3D


@pytest.mark.slow
class TestEmptyBCQuasi3D:
    """Verify empty BC on a quasi-3D extruded mesh."""

    def test_quasi_3d_duct_flow(self, tmp_path):
        gmsh.initialize()
        try:
            gmsh.model.add("duct")
            gmsh.model.occ.addBox(0, -H / 2, 0, L, H, 0.1)
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
            inlets, outlets, walls, empty_faces = [], [], [], []
            x_min, x_max = 0.0, L
            for dim, tag in surfaces:
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                cz = (zmin + zmax) / 2
                if abs(cx - x_min) < tol:  # x=0 face
                    inlets.append(tag)
                elif abs(cx - x_max) < tol:  # x=L face
                    outlets.append(tag)
                elif abs(cz - 0.05) < tol:  # z=0 or z=0.1 faces
                    # Only classify as empty if y-span is full
                    if abs(cy) < H / 2 + tol:
                        empty_faces.append(tag)
                    else:
                        walls.append(tag)
                else:
                    walls.append(tag)

            assert len(inlets) > 0, "No inlet surfaces found"
            assert len(outlets) > 0, "No outlet surfaces found"
            assert len(walls) > 0, "No wall surfaces found"
            assert len(empty_faces) > 0, "No empty surfaces found"

            gmsh.model.addPhysicalGroup(2, inlets, 1, "inlet")
            gmsh.model.addPhysicalGroup(2, outlets, 2, "outlet")
            gmsh.model.addPhysicalGroup(2, walls, 3, "walls")
            gmsh.model.addPhysicalGroup(2, empty_faces, 4, "empty")

            imp = GmshImporter()
            mesh = imp.get_mesh_data()
        finally:
            gmsh.finalize()

        boundary_names = {b["name"] for b in mesh["boundary"]}
        assert "inlet" in boundary_names, f"Missing inlet patch in mesh: {boundary_names}"
        assert "outlet" in boundary_names, f"Missing outlet patch: {boundary_names}"
        assert "walls" in boundary_names, f"Missing walls patch: {boundary_names}"
        assert "empty" in boundary_names, f"Missing empty patch: {boundary_names}"

        config = FVMSetup(
            case_name="quasi3d_duct",
            time=TimeConfig.transient(dt=0.05, duration=2.0, write_interval=100),
            schemes=SchemesConfig(convection_scheme="upwind"),
            linear=LinearSolverConfig(linear_solver="spsolve"),
            pimple=PimpleControl(n_correctors=2, n_outer_correctors=1),
            transport=TransportConfig(density=1.0, nu=NU),
            boundaries=[
                BoundaryConfig.inlet("inlet", [U_IN, 0.0, 0.0]),
                BoundaryConfig.outlet("outlet", 0.0),
                BoundaryConfig.wall("walls"),
                BoundaryConfig.empty("empty"),
            ],
            initial_U=[0.0, 0.0, 0.0],
            initial_p=0.0,
        )

        solver = Solver(config, str(tmp_path / "case"), mesh_data=mesh)
        n_steps = int(2.0 / 0.05)
        for _ in range(n_steps):
            solver.evolve()

        U = solver.U[: mesh["n_elements"]]
        p = solver.p[: mesh["n_elements"]]

        assert np.isfinite(U).all(), "NaN/Inf in velocity — solver diverged"
        assert np.isfinite(p).all(), "NaN/Inf in pressure — solver diverged"

        # 1. Single z-layer
        cents = solver.geo_data["element_centroids"]
        z_vals = np.unique(np.round(cents[:, 2], decimals=10))
        assert len(z_vals) == 1, f"Expected single z-layer for quasi-3D mesh, got {len(z_vals)}"

        # 2. Forward flow: mean Ux > 0
        assert np.mean(U[:, 0]) > 0, "Net forward flow should be positive"

        # 3. Uz must be near zero everywhere (empty BC enforces 2D)
        Uz_max = np.max(np.abs(U[:, 2]))
        Uz_mean = np.mean(np.abs(U[:, 2]))
        assert Uz_max < 0.05, f"2D flow: max|Uz|={Uz_max:.6f}"
        assert Uz_mean < 0.005, f"2D flow: mean|Uz|={Uz_mean:.6f}"

        # 4. Pressure drop along duct
        order = np.argsort(cents[:, 0])
        p_sorted = p[order]
        n10 = max(1, len(p_sorted) // 10)
        p_in = np.mean(p_sorted[:n10])
        p_out = np.mean(p_sorted[-n10:])
        assert p_in > p_out, f"Pressure should drop along duct: p_in={p_in:.4f} p_out={p_out:.4f}"

        # 5. Surface forces are finite (for all mesh patches)
        from source.solvers.FVM.fields.diagnostics import compute_surface_forces

        patch_names = [b["name"] for b in mesh["boundary"]]
        result = compute_surface_forces(
            solver.U,
            solver.p,
            NU * 1.0,
            1.0,
            mesh,
            solver.geo_data,
            mesh["boundary"],
            patch_names=patch_names,
            ref_U=U_IN,
            ref_area=H * L,
        )
        for name in patch_names:
            assert name in result, f"Missing patch '{name}' in force results"
            assert np.isfinite(result[name]["Ftot"]).all(), (
                f"Non-finite force on {name}: {result[name]['Ftot']}"
            )
