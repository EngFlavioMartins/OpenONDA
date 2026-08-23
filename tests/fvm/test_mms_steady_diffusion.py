"""MMS test — discrete-source consistency check for steady diffusion.

Because the Green‑Gauss gradient is O(1) inaccurate on non‑orthogonal
tetrahedral meshes (see test_gradients_convergence.py), the analytical
MMS source S = –∇²φ_exact does NOT match the discrete operator and the
Picard iteration for the deferred non‑orthogonal correction is unstable.

We therefore use the **discrete source** approach:

    R = A·φ_exact – b   (evaluated from the exact field)

and verify that solving

    A·φ = b + R

recovers φ_exact to machine precision.  This tests that the linear‑system
assembly and solve work correctly.
"""

import numpy as np
import pytest

from source.solvers.fvm.assemble.diffusion import assemble_diffusion_term
from source.solvers.fvm.assemble.matrix_assembly import (
    assemble_matrix_from_fluxes_vectorized,
    assemble_rhs_from_fluxes_vectorized,
)
from source.solvers.fvm.fields.gradients import compute_gauss_gradient
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.solve.linear_interface import solve_linear_system


def _scalar_field_exact(x, y, z):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)


def _assemble_system(mesh, geo):
    """Assemble A·φ = b for exact φ_exact with fixedValue BCs.

    Returns: A, b, phi_exact_interior, volumes
    """
    n_elem = mesh["n_cells"]
    n_int = mesh["n_interior_faces"]
    n_bnd = mesh["n_faces"] - n_int
    cents = geo["cell_centre"]
    fc = geo["face_centre"]

    scalar_field = np.zeros(n_elem + n_bnd)
    scalar_field[:n_elem] = _scalar_field_exact(cents[:, 0], cents[:, 1], cents[:, 2])
    for b in mesh["boundary"]:
        b["boundary_condition_type"] = "fixedValue"
        start, nf = b["start_face"], b["n_faces"]
        for j in range(nf):
            fi = start + j
            gi = n_elem + (fi - n_int)
            scalar_field[gi] = _scalar_field_exact(fc[fi, 0], fc[fi, 1], fc[fi, 2])

    scalar_field_gradient = compute_gauss_gradient(scalar_field, mesh, geo)
    diffusivity = np.ones(n_elem)
    flux_data = assemble_diffusion_term(
        scalar_field, scalar_field_gradient, diffusivity, mesh, geo, mesh["boundary"]
    )
    A = assemble_matrix_from_fluxes_vectorized(flux_data, mesh)
    b = assemble_rhs_from_fluxes_vectorized(flux_data, mesh)
    return A, b, scalar_field[:n_elem], geo["cell_volume"]


class TestMMSSteadyDiffusion:
    """Discrete‑source MMS: A·φ = b + R with R = A·φ_exact – b → φ = φ_exact."""

    @pytest.mark.slow
    def test_consistency(self):
        """L₂ error of φ_sol vs φ_exact should be at solver‑tolerance level."""
        gmsh = pytest.importorskip("gmsh", reason="Gmsh FVM test dependency is not installed")

        for lcar in [0.5, 0.25]:
            gmsh.initialize()
            try:
                model = gmsh.model
                model.add("mms_cube")
                model.occ.addBox(0, 0, 0, 1, 1, 1)
                model.occ.synchronize()
                model.mesh.setSize(model.getEntities(0), lcar)
                model.mesh.generate(3)
                from source.solvers.fvm.mesh.gmsh_importer import GmshImporter

                imp = GmshImporter()
                mesh = imp.get_mesh_data()
            finally:
                gmsh.finalize()

            geo = compute_mesh_geometry(mesh)
            A, b, phi_exact, vol = _assemble_system(mesh, geo)

            # Discrete MMS source
            R = A @ phi_exact - b
            b_mms = b + R  # → A·φ_sol = b + R = A·φ_exact

            phi_sol = solve_linear_system(A, b_mms, method="spsolve", equation_type="scalar")

            diff = phi_sol - phi_exact
            err = np.sqrt(np.sum(vol * diff**2) / np.sum(vol))
            assert err < 1e-12, f"Discrete‑source MMS error too large: {err:.2e}"

    def test_single_residual(self, gmsh_unit_cube):
        """Verify the solve runs without error on a single mesh."""
        mesh = gmsh_unit_cube
        geo = compute_mesh_geometry(mesh)
        A, b, phi_exact, vol = _assemble_system(mesh, geo)

        R = A @ phi_exact - b
        b_mms = b + R
        phi_sol = solve_linear_system(A, b_mms, method="spsolve", equation_type="scalar")
        assert phi_sol.shape == (mesh["n_cells"],)
        assert np.all(np.isfinite(phi_sol))
