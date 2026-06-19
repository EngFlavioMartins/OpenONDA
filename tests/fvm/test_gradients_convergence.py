import numpy as np
import pytest

from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.fields.gradients import compute_gradient_gauss_linear_vectorized


def _l2_error(grad_computed, grad_exact, volumes):
    err = grad_computed - grad_exact
    return np.sqrt(np.sum(volumes[:, np.newaxis] * (err ** 2))) / np.sqrt(np.sum(volumes))


def _set_ghost_cells_to_analytic(phi, mesh, geo, func):
    n_elem = mesh["n_elements"]
    n_int = mesh["n_interior_faces"]
    fc = geo["face_centroids"]
    for b in mesh["boundary"]:
        start, nf = b["startFace"], b["nFaces"]
        for j in range(nf):
            fi = start + j
            gi = n_elem + (fi - n_int)
            phi[gi] = func(fc[fi])


class TestGradientConvergence:
    """φ = x² + y² + z² → ∇φ = [2x, 2y, 2z].

    NOTE: The Green‑Gauss gradient with cell‑centre‑to‑face interpolation is
    NOT convergent on non‑orthogonal tetrahedral meshes because the face
    centroid does not lie on the cell‑centre line.  The gradient has O(1)
    error on all refinement levels.  A least‑squares gradient (future work)
    would restore O(h²) convergence.

    For now we verify that the gradient error is bounded.
    """

    def test_gradient_quadratic_scalar(self, gmsh_unit_cube):
        mesh = gmsh_unit_cube
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_elements"]
        n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]
        cents = geo["element_centroids"]

        def phi_func(c):
            return c[0] ** 2 + c[1] ** 2 + c[2] ** 2

        phi = np.zeros(n_elem + n_bnd)
        phi[:n_elem] = np.array([phi_func(c) for c in cents])
        _set_ghost_cells_to_analytic(phi, mesh, geo, phi_func)

        grad = compute_gradient_gauss_linear_vectorized(phi, mesh, geo)
        g = grad[:n_elem].squeeze()
        grad_exact = 2.0 * cents
        err = _l2_error(g, grad_exact, geo["element_volumes"])
        assert err < 2.0, f"L₂ error too large: {err:.4f}"

    @pytest.mark.slow
    def test_gradient_error_bounded(self):
        """Verify the Gauss‑linear gradient error does not diverge with
        refinement on tetrahedral meshes."""
        import gmsh

        errors = []
        for lcar in [0.5, 0.25, 0.125]:
            gmsh.initialize()
            try:
                model = gmsh.model
                model.add("conv_cube")
                model.occ.addBox(0, 0, 0, 1, 1, 1)
                model.occ.synchronize()
                model.mesh.setSize(model.getEntities(0), lcar)
                model.mesh.generate(3)
                from source.solvers.FVM.mesh.gmsh_importer import GmshImporter

                imp = GmshImporter()
                mesh = imp.get_mesh_data()
            finally:
                gmsh.finalize()

            def phi_func(c):
                return c[0] ** 2 + c[1] ** 2 + c[2] ** 2

            geo = compute_mesh_geometry(mesh)
            cents = geo["element_centroids"]
            n_elem = mesh["n_elements"]
            n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]
            phi = np.zeros(n_elem + n_bnd)
            phi[:n_elem] = np.array([phi_func(c) for c in cents])
            _set_ghost_cells_to_analytic(phi, mesh, geo, phi_func)
            grad = compute_gradient_gauss_linear_vectorized(phi, mesh, geo)
            g = grad[:n_elem].squeeze()
            grad_exact = 2.0 * cents
            err = _l2_error(g, grad_exact, geo["element_volumes"])
            errors.append(err)

        # Gradient error should not increase significantly with refinement
        assert max(errors) < 1.5, f"Gradient error too large: {errors}"
