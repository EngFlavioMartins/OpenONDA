import numpy as np
import pytest

from source.solvers.FVM.fields.gradients import compute_gradient_lsq_vectorized
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry

from ._polyhedral_mesh import split_prism_box


def _l2_error(grad_computed, grad_exact, volumes):
    err = grad_computed - grad_exact
    return np.sqrt(np.sum(volumes[:, np.newaxis] * (err**2))) / np.sqrt(np.sum(volumes))


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
    """Weighted LSQ convergence for φ=x²+y²+z² on tetrahedra."""

    def test_gradient_quadratic_scalar(self, gmsh_unit_cube):
        mesh = gmsh_unit_cube
        geo = compute_mesh_geometry(mesh, gradient_scheme="lsq")
        n_elem = mesh["n_elements"]
        n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]
        cents = geo["element_centroids"]

        def phi_func(c):
            return c[0] ** 2 + c[1] ** 2 + c[2] ** 2

        phi = np.zeros(n_elem + n_bnd)
        phi[:n_elem] = np.array([phi_func(c) for c in cents])
        _set_ghost_cells_to_analytic(phi, mesh, geo, phi_func)

        grad = compute_gradient_lsq_vectorized(phi, mesh, geo)
        g = grad[:n_elem].squeeze()
        grad_exact = 2.0 * cents
        err = _l2_error(g, grad_exact, geo["element_volumes"])
        assert err < 0.7, f"L₂ error too large: {err:.4f}"

    @pytest.mark.slow
    def test_gradient_converges_on_three_tetrahedral_levels(self):
        gmsh = pytest.importorskip("gmsh", reason="Gmsh FVM test dependency is not installed")

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

            geo = compute_mesh_geometry(mesh, gradient_scheme="lsq")
            cents = geo["element_centroids"]
            n_elem = mesh["n_elements"]
            n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]
            phi = np.zeros(n_elem + n_bnd)
            phi[:n_elem] = np.array([phi_func(c) for c in cents])
            _set_ghost_cells_to_analytic(phi, mesh, geo, phi_func)
            grad = compute_gradient_lsq_vectorized(phi, mesh, geo)
            g = grad[:n_elem].squeeze()
            grad_exact = 2.0 * cents
            err = _l2_error(g, grad_exact, geo["element_volumes"])
            errors.append(err)

        observed_order = np.polyfit(
            np.log(np.array([0.5, 0.25, 0.125])), np.log(np.asarray(errors)), 1
        )[0]
        assert np.all(np.diff(errors) < 0.0), f"LSQ error is not monotone: {errors}"
        assert observed_order > 0.7, f"Observed LSQ order {observed_order:.3f}; errors={errors}"

    @pytest.mark.parametrize("mixed", [False, True], ids=["prism", "mixed_hex_prism"])
    def test_gradient_converges_on_three_polyhedral_levels(self, mixed):
        errors = []
        sizes = []
        for n in (2, 4, 8):
            mesh = split_prism_box(n, mixed=mixed)
            geo = compute_mesh_geometry(mesh, gradient_scheme="lsq")
            centres = geo["element_centroids"]
            n_cells = mesh["n_elements"]
            n_boundary = mesh["n_faces"] - mesh["n_interior_faces"]
            field = np.zeros(n_cells + n_boundary)
            field[:n_cells] = np.sum(centres**2, axis=1)
            _set_ghost_cells_to_analytic(field, mesh, geo, lambda point: np.sum(point**2))
            computed = compute_gradient_lsq_vectorized(field, mesh, geo)[:n_cells].squeeze()
            errors.append(_l2_error(computed, 2.0 * centres, geo["element_volumes"]))
            sizes.append(1.0 / n)

        observed_order = np.polyfit(np.log(sizes), np.log(errors), 1)[0]
        assert np.all(np.diff(errors) < 0.0), f"LSQ error is not monotone: {errors}"
        assert observed_order > 0.7, f"Observed LSQ order {observed_order:.3f}; errors={errors}"
