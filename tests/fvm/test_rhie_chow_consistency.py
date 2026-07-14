import numpy as np

from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.solve.simple_solver import (
    _compute_pressure_face_conductance,
    _compute_rhie_chow_coefficients,
    assemble_pressure_correction_equation_rhie_chow,
    correct_velocity_and_flux,
    update_scalar_boundaries,
)


class TestRhieChowConsistency:
    """A_U_physical = A_U * alpha_u is used identically in assembly and correction."""

    def test_du_coefficient_identity(self, hand_built_3d_mesh):
        """Both assembly and correction produce the same DU for given A_U and alpha_u."""
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        for b in mesh["boundary"]:
            b["bc_type"] = "zeroGradient"
            b["bc_type_U"] = "zeroGradient"

        n_elem = mesh["n_elements"]
        volumes = geo["element_volumes"]
        A_U = np.ones((n_elem, 3)) * 10.0  # per-component diagonal
        alpha_u = 0.7

        # Expected DUs (per component)
        A_U_phys = A_U * alpha_u  # (n_elem, 3)
        expected_DU = volumes[:, np.newaxis] / A_U_phys  # (n_elem, 3)

        # From assembly (uses A_U * alpha_u internally)
        U = np.zeros((n_elem + mesh["n_faces"] - mesh["n_interior_faces"], 3))
        p = np.zeros(n_elem + mesh["n_faces"] - mesh["n_interior_faces"])
        A_p, b_p, phi_star = assemble_pressure_correction_equation_rhie_chow(
            U,
            A_U,
            p,
            1.0,
            mesh,
            geo,
            mesh["boundary"],
            alpha_u=alpha_u,
        )

        # From correction (uses A_U * alpha_u internally)
        p_prime = np.zeros(n_elem)
        U_corr, phi_corr = correct_velocity_and_flux(
            U.copy(),
            phi_star.copy(),
            p_prime,
            A_U,
            mesh,
            geo,
            mesh["boundary"],
            rho=1.0,
            alpha_u=alpha_u,
        )

        # Direct computation
        direct_DU = _compute_rhie_chow_coefficients(volumes, A_U_phys)
        assert np.allclose(expected_DU, direct_DU), "expected DU mismatch"

    def test_du_differs_with_alpha(self, hand_built_3d_mesh):
        """DU(alpha_u=0.5) should be 2× DU(alpha_u=1.0)."""
        mesh = hand_built_3d_mesh
        geo = compute_mesh_geometry(mesh)
        volumes = geo["element_volumes"]
        A_U = np.ones((mesh["n_elements"], 3)) * 10.0

        DU_1 = _compute_rhie_chow_coefficients(volumes, A_U * 1.0)
        DU_05 = _compute_rhie_chow_coefficients(volumes, A_U * 0.5)
        assert np.allclose(DU_05, 2.0 * DU_1)

    def test_pressure_solve_removes_flux_defect_on_skewed_mesh(self):
        """Assembly and correction must conserve with the same non-orthogonal metric."""
        from scipy.sparse.linalg import spsolve

        from source.solvers.FVM.fields.diagnostics import compute_continuity_error

        from ._structured_mesh import structured_box

        mesh = structured_box(3, 3, 2)
        # Affine shear makes Sf non-parallel to the owner-neighbour vector while
        # preserving a valid mesh and exact cell connectivity.
        mesh["points"][:, 0] += 0.35 * mesh["points"][:, 1]
        geo = compute_mesh_geometry(mesh)

        for patch in mesh["boundary"]:
            patch["bc_type_U"] = "zeroGradient"
            patch["bc_type_p"] = "fixedValue" if patch["name"] == "xmax" else "zeroGradient"
            patch["value_p"] = 0.0

        n = mesh["n_elements"]
        nb = mesh["n_faces"] - mesh["n_interior_faces"]
        rng = np.random.default_rng(17)
        U = rng.normal(scale=0.1, size=(n + nb, 3))
        p = rng.normal(scale=0.05, size=n + nb)
        update_scalar_boundaries(p, mesh, mesh["boundary"], field_name="p")
        A_U = np.full((n, 3), 4.0)

        A_p, b_p, phi_star = assemble_pressure_correction_equation_rhie_chow(
            U, A_U, p, 1.0, mesh, geo, mesh["boundary"]
        )
        p_prime = spsolve(A_p, b_p)
        _, phi = correct_velocity_and_flux(
            U.copy(), phi_star.copy(), p_prime, A_U, mesh, geo, mesh["boundary"]
        )

        linear_residual = np.linalg.norm(A_p @ p_prime - b_p)
        continuity = compute_continuity_error(phi, mesh, geo)
        assert linear_residual < 1e-11
        assert np.max(np.abs(continuity)) < 2e-11

    def test_face_conductance_is_positive_on_skewed_mesh(self):
        from ._structured_mesh import structured_box

        mesh = structured_box(2, 2, 2)
        mesh["points"][:, 0] += 0.25 * mesh["points"][:, 1]
        geo = compute_mesh_geometry(mesh)
        DU = np.ones((mesh["n_elements"], 3))
        conductance = _compute_pressure_face_conductance(mesh, geo, DU)
        assert conductance.shape == (mesh["n_faces"],)
        assert np.all(conductance > 0.0)
