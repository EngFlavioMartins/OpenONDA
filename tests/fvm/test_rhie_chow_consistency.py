import numpy as np
import pytest

from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.fields.gradients import compute_gradient_gauss_linear_vectorized
from source.solvers.FVM.assemble.matrix_assembly import assemble_matrix_from_fluxes_vectorized
from source.solvers.FVM.assemble.diffusion import assemble_diffusion_term
from source.solvers.FVM.assemble.convection import compute_mass_flow_rate
from source.solvers.FVM.solve.simple_solver import (
    assemble_pressure_correction_equation_rhie_chow,
    correct_velocity_and_flux,
    _compute_rhie_chow_coefficients,
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
            U, A_U, p, 1.0, mesh, geo, mesh["boundary"], alpha_u=alpha_u,
        )

        # From correction (uses A_U * alpha_u internally)
        p_prime = np.zeros(n_elem)
        U_corr, phi_corr = correct_velocity_and_flux(
            U.copy(), phi_star.copy(), p_prime, A_U, mesh, geo, mesh["boundary"],
            rho=1.0, alpha_u=alpha_u,
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
