import numpy as np

from source.solvers.FVM.assemble.convection import (
    assemble_convection_term,
    compute_volumetric_face_flux,
)
from source.solvers.FVM.assemble.diffusion import assemble_diffusion_term
from source.solvers.FVM.assemble.matrix_assembly import (
    MatrixAssemblyWorkspace,
    assemble_matrix_from_fluxes_vectorized,
    assemble_rhs_from_fluxes_vectorized,
)
from source.solvers.FVM.fields.gradients import compute_gauss_gradient
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry


class TestMatrixAssembly:
    """Identity check: φ = 1 → matrix·1 − b = 0 for both diffusion and convection."""

    def test_diffusion_identity(self, hand_built_3d_mesh):
        mesh = hand_built_3d_mesh
        for b in mesh["boundary"]:
            b["bc_type"] = "zeroGradient"
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_elements"]
        n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]
        phi = np.ones(n_elem + n_bnd)
        grad_phi = compute_gauss_gradient(phi, mesh, geo)
        gamma = np.ones(n_elem)

        flux_data = assemble_diffusion_term(phi, grad_phi, gamma, mesh, geo, mesh["boundary"])
        A = assemble_matrix_from_fluxes_vectorized(flux_data, mesh)
        b = assemble_rhs_from_fluxes_vectorized(flux_data, mesh)
        assert np.allclose(A @ np.ones(n_elem) - b, 0.0, atol=1e-12)

    def test_convection_identity(self, hand_built_3d_mesh):
        mesh = hand_built_3d_mesh
        for b in mesh["boundary"]:
            b["bc_type"] = "zeroGradient"
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_elements"]
        n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]
        U = np.tile([1.0, 0.0, 0.0], (n_elem + n_bnd, 1))
        mdot = compute_volumetric_face_flux(U, mesh, geo)
        phi = np.ones(n_elem + n_bnd)

        flux_data = assemble_convection_term(
            phi, mdot, mesh, geo, mesh["boundary"], scheme="upwind"
        )
        A = assemble_matrix_from_fluxes_vectorized(flux_data, mesh)
        b = assemble_rhs_from_fluxes_vectorized(flux_data, mesh)
        assert np.allclose(A @ np.ones(n_elem) - b, 0.0, atol=1e-12)

    def test_workspace_updates_csr_coefficients_in_place(self, hand_built_3d_mesh):
        mesh = hand_built_3d_mesh
        n_faces = mesh["n_faces"]
        first_flux = {
            "flux_cf": np.linspace(1.0, 2.0, n_faces),
            "flux_ff": np.linspace(-0.5, -0.1, n_faces),
        }
        second_flux = {
            "flux_cf": 3.0 * first_flux["flux_cf"],
            "flux_ff": 2.0 * first_flux["flux_ff"],
        }
        workspace = MatrixAssemblyWorkspace.create(mesh)

        first = assemble_matrix_from_fluxes_vectorized(first_flux, mesh, workspace=workspace)
        first_values = first.data.copy()
        second = assemble_matrix_from_fluxes_vectorized(second_flux, mesh, workspace=workspace)
        reference = assemble_matrix_from_fluxes_vectorized(second_flux, mesh)

        assert second is first
        assert second.data is first.data
        assert not np.array_equal(second.data, first_values)
        np.testing.assert_allclose(second.toarray(), reference.toarray())
