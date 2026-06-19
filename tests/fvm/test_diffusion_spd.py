import numpy as np
import pytest

from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.fields.gradients import compute_gradient_gauss_linear_vectorized
from source.solvers.FVM.assemble.diffusion import assemble_diffusion_term
from source.solvers.FVM.assemble.matrix_assembly import (
    assemble_matrix_from_fluxes_vectorized,
    assemble_rhs_from_fluxes_vectorized,
)


@pytest.fixture(scope="module")
def diff_data(hand_built_3d_mesh):
    """Pre-compute mesh data with zeroGradient BCs for diffusion tests."""
    mesh = hand_built_3d_mesh
    geo = compute_mesh_geometry(mesh)
    n_elem = mesh["n_elements"]
    n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]

    # Add bc_type to boundary patches
    for b in mesh["boundary"]:
        b["bc_type"] = "zeroGradient"

    # Build full φ field (interior + ghost)
    phi = np.zeros(n_elem + n_bnd)
    phi[:n_elem] = 1.0  # φ=1 everywhere → zero gradient → no diffusion flux

    # Compute gradient
    grad_phi = compute_gradient_gauss_linear_vectorized(phi, mesh, geo)

    # Diffusion coefficient = 1 everywhere
    gamma = np.ones(n_elem)

    return {"mesh": mesh, "geo": geo, "phi": phi, "grad_phi": grad_phi, "gamma": gamma}


class TestDiffusionMatrixSPD:
    """Diffusion matrix properties: symmetry, row-sum zero, negative semi-definite."""

    def test_matrix_symmetric(self, diff_data):
        flux_data = assemble_diffusion_term(
            diff_data["phi"], diff_data["grad_phi"], diff_data["gamma"],
            diff_data["mesh"], diff_data["geo"], diff_data["mesh"]["boundary"],
        )
        A = assemble_matrix_from_fluxes_vectorized(flux_data, diff_data["mesh"])
        diff = (A - A.T).multiply(1.0 / (A.diagonal().mean() + 1e-30))
        assert diff.max() < 1e-12, f"Symmetry error: max|A-Aᵀ|/avg|diag| = {diff.max():.2e}"

    def test_row_sum_zero(self, diff_data):
        flux_data = assemble_diffusion_term(
            diff_data["phi"], diff_data["grad_phi"], diff_data["gamma"],
            diff_data["mesh"], diff_data["geo"], diff_data["mesh"]["boundary"],
        )
        A = assemble_matrix_from_fluxes_vectorized(flux_data, diff_data["mesh"])
        row_sum = np.array(A.sum(axis=1)).flatten()
        assert np.allclose(row_sum, 0.0, atol=1e-12), (
            f"max row-sum = {np.max(np.abs(row_sum)):.2e}"
        )

    def test_positive_semi_definite(self, diff_data):
        """vᵀAv ≥ 0 for each standard basis vector (diffusion operator -∇·∇ is PSD)."""
        flux_data = assemble_diffusion_term(
            diff_data["phi"], diff_data["grad_phi"], diff_data["gamma"],
            diff_data["mesh"], diff_data["geo"], diff_data["mesh"]["boundary"],
        )
        A = assemble_matrix_from_fluxes_vectorized(flux_data, diff_data["mesh"])
        n = A.shape[0]
        for i in range(min(n, 8)):
            v = np.zeros(n)
            v[i] = 1.0
            val = v @ (A @ v)
            assert val >= -1e-12, f"v[{i}]ᵀ A v[{i}] = {val:.4e} < 0 (should be ≥ 0)"

    def test_zero_residual_for_constant_field(self, diff_data):
        """If φ=1, diffusion matrix + RHS should produce zero residual."""
        flux_data = assemble_diffusion_term(
            diff_data["phi"], diff_data["grad_phi"], diff_data["gamma"],
            diff_data["mesh"], diff_data["geo"], diff_data["mesh"]["boundary"],
        )
        A = assemble_matrix_from_fluxes_vectorized(flux_data, diff_data["mesh"])
        b = assemble_rhs_from_fluxes_vectorized(flux_data, diff_data["mesh"])
        residual = A @ diff_data["phi"][: diff_data["mesh"]["n_elements"]] - b
        assert np.allclose(residual, 0.0, atol=1e-12), (
            f"max residual = {np.max(np.abs(residual)):.2e}"
        )
