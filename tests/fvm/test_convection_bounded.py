import numpy as np
import pytest

from source.solvers.fvm.assemble.convection import (
    assemble_convection_term,
    compute_volumetric_face_flux,
)
from source.solvers.fvm.assemble.matrix_assembly import (
    assemble_matrix_from_fluxes_vectorized,
    assemble_rhs_from_fluxes_vectorized,
)
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry


@pytest.fixture
def conv_data(hand_built_3d_mesh):
    """Pre-compute mesh with uniform U=(1,0,0) for convection tests."""
    mesh = hand_built_3d_mesh
    geo = compute_mesh_geometry(mesh)
    n_elem = mesh["n_cells"]
    n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]

    for b in mesh["boundary"]:
        b["boundary_condition_type"] = "zeroGradient"

    # Uniform velocity field (interior + ghost)
    velocity = np.tile([1.0, 0.0, 0.0], (n_elem + n_bnd, 1))

    # Mass flow rate
    volumetric_face_flux = compute_volumetric_face_flux(velocity, mesh, geo)

    return {
        "mesh": mesh,
        "geo": geo,
        "velocity": velocity,
        "volumetric_face_flux": volumetric_face_flux,
    }


class TestConvectionBounded:
    """Convection matrix properties: M-matrix, boundedness."""

    def test_upwind_is_m_matrix(self, conv_data):
        """Upwind convection matrix has off-diag ≤ 0 and diag > 0 (M-matrix)."""
        n_elem = conv_data["mesh"]["n_cells"]
        n_bnd = conv_data["mesh"]["n_faces"] - conv_data["mesh"]["n_interior_faces"]
        scalar_field = np.zeros(n_elem + n_bnd)
        flux_data = assemble_convection_term(
            scalar_field,
            conv_data["volumetric_face_flux"],
            conv_data["mesh"],
            conv_data["geo"],
            conv_data["mesh"]["boundary"],
            scheme="upwind",
        )
        A = assemble_matrix_from_fluxes_vectorized(flux_data, conv_data["mesh"]).toarray()
        off_diag = A - np.diagflat(np.diag(A))
        assert np.all(off_diag <= 1e-12), "Upwind: off-diagonal entries must be ≤ 0"
        assert np.all(np.diag(A) > 0), "Upwind: diagonal entries must be > 0"

    def test_constant_field_zero_residual(self, conv_data):
        """If φ=const=1, convection RHS and matrix product should cancel."""
        n_elem = conv_data["mesh"]["n_cells"]
        n_bnd = conv_data["mesh"]["n_faces"] - conv_data["mesh"]["n_interior_faces"]
        scalar_field = np.ones(n_elem + n_bnd)
        flux_data = assemble_convection_term(
            scalar_field,
            conv_data["volumetric_face_flux"],
            conv_data["mesh"],
            conv_data["geo"],
            conv_data["mesh"]["boundary"],
            scheme="upwind",
        )
        A = assemble_matrix_from_fluxes_vectorized(flux_data, conv_data["mesh"])
        b = assemble_rhs_from_fluxes_vectorized(flux_data, conv_data["mesh"])
        residual = A @ np.ones(n_elem) - b
        assert np.allclose(residual, 0.0, atol=1e-12), (
            f"max residual = {np.max(np.abs(residual)):.2e}"
        )
