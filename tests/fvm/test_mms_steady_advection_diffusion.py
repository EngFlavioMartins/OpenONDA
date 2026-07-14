import numpy as np
import pytest

from source.solvers.FVM.assemble.convection import assemble_convection_term, compute_mass_flow_rate
from source.solvers.FVM.assemble.diffusion import assemble_diffusion_term
from source.solvers.FVM.assemble.matrix_assembly import (
    assemble_matrix_from_fluxes_vectorized,
    assemble_rhs_from_fluxes_vectorized,
)
from source.solvers.FVM.fields.gradients import compute_gradient_gauss_linear_vectorized
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.solve.linear_interface import solve_linear_system


def _phi_exact(x, y, z):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)


def _grad_phi_exact(x, y, z):
    """∇φ = [∂φ/∂x, ∂φ/∂y, ∂φ/∂z]."""
    sx, sy, sz = np.sin(np.pi * x), np.sin(np.pi * y), np.sin(np.pi * z)
    cx, cy, cz = np.cos(np.pi * x), np.cos(np.pi * y), np.cos(np.pi * z)
    return np.column_stack(
        [
            np.pi * cx * sy * sz,
            np.pi * sx * cy * sz,
            np.pi * sx * sy * cz,
        ]
    )


def _source_adv_diff(x, y, z, U, nu):
    """∇·(Uφ) - ν∇²φ with φ = sin(πx)sin(πy)sin(πz)."""
    sx, sy, sz = np.sin(np.pi * x), np.sin(np.pi * y), np.sin(np.pi * z)
    cx, cy, cz = np.cos(np.pi * x), np.cos(np.pi * y), np.cos(np.pi * z)
    div_conv = (
        U[0] * np.pi * cx * sy * sz + U[1] * np.pi * sx * cy * sz + U[2] * np.pi * sx * sy * cz
    )
    lap = -3.0 * np.pi**2 * sx * sy * sz
    return div_conv - nu * lap


def _l2_error(computed, exact, volumes):
    diff = computed - exact
    return np.sqrt(np.sum(volumes * diff**2) / np.sum(volumes))


def _make_mms_mesh(lcar):
    gmsh = pytest.importorskip("gmsh", reason="Gmsh FVM test dependency is not installed")
    gmsh.initialize()
    try:
        model = gmsh.model
        model.add("mms_ad")
        model.occ.addBox(0, 0, 0, 1, 1, 1)
        model.occ.synchronize()
        model.mesh.setSize(model.getEntities(0), lcar)
        model.mesh.generate(3)
        from source.solvers.FVM.mesh.gmsh_importer import GmshImporter

        imp = GmshImporter()
        mesh = imp.get_mesh_data()
    finally:
        gmsh.finalize()
    return mesh


def _setup_dirichlet_bcs(mesh, geo):
    """Set φ = φ_exact on all boundary faces."""
    n_elem = mesh["n_elements"]
    n_int = mesh["n_interior_faces"]
    fc = geo["face_centroids"]
    phi = np.zeros(n_elem + mesh["n_faces"] - n_int)
    phi[:n_elem] = _phi_exact(
        geo["element_centroids"][:, 0],
        geo["element_centroids"][:, 1],
        geo["element_centroids"][:, 2],
    )
    for b in mesh["boundary"]:
        b["bc_type"] = "fixedValue"
        b["bc_type_U"] = "fixedValue"
        b["value_U"] = [1.0, 1.0, 1.0]
        start, nf = b["startFace"], b["nFaces"]
        for j in range(nf):
            fi = start + j
            gi = n_elem + (fi - n_int)
            phi[gi] = _phi_exact(fc[fi, 0], fc[fi, 1], fc[fi, 2])
    return phi


class TestMMSSteadyAdvectionDiffusion:
    """∇·(Uφ) - ν∇²φ = S on [0,1]³ with U=(1,1,1), ν=0.01."""

    @pytest.mark.slow
    def test_convergence_upwind(self):
        nu = 0.01
        U = np.array([1.0, 1.0, 1.0])

        errors = []
        h_vals = []
        for lcar in [0.5, 0.25, 0.125]:
            mesh = _make_mms_mesh(lcar)
            geo = compute_mesh_geometry(mesh)
            n_elem = mesh["n_elements"]
            n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]
            cents = geo["element_centroids"]

            phi = _setup_dirichlet_bcs(mesh, geo)
            vol = geo["element_volumes"]

            # Velocity field
            U_field = np.tile(U, (n_elem + n_bnd, 1))

            # Mass flow rate
            mdot = compute_mass_flow_rate(U_field, mesh, geo)

            # Diffusion
            grad_phi = compute_gradient_gauss_linear_vectorized(phi, mesh, geo)
            gamma = nu * np.ones(n_elem)
            diff_flux = assemble_diffusion_term(phi, grad_phi, gamma, mesh, geo, mesh["boundary"])

            # Convection (upwind)
            conv_flux = assemble_convection_term(
                phi, mdot, mesh, geo, mesh["boundary"], scheme="upwind"
            )

            # Combine fluxes
            flux_data = {
                "flux_cf": diff_flux["flux_cf"] + conv_flux["flux_cf"],
                "flux_ff": diff_flux["flux_ff"] + conv_flux["flux_ff"],
                "flux_vf": diff_flux["flux_vf"] + conv_flux["flux_vf"],
                "flux_tf": diff_flux["flux_tf"] + conv_flux["flux_tf"],
            }
            A = assemble_matrix_from_fluxes_vectorized(flux_data, mesh)
            b = assemble_rhs_from_fluxes_vectorized(flux_data, mesh)

            # The PDE is: ∇·(Uφ) - ν∇²φ = S.
            # A·φ = b represents the LHS operator (convection + diffusion).
            # So: A·φ = b_boundary + S·V → b_solved = b_assembled + S·V
            S = _source_adv_diff(cents[:, 0], cents[:, 1], cents[:, 2], U, nu)
            b += S * vol

            phi_sol = solve_linear_system(A, b, method="spsolve", equation_type="scalar")
            phi_exact = _phi_exact(cents[:, 0], cents[:, 1], cents[:, 2])
            err = _l2_error(phi_sol, phi_exact, vol)
            errors.append(err)
            h_vals.append(lcar)

        coeffs = np.polyfit(np.log(h_vals), np.log(errors), 1)
        p = coeffs[0]
        # Upwind is first-order, so p should be ~1
        assert p > 0.5, f"Observed order {p:.2f} < 0.5 (expected ~1 for upwind)"
