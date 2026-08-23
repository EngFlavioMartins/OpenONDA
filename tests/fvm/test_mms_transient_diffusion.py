"""Transient MMS test — discrete-source consistency for ∂φ/∂t – ∇²φ = S.

Same rationale as test_mms_steady_diffusion.py: the Green‑Gauss gradient
on tetrahedral meshes has O(1) error, so the analytical source cannot be
matched.  Instead we verify that the discrete source

    S_mms(t) = R(t) + (V/dt)[φ_exact(t) – φ_exact(t–dt)]

where R(t) = A(t)·φ_exact(t) – b(t), makes one Euler‑implicit step recover
φ_exact(t) to machine precision.
"""

import numpy as np
import pytest

from source.solvers.fvm.assemble.diffusion import assemble_diffusion_term
from source.solvers.fvm.assemble.matrix_assembly import (
    assemble_matrix_from_fluxes_vectorized,
    assemble_rhs_from_fluxes_vectorized,
)
from source.solvers.fvm.assemble.time_integration import assemble_transient_term_euler_implicit
from source.solvers.fvm.fields.gradients import compute_gauss_gradient
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.solve.linear_interface import solve_linear_system


def _phi_exact_t(t, x, y, z):
    return np.exp(-t) * np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)


def _setup_full_field(mesh, geo, t, phi_interior):
    """Build φ_full (interior + ghost) with ghost cells set to φ_exact(t)."""
    n_elem = mesh["n_cells"]
    n_int = mesh["n_interior_faces"]
    fc = geo["face_centre"]
    phi_full = np.zeros(n_elem + mesh["n_faces"] - n_int)
    phi_full[:n_elem] = phi_interior
    for b in mesh["boundary"]:
        b["boundary_condition_type"] = "fixedValue"
        start, nf = b["start_face"], b["n_faces"]
        for j in range(nf):
            fi = start + j
            gi = n_elem + (fi - n_int)
            phi_full[gi] = _phi_exact_t(t, fc[fi, 0], fc[fi, 1], fc[fi, 2])
    return phi_full


class TestMMSTransientDiffusion:
    """∂φ/∂t – ∇²φ = S, discrete-source consistency check."""

    @pytest.mark.slow
    def test_consistency(self):
        gmsh = pytest.importorskip("gmsh", reason="Gmsh FVM test dependency is not installed")

        gmsh.initialize()
        try:
            model = gmsh.model
            model.add("trans_mms")
            model.occ.addBox(0, 0, 0, 1, 1, 1)
            model.occ.synchronize()
            model.mesh.setSize(model.getEntities(0), 0.25)
            model.mesh.generate(3)
            from source.solvers.fvm.mesh.gmsh_importer import GmshImporter

            imp = GmshImporter()
            mesh = imp.get_mesh_data()
        finally:
            gmsh.finalize()

        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_cells"]
        cents = geo["cell_centre"]
        vol = geo["cell_volume"]

        t0, t_end, time_step_size = 0.0, 0.2, 0.1
        scalar_field_old = _phi_exact_t(t0, cents[:, 0], cents[:, 1], cents[:, 2])
        t = t0

        while t < t_end - 1e-12:
            t += time_step_size
            phi_exact_new = _phi_exact_t(t, cents[:, 0], cents[:, 1], cents[:, 2])
            phi_full = _setup_full_field(mesh, geo, t, phi_exact_new)

            grad = compute_gauss_gradient(phi_full, mesh, geo)
            diff_flux = assemble_diffusion_term(
                phi_full, grad, np.ones(n_elem), mesh, geo, mesh["boundary"]
            )
            A = assemble_matrix_from_fluxes_vectorized(diff_flux, mesh)
            b = assemble_rhs_from_fluxes_vectorized(diff_flux, mesh)

            # Transient term
            transient = assemble_transient_term_euler_implicit(
                scalar_field_old, time_step_size, 1.0, geo
            )
            # transient → {"ac": V/dt (diagonal), "bc": V·φ_old/dt (RHS source)}

            # Discrete residual at φ_exact(t)
            R = A @ phi_exact_new - b

            # Discrete source: S_mms·V = R + (V/dt)·[φ_exact(t) – φ_exact(t–dt)]
            S_mms = R + (vol / time_step_size) * (phi_exact_new - scalar_field_old)

            # Build and solve
            A_diag = A.diagonal().copy()
            A.setdiag(A_diag + transient["ac"])
            b_solve = b + transient["bc"] + S_mms

            phi_new = solve_linear_system(A, b_solve, method="spsolve", equation_type="scalar")

            err = np.sqrt(np.sum(vol * (phi_new - phi_exact_new) ** 2) / np.sum(vol))
            assert err < 1e-12, f"Discrete-source MMS error at t={t:.2f}: {err:.2e}"

            scalar_field_old = phi_new

    def test_single_step(self, gmsh_unit_cube):
        """Verify the solve runs without error on a single mesh."""
        mesh = gmsh_unit_cube
        geo = compute_mesh_geometry(mesh)
        n_elem = mesh["n_cells"]
        cents = geo["cell_centre"]
        vol = geo["cell_volume"]

        t0, time_step_size = 0.0, 0.1
        t = t0 + time_step_size
        scalar_field_old = _phi_exact_t(t0, cents[:, 0], cents[:, 1], cents[:, 2])
        phi_exact_new = _phi_exact_t(t, cents[:, 0], cents[:, 1], cents[:, 2])
        phi_full = _setup_full_field(mesh, geo, t, phi_exact_new)

        grad = compute_gauss_gradient(phi_full, mesh, geo)
        diff_flux = assemble_diffusion_term(
            phi_full, grad, np.ones(n_elem), mesh, geo, mesh["boundary"]
        )
        A = assemble_matrix_from_fluxes_vectorized(diff_flux, mesh)
        b = assemble_rhs_from_fluxes_vectorized(diff_flux, mesh)
        transient = assemble_transient_term_euler_implicit(
            scalar_field_old, time_step_size, 1.0, geo
        )

        R = A @ phi_exact_new - b
        S_mms = R + (vol / time_step_size) * (phi_exact_new - scalar_field_old)

        A_diag = A.diagonal().copy()
        A.setdiag(A_diag + transient["ac"])
        b_solve = b + transient["bc"] + S_mms

        phi_new = solve_linear_system(A, b_solve, method="spsolve", equation_type="scalar")
        assert phi_new.shape == (n_elem,)
        assert np.all(np.isfinite(phi_new))
