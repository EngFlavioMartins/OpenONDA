"""Convection numerical-dissipation gate — the quantitative "less diffusive" test.

The Taylor–Green field ``u = (sin πx cos πy, −cos πx sin πy, 0)`` is
divergence-free and has **zero normal velocity on every face of the unit box**
(``u·n̂ = 0`` on x=0,1 and y=0,1).  Therefore the exact convective energy budget

    ∫_Ω u·(u·∇)u dV = ½ ∮_∂Ω |u|² (u·n̂) dS = 0

is identically zero: a *discretely energy-conserving* convection scheme must
reproduce ``P ≈ 0`` to machine precision, while an upwind scheme injects
numerical diffusion and yields ``P < 0`` (kinetic energy is destroyed).

We assemble the convective contribution to ``du/dt`` on a structured orthogonal
mesh and form the global energy-production rate ``P = Σ_c u_c·(du/dt|_conv)_c V_c``.
This is the headline acceptance gate for the Phase-2 energy-conserving scheme and
the baseline number for the existing schemes.
"""

import numpy as np

from source.solvers.FVM.assemble.convection import assemble_convection_term, compute_mass_flow_rate
from source.solvers.FVM.assemble.matrix_assembly import (
    assemble_matrix_from_fluxes_vectorized,
    assemble_rhs_from_fluxes_vectorized,
)
from source.solvers.FVM.fields.gradients import compute_gradient_gauss_linear_vectorized
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry

from ._structured_mesh import structured_box

PI = np.pi


def _tgv(x, y):
    return np.column_stack(
        [np.sin(PI * x) * np.cos(PI * y), -np.cos(PI * x) * np.sin(PI * y), np.zeros_like(x)]
    )


def _setup():
    mesh = structured_box(24, 24, 1)
    geo = compute_mesh_geometry(mesh)
    n_elem = mesh["n_elements"]
    n_int = mesh["n_interior_faces"]
    cc, fc = geo["element_centroids"], geo["face_centroids"]

    U = np.zeros((n_elem + mesh["n_faces"] - n_int, 3))
    U[:n_elem] = _tgv(cc[:, 0], cc[:, 1])
    for b in mesh["boundary"]:
        b["bc_type"] = "fixedValue"
        for j in range(b["nFaces"]):
            fi = b["startFace"] + j
            gi = n_elem + (fi - n_int)
            U[gi] = _tgv(np.array([fc[fi, 0]]), np.array([fc[fi, 1]])).ravel()
    return mesh, geo, U


def _energy_production(mesh, geo, U, scheme):
    """P = Σ_c u_c·(du/dt|_conv)_c V_c, with du/dt|_conv = (−A_conv u + b_conv)/V."""
    n_elem = mesh["n_elements"]
    vol = geo["element_volumes"]
    mdot = compute_mass_flow_rate(U, mesh, geo)
    P = 0.0
    for i in range(3):
        u_comp = U[:, i]
        grad = compute_gradient_gauss_linear_vectorized(u_comp, mesh, geo)[:, :, 0]
        conv = assemble_convection_term(
            u_comp, mdot, mesh, geo, mesh["boundary"], scheme=scheme, grad_phi=grad
        )
        A = assemble_matrix_from_fluxes_vectorized(conv, mesh)
        b = assemble_rhs_from_fluxes_vectorized(conv, mesh)
        dudt = (-(A @ u_comp[:n_elem]) + b) / vol
        P += float(np.sum(u_comp[:n_elem] * dudt * vol))
    return P


class TestConvectionDissipation:
    def test_tgv_zero_boundary_flux(self):
        """Precondition: the TGV normal velocity vanishes on all box faces."""
        mesh, geo, U = _setup()
        mdot = compute_mass_flow_rate(U, mesh, geo)
        n_int = mesh["n_interior_faces"]
        assert np.abs(mdot[n_int:]).max() < 1e-12

    def test_central_conserves_upwind_dissipates(self):
        mesh, geo, U = _setup()
        n_elem = mesh["n_elements"]
        ke = 0.5 * float(np.sum(np.sum(U[:n_elem] ** 2, axis=1) * geo["element_volumes"]))

        p_upwind = _energy_production(mesh, geo, U, "upwind") / ke
        p_central = _energy_production(mesh, geo, U, "central") / ke
        p_deferred = _energy_production(mesh, geo, U, "deferred") / ke

        # Upwind is strongly dissipative (destroys kinetic energy).
        assert p_upwind < -0.05, f"upwind P/KE={p_upwind:.3e} not dissipative"
        # Central and deferred conserve energy to machine precision (exact, since
        # u·n̂ = 0 on the boundary removes the surface term).
        assert abs(p_central) < 1e-10, f"central P/KE={p_central:.3e} not conservative"
        assert abs(p_deferred) < 1e-10, f"deferred P/KE={p_deferred:.3e} not conservative"
        # The "less diffusive" claim, quantified.
        assert abs(p_central) < 1e-6 * abs(p_upwind)

    def test_tvd_family_far_less_dissipative_than_upwind(self):
        """The bounded high-resolution (TVD) and blended schemes must all be
        dramatically less dissipative than upwind on the smooth TGV field."""
        mesh, geo, U = _setup()
        n_elem = mesh["n_elements"]
        ke = 0.5 * float(np.sum(np.sum(U[:n_elem] ** 2, axis=1) * geo["element_volumes"]))
        p_upwind = _energy_production(mesh, geo, U, "upwind") / ke

        for scheme in ("limitedLinear", "vanLeer", "MUSCL", "minmod", "LUST"):
            p = _energy_production(mesh, geo, U, scheme) / ke
            # Dissipative (P<=0) but at least 3x less so than upwind on a smooth field.
            assert p <= 1e-12, f"{scheme} P/KE={p:.3e} should not produce energy"
            assert abs(p) < abs(p_upwind) / 3.0, (
                f"{scheme} P/KE={p:.3e} not far below upwind {p_upwind:.3e}"
            )
