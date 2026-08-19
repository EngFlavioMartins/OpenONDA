"""Method-of-Manufactured-Solutions verification of the discrete *momentum
operator* (convection + diffusion + volumetric source), independent of the
pressure–velocity coupling.

This pins the spatial accuracy of ``assemble_momentum_equation`` and, in doing
so, exercises the generic ``source_explicit`` hook used by both MMS forcing and
the future coupling blending source ``S = λ(Utarget − U)``.

Manufactured field (divergence-free 2D Taylor–Green, uniform in z, on [0,1]³)::

    u = (  sin(πx) cos(πy),  −cos(πx) sin(πy),  0 )

For ∇·u = 0 the convective term reduces to (u·∇)u, and with p ≡ 0 the steady
momentum operator ``L(u)_i = ∇·(u u_i) − ν∇²u_i`` is balanced by the analytic
source::

    S_x = (π/2) sin(2πx) + 2νπ² sin(πx) cos(πy)
    S_y = (π/2) sin(2πy) − 2νπ² cos(πx) sin(πy)
    S_z = 0

Solving ``A·u = b + S·V`` per component must recover u_exact; the L2 error must
converge under mesh refinement.  We characterise both ``upwind`` (1st order,
diffusive) and ``deferred`` (2nd order, less diffusive) and assert that the
deferred scheme is the more accurate of the two — the "less diffusive" claim in
its most basic form.
"""

import numpy as np

from source.solvers.FVM.assemble.convection import compute_volumetric_face_flux
from source.solvers.FVM.assemble.momentum import assemble_momentum_equation
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.solve.linear_interface import solve_linear_system

from ._structured_mesh import structured_box

PI = np.pi


def _u_exact(x, y, z):
    ux = np.sin(PI * x) * np.cos(PI * y)
    uy = -np.cos(PI * x) * np.sin(PI * y)
    uz = np.zeros_like(x)
    return np.column_stack([ux, uy, uz])


def _momentum_source(x, y, z, nu):
    sx = 0.5 * PI * np.sin(2 * PI * x) + 2.0 * nu * PI**2 * np.sin(PI * x) * np.cos(PI * y)
    sy = 0.5 * PI * np.sin(2 * PI * y) - 2.0 * nu * PI**2 * np.cos(PI * x) * np.sin(PI * y)
    sz = np.zeros_like(x)
    return np.column_stack([sx, sy, sz])


def _l2_vector(computed, exact, volumes):
    diff = computed - exact
    return np.sqrt(np.sum(volumes[:, None] * diff**2) / np.sum(volumes))


def _setup_exact_field(mesh, geo):
    """Cell-centre + boundary-ghost velocity set to u_exact (Dirichlet)."""
    n_elem = mesh["n_elements"]
    n_int = mesh["n_interior_faces"]
    n_bnd = mesh["n_faces"] - n_int
    cc = geo["element_centroids"]
    fc = geo["face_centroids"]

    U = np.zeros((n_elem + n_bnd, 3))
    U[:n_elem] = _u_exact(cc[:, 0], cc[:, 1], cc[:, 2])
    for b in mesh["boundary"]:
        b["bc_type"] = "fixedValue"
        b["bc_type_velocity"] = "fixedValue"
        b["value_velocity"] = [0.0, 0.0, 0.0]  # unused: ghosts below carry the exact value
        start, nf = b["startFace"], b["nFaces"]
        for j in range(nf):
            fi = start + j
            gi = n_elem + (fi - n_int)
            U[gi] = _u_exact(
                np.array([fc[fi, 0]]), np.array([fc[fi, 1]]), np.array([fc[fi, 2]])
            ).ravel()
    return U


def _solve_momentum_operator(mesh, geo, nu, scheme):
    """Assemble + solve the steady momentum operator with the manufactured
    source; return (U_solution[n_elem, 3], U_exact[n_elem, 3], volumes)."""
    n_elem = mesh["n_elements"]
    n_bnd = mesh["n_faces"] - mesh["n_interior_faces"]
    cc = geo["element_centroids"]

    U = _setup_exact_field(mesh, geo)
    p = np.zeros(n_elem + n_bnd)
    phi = compute_volumetric_face_flux(U, mesh, geo)
    S = _momentum_source(cc[:, 0], cc[:, 1], cc[:, 2], nu)

    mom = assemble_momentum_equation(
        U,
        p,
        phi,
        1.0,
        nu,
        mesh,
        geo,
        mesh["boundary"],
        convection_scheme=scheme,
        time_step_size=None,
        source_explicit=S,
    )

    U_sol = np.zeros((n_elem, 3))
    for i, comp in enumerate(["x", "y", "z"]):
        A = mom[comp]["A"]
        b = mom[comp]["b"]
        U_sol[:, i] = solve_linear_system(A, b, method="spsolve", equation_type="momentum")

    U_exact = _u_exact(cc[:, 0], cc[:, 1], cc[:, 2])
    return U_sol, U_exact, geo["element_volumes"]


def _observed_order(errors, h_vals):
    return float(np.polyfit(np.log(h_vals), np.log(errors), 1)[0])


class TestMomentumOperatorMMS:
    NU = 0.05

    def test_source_hook_recovers_field(self):
        """Fast sanity: on a single mesh the operator + manufactured source
        recovers u_exact to within discretisation error."""
        mesh = structured_box(8, 8, 8)
        geo = compute_mesh_geometry(mesh)
        U_sol, U_exact, vol = _solve_momentum_operator(mesh, geo, self.NU, "deferred")
        err = _l2_vector(U_sol, U_exact, vol)
        u_ref = _l2_vector(U_exact, 0.0 * U_exact, vol)
        assert err / u_ref < 0.05, f"relative L2 error {err / u_ref:.3f} too large"

    def test_convergence_and_relative_diffusivity(self):
        """On a structured orthogonal hex mesh (non-orthogonal correction ≡ 0)
        the deferred/central correction must reach its design 2nd order while
        upwind stays ~1st — the basic 'less diffusive' verification."""
        ns = [8, 16, 32]
        h_vals = [1.0 / n for n in ns]
        results = {}
        for scheme in ("upwind", "deferred"):
            errors = []
            for n in ns:
                mesh = structured_box(n, n, n)
                geo = compute_mesh_geometry(mesh)
                U_sol, U_exact, vol = _solve_momentum_operator(mesh, geo, self.NU, scheme)
                errors.append(_l2_vector(U_sol, U_exact, vol))
            results[scheme] = (errors, _observed_order(errors, h_vals))

        upwind_err, upwind_p = results["upwind"]
        deferred_err, deferred_p = results["deferred"]

        assert upwind_p > 0.85, f"upwind order {upwind_p:.2f} < 0.85"
        assert deferred_p > 1.8, f"deferred order {deferred_p:.2f} < 1.8 (design 2nd order)"
        # Headline "less diffusive": the 2nd-order correction beats upwind.
        assert deferred_err[-1] < 0.25 * upwind_err[-1], (
            f"deferred finest error {deferred_err[-1]:.3e} not << upwind {upwind_err[-1]:.3e}"
        )
