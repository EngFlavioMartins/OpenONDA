"""Boundedness of the high-resolution convection schemes.

A single explicit Euler transport step of a sharp step profile (front at
x = 0.5, advected by U = (1,0,0)) on a structured mesh.  Under CFL < 1 a *TVD*
scheme must not create new extrema — the updated field stays within the data
range [0, 1] — whereas pure central (``linear``) overshoots/undershoots (Gibbs).

This is the boundedness counterpart to ``test_convection_dissipation.py``: the
TVD schemes give the best of both — far less dissipative than upwind, yet
oscillation-free where central is not.
"""

import numpy as np
import pytest

from source.solvers.FVM.assemble.convection import assemble_convection_term, compute_mass_flow_rate
from source.solvers.FVM.assemble.matrix_assembly import (
    assemble_matrix_from_fluxes_vectorized,
    assemble_rhs_from_fluxes_vectorized,
)
from source.solvers.FVM.fields.gradients import compute_gradient_gauss_linear_vectorized
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry
from source.solvers.FVM.schemes.limiters import apply_limiter

from ._structured_mesh import structured_box

N = 20


def _setup():
    mesh = structured_box(N, N, 1)
    geo = compute_mesh_geometry(mesh)
    n_elem = mesh["n_elements"]
    n_int = mesh["n_interior_faces"]
    cc = geo["element_centroids"]

    # Uniform +x advection; ghosts included.
    U = np.tile([1.0, 0.0, 0.0], (n_elem + mesh["n_faces"] - n_int, 1)).astype(float)
    mdot = compute_mass_flow_rate(U, mesh, geo)

    # Step initial field: 1 upstream (x<0.5), 0 downstream.
    phi = np.zeros(n_elem + mesh["n_faces"] - n_int)
    phi[:n_elem] = (cc[:, 0] < 0.5).astype(float)
    # Boundary ghosts: inlet (xmin) holds the upstream value 1; all others
    # zero-gradient (copy owner).
    for b in mesh["boundary"]:
        own = mesh["owners"][b["startFace"] : b["startFace"] + b["nFaces"]]
        idx = n_elem + (b["startFace"] - n_int)
        if b["name"] == "xmin":
            b["bc_type"] = "fixedValue"
            phi[idx : idx + b["nFaces"]] = 1.0
        else:
            b["bc_type"] = "zeroGradient"
            phi[idx : idx + b["nFaces"]] = phi[own]
    return mesh, geo, U, mdot, phi


def _one_explicit_step(mesh, geo, mdot, phi, scheme, cfl=0.5):
    n_elem = mesh["n_elements"]
    vol = geo["element_volumes"]
    grad = compute_gradient_gauss_linear_vectorized(phi, mesh, geo)[:, :, 0]
    conv = assemble_convection_term(
        phi, mdot, mesh, geo, mesh["boundary"], scheme=scheme, grad_phi=grad
    )
    A = assemble_matrix_from_fluxes_vectorized(conv, mesh)
    b = assemble_rhs_from_fluxes_vectorized(conv, mesh)
    dudt = (-(A @ phi[:n_elem]) + b) / vol
    dt = cfl * (1.0 / N)  # |U| = 1, dx = 1/N
    return phi[:n_elem] + dt * dudt


class TestTVDBoundedness:
    TOL = 1e-9

    def test_named_limiters_keep_their_standard_compressive_range(self):
        ratio = np.array([0.0, 0.5, 1.0, 2.0, 10.0])
        assert apply_limiter("vanLeer", ratio)[3] == pytest.approx(4.0 / 3.0)
        assert apply_limiter("MUSCL", ratio)[3] == pytest.approx(1.5)
        assert apply_limiter("superbee", ratio)[3] == pytest.approx(2.0)

    def test_tvd_schemes_are_bounded(self):
        mesh, geo, U, mdot, phi = _setup()
        for scheme in ("upwind", "limitedLinear", "vanLeer", "MUSCL", "minmod"):
            phi_new = _one_explicit_step(mesh, geo, mdot, phi, scheme)
            assert phi_new.min() > -self.TOL, f"{scheme} undershoot: min={phi_new.min():.3e}"
            assert phi_new.max() < 1.0 + self.TOL, f"{scheme} overshoot: max={phi_new.max():.3e}"

    def test_central_overshoots(self):
        """Contrast: pure central (linear) is not monotone and produces
        over/undershoot on the same step — motivating the TVD family."""
        mesh, geo, U, mdot, phi = _setup()
        phi_new = _one_explicit_step(mesh, geo, mdot, phi, "central")
        overshoot = max(phi_new.max() - 1.0, -phi_new.min())
        assert overshoot > 1e-3, f"central unexpectedly bounded (overshoot={overshoot:.3e})"
