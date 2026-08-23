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

from source.solvers.fvm.assemble.convection import (
    assemble_convection_term,
    compute_volumetric_face_flux,
)
from source.solvers.fvm.assemble.matrix_assembly import (
    assemble_matrix_from_fluxes_vectorized,
    assemble_rhs_from_fluxes_vectorized,
)
from source.solvers.fvm.fields.gradients import compute_gauss_gradient
from source.solvers.fvm.mesh.geometry import compute_mesh_geometry
from source.solvers.fvm.schemes.limiters import apply_limiter

from ._structured_mesh import structured_box

N = 20


def _setup():
    mesh = structured_box(N, N, 1)
    geo = compute_mesh_geometry(mesh)
    n_elem = mesh["n_cells"]
    n_int = mesh["n_interior_faces"]
    cc = geo["cell_centre"]

    # Uniform +x advection; ghosts included.
    velocity = np.tile([1.0, 0.0, 0.0], (n_elem + mesh["n_faces"] - n_int, 1)).astype(float)
    volumetric_face_flux = compute_volumetric_face_flux(velocity, mesh, geo)

    # Step initial field: 1 upstream (x<0.5), 0 downstream.
    scalar_field = np.zeros(n_elem + mesh["n_faces"] - n_int)
    scalar_field[:n_elem] = (cc[:, 0] < 0.5).astype(float)
    # Boundary ghosts: inlet (xmin) holds the upstream value 1; all others
    # zero-gradient (copy owner).
    for b in mesh["boundary"]:
        own = mesh["owners"][b["start_face"] : b["start_face"] + b["n_faces"]]
        idx = n_elem + (b["start_face"] - n_int)
        if b["name"] == "xmin":
            b["boundary_condition_type"] = "fixedValue"
            scalar_field[idx : idx + b["n_faces"]] = 1.0
        else:
            b["boundary_condition_type"] = "zeroGradient"
            scalar_field[idx : idx + b["n_faces"]] = scalar_field[own]
    return mesh, geo, velocity, volumetric_face_flux, scalar_field


def _one_explicit_step(mesh, geo, volumetric_face_flux, scalar_field, scheme, courant_number=0.5):
    n_elem = mesh["n_cells"]
    vol = geo["cell_volume"]
    grad = compute_gauss_gradient(scalar_field, mesh, geo)[:, :, 0]
    conv = assemble_convection_term(
        scalar_field,
        volumetric_face_flux,
        mesh,
        geo,
        mesh["boundary"],
        scheme=scheme,
        scalar_field_gradient=grad,
    )
    A = assemble_matrix_from_fluxes_vectorized(conv, mesh)
    b = assemble_rhs_from_fluxes_vectorized(conv, mesh)
    dudt = (-(A @ scalar_field[:n_elem]) + b) / vol
    time_step_size = courant_number * (1.0 / N)  # |U| = 1, dx = 1/N
    return scalar_field[:n_elem] + time_step_size * dudt


class TestTVDBoundedness:
    TOL = 1e-9

    def test_named_limiters_keep_their_standard_compressive_range(self):
        ratio = np.array([0.0, 0.5, 1.0, 2.0, 10.0])
        assert apply_limiter("vanLeer", ratio)[3] == pytest.approx(4.0 / 3.0)
        assert apply_limiter("MUSCL", ratio)[3] == pytest.approx(1.5)
        assert apply_limiter("superbee", ratio)[3] == pytest.approx(2.0)

    def test_tvd_schemes_are_bounded(self):
        mesh, geo, velocity, volumetric_face_flux, scalar_field = _setup()
        for scheme in ("upwind", "limitedLinear", "vanLeer", "MUSCL", "minmod"):
            scalar_field_new = _one_explicit_step(
                mesh, geo, volumetric_face_flux, scalar_field, scheme
            )
            assert scalar_field_new.min() > -self.TOL, (
                f"{scheme} undershoot: min={scalar_field_new.min():.3e}"
            )
            assert scalar_field_new.max() < 1.0 + self.TOL, (
                f"{scheme} overshoot: max={scalar_field_new.max():.3e}"
            )

    def test_central_overshoots(self):
        """Contrast: pure central (linear) is not monotone and produces
        over/undershoot on the same step — motivating the TVD family."""
        mesh, geo, velocity, volumetric_face_flux, scalar_field = _setup()
        scalar_field_new = _one_explicit_step(
            mesh, geo, volumetric_face_flux, scalar_field, "central"
        )
        overshoot = max(scalar_field_new.max() - 1.0, -scalar_field_new.min())
        assert overshoot > 1e-3, f"central unexpectedly bounded (overshoot={overshoot:.3e})"
