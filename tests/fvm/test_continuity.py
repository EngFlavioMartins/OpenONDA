"""Continuity (incompressibility) diagnostic checks.

``compute_continuity_error`` returns the per-cell net face flux ∮U·dS.  On a
divergence-free field it must be ~0; on a known-divergence field it must equal
∫∇·U dV exactly (the discrete divergence of a linear field via face fluxes is
exact).
"""

import numpy as np

from source.solvers.FVM.assemble.convection import compute_mass_flow_rate
from source.solvers.FVM.fields.diagnostics import compute_continuity_error
from source.solvers.FVM.mesh.geometry import compute_mesh_geometry

from ._structured_mesh import structured_box


def _field(mesh, geo, fn):
    n_elem = mesh["n_elements"]
    n_int = mesh["n_interior_faces"]
    cc, fc = geo["element_centroids"], geo["face_centroids"]
    U = np.zeros((n_elem + mesh["n_faces"] - n_int, 3))
    U[:n_elem] = fn(cc[:, 0], cc[:, 1], cc[:, 2])
    for b in mesh["boundary"]:
        for j in range(b["nFaces"]):
            fi = b["startFace"] + j
            U[n_elem + (fi - n_int)] = fn(
                np.array([fc[fi, 0]]), np.array([fc[fi, 1]]), np.array([fc[fi, 2]])
            ).ravel()
    return U


def test_uniform_flow_is_divergence_free():
    mesh = structured_box(8, 8, 8)
    geo = compute_mesh_geometry(mesh)
    U = _field(
        mesh,
        geo,
        lambda x, y, z: np.column_stack(
            [np.ones_like(x), 0.3 * np.ones_like(x), -0.5 * np.ones_like(x)]
        ),
    )
    phi = compute_mass_flow_rate(U, mesh, geo)
    div = compute_continuity_error(phi, mesh, geo)
    assert np.max(np.abs(div)) < 1e-12, (
        f"uniform flow not divergence-free: {np.max(np.abs(div)):.2e}"
    )


def test_linear_field_recovers_known_divergence():
    """U = (x, 2y, 0) ⇒ ∇·U = 3; net flux per cell must equal 3·V."""
    mesh = structured_box(8, 8, 8)
    geo = compute_mesh_geometry(mesh)
    U = _field(mesh, geo, lambda x, y, z: np.column_stack([x, 2.0 * y, np.zeros_like(x)]))
    phi = compute_mass_flow_rate(U, mesh, geo)
    div = compute_continuity_error(phi, mesh, geo)
    local_div = div / geo["element_volumes"]
    assert np.allclose(local_div, 3.0, atol=1e-10), (
        f"max dev: {np.max(np.abs(local_div - 3.0)):.2e}"
    )
